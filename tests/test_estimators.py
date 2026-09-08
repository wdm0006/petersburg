"""
Behavioral tests for the scikit-learn-style estimators: FrequencyEstimator and
MixedModeEstimator. These exercise the real fit/partial_fit/predict paths (building
and simulating an actual petersburg Graph) on small deterministic datasets.
"""

import os
import pickle
import random
import subprocess
import sys
import textwrap
import unittest
from functools import partial
from unittest import mock

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from petersburg import FrequencyEstimator, MixedModeEstimator
from petersburg import graph as pg

__author__ = "willmcginnis"


def _classifier_training_data():
    """
    Training data with enough support for MixedModeEstimator to train real classifiers.

    Source (0,0) transitions to two terminals with a clean feature split:
      150 rows -> (1,0) with feature near 0
      100 rows -> (1,1) with feature near 10  (exactly the 100-sample threshold)
    """
    np.random.seed(0)
    rows, feats = [], []
    for _ in range(150):
        rows.append([0, 0])
        feats.append([np.random.normal(0.0, 0.5)])
    for _ in range(100):
        rows.append([0, 1])
        feats.append([np.random.normal(10.0, 0.5)])
    return np.array(feats), np.array(rows)


def _terminal_labels(categories):
    """Fitted labels of the last-layer (terminal) categories."""
    last_layer = max(layer for layer, _ in categories)
    return {value for layer, value in categories if layer == last_layer}


class TestFrequencyEstimatorFit(unittest.TestCase):
    """FrequencyEstimator.fit learns categories and transition counts."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_fit_learns_categories(self):
        # Two layers: column 0 is the single root (always 0), column 1 branches.
        y = np.array([[0, 0], [0, 0], [0, 1]])
        X = np.zeros((3, 2))

        est = FrequencyEstimator()
        self.assertIs(est.fit(X, y), est)

        # One category per (layer, value): (0,0) in layer 0; (1,0) and (1,1) in layer 1.
        self.assertEqual(set(est._categories), {(0, 0), (1, 0), (1, 1)})
        self.assertEqual(est._frequency_matrix.shape, (3, 3))

    def test_fit_counts_transitions(self):
        y = np.array([[0, 0], [0, 0], [0, 1]])
        X = np.zeros((3, 2))

        est = FrequencyEstimator().fit(X, y)

        i00 = est._categories.index((0, 0))
        i10 = est._categories.index((1, 0))
        i11 = est._categories.index((1, 1))

        # Two rows went 0 -> 0 and one went 0 -> 1.
        self.assertEqual(est._frequency_matrix[i00, i10], 2)
        self.assertEqual(est._frequency_matrix[i00, i11], 1)
        # Terminal (layer-1) categories have no outgoing transitions.
        self.assertEqual(est._frequency_matrix[i10].sum(), 0)
        self.assertEqual(est._frequency_matrix[i11].sum(), 0)


class TestFrequencyEstimatorPartialFit(unittest.TestCase):
    """partial_fit incrementally updates an already-fitted model."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_partial_fit_updates_existing_counts(self):
        y = np.array([[0, 0], [0, 0], [0, 1]])
        X = np.zeros((3, 2))
        est = FrequencyEstimator().fit(X, y)

        i00 = est._categories.index((0, 0))
        i11 = est._categories.index((1, 1))
        before = est._frequency_matrix[i00, i11]

        # Feed two more 0 -> 1 rows; the existing count must grow by exactly two.
        est.partial_fit(np.zeros((2, 2)), np.array([[0, 1], [0, 1]]))
        self.assertEqual(est._frequency_matrix[i00, i11], before + 2)

    def test_partial_fit_without_fit_falls_back_to_fit(self):
        # With no existing model, partial_fit builds one from scratch.
        y = np.array([[0, 0], [0, 1]])
        est = FrequencyEstimator()
        est.partial_fit(np.zeros((2, 2)), y)
        self.assertEqual(set(est._categories), {(0, 0), (1, 0), (1, 1)})

    def test_partial_fit_unseen_category_raises(self):
        y = np.array([["a", "x"], ["a", "y"]])
        est = FrequencyEstimator().fit(np.zeros((2, 2)), y)
        with self.assertRaises(ValueError) as ctx:
            est.partial_fit(np.zeros((1, 2)), np.array([["b", "x"]]))

        message = str(ctx.exception)
        self.assertIn("FrequencyEstimator", message)
        self.assertIn("'b'", message)
        self.assertIn("column 0", message)
        self.assertIn("categories seen during fit", message)
        self.assertNotIn("is not in list", message)


class TestFrequencyEstimatorPredict(unittest.TestCase):
    """predict builds a real Graph and simulates terminal outcomes."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_predict_single_terminal_is_deterministic(self):
        # Every row transitions 0 -> 0, so the only reachable terminal is (1, 0) and
        # predict must return that category's fitted label, not its index.
        y = np.array([[0, 0], [0, 0], [0, 0]])
        X = np.zeros((3, 2))
        est = FrequencyEstimator(num_simulations=5).fit(X, y)

        y_hat = est.predict(X)
        self.assertEqual(y_hat.shape, (3,))
        self.assertTrue((y_hat == 0).all())

    def test_predict_builds_graph_from_non_negative_count_matrix(self):
        y = np.array([[0, 0], [0, 0], [0, 1]])
        X = np.zeros((3, 2))
        est = FrequencyEstimator(num_simulations=1).fit(X, y)

        original = pg.Graph.from_adj_matrix
        with mock.patch.object(pg.Graph, "from_adj_matrix", autospec=True) as build:
            build.side_effect = original
            predictions = est.predict(X)

        build.assert_called_once()
        matrix = build.call_args.args[1]
        self.assertTrue((matrix >= 0).all())
        self.assertTrue(set(predictions.ravel()).issubset({0, 1}))

    def test_predict_output_shape_and_valid_labels(self):
        # A branching graph: predictions must be labels of terminal categories.
        y = np.array([[0, 0], [0, 0], [0, 1], [0, 1]])
        X = np.zeros((4, 2))
        est = FrequencyEstimator(num_simulations=25).fit(X, y)

        y_hat = est.predict(X)
        self.assertEqual(y_hat.shape, (4,))
        self.assertTrue(set(y_hat.ravel()).issubset(_terminal_labels(est._categories)))

    def test_predict_round_trips_string_labels(self):
        # Non-numeric labels must survive predict; a float y_hat could not hold them.
        y = np.array([["apply", "approved"]] * 3)
        X = np.zeros((3, 2))
        est = FrequencyEstimator(num_simulations=5).fit(X, y)

        y_hat = est.predict(X)
        self.assertEqual(y_hat.dtype, object)
        self.assertEqual(y_hat.ravel().tolist(), ["approved", "approved", "approved"])

    def test_predict_rejects_terminal_id_outside_the_category_range(self):
        # The synthetic root -1 injected by from_adj_matrix is not a category index;
        # indexing _categories with it would silently return the LAST category.
        y = np.array([["apply", "approved"], ["apply", "rejected"]])
        X = np.zeros((2, 2))
        est = FrequencyEstimator(num_simulations=5).fit(X, y)

        for node_id in (-1, len(est._categories)):
            with self.subTest(node_id=node_id):
                with mock.patch.object(pg.Graph, "get_outcome_node", return_value=node_id):
                    with self.assertRaises(ValueError) as ctx:
                        est.predict(X)
                self.assertIn(str(node_id), str(ctx.exception))


class TestMixedModeEstimatorFrequencyFallback(unittest.TestCase):
    """Below the classifier threshold, MixedModeEstimator behaves frequency-only."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_no_classifiers_trained_below_threshold(self):
        y = np.array([[0, 0], [0, 0], [0, 1]])
        X = np.array([[0.0], [0.1], [9.0]])
        est = MixedModeEstimator(num_simulations=5).fit(X, y)

        # Not enough samples for any transition -> the clf matrix is entirely empty.
        self.assertTrue(all(clf is None for row in est._clf_matrix for clf in row))

    def test_predict_falls_back_to_frequency(self):
        # Single deterministic terminal, so the frequency-only path is exact.
        y = np.array([[0, 0], [0, 0], [0, 0]])
        X = np.zeros((3, 1))
        est = MixedModeEstimator(num_simulations=5).fit(X, y)

        y_hat = est.predict(X)
        self.assertEqual(y_hat.shape, (3,))
        self.assertTrue((y_hat == 0).all())


class TestMixedModeEstimatorClassifierBacked(unittest.TestCase):
    """At/above the sample threshold, transitions get feature-dependent classifiers."""

    def test_single_class_transition_keeps_frequency_fallback(self):
        X = np.zeros((100, 1))
        y = np.array([["apply", "approved"]] * 100)

        est = MixedModeEstimator().fit(X, y)

        source = est._categories.index((0, "apply"))
        terminal = est._categories.index((1, "approved"))
        self.assertEqual(est._frequency_matrix[source, terminal], est.min_samples)
        self.assertIsNone(est._clf_matrix[source][terminal])

    def test_classifier_training_value_error_propagates(self):
        X = np.zeros((240, 1))
        X[0, 0] = np.nan
        y = np.array([["apply", "approved"]] * 120 + [["apply", "rejected"]] * 120)

        with self.assertRaises(ValueError):
            MixedModeEstimator().fit(X, y)

    def test_classifier_trained_at_threshold(self):
        X, y = _classifier_training_data()
        est = MixedModeEstimator(num_simulations=25).fit(X, y)

        i00 = est._categories.index((0, 0))
        i10 = est._categories.index((1, 0))
        i11 = est._categories.index((1, 1))

        # Both transitions meet the >= min_samples (100) threshold, so both are modeled.
        self.assertEqual(est.min_samples, 100)
        self.assertIsNotNone(est._clf_matrix[i00][i10])
        self.assertIsNotNone(est._clf_matrix[i00][i11])
        # The terminal nodes never originate a transition, so they train nothing.
        self.assertTrue(all(clf is None for clf in est._clf_matrix[i10]))
        self.assertTrue(all(clf is None for clf in est._clf_matrix[i11]))

    def test_prediction_is_feature_dependent(self):
        X, y = _classifier_training_data()
        est = MixedModeEstimator(num_simulations=25).fit(X, y)

        # A low-feature row routes to the low-feature terminal; a high one to the other,
        # and each prediction is that terminal's fitted label.
        random.seed(1)
        np.random.seed(1)
        self.assertEqual(est.predict(np.array([[0.0]]))[0], 0)
        self.assertEqual(est.predict(np.array([[10.0]]))[0], 1)

    def test_predict_returns_string_label_and_rejects_the_synthetic_root(self):
        y = np.array([["apply", "approved"]] * 3)
        X = np.zeros((3, 1))
        est = MixedModeEstimator(num_simulations=5).fit(X, y)

        self.assertEqual(est.predict(X).ravel().tolist(), ["approved"] * 3)

        with mock.patch.object(pg.Graph, "get_outcome_node", return_value=-1):
            with self.assertRaises(ValueError):
                est.predict(X)


class TestEstimatorScore(unittest.TestCase):
    """score compares predictions with the terminal column of a path target."""

    ESTIMATORS = (FrequencyEstimator, MixedModeEstimator)

    def test_terminal_label_accuracy(self):
        X = np.zeros((3, 1))
        y = np.array([["apply", "approved"]] * 3)

        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=1).fit(X, y)

                self.assertEqual(est.score(X, y), 1.0)
                incorrect = np.array([["apply", "rejected"]] * 3)
                self.assertEqual(est.score(X, incorrect), 0.0)

                y_hat = est.predict(X)
                self.assertEqual(y_hat.shape, (3,))
                self.assertEqual(y_hat.dtype, object)
                self.assertEqual(y_hat.ravel().tolist(), ["approved"] * 3)

    def test_score_rejects_invalid_path_target_shape(self):
        X = np.zeros((3, 1))
        y = np.array([["apply", "approved"]] * 3)

        for cls in self.ESTIMATORS:
            est = cls(num_simulations=1).fit(X, y)
            for invalid_y in (y[:, -1], np.empty((3, 0))):
                with self.subTest(estimator=cls.__name__, shape=invalid_y.shape):
                    with self.assertRaisesRegex(ValueError, "non-empty 2D path target"):
                        est.score(X, invalid_y)


class TestEstimatorPathTargetValidation(unittest.TestCase):
    """fit and partial_fit reject targets that cannot encode a layer-to-layer transition."""

    ESTIMATORS = (FrequencyEstimator, MixedModeEstimator)

    # (case name, malformed target, expected message fragment)
    MALFORMED = (
        ("one_dimensional", np.array(["approved", "rejected", "approved"]), "2D path target"),
        ("zero_rows", np.empty((0, 2), dtype=object), "no rows"),
        ("zero_columns", np.empty((3, 0), dtype=object), "at least two decision layers"),
        (
            "single_column",
            np.array([["apply"], ["apply"], ["decline"]]),
            "at least two decision layers",
        ),
    )

    def _valid_data(self):
        X = np.zeros((3, 1))
        y = np.array([["apply", "approved"], ["apply", "rejected"], ["apply", "approved"]])
        return X, y

    def test_fit_rejects_malformed_path_targets(self):
        X = np.zeros((3, 1))

        for cls in self.ESTIMATORS:
            for case, bad_y, fragment in self.MALFORMED:
                with self.subTest(estimator=cls.__name__, case=case):
                    with self.assertRaises(ValueError) as ctx:
                        cls(num_simulations=1).fit(X, bad_y)

                    message = str(ctx.exception)
                    self.assertIn(cls.__name__, message)
                    self.assertIn(fragment, message)

    def test_partial_fit_rejects_malformed_path_targets_before_and_after_fit(self):
        X, y = self._valid_data()

        for case, bad_y, fragment in self.MALFORMED:
            # Before any fit, partial_fit would otherwise delegate straight to fit.
            with self.subTest(case=case, fitted=False):
                with self.assertRaises(ValueError) as ctx:
                    FrequencyEstimator(num_simulations=1).partial_fit(X, bad_y)
                self.assertIn("FrequencyEstimator", str(ctx.exception))
                self.assertIn(fragment, str(ctx.exception))

            with self.subTest(case=case, fitted=True):
                est = FrequencyEstimator(num_simulations=1).fit(X, y)
                with self.assertRaises(ValueError) as ctx:
                    est.partial_fit(X, bad_y)
                self.assertIn(fragment, str(ctx.exception))

    def test_rejected_training_input_leaves_fitted_state_unchanged(self):
        X, y = self._valid_data()

        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=1).fit(X, y)
                categories = list(est._categories)
                frequencies = est._frequency_matrix.copy()
                clf_matrix = None if cls is FrequencyEstimator else list(est._clf_matrix)

                for _, bad_y, _ in self.MALFORMED:
                    with self.assertRaises(ValueError):
                        est.fit(X, bad_y)
                    if cls is FrequencyEstimator:
                        with self.assertRaises(ValueError):
                            est.partial_fit(X, bad_y)

                self.assertEqual(est._categories, categories)
                self.assertTrue(np.array_equal(est._frequency_matrix, frequencies))
                if clf_matrix is not None:
                    self.assertEqual(est._clf_matrix, clf_matrix)

                # The still-fitted model keeps predicting its terminal labels.
                predicted = set(est.predict(X).ravel().tolist())
                self.assertTrue(predicted <= _terminal_labels(est._categories))
                self.assertTrue(predicted)

    def test_ordinary_two_layer_target_fits_and_predicts_terminal_labels(self):
        X, y = self._valid_data()
        terminals = {"approved", "rejected"}

        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=5).fit(X, y)

                self.assertEqual(_terminal_labels(est._categories), terminals)

                y_hat = est.predict(X)
                self.assertEqual(y_hat.shape, (3,))
                self.assertTrue(set(y_hat.ravel().tolist()) <= terminals)

    def test_list_of_lists_target_is_accepted(self):
        X = np.zeros((3, 1))
        y = [["apply", "approved"], ["apply", "rejected"], ["apply", "approved"]]

        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=1).fit(X, y)
                self.assertEqual(_terminal_labels(est._categories), {"approved", "rejected"})

    def test_mixed_mode_trains_classifiers_from_a_list_target(self):
        # The converted target has to reach the classifier-training filter too, which is the
        # only part of MixedModeEstimator.fit that indexes y outside _update_frequencies.
        X, y = _classifier_training_data()

        est = MixedModeEstimator(num_simulations=1).fit(X, y.tolist())

        trained = [clf for row in est._clf_matrix for clf in row if clf is not None]
        self.assertTrue(trained)


class TestEstimatorFeatureMatrixValidation(unittest.TestCase):
    """predict, score, and MixedModeEstimator.fit coerce and shape-check the feature matrix."""

    ESTIMATORS = (FrequencyEstimator, MixedModeEstimator)

    # (case name, malformed feature matrix, expected dimension count)
    MALFORMED = (
        ("one_dimensional_array", np.array([1.0, 2.0, 3.0]), 1),
        ("one_dimensional_list", [1.0, 2.0, 3.0], 1),
        ("scalar", 1.0, 0),
        ("three_dimensional", np.zeros((3, 1, 1)), 3),
    )

    def _valid_data(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([["a", "x"], ["a", "y"], ["b", "x"]])
        return X, y

    def test_predict_and_score_accept_a_list_of_lists_feature_matrix(self):
        X, y = self._valid_data()

        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=5, random_state=7).fit(X, y)

                expected = est.predict(X)
                predicted = est.predict(X.tolist())

                self.assertEqual(predicted.shape, (3,))
                self.assertEqual(predicted.dtype, object)
                self.assertEqual(predicted.ravel().tolist(), expected.ravel().tolist())
                self.assertEqual(est.score(X.tolist(), y), est.score(X, y))

    def test_predict_rejects_a_feature_matrix_that_is_not_2d(self):
        X, y = self._valid_data()

        for cls in self.ESTIMATORS:
            est = cls(num_simulations=5).fit(X, y)
            for case, bad_X, ndim in self.MALFORMED:
                with self.subTest(estimator=cls.__name__, case=case):
                    with self.assertRaises(ValueError) as ctx:
                        est.predict(bad_X)

                    message = str(ctx.exception)
                    self.assertIn(cls.__name__, message)
                    self.assertIn("2D feature matrix", message)
                    self.assertIn(f"{ndim} dimension(s)", message)

    def test_unfitted_estimator_still_reports_the_fitted_state_first(self):
        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                with self.assertRaises(NotFittedError):
                    cls(num_simulations=5).predict(np.array([1.0, 2.0, 3.0]))

    def test_mixed_mode_fit_trains_the_same_classifiers_from_a_list_feature_matrix(self):
        # The classifier-training loop only runs once a transition reaches _min_samples, so
        # this is the size at which fit actually masks X with a boolean array derived from y.
        X, y = _classifier_training_data()

        from_array = MixedModeEstimator(num_simulations=1).fit(X, y)
        from_list = MixedModeEstimator(num_simulations=1).fit(X.tolist(), y)

        def trained(est):
            return [
                (r, c)
                for r, row in enumerate(est._clf_matrix)
                for c, clf in enumerate(row)
                if clf is not None
            ]

        self.assertEqual(len(trained(from_array)), 2)
        self.assertEqual(trained(from_list), trained(from_array))

    def test_mixed_mode_fit_rejects_mismatched_row_counts(self):
        # Below _min_samples the mismatch used to fit silently; above it, it died in the
        # classifier-training mask. Both sizes must report the same rule.
        X, y = self._valid_data()
        big_X, big_y = _classifier_training_data()

        for case, bad_X, bad_y in (
            ("below_threshold", X[:1], y),
            ("at_threshold", big_X[:100], big_y),
        ):
            with self.subTest(case=case):
                with self.assertRaises(ValueError) as ctx:
                    MixedModeEstimator(num_simulations=1).fit(bad_X, bad_y)

                message = str(ctx.exception)
                self.assertIn("MixedModeEstimator", message)
                self.assertIn(f"X has {bad_X.shape[0]} row(s)", message)
                self.assertIn(f"y has {bad_y.shape[0]} row(s)", message)

    def test_mixed_mode_rejected_feature_matrix_leaves_fitted_state_unchanged(self):
        X, y = self._valid_data()
        est = MixedModeEstimator(num_simulations=1).fit(X, y)
        categories = list(est._categories)
        frequencies = est._frequency_matrix.copy()

        with self.assertRaises(ValueError):
            est.fit(np.array([1.0, 2.0, 3.0]), y)
        with self.assertRaises(ValueError):
            est.fit(X[:1], y)

        self.assertEqual(est._categories, categories)
        self.assertTrue(np.array_equal(est._frequency_matrix, frequencies))

    def test_frequency_estimator_still_ignores_x(self):
        # fit documents X as ignored, so None must keep working there and in partial_fit.
        _, y = self._valid_data()

        est = FrequencyEstimator(num_simulations=1).fit(None, y)
        self.assertEqual(_terminal_labels(est._categories), {"x", "y"})

        est.partial_fit(None, y)
        self.assertEqual(est._frequency_matrix.sum(), 6)


class TestEstimatorFittedState(unittest.TestCase):
    """predict and score report an estimator that has not been fitted."""

    ESTIMATORS = (FrequencyEstimator, MixedModeEstimator)

    def test_unfitted_predict_raises_not_fitted_error(self):
        X = np.zeros((1, 1))
        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                with self.assertRaises(NotFittedError) as ctx:
                    cls().predict(X)
                self.assertIn(cls.__name__, str(ctx.exception))
                self.assertIn("fit", str(ctx.exception))

    def test_unfitted_score_raises_not_fitted_error(self):
        X = np.zeros((1, 1))
        y = np.array([["start", "end"]])
        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                with self.assertRaises(NotFittedError) as ctx:
                    cls().score(X, y)
                self.assertIn(cls.__name__, str(ctx.exception))
                self.assertIn("fit", str(ctx.exception))


class TestCategoryOrderDeterminism(unittest.TestCase):
    """Category indices must depend on the data alone, never on set iteration order."""

    LABELS = ["approved", "rejected", "delayed", "appealed", "withdrawn"]

    def _string_labelled_y(self):
        return np.array([["apply", self.LABELS[idx % len(self.LABELS)]] for idx in range(20)])

    def test_categories_follow_first_appearance_order(self):
        y = self._string_labelled_y()
        expected = [(0, "apply")] + [(1, label) for label in self.LABELS]

        self.assertEqual(FrequencyEstimator().fit(np.zeros((20, 2)), y)._categories, expected)
        self.assertEqual(MixedModeEstimator().fit(np.zeros((20, 1)), y)._categories, expected)

    def test_fresh_processes_agree_on_category_order(self):
        # str hashing is randomized per process, so iterating a set of string labels gives
        # each label a different index in each interpreter run. Only a fresh process shows it.
        script_source = f"""
            import numpy as np
            from petersburg.estimators import FrequencyEstimator

            labels = {self.LABELS!r}
            y = np.array([["apply", labels[idx % len(labels)]] for idx in range(20)])
            est = FrequencyEstimator().fit(np.zeros((20, 2)), y)
            print(est._categories)
            """
        script = textwrap.dedent(script_source)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        outputs = set()
        for _ in range(3):
            proc = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, cwd=repo_root
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.add(proc.stdout)

        self.assertEqual(len(outputs), 1)
        expected = [(0, "apply")] + [(1, label) for label in self.LABELS]
        self.assertEqual(outputs.pop().strip(), str(expected))


class TestEstimatorReproducibility(unittest.TestCase):
    """random_state pins predict without touching the process-global np.random stream."""

    ESTIMATORS = (FrequencyEstimator, MixedModeEstimator)
    TERMINALS = ["approved", "rejected", "delayed", "appealed"]

    def _fixture(self):
        # One source category branching to four near-equally weighted terminals, so a
        # majority vote over num_simulations walks genuinely varies when it is unseeded.
        y = np.array([["apply", self.TERMINALS[idx % len(self.TERMINALS)]] for idx in range(20)])
        return np.zeros((20, 1)), y, np.zeros((12, 1))

    def _predict(self, est, X_predict):
        return est.predict(X_predict).ravel().tolist()

    def test_random_state_is_stored_verbatim_and_survives_clone(self):
        generator = np.random.default_rng(3)
        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                self.assertIsNone(cls().get_params()["random_state"])
                self.assertEqual(cls(random_state=7).get_params()["random_state"], 7)
                self.assertEqual(clone(cls(random_state=7)).get_params()["random_state"], 7)

                # Not normalized into a Generator in __init__, or get_params/clone would
                # hand back something the constructor was never called with.
                self.assertIs(cls(random_state=generator).random_state, generator)

    def test_predict_is_repeatable_with_a_seed(self):
        X_fit, y, X_predict = self._fixture()
        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                est = cls(random_state=7).fit(X_fit, y)
                first = self._predict(est, X_predict)

                # Every call rebuilds the graph from the same seed, so the answer is stable.
                for _ in range(5):
                    self.assertEqual(self._predict(est, X_predict), first)

    def test_predict_varies_without_a_seed(self):
        # Non-vacuity check for the test above: this fixture really does move around.
        X_fit, y, X_predict = self._fixture()
        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                random.seed(42)
                np.random.seed(42)
                est = cls().fit(X_fit, y)
                results = {tuple(self._predict(est, X_predict)) for _ in range(5)}

                self.assertGreaterEqual(len(results), 2)

    def test_same_seed_agrees_and_different_seeds_differ(self):
        X_fit, y, X_predict = self._fixture()
        for cls in self.ESTIMATORS:
            with self.subTest(estimator=cls.__name__):
                same = [
                    self._predict(cls(random_state=7).fit(X_fit, y), X_predict) for _ in range(2)
                ]
                other = self._predict(cls(random_state=8).fit(X_fit, y), X_predict)

                self.assertEqual(same[0], same[1])
                self.assertNotEqual(same[0], other)

    def test_mixed_mode_seeds_its_classifiers(self):
        # A fitted model has to be reproducible end to end, not just at simulation time.
        X, y = _classifier_training_data()
        est = MixedModeEstimator(random_state=7).fit(X, y)

        trained = [clf for row in est._clf_matrix for clf in row if clf is not None]
        self.assertTrue(trained)
        self.assertTrue(all(clf.random_state == 7 for clf in trained))

        # The seed is threaded in for the call only; clf_args stays as __init__ set it.
        self.assertIsNone(est.clf_args)

    def test_mixed_mode_keeps_an_explicit_classifier_seed(self):
        X, y = _classifier_training_data()
        est = MixedModeEstimator(random_state=7, clf_args={"random_state": 99})
        est.fit(X, y)

        trained = [clf for row in est._clf_matrix for clf in row if clf is not None]
        self.assertTrue(trained)
        self.assertTrue(all(clf.random_state == 99 for clf in trained))

    def test_mixed_mode_does_not_hand_a_generator_to_its_classifiers(self):
        # LogisticRegression accepts None, an int, or a RandomState -- never a Generator.
        X, y = _classifier_training_data()
        est = MixedModeEstimator(random_state=np.random.default_rng(7)).fit(X, y)

        trained = [clf for row in est._clf_matrix for clf in row if clf is not None]
        self.assertTrue(trained)
        self.assertTrue(all(clf.random_state is None for clf in trained))


if __name__ == "__main__":
    unittest.main()


class TestEstimatorSimulationCountValidation(unittest.TestCase):
    """A fitted estimator rejects a non-positive num_simulations before it simulates."""

    NON_POSITIVE = (0, -1, -50)

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        self.X = np.array([[1.0], [2.0], [3.0]])
        self.y = np.array([["apply", "win"], ["apply", "lose"], ["apply", "win"]])

    def _fitted(self, cls, num_simulations):
        return cls(num_simulations=num_simulations).fit(self.X, self.y)

    def test_non_positive_num_simulations_is_rejected(self):
        for cls in (FrequencyEstimator, MixedModeEstimator):
            for num_simulations in self.NON_POSITIVE:
                estimator = self._fitted(cls, num_simulations)

                with self.assertRaises(ValueError) as ctx:
                    estimator.predict(self.X)

                message = str(ctx.exception)
                self.assertIn("num_simulations", message)
                self.assertIn(repr(num_simulations), message)

    def test_rejection_precedes_graph_construction(self):
        # Without the guard, predict builds the simulation graph and only fails later, at
        # most_common(1)[0] on an empty Counter.
        def unexpected_from_adj_matrix(*args, **kwargs):
            raise AssertionError("a graph was built before num_simulations was validated")

        for cls in (FrequencyEstimator, MixedModeEstimator):
            estimator = self._fitted(cls, 0)

            with mock.patch.object(pg.Graph, "from_adj_matrix", unexpected_from_adj_matrix):
                with self.assertRaises(ValueError):
                    estimator.predict(self.X)

    def test_unfitted_estimator_still_reports_the_fitted_state_first(self):
        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.assertRaises(NotFittedError):
                cls(num_simulations=0).predict(self.X)

    def test_score_inherits_the_rule(self):
        for cls in (FrequencyEstimator, MixedModeEstimator):
            estimator = self._fitted(cls, 0)

            with self.assertRaises(ValueError) as ctx:
                estimator.score(self.X, self.y)

            self.assertIn("num_simulations", str(ctx.exception))

    def test_non_integer_num_simulations_is_rejected(self):
        for cls in (FrequencyEstimator, MixedModeEstimator):
            for num_simulations in (10.5, 5.0, "10", None):
                estimator = self._fitted(cls, num_simulations)

                with self.assertRaises(ValueError) as ctx:
                    estimator.predict(self.X)

                self.assertIn("num_simulations", str(ctx.exception))

    def test_positive_num_simulations_is_unchanged(self):
        for cls in (FrequencyEstimator, MixedModeEstimator):
            estimator = self._fitted(cls, 5)

            predictions = estimator.predict(self.X)

            self.assertEqual(predictions.shape, (3,))
            self.assertEqual(predictions.dtype, object)
            self.assertTrue(set(predictions.ravel().tolist()) <= {"win", "lose"})


class TestPredictProba(unittest.TestCase):
    """predict_proba exposes the per-sample terminal distribution predict consumes."""

    def test_single_terminal_distribution_is_one_hot(self):
        X = np.zeros((3, 2))
        y = np.array([[0, 0], [0, 0], [0, 0]])

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                proba = cls(num_simulations=5).fit(X, y).predict_proba(X)

                self.assertEqual(proba.shape, (3, 1))
                self.assertTrue((proba == 1.0).all())

    def test_rows_sum_to_one_and_columns_match_classes(self):
        X = np.array([[0.0], [0.1], [9.0], [9.5]])
        y = np.array([["a", "x"], ["a", "x"], ["a", "y"], ["a", "y"]])

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=25, random_state=7).fit(X, y)

                proba = est.predict_proba(X)

                self.assertEqual(proba.shape, (4, len(est.classes_)))
                self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
                # Both terminals are reachable, so no column is dead.
                self.assertTrue((proba.min(axis=0) > 0.0).all())

    def test_predict_matches_the_distribution_argmax_under_a_seed(self):
        X = np.array([[0.0], [0.1], [9.0], [9.5]])
        y = np.array([["a", "x"], ["a", "x"], ["a", "y"], ["a", "y"]])

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=25, random_state=7).fit(X, y)

                argmax_labels = est.classes_[est.predict_proba(X).argmax(axis=1)]

                self.assertEqual(argmax_labels.tolist(), est.predict(X).tolist())

    def test_classifier_backed_distributions_are_feature_dependent(self):
        X, y = _classifier_training_data()
        est = MixedModeEstimator(num_simulations=50, random_state=7).fit(X, y)

        low = est.predict_proba(np.array([[0.0]]))[0]
        high = est.predict_proba(np.array([[10.0]]))[0]

        # classes_ orders the columns; the terminal labels here are ints 0 and 1.
        terminal_one = list(est.classes_).index(1)

        self.assertGreater(high[terminal_one], low[terminal_one])
        self.assertGreater(low.sum() - low[terminal_one], high.sum() - high[terminal_one])

    def test_predict_proba_requires_fitted_estimator(self):
        X = np.array([[0.0], [9.0]])

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                with self.assertRaises(NotFittedError):
                    cls().predict_proba(X)

    def test_predict_proba_validates_simulation_count_like_predict(self):
        X = np.array([[0.0], [9.0]])
        y = np.array([["a", "x"], ["a", "y"]])

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=0).fit(X, y)

                with self.assertRaises(ValueError) as ctx:
                    est.predict_proba(X)

                self.assertIn("num_simulations", str(ctx.exception))


class TestFittedAttributes(unittest.TestCase):
    """classes_ and n_features_in_ follow the scikit-learn fitted-attribute convention."""

    @staticmethod
    def _data():
        X = np.array([[0.0], [0.1], [9.0], [9.5]])
        return X, np.array([["a", "x"], ["a", "x"], ["a", "y"], ["a", "y"]])

    def test_classes_and_n_features_in_are_recorded_after_fit(self):
        X, y = self._data()

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=5).fit(X, y)

                self.assertEqual(list(est.classes_), ["x", "y"])
                self.assertEqual(est.n_features_in_, 1)

    def test_classes_follow_first_appearance_order(self):
        X = np.zeros((3, 1))
        y = np.array([["a", "late"], ["a", "early"], ["a", "early"]])

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=5).fit(X, y)

                self.assertEqual(list(est.classes_), ["late", "early"])

    def test_check_is_fitted_passes_after_fit_and_fails_before(self):
        X, y = self._data()

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=5)

                with self.assertRaises(NotFittedError):
                    check_is_fitted(est)

                est.fit(X, y)

                check_is_fitted(est)

    def test_fit_without_features_still_records_classes(self):
        y = np.array([["a", "x"], ["a", "x"], ["a", "y"]])

        est = FrequencyEstimator(num_simulations=5).fit(None, y)

        self.assertEqual(list(est.classes_), ["x", "y"])
        self.assertFalse(hasattr(est, "n_features_in_"))


class TestMixedModePartialFitParity(unittest.TestCase):
    """MixedModeEstimator.partial_fit matches FrequencyEstimator.partial_fit semantics."""

    def test_unfitted_partial_fit_delegates_to_fit(self):
        X = np.zeros((3, 1))
        y = np.array([["a", "x"], ["a", "x"], ["a", "y"]])

        est = MixedModeEstimator(num_simulations=5)
        est.partial_fit(X, y)

        self.assertEqual(_terminal_labels(est._categories), {"x", "y"})

    def test_partial_fit_updates_transition_counts_after_fit(self):
        X = np.zeros((3, 1))
        y = np.array([["a", "x"], ["a", "x"], ["a", "y"]])

        est = MixedModeEstimator(num_simulations=5).fit(X, y)
        est.partial_fit(X, y)

        # Three path transitions per pass, two passes.
        self.assertEqual(est._frequency_matrix.sum(), 6)

    def test_partial_fit_rejects_unseen_categories_like_frequency(self):
        X = np.zeros((3, 1))
        y = np.array([["a", "x"], ["a", "x"], ["a", "y"]])

        frequency = FrequencyEstimator(num_simulations=5).fit(X, y)
        mixed = MixedModeEstimator(num_simulations=5).fit(X, y)

        with self.assertRaises(ValueError) as frequency_ctx:
            frequency.partial_fit(X, np.array([["b", "x"]]))

        with self.assertRaises(ValueError) as mixed_ctx:
            mixed.partial_fit(X, np.array([["b", "x"]]))

        # Same rule, same message, up to the class name that prefixes it.
        self.assertIn("categories seen", str(frequency_ctx.exception))
        self.assertEqual(
            str(mixed_ctx.exception).replace("MixedModeEstimator", ""),
            str(frequency_ctx.exception).replace("FrequencyEstimator", ""),
        )

    def test_partial_fit_never_retrains_transition_classifiers(self):
        X, y = _classifier_training_data()
        random.seed(1)
        np.random.seed(1)

        est = MixedModeEstimator().fit(X, y)
        trained_before = [clf for row in est._clf_matrix for clf in row if clf is not None]
        self.assertTrue(trained_before)

        est.partial_fit(X, y)

        trained_after = [clf for row in est._clf_matrix for clf in row if clf is not None]

        self.assertTrue(all(a is b for a, b in zip(trained_before, trained_after)))


class TestSklearnHyperparameterContract(unittest.TestCase):
    """The MixedModeEstimator hyperparameters participate in the sklearn machinery."""

    def test_get_params_reports_the_constructor_parameters(self):
        params = MixedModeEstimator().get_params()

        self.assertEqual(
            set(params),
            {"verbose", "num_simulations", "random_state", "min_samples", "clf", "clf_args"},
        )
        self.assertEqual(params["min_samples"], 100)
        # sklearn's constructibility check forbids estimator instances as constructor
        # defaults, so None is stored verbatim and resolves to LogisticRegression() at
        # fit time (exercised in test_default_classifier_is_logistic_regression).
        self.assertIsNone(params["clf"])
        self.assertIsNone(params["clf_args"])

    def test_set_params_updates_public_parameters(self):
        est = MixedModeEstimator()

        self.assertIs(est.set_params(min_samples=25, clf_args={"max_iter": 10}), est)
        self.assertEqual(est.min_samples, 25)
        self.assertEqual(est.clf_args, {"max_iter": 10})

    def test_clone_round_trips_init_parameters(self):
        est = MixedModeEstimator(
            verbose=True,
            num_simulations=3,
            random_state=7,
            min_samples=42,
            clf=LogisticRegression(C=0.5),
            clf_args={"max_iter": 200},
        )

        cloned = clone(est)

        self.assertTrue(cloned.verbose)
        self.assertEqual(cloned.num_simulations, 3)
        self.assertEqual(cloned.random_state, 7)
        self.assertEqual(cloned.min_samples, 42)
        self.assertIsInstance(cloned.clf, LogisticRegression)
        self.assertEqual(cloned.clf.C, 0.5)
        self.assertIsNot(cloned.clf, est.clf)
        self.assertEqual(cloned.clf_args, {"max_iter": 200})

    def test_default_classifier_is_logistic_regression(self):
        X, y = _classifier_training_data()
        random.seed(1)
        np.random.seed(1)

        est = MixedModeEstimator().fit(X, y)
        trained = [clf for row in est._clf_matrix for clf in row if clf is not None]

        self.assertTrue(trained)
        self.assertTrue(all(isinstance(clf, LogisticRegression) for clf in trained))

    def test_fit_leaves_constructor_parameters_untouched(self):
        X, y = _classifier_training_data()
        est = MixedModeEstimator(num_simulations=5, random_state=7, min_samples=100).fit(X, y)

        self.assertEqual(est.num_simulations, 5)
        self.assertEqual(est.random_state, 7)
        self.assertEqual(est.min_samples, 100)
        self.assertIsNone(est.clf)
        self.assertIsNone(est.clf_args)

    def test_grid_search_tunes_min_samples(self):
        X, y = _classifier_training_data()
        search = GridSearchCV(
            MixedModeEstimator(num_simulations=5, random_state=7),
            {"min_samples": [50, 100]},
            cv=KFold(n_splits=2, shuffle=True, random_state=0),
        )

        search.fit(X, y)

        self.assertIn(search.best_params_["min_samples"], [50, 100])
        self.assertGreaterEqual(search.best_score_, 0.0)

    def test_grid_search_tunes_the_nested_classifier_penalty(self):
        X, y = _classifier_training_data()
        search = GridSearchCV(
            MixedModeEstimator(num_simulations=5, random_state=7, clf=LogisticRegression()),
            {"clf__C": [0.1, 10.0]},
            cv=KFold(n_splits=2, shuffle=True, random_state=0),
        )

        search.fit(X, y)

        self.assertIn(search.best_params_["clf__C"], [0.1, 10.0])
        self.assertGreaterEqual(search.best_score_, 0.0)

    def test_grid_search_can_replace_the_whole_classifier(self):
        # The blessed pattern for searching over the classifier itself when the
        # constructor default is None.
        X, y = _classifier_training_data()
        search = GridSearchCV(
            MixedModeEstimator(num_simulations=5, random_state=7),
            [{"clf": [LogisticRegression(C=0.1), LogisticRegression(C=10.0)]}],
            cv=KFold(n_splits=2, shuffle=True, random_state=0),
        )

        search.fit(X, y)

        self.assertIsNotNone(search.best_estimator_)
        self.assertGreaterEqual(search.best_score_, 0.0)

    def test_pipeline_fit_and_predict_smoke_test(self):
        X, y = _classifier_training_data()

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                pipe = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("est", cls(num_simulations=5, random_state=7)),
                    ]
                )
                pipe.fit(X, y)

                self.assertEqual(pipe.predict(X[:5]).shape, (5,))
                self.assertEqual(pipe.predict_proba(X[:5]).shape, (5, len(pipe.classes_)))

    def test_pipeline_can_override_nested_estimator_parameters(self):
        X, y = _classifier_training_data()
        pipe = Pipeline(
            [
                (
                    "est",
                    MixedModeEstimator(num_simulations=5, random_state=7, clf=LogisticRegression()),
                ),
            ]
        )
        pipe.set_params(est__clf__C=0.5)

        pipe.fit(X, y)

        self.assertEqual(pipe.predict(X[:3]).shape, (3,))


class TestSkippedCheckIntentCoverage(unittest.TestCase):
    """Path-target equivalents for the sklearn checks skipped on the 1-D y harness."""

    @staticmethod
    def _data():
        X = np.array([[0.0], [0.1], [9.0], [9.5]])
        return X, np.array([["a", "x"], ["a", "x"], ["a", "y"], ["a", "y"]])

    def test_pickle_round_trip_preserves_predictions(self):
        X, y = self._data()

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                est = cls(num_simulations=5, random_state=7).fit(X, y)
                restored = pickle.loads(pickle.dumps(est))

                np.testing.assert_array_equal(restored.predict(X), est.predict(X))

    def test_fit_is_idempotent_under_a_fixed_seed(self):
        X, y = self._data()

        for cls in (FrequencyEstimator, MixedModeEstimator):
            with self.subTest(estimator=cls.__name__):
                once = cls(num_simulations=5, random_state=7).fit(X, y)
                twice = cls(num_simulations=5, random_state=7).fit(X, y).fit(X, y)

                np.testing.assert_array_equal(twice.predict(X), once.predict(X))


# Written justifications for the estimator checks that cannot apply to these estimators.
# Their contract differs from generic scikit-learn classifiers in exactly three ways:
# fit consumes a 2D multi-column path target y (not a 1-D y); every y value is a
# discrete category, so any value -- float, string, or non-finite -- is a legal label;
# and features are read only through classifier-backed edges (FrequencyEstimator
# ignores X entirely). Where a check's intent survives those differences, an equivalent
# test built on real path targets lives in this module and is named in the reason.
_SKLEARN_CHECK_SKIP_REASONS = {
    "check_fit_score_takes_y": "the harness calls fit with a 1-D y, which the 2D path-target contract rejects before signatures are exercised; returning self from fit is exercised by every chained .fit(...) call in this module",
    "check_estimators_fit_returns_self": "the harness fits a 1-D y, rejected by the 2D path-target contract; returning self is exercised by every chained .fit(...) call in this module",
    "check_estimators_overwrite_params": "the harness fits a 1-D y, rejected by the 2D path-target contract; constructor parameters staying untouched is covered by test_fit_leaves_constructor_parameters_untouched",
    "check_dont_overwrite_parameters": "the harness fits a 1-D y, rejected by the 2D path-target contract; constructor parameters staying untouched is covered by test_fit_leaves_constructor_parameters_untouched",
    "check_dict_unchanged": "the harness fits a 1-D y, rejected by the 2D path-target contract; fit never mutates or retains caller inputs, and dict features are not read at all",
    "check_readonly_memmap_input": "the harness fits a read-only memmap alongside a 1-D y, which the 2D path-target contract rejects",
    "check_n_features_in_after_fitting": "the harness fits a 1-D y, rejected by the 2D path-target contract; n_features_in_ recording is covered by test_classes_and_n_features_in_are_recorded_after_fit",
    "check_n_features_in": "the harness fits a 1-D y, rejected by the 2D path-target contract; n_features_in_ recording is covered by test_classes_and_n_features_in_are_recorded_after_fit",
    "check_fit_check_is_fitted": "the harness fits a 1-D y, rejected by the 2D path-target contract; check_is_fitted behavior is covered by test_check_is_fitted_passes_after_fit_and_fails_before",
    "check_fit_idempotent": "the harness fits a 1-D y, rejected by the 2D path-target contract; refit determinism under a fixed seed is covered by test_fit_is_idempotent_under_a_fixed_seed",
    "check_estimators_pickle": "the harness fits a 1-D y, rejected by the 2D path-target contract; pickle round-trips are covered by test_pickle_round_trip_preserves_predictions",
    "check_pipeline_consistency": "the harness fits a 1-D y, rejected by the 2D path-target contract; Pipeline behavior with path targets is covered by test_pipeline_fit_and_predict_smoke_test",
    "check_estimators_dtypes": "the harness fits a 1-D y, rejected by the 2D path-target contract; feature dtypes only matter inside classifier-backed edges, which scikit-learn classifiers validate themselves",
    "check_dtype_object": "the harness fits a 1-D y, rejected by the 2D path-target contract; feature dtypes only matter inside classifier-backed edges, which scikit-learn classifiers validate themselves",
    "check_f_contiguous_array_estimator": "the harness fits a 1-D y, rejected by the 2D path-target contract; F-ordered inputs are converted with np.asarray before any use",
    "check_classifier_data_not_an_array": "the harness fits a 1-D y, rejected by the 2D path-target contract; list-of-lists features are covered by test_predict_and_score_accept_a_list_of_lists_feature_matrix",
    "check_classifiers_classes": "the harness fits a 1-D y, rejected by the 2D path-target contract; classes_ population is covered by test_classes_and_n_features_in_are_recorded_after_fit",
    "check_classifiers_train": "the harness fits a 1-D y, rejected by the 2D path-target contract; train-then-predict behavior is exercised by every predict-based test in this module",
    "check_classifiers_one_label": "the harness fits a 1-D y, rejected by the 2D path-target contract; single-terminal targets are covered by test_predict_single_terminal_is_deterministic",
    "check_supervised_y_2d": "the check requires supporting 1-D y before testing 2-D handling, but the path-target contract requires 2-D and rejects 1-D by design",
    "check_methods_sample_order_invariance": "the harness drives predict and score with a 1-D y, rejected by the 2D path-target contract",
    "check_methods_subset_invariance": "the harness drives predict and score with a 1-D y, rejected by the 2D path-target contract",
    "check_fit2d_predict1d": "the harness fits a 1-D y, rejected by the 2D path-target contract; rejecting 1-D feature matrices at predict is covered by test_predict_rejects_a_feature_matrix_that_is_not_2d",
    "check_requires_y_none": "the path-target contract requires y on every call, so y=None is rejected by design, and the harness also fits with a 1-D y in its remaining steps",
    "check_fit2d_1sample": "the harness's 1-D y is rejected before sample counts matter, and one-row path targets are themselves legal fits that must not raise",
    "check_fit2d_1feature": "the harness's 1-D y is rejected before feature counts matter, and single-feature path targets are themselves legal fits that must not raise",
    "check_positive_only_tag_during_fit": "the harness refits with negated 1-D y values, so the observed rejection is the path-target shape error rather than any positivity rule",
    "check_estimators_partial_fit_n_features": "the harness drives partial_fit with a 1-D y and a classes array, and the path-target contract rejects the 1-D y; the classes kwarg itself is accepted for API compatibility",
    "check_estimators_nan_inf": "path-target values are discrete categories, so non-finite y values are legal labels, and FrequencyEstimator never reads features; a blanket NaN/inf rejection would change documented behavior for inputs these estimators legitimately handle",
    "check_complex_data": "complex features are not validated: FrequencyEstimator ignores X entirely and MixedModeEstimator only reads features through classifier-backed edges, while the harness also fits with a 1-D y",
    "check_estimator_sparse_tag": "no sparse support is declared or implemented: FrequencyEstimator ignores X, MixedModeEstimator hands features to per-transition scikit-learn classifiers, and path targets are dense by construction",
    "check_estimator_sparse_array": "no sparse support is declared or implemented: FrequencyEstimator ignores X, MixedModeEstimator hands features to per-transition scikit-learn classifiers, and path targets are dense by construction",
    "check_estimator_sparse_matrix": "no sparse support is declared or implemented: FrequencyEstimator ignores X, MixedModeEstimator hands features to per-transition scikit-learn classifiers, and path targets are dense by construction",
    "check_estimators_empty_data_messages": "empty path targets are rejected with the estimators' own documented wording; the check requires scikit-learn's exact '0 feature(s)/0 sample(s)' patterns, and validation messages are owned by the parallel error-hierarchy change",
    "check_classifiers_regression_target": "path-target values are discrete categories by construction, so float values are legitimate labels rather than continuous targets and no 'Unknown label type' rejection exists",
}


def _estimator_checks(estimator):
    """Yield every applicable check as (estimator, check) across scikit-learn versions."""

    try:
        from sklearn.utils.estimator_checks import estimator_checks_generator

        yield from estimator_checks_generator(estimator)

    except ImportError:
        # scikit-learn < 1.6 exposes the same checks through a private generator whose
        # entries take (name, estimator); bind the name the same way the public
        # generator does.
        from sklearn.utils.estimator_checks import _yield_all_checks

        for check in _yield_all_checks(estimator):
            func = getattr(check, "func", check)
            if func is check:
                check = partial(check, type(estimator).__name__)

            yield estimator, check


def _estimator_check_params():
    """Build the parametrized check list with unique ids and justified skips."""

    params = []
    occurrences = {}

    for cls in (FrequencyEstimator, MixedModeEstimator):
        for _, check in _estimator_checks(cls()):
            name = check.func.__name__
            marks = []
            if name in _SKLEARN_CHECK_SKIP_REASONS:
                marks.append(pytest.mark.skip(reason=_SKLEARN_CHECK_SKIP_REASONS[name]))

            key = (cls.__name__, name)
            occurrences[key] = occurrences.get(key, 0) + 1
            suffix = f"[{occurrences[key]}]" if occurrences[key] > 1 else ""

            params.append(
                pytest.param(cls, check, id=f"{cls.__name__}-{name}{suffix}", marks=marks)
            )

    return params


@pytest.mark.parametrize("estimator_cls,check", _estimator_check_params())
def test_sklearn_estimator_checks(estimator_cls, check):
    check(estimator_cls())
