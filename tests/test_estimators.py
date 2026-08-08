"""
Behavioral tests for the scikit-learn-style estimators: FrequencyEstimator and
MixedModeEstimator. These exercise the real fit/partial_fit/predict paths (building
and simulating an actual petersburg Graph) on small deterministic datasets.
"""

import os
import random
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import numpy as np
from sklearn.base import clone

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
        # Categories are fixed at fit time; an unseen value has no index and
        # partial_fit surfaces a ValueError rather than silently updating.
        y = np.array([[0, 0], [0, 1]])
        est = FrequencyEstimator().fit(np.zeros((2, 2)), y)
        with self.assertRaises(ValueError):
            est.partial_fit(np.zeros((1, 2)), np.array([[0, 9]]))


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
        self.assertEqual(y_hat.shape, (3, 1))
        self.assertTrue((y_hat == 0).all())

    def test_predict_output_shape_and_valid_labels(self):
        # A branching graph: predictions must be labels of terminal categories.
        y = np.array([[0, 0], [0, 0], [0, 1], [0, 1]])
        X = np.zeros((4, 2))
        est = FrequencyEstimator(num_simulations=25).fit(X, y)

        y_hat = est.predict(X)
        self.assertEqual(y_hat.shape, (4, 1))
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
        self.assertEqual(y_hat.shape, (3, 1))
        self.assertTrue((y_hat == 0).all())


class TestMixedModeEstimatorClassifierBacked(unittest.TestCase):
    """At/above the sample threshold, transitions get feature-dependent classifiers."""

    def test_classifier_trained_at_threshold(self):
        X, y = _classifier_training_data()
        est = MixedModeEstimator(num_simulations=25).fit(X, y)

        i00 = est._categories.index((0, 0))
        i10 = est._categories.index((1, 0))
        i11 = est._categories.index((1, 1))

        # Both transitions meet the >= _min_samples (100) threshold, so both are modeled.
        self.assertEqual(est._min_samples, 100)
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
        self.assertEqual(est.predict(np.array([[0.0]]))[0, 0], 0)
        self.assertEqual(est.predict(np.array([[10.0]]))[0, 0], 1)

    def test_predict_returns_string_label_and_rejects_the_synthetic_root(self):
        y = np.array([["apply", "approved"]] * 3)
        X = np.zeros((3, 1))
        est = MixedModeEstimator(num_simulations=5).fit(X, y)

        self.assertEqual(est.predict(X).ravel().tolist(), ["approved"] * 3)

        with mock.patch.object(pg.Graph, "get_outcome_node", return_value=-1):
            with self.assertRaises(ValueError):
                est.predict(X)


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

        # The seed is threaded in for the call only; _clf_args is left as the class set it.
        self.assertEqual(est._clf_args, {})

    def test_mixed_mode_keeps_an_explicit_classifier_seed(self):
        X, y = _classifier_training_data()
        est = MixedModeEstimator(random_state=7)
        est._clf_args = {"random_state": 99}
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
