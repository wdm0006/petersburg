"""
Behavioral tests for the scikit-learn-style estimators: FrequencyEstimator and
MixedModeEstimator. These exercise the real fit/partial_fit/predict paths (building
and simulating an actual petersburg Graph) on small deterministic datasets.
"""

import random
import unittest

import numpy as np

from petersburg import FrequencyEstimator, MixedModeEstimator

__author__ = "willmcginnis"


def _terminal_indices(categories):
    """Indices into ``categories`` for the last-layer (terminal) nodes."""
    last_layer = max(layer for layer, _ in categories)
    return {idx for idx, (layer, _) in enumerate(categories) if layer == last_layer}


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
        # Every row transitions 0 -> 0, so the only reachable terminal is (1, 0).
        y = np.array([[0, 0], [0, 0], [0, 0]])
        X = np.zeros((3, 2))
        est = FrequencyEstimator(num_simulations=5).fit(X, y)

        y_hat = est.predict(X)
        terminal = est._categories.index((1, 0))
        self.assertEqual(y_hat.shape, (3, 1))
        self.assertTrue((y_hat == terminal).all())

    def test_predict_output_shape_and_valid_labels(self):
        # A branching graph: predictions must be valid terminal category indices.
        y = np.array([[0, 0], [0, 0], [0, 1], [0, 1]])
        X = np.zeros((4, 2))
        est = FrequencyEstimator(num_simulations=25).fit(X, y)

        y_hat = est.predict(X)
        self.assertEqual(y_hat.shape, (4, 1))
        valid = _terminal_indices(est._categories)
        self.assertTrue(set(y_hat.ravel().astype(int)).issubset(valid))


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
        terminal = est._categories.index((1, 0))
        self.assertEqual(y_hat.shape, (3, 1))
        self.assertTrue((y_hat == terminal).all())


class TestMixedModeEstimatorClassifierBacked(unittest.TestCase):
    """At/above the sample threshold, transitions get feature-dependent classifiers."""

    def _training_data(self):
        # Source (0,0) transitions to two terminals with a clean feature split:
        #   150 rows -> (1,0) with feature near 0
        #   100 rows -> (1,1) with feature near 10  (exactly the 100-sample threshold)
        np.random.seed(0)
        rows, feats = [], []
        for _ in range(150):
            rows.append([0, 0])
            feats.append([np.random.normal(0.0, 0.5)])
        for _ in range(100):
            rows.append([0, 1])
            feats.append([np.random.normal(10.0, 0.5)])
        return np.array(feats), np.array(rows)

    def test_classifier_trained_at_threshold(self):
        X, y = self._training_data()
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
        X, y = self._training_data()
        est = MixedModeEstimator(num_simulations=25).fit(X, y)

        i10 = est._categories.index((1, 0))
        i11 = est._categories.index((1, 1))

        # A low-feature row routes to the low-feature terminal; a high one to the other.
        random.seed(1)
        np.random.seed(1)
        self.assertEqual(int(est.predict(np.array([[0.0]]))[0, 0]), i10)
        self.assertEqual(int(est.predict(np.array([[10.0]]))[0, 0]), i11)


if __name__ == "__main__":
    unittest.main()
