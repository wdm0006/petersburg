"""
Tests for transition-weight validation in the edge-selection path.
"""

import unittest

import numpy as np

from petersburg import Graph, Node

__author__ = "willmcginnis"


class ConstantProbaClassifier:
    """Minimal stand-in for a fitted classifier, returning a fixed positive-class probability."""

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, feature_vector):
        return [[1 - self.probability, self.probability]]


class ZeroUniform:
    """RNG stub whose uniform draw is always the low end of the interval."""

    def uniform(self, low, high):
        return low


def _two_branch_node(first_weight, second_weight):
    """Node 1 with two outgoing edges, so weighted_choice has a real choice to make."""
    node = Node(node_id=1)
    node.add_outcome(Node(node_id=2, payoff=10), cost=0, weight=first_weight)
    node.add_outcome(Node(node_id=3, payoff=20), cost=0, weight=second_weight)
    return node


class TestInvalidTransitionWeights(unittest.TestCase):
    """Malformed weights raise a descriptive ValueError instead of being sampled from."""

    def setUp(self):
        np.random.seed(42)

    def test_negative_weight_is_rejected(self):
        node = _two_branch_node(2, -1)

        with self.assertRaises(ValueError) as ctx:
            node.weighted_choice()

        message = str(ctx.exception)
        self.assertIn("-1", message)
        self.assertIn("Node 1", message)
        self.assertIn("non-negative", message)

    def test_non_finite_weights_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            node = _two_branch_node(1, bad)

            with self.assertRaises(ValueError) as ctx:
                node.weighted_choice()

            message = str(ctx.exception)
            self.assertIn("finite", message)
            self.assertIn("Node 1", message)

    def test_all_zero_weights_are_rejected(self):
        node = _two_branch_node(0, 0)

        with self.assertRaises(ValueError) as ctx:
            node.weighted_choice()

        message = str(ctx.exception)
        self.assertIn("Node 1", message)
        self.assertIn("positive", message)

    def test_overflowing_total_is_rejected(self):
        # Each weight is finite, but their sum is not, so the sampling interval is unusable.
        node = _two_branch_node(1e308, 1e308)

        with self.assertRaises(ValueError) as ctx:
            node.weighted_choice()

        self.assertIn("finite", str(ctx.exception))

    def test_classifier_weights_get_the_same_validation(self):
        for bad in (float("nan"), float("inf"), -0.5):
            node = Node(node_id=1)
            node.add_outcome(Node(node_id=2, payoff=10), weight=1)
            node.add_outcome(Node(node_id=3, payoff=20), classifier=ConstantProbaClassifier(bad))

            with self.assertRaises(ValueError) as ctx:
                node.weighted_choice(feature_vector=[[0.0]])

            self.assertIn("Node 1", str(ctx.exception))

    def test_graph_simulation_rejects_invalid_weights(self):
        graph = Graph(random_state=7).from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 100, "after": [{"node_id": 1, "cost": 0, "weight": 2}]},
                3: {"payoff": 40, "after": [{"node_id": 1, "cost": 0, "weight": -1}]},
            }
        )

        with self.assertRaises(ValueError) as ctx:
            graph.get_outcome()

        self.assertIn("-1", str(ctx.exception))

        with self.assertRaises(ValueError):
            graph.get_outcome_node()


class TestValidTransitionWeights(unittest.TestCase):
    """Relative-weight semantics are preserved: zeros beside positives, and sums above one."""

    def setUp(self):
        np.random.seed(42)

    def test_zero_weight_beside_a_positive_weight_is_never_selected(self):
        # The zero edge comes first, so only the lowest possible draw can reach it; pin that
        # draw with a stub rng rather than relying on it never showing up in random samples.
        node = _two_branch_node(0, 5)
        node.rng = ZeroUniform()
        self.assertEqual(node.weighted_choice().to_node.node_id, 3)

        node = _two_branch_node(0, 5)
        selected = {node.weighted_choice().to_node.node_id for _ in range(500)}
        self.assertEqual(selected, {3})

    def test_zero_weight_after_a_positive_weight_is_never_selected(self):
        node = _two_branch_node(5, 0)
        node.rng = ZeroUniform()
        self.assertEqual(node.weighted_choice().to_node.node_id, 2)

        node = _two_branch_node(5, 0)
        selected = {node.weighted_choice().to_node.node_id for _ in range(500)}
        self.assertEqual(selected, {2})

    def test_relative_weights_above_one_are_supported(self):
        # 30/70 expressed as raw counts rather than probabilities.
        node = _two_branch_node(30, 70)
        node.rng = np.random.default_rng(11)

        selected = [node.weighted_choice().to_node.node_id for _ in range(4000)]
        share_of_2 = selected.count(2) / len(selected)
        self.assertAlmostEqual(share_of_2, 0.3, delta=0.05)

    def test_graph_simulation_accepts_a_zero_weight_branch(self):
        graph = Graph(random_state=3).from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 100, "after": [{"node_id": 1, "cost": 0, "weight": 0}]},
                3: {"payoff": 40, "after": [{"node_id": 1, "cost": 0, "weight": 7}]},
            }
        )

        self.assertEqual({graph.get_outcome_node() for _ in range(200)}, {3})


if __name__ == "__main__":
    unittest.main()
