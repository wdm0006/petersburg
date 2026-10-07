"""Exact outcome variance, including branch costs and independent payoff uncertainty."""

import math
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from petersburg import Graph, Node, ValidationError


def _mixed_graph(seed=None):
    return Graph(random_state=seed).from_dict(
        {
            1: {"payoff": 5, "after": []},
            2: {
                "type": "uniform",
                "min_payoff": 2,
                "max_payoff": 8,
                "after": [{"node_id": 1, "cost": 1, "weight": 3}],
            },
            3: {
                "type": "gaussian",
                "mean": -2,
                "std": 2,
                "after": [{"node_id": 1, "cost": 3, "weight": 1}],
            },
            4: {
                "type": "lognormal",
                "mu": 0,
                "sigma": 0.5,
                "after": [{"node_id": 2, "cost": 2}, {"node_id": 3, "cost": 2}],
            },
            5: {
                "type": "powerlaw",
                "scale": 2,
                "alpha": 8,
                "after": [{"node_id": 4, "cost": 1}],
            },
        }
    )


class TestVariance(unittest.TestCase):
    def test_fixed_chain(self):
        g = Graph().from_dict(
            {
                1: {"payoff": 5, "after": []},
                2: {"payoff": -3, "after": [{"node_id": 1, "cost": 2}]},
                3: {"payoff": 11, "after": [{"node_id": 2, "cost": 4}]},
            }
        )
        self.assertEqual(g.variance(), 0.0)
        self.assertEqual(g.std(), 0.0)

    def test_branching_and_parallel_costs(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                root = Node(1, 5)
                left, right = Node(2, 10), Node(3, -2)
                root.add_outcome(left, cost=2, weight=3)
                root.add_outcome(left if parallel else right, cost=4, weight=1)
                g = Graph()
                g.start_node = root
                # Net outcomes are 13 and (11 or -1), with probabilities .75/.25.
                expected = 0.75 * 0.25 * (2 if parallel else 14) ** 2
                self.assertAlmostEqual(g.variance(), expected)
                self.assertAlmostEqual(g.std(), math.sqrt(expected))

    def test_mixing_every_node_type(self):
        # Branch means (after their first costs) are 4 and -5; variances 3 and 4.
        # The shared tail adds independent lognormal and Pareto variance.
        expected = (
            0.75 * 3
            + 0.25 * 4
            + 0.75 * 0.25 * 9**2
            + math.expm1(0.25) * math.exp(0.25)
            + 4 * 8 / (7**2 * 6)
        )
        self.assertAlmostEqual(_mixed_graph().variance(), expected)

    def test_sample_variance_across_seeds(self):
        exact = _mixed_graph().variance()
        for seed in range(5):
            with self.subTest(seed=seed):
                graph = _mixed_graph(seed)
                samples = [graph.get_outcome() for _ in range(30000)]
                self.assertAlmostEqual(np.var(samples, ddof=1), exact, delta=exact * 0.04)

    def test_memoized_converging_dag_and_no_sampling(self):
        graph = _mixed_graph(42)
        counts = Counter()
        original = graph._payoff_variance

        def record(node):
            counts[node.node_id] += 1
            return original(node)

        state = graph.rng.bit_generator.state
        with (
            patch.object(Graph, "_payoff_variance", side_effect=record),
            patch.object(
                Node, "sample_payoff", side_effect=AssertionError("unexpected simulation")
            ),
        ):
            graph.variance()
        self.assertEqual(counts, Counter(dict.fromkeys(range(1, 6), 1)))
        self.assertEqual(graph.rng.bit_generator.state, state)

    def test_unbuilt_and_classifier_weights(self):
        for method in ("variance", "std"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValidationError, "built graph"):
                    getattr(Graph(), method)()
                graph = _mixed_graph()
                edge, _ = graph.start_node.outcomes[0]
                graph.start_node.outcomes[0] = (edge, object())
                with self.assertRaisesRegex(ValidationError, "numeric transition weights"):
                    getattr(graph, method)()

    def test_infinite_powerlaw_variance(self):
        for alpha in (0.5, 1, 1.5, 2):
            with self.subTest(alpha=alpha):
                graph = Graph().from_dict(
                    {1: {"type": "powerlaw", "scale": 2, "alpha": alpha, "after": []}}
                )
                with self.assertRaisesRegex(ValidationError, "infinite variance"):
                    graph.variance()

    def test_zero_weight_infinite_variance_is_skipped(self):
        graph = Graph().from_dict(
            {
                1: {"payoff": 5, "after": []},
                2: {"type": "powerlaw", "alpha": 1, "after": [{"node_id": 1, "weight": 0}]},
                3: {"payoff": 10, "after": [{"node_id": 1}]},
            }
        )
        self.assertEqual(graph.variance(), 0)

    def test_invalid_weights(self):
        for weight in (-1, float("nan"), float("inf"), 0):
            with self.subTest(weight=weight):
                graph = Graph().from_dict(
                    {1: {"after": []}, 2: {"after": [{"node_id": 1, "weight": weight}]}}
                )
                with self.assertRaises(ValidationError):
                    graph.variance()

    def test_non_finite_moments_and_overflow(self):
        for spec in (
            {"payoff": float("inf")},
            {"payoff": float("nan")},
            {"payoff": 1e200},
            {"type": "lognormal", "mu": 400, "sigma": 1},
        ):
            with self.subTest(spec=spec):
                graph = Graph().from_dict({1: dict(spec, after=[])})
                with self.assertRaisesRegex(ValidationError, "non-finite"):
                    graph.variance()

    def test_deterministic_distribution_parameters(self):
        for spec in (
            {"type": "uniform", "min_payoff": 3, "max_payoff": 3},
            {"type": "gaussian", "mean": 3, "std": 0},
            {"type": "lognormal", "mu": 1, "sigma": 0},
        ):
            with self.subTest(spec=spec):
                self.assertEqual(Graph().from_dict({1: dict(spec, after=[])}).variance(), 0)
