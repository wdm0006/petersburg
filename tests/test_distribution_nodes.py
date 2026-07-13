"""
Tests for distribution-based node types.
"""

import math
import unittest
from unittest.mock import patch

import numpy as np

from petersburg import GaussianNode, Graph, LogNormalNode, PowerLawNode, UniformNode


class TestDistributionNodes(unittest.TestCase):
    """Test cases for stochastic node types."""

    def test_uniform_node_creation(self):
        """Test that UniformNode can be created and returns values in range."""
        node = UniformNode(node_id=1, min_payoff=10, max_payoff=20)
        self.assertEqual(node.node_id, 1)
        self.assertEqual(node.min_payoff, 10)
        self.assertEqual(node.max_payoff, 20)

        # Test that sampled values are in range
        samples = [node.sample_payoff() for _ in range(100)]
        self.assertTrue(all(10 <= s <= 20 for s in samples))
        self.assertGreater(max(samples), 15)  # Should get some high values
        self.assertLess(min(samples), 15)  # Should get some low values

    def test_gaussian_node_creation(self):
        """Test that GaussianNode can be created and returns reasonable values."""
        node = GaussianNode(node_id=2, mean=100, std=10)
        self.assertEqual(node.node_id, 2)
        self.assertEqual(node.mean, 100)
        self.assertEqual(node.std, 10)

        # Test that sampled values have roughly correct mean and std
        samples = [node.sample_payoff() for _ in range(1000)]
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        self.assertAlmostEqual(sample_mean, 100, delta=5)
        self.assertAlmostEqual(sample_std, 10, delta=2)

    def test_lognormal_node_creation(self):
        """Test that LogNormalNode can be created and returns positive values."""
        node = LogNormalNode(node_id=3, mu=0, sigma=1)
        self.assertEqual(node.node_id, 3)
        self.assertEqual(node.mu, 0)
        self.assertEqual(node.sigma, 1)

        # Test that all sampled values are positive
        samples = [node.sample_payoff() for _ in range(100)]
        self.assertTrue(all(s > 0 for s in samples))

    def test_powerlaw_node_creation(self):
        """Test that PowerLawNode can be created and returns values >= scale."""
        node = PowerLawNode(node_id=4, scale=5, alpha=2)
        self.assertEqual(node.node_id, 4)
        self.assertEqual(node.scale, 5)
        self.assertEqual(node.alpha, 2)

        # Test that all sampled values are >= scale
        samples = [node.sample_payoff() for _ in range(100)]
        self.assertTrue(all(s >= 5 for s in samples))

    def test_uniform_node_in_graph(self):
        """Test UniformNode integration in a graph."""
        g = Graph()
        g.from_dict(
            {
                1: {"type": "uniform", "min_payoff": 50, "max_payoff": 150, "after": []},
                2: {"type": "fixed", "payoff": 0, "after": [{"node_id": 1, "cost": 10}]},
            }
        )

        # Run multiple simulations and check results vary
        outcomes = [g.get_outcome() for _ in range(100)]
        # All outcomes should be between 40 and 140 (payoff minus cost)
        self.assertTrue(all(40 <= o <= 140 for o in outcomes))
        # Should have variation
        self.assertGreater(np.std(outcomes), 10)

    def test_gaussian_node_in_graph(self):
        """Test GaussianNode integration in a graph."""
        g = Graph()
        g.from_dict(
            {
                1: {"type": "gaussian", "mean": 100, "std": 20, "after": []},
                2: {
                    "type": "fixed",
                    "payoff": 0,
                    "after": [{"node_id": 1, "cost": 0, "weight": 1}],
                },
            }
        )

        # Run simulations and check mean is roughly correct
        outcomes = [g.get_outcome() for _ in range(1000)]
        mean_outcome = np.mean(outcomes)
        self.assertAlmostEqual(mean_outcome, 100, delta=5)

    def test_lognormal_node_in_graph(self):
        """Test LogNormalNode integration in a graph."""
        g = Graph()
        g.from_dict(
            {
                1: {"type": "lognormal", "mu": 2, "sigma": 0.5, "after": []},
                2: {"type": "fixed", "payoff": 0, "after": [{"node_id": 1, "cost": 0}]},
            }
        )

        # Run simulations and check all values are positive
        outcomes = [g.get_outcome() for _ in range(100)]
        self.assertTrue(all(o > 0 for o in outcomes))

    def test_powerlaw_node_in_graph(self):
        """Test PowerLawNode integration in a graph."""
        g = Graph()
        g.from_dict(
            {
                1: {"type": "powerlaw", "scale": 10, "alpha": 3, "after": []},
                2: {"type": "fixed", "payoff": 0, "after": [{"node_id": 1, "cost": 0}]},
            }
        )

        # Run simulations and check values are >= scale
        outcomes = [g.get_outcome() for _ in range(100)]
        self.assertTrue(all(o >= 10 for o in outcomes))

    def test_mixed_node_types_in_graph(self):
        """Test a graph with multiple different node types."""
        g = Graph()
        g.from_dict(
            {
                1: {"type": "uniform", "min_payoff": 0, "max_payoff": 100, "after": []},
                2: {
                    "type": "gaussian",
                    "mean": 50,
                    "std": 10,
                    "after": [{"node_id": 1, "cost": 0, "weight": 1}],
                },
                3: {
                    "type": "fixed",
                    "payoff": 0,
                    "after": [
                        {"node_id": 1, "cost": 10, "weight": 0.5},
                        {"node_id": 2, "cost": 10, "weight": 0.5},
                    ],
                },
            }
        )

        # Run simulations - should get mix of outcomes from both distributions
        outcomes = [g.get_outcome() for _ in range(1000)]
        mean_outcome = np.mean(outcomes)
        # Expected: 0.5 * (50 - 10) + 0.5 * ((50 + 50) - 10) = 0.5 * 40 + 0.5 * 90 = 65
        # Using wider delta due to variance
        self.assertAlmostEqual(mean_outcome, 65, delta=20)

    def test_fixed_node_backward_compatibility(self):
        """Test that regular Node still works as before."""
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 100, "after": []},
                2: {"payoff": 0, "after": [{"node_id": 1, "cost": 20}]},
            }
        )

        # Should always return exactly 80
        outcomes = [g.get_outcome() for _ in range(10)]
        self.assertTrue(all(o == 80 for o in outcomes))

    def test_node_with_explicit_fixed_type(self):
        """Test that explicit 'fixed' type works."""
        g = Graph()
        g.from_dict(
            {
                1: {"type": "fixed", "payoff": 100, "after": []},
                2: {"type": "fixed", "payoff": 0, "after": [{"node_id": 1, "cost": 20}]},
            }
        )

        # Should always return exactly 80
        outcomes = [g.get_outcome() for _ in range(10)]
        self.assertTrue(all(o == 80 for o in outcomes))


class TestPayoffSensitivity(unittest.TestCase):
    """Payoff sensitivity scales fixed and stochastic node samples."""

    def _analyze(self, node_spec):
        graph = Graph().from_dict({1: {**node_spec, "after": []}})
        result = graph.analyze_sensitivity(
            parameter_type="payoffs", num_simulations=1, perturbation=0.1
        )
        return graph.start_node, result["results"][0]

    def test_fixed_payoff_sensitivity(self):
        node, result = self._analyze({"type": "fixed", "payoff": 100})

        self.assertEqual(result["baseline_ev"], 100)
        self.assertAlmostEqual(result["increased_ev"], 110)
        self.assertAlmostEqual(result["decreased_ev"], 90)
        self.assertAlmostEqual(result["sensitivity"], 10)
        self.assertEqual(node.payoff, 100)

    def test_uniform_payoff_sensitivity_scales_bounds(self):
        draws = []

        def uniform(low, high):
            draws.append((low, high))
            return (low + high) / 2

        with patch("numpy.random.uniform", side_effect=uniform):
            node, result = self._analyze({"type": "uniform", "min_payoff": 10, "max_payoff": 20})

        self.assertEqual(draws, [(10, 20), (11, 22), (9, 18)])
        self.assertAlmostEqual(result["increased_ev"], result["baseline_ev"] * 1.1)
        self.assertAlmostEqual(result["decreased_ev"], result["baseline_ev"] * 0.9)
        self.assertEqual((node.min_payoff, node.max_payoff, node.payoff), (10, 20, 15))

    def test_gaussian_payoff_sensitivity_preserves_relative_spread(self):
        draws = []

        def normal(mean, std):
            draws.append((mean, std))
            return mean + std

        with patch("numpy.random.normal", side_effect=normal):
            node, result = self._analyze({"type": "gaussian", "mean": 100, "std": 10})

        expected_draws = [(100, 10), (110, 11), (90, 9)]
        for actual, expected in zip(draws, expected_draws):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])
        self.assertAlmostEqual(result["increased_ev"], result["baseline_ev"] * 1.1)
        self.assertAlmostEqual(result["decreased_ev"], result["baseline_ev"] * 0.9)
        self.assertEqual((node.mean, node.std, node.payoff), (100, 10, 100))

    def test_lognormal_payoff_sensitivity_preserves_sigma(self):
        draws = []

        def lognormal(mu, sigma):
            draws.append((mu, sigma))
            return math.exp(mu + sigma)

        with patch("numpy.random.lognormal", side_effect=lognormal):
            node, result = self._analyze({"type": "lognormal", "mu": 2, "sigma": 0.5})

        self.assertAlmostEqual(draws[1][0], 2 + math.log(1.1))
        self.assertAlmostEqual(draws[2][0], 2 + math.log(0.9))
        self.assertEqual([sigma for _, sigma in draws], [0.5, 0.5, 0.5])
        self.assertAlmostEqual(result["increased_ev"], result["baseline_ev"] * 1.1)
        self.assertAlmostEqual(result["decreased_ev"], result["baseline_ev"] * 0.9)
        self.assertEqual((node.mu, node.sigma), (2, 0.5))
        self.assertAlmostEqual(node.payoff, math.exp(2 + 0.5**2 / 2))

    def test_powerlaw_payoff_sensitivity_preserves_alpha(self):
        alphas = []

        def pareto(alpha):
            alphas.append(alpha)
            return 0.5

        with patch("numpy.random.pareto", side_effect=pareto):
            node, result = self._analyze({"type": "powerlaw", "scale": 10, "alpha": 3})

        self.assertEqual(alphas, [3, 3, 3])
        self.assertAlmostEqual(result["increased_ev"], result["baseline_ev"] * 1.1)
        self.assertAlmostEqual(result["decreased_ev"], result["baseline_ev"] * 0.9)
        self.assertEqual((node.scale, node.alpha, node.payoff), (10, 3, 15))

    def test_parameters_are_restored_when_simulation_raises(self):
        graph = Graph().from_dict({1: {"type": "gaussian", "mean": 100, "std": 10, "after": []}})
        node = graph.start_node

        with patch("numpy.random.normal", side_effect=[100, RuntimeError("draw failed")]):
            with self.assertRaisesRegex(RuntimeError, "draw failed"):
                graph.analyze_sensitivity(
                    parameter_type="payoffs", num_simulations=1, perturbation=0.1
                )

        self.assertEqual((node.mean, node.std, node.payoff), (100, 10, 100))


if __name__ == "__main__":
    unittest.main()
