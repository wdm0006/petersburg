"""
Tests for distribution-based node types.
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
