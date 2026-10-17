"""Exact per-option risk summaries: values, alpha validation, single PMF call, RNG."""

import math
import unittest
from unittest.mock import patch

from petersburg import Graph, ValidationError


def _graph(seed=None, start_payoff=0):
    return Graph(random_state=seed).from_dict(
        {
            1: {"payoff": start_payoff, "after": []},
            2: {"after": [{"node_id": 1}]},
            3: {"payoff": -10, "after": [{"node_id": 2, "weight": 1}]},
            4: {"payoff": 2, "after": [{"node_id": 2, "weight": 3}]},
        }
    )


class TestOptionStatistics(unittest.TestCase):
    def test_unequal_mass_values(self):
        g = _graph()
        self.assertEqual(g.option_distributions(), {2: {-10: 0.25, 2: 0.75}})
        s = g.option_statistics(alpha=0.5)[2]
        self.assertEqual(s["mean"], -1.0)
        self.assertAlmostEqual(s["std"], math.sqrt(27))
        self.assertEqual(s["p_loss"], 0.25)
        self.assertEqual(s["var_alpha"], 2.0)
        # unweighted mean over the keys would be -4
        self.assertEqual(s["cvar_alpha"], -1.0)

    def test_quantile_boundaries(self):
        g = _graph()
        for alpha, var in [(0, -10.0), (0.25, -10.0), (0.2500001, 2.0), (1, 2.0)]:
            self.assertEqual(g.option_statistics(alpha=alpha)[2]["var_alpha"], var, alpha)
        self.assertEqual(g.option_statistics(alpha=0.25)[2]["cvar_alpha"], -10.0)
        self.assertEqual(g.option_statistics(alpha=1)[2]["cvar_alpha"], -1.0)

    def test_parallel_costs_start_payoff_and_zero_weight(self):
        g = Graph().from_dict(
            {
                1: {"payoff": 5, "after": []},
                2: {
                    "payoff": 10,
                    "after": [
                        {"node_id": 1, "cost": 2, "weight": 0},
                        {"node_id": 1, "cost": 20},
                    ],
                },
            }
        )
        s = g.option_statistics()
        self.assertEqual(set(s), {2, (2, 1)})
        self.assertEqual(s[2]["mean"], 13.0)
        self.assertEqual(s[(2, 1)]["mean"], -5.0)
        self.assertEqual(s[(2, 1)]["p_loss"], 1.0)
        self.assertEqual(s[2]["std"], 0.0)

    def test_terminal_start(self):
        self.assertEqual(Graph().from_dict({1: {"after": []}}).option_statistics(), {})

    def test_invalid_alpha_checked_before_pmf(self):
        terminal = Graph().from_dict({1: {"after": []}})
        for g in (_graph(), terminal):
            with patch.object(g, "option_distributions", side_effect=AssertionError):
                for bad in (
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                    -0.1,
                    1.1,
                    True,
                    "0.5",
                    None,
                ):
                    with self.assertRaises(ValidationError, msg=repr(bad)):
                        g.option_statistics(alpha=bad)

    def test_cap_errors_propagate(self):
        with self.assertRaises(ValidationError):
            _graph().option_statistics(max_outcomes=1)

    def test_single_pmf_call_and_rng_unchanged(self):
        g = _graph(seed=3)
        state = g.rng.bit_generator.state
        original = g.option_distributions
        calls = []
        g.option_distributions = lambda **kw: calls.append(kw) or original(**kw)
        g.option_statistics(max_outcomes=50)
        self.assertEqual(calls, [{"max_outcomes": 50}])
        self.assertEqual(g.rng.bit_generator.state, state)


if __name__ == "__main__":
    unittest.main()
