"""Value-based checks for bounded exact fixed-payoff distributions."""

import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from petersburg import Graph, Node, ValidationError


def _graph(seed=None):
    return Graph(random_state=seed).from_dict(
        {
            1: {"payoff": 5, "after": []},
            2: {"payoff": 10, "after": [{"node_id": 1, "cost": 2, "weight": 3}]},
            3: {"payoff": -4, "after": [{"node_id": 1, "cost": 3, "weight": 1}]},
        }
    )


class TestOutcomeDistribution(unittest.TestCase):
    def test_costs_start_payoff_order_and_mean(self):
        g = _graph()
        masses = g.outcome_distribution()
        self.assertEqual(list(masses.items()), [(-2, 0.25), (13, 0.75)])
        self.assertEqual(sum(masses.values()), 1)
        self.assertEqual(sum(x * p for x, p in masses.items()), g.expected_value())

    def test_converging_equal_totals_merge(self):
        g = Graph().from_dict(
            {
                1: {"payoff": 5, "after": []},
                2: {"payoff": 4, "after": [{"node_id": 1, "cost": 2}]},
                3: {"payoff": 3, "after": [{"node_id": 1, "cost": 1}]},
                4: {"payoff": 10, "after": [{"node_id": 2}, {"node_id": 3}]},
            }
        )
        self.assertEqual(g.outcome_distribution(max_outcomes=1), {17: 1})

    def test_parallel_costs_and_zero_mass(self):
        g = Graph()
        g.start_node = Node(1, 5)
        child = Node(2, 10)
        for cost, weight in [(2, 3), (3, 1), (100, 0)]:
            g.start_node.add_outcome(child, cost=cost, weight=weight)
        self.assertEqual(g.outcome_distribution(), {12: 0.25, 13: 0.75})

    def test_terminal_and_local_state(self):
        g = Graph().from_dict({1: {"payoff": -7, "after": []}})
        masses = g.outcome_distribution(np.int64(1))
        self.assertEqual(masses, {-7: 1})
        masses.clear()
        g.start_node.payoff = 9
        self.assertEqual(g.outcome_distribution(), {9: 1})

    def test_no_sampling_or_rng_consumption(self):
        g, control = _graph(19), _graph(19)
        with patch.object(g, "get_outcome", side_effect=AssertionError("draw")):
            with patch.object(Node, "sample_payoff", side_effect=AssertionError("draw")):
                with patch.object(Node, "weighted_choice", side_effect=AssertionError("draw")):
                    self.assertEqual(g.outcome_distribution(), {-2: 0.25, 13: 0.75})
        self.assertEqual(
            [g.get_outcome() for _ in range(100)], [control.get_outcome() for _ in range(100)]
        )

    def test_invalid_caps_and_unbuilt(self):
        for g in (Graph(), _graph()):
            for cap in (True, False, 0, -1, 1.5, float("nan"), None, "10"):
                with self.subTest(cap=cap), self.assertRaisesRegex(ValidationError, "max_outcomes"):
                    g.outcome_distribution(cap)
        with self.assertRaisesRegex(ValidationError, "built graph"):
            Graph().outcome_distribution()

    def test_unsupported_nodes_even_on_zero_branches(self):
        for kind in ("uniform", "gaussian", "lognormal", "powerlaw"):
            g = _graph()
            g.start_node.add_outcome(
                Graph().from_dict({9: {"type": kind, "after": []}}).start_node, weight=0
            )
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValidationError, "fixed payoff nodes"),
            ):
                g.outcome_distribution()

    def test_invalid_weights_including_zero_branch_descendants(self):
        for weights in (
            (-1, 1),
            (float("nan"), 1),
            (float("inf"), 1),
            (0, 0),
            (1j, 1),
            (1e308, 1e308),
            ("x", 1),
            (Mock(predict_proba=Mock()), 1),
        ):
            g = _graph()
            hidden = Node(9)
            for weight in weights:
                hidden.add_outcome(Node(10), weight=weight)
            g.start_node.add_outcome(hidden, weight=0)
            with self.subTest(weights=weights), self.assertRaisesRegex(ValidationError, "weight"):
                g.outcome_distribution()

    def test_invalid_payoffs_and_costs_on_zero_branches(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1j, "5", None):
            for field in ("payoff", "cost"):
                g = _graph()
                hidden = Node(9)
                g.start_node.add_outcome(hidden, weight=0)
                if field == "payoff":
                    hidden.payoff = value
                else:
                    g.start_node.outcomes[-1][0].cost = value
                with (
                    self.subTest(value=value, field=field),
                    self.assertRaisesRegex(ValidationError, "finite real"),
                ):
                    g.outcome_distribution()

    def test_shift_overflow(self):
        g = _graph()
        g.start_node.payoff = 1e308
        g.start_node.outcomes[0][0].to_node.payoff = 1e308
        with self.assertRaisesRegex(ValidationError, "shifted outcome"):
            g.outcome_distribution()

    def test_branching_chain_cap_during_accumulation(self):
        g = Graph()
        g.start_node = Node(0)
        current = g.start_node
        for i in range(8):
            child = Node(i + 1)
            current.add_outcome(child, cost=0)
            current.add_outcome(child, cost=2**i)
            current = child
        with self.assertRaisesRegex(ValidationError, "Node 5 .*max_outcomes=4"):
            g.outcome_distribution(4)
        self.assertEqual(len(g.outcome_distribution(256)), 256)

    def test_cap_applies_to_child_before_parent_float_collapses_atoms(self):
        g = Graph()
        g.start_node = Node(0, 1e20)
        child = Node(1)
        g.start_node.add_outcome(child)
        child.add_outcome(Node(2, 1))
        child.add_outcome(Node(3, 2))
        with self.assertRaisesRegex(ValidationError, "Node 1 .*max_outcomes=1"):
            g.outcome_distribution(1)
        self.assertEqual(g.outcome_distribution(2), {1e20: 1})

    def test_validation_precedes_support_accumulation(self):
        g = _graph()
        g.start_node.outcomes[1][0].to_node.payoff = float("nan")
        with self.assertRaisesRegex(ValidationError, "payoff.*finite real"):
            g.outcome_distribution(1)

    def test_many_paths_small_support_is_memoized(self):
        spec = {0: {"payoff": 1, "after": []}}
        previous = [0]
        for layer in range(30):
            ids = [2 * layer + 1, 2 * layer + 2]
            for node_id in ids:
                spec[node_id] = {"payoff": 1, "after": [{"node_id": parent} for parent in previous]}
            previous = ids
        g = Graph().from_dict(spec)
        start = time.monotonic()
        self.assertEqual(g.outcome_distribution(1), {31: 1})
        self.assertLess(time.monotonic() - start, 2)

    def test_float_atoms_are_not_rounded(self):
        g = Graph()
        g.start_node = Node(0)
        g.start_node.add_outcome(Node(1, 0.3))
        g.start_node.add_outcome(Node(2, 0.1 + 0.2))
        self.assertEqual(g.outcome_distribution(), {0.3: 0.5, 0.1 + 0.2: 0.5})
