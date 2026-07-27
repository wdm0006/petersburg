"""
Tests for the core Graph API: construction, simulation, options, and export.
"""

import random
import time
import unittest

import numpy as np

from petersburg import Graph, Node

__author__ = "willmcginnis"


class TestGraphFromDict(unittest.TestCase):
    """Construction of graphs from dictionary specifications."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_from_dict_sets_single_start_node(self):
        g = Graph()
        result = g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 50, "after": [{"node_id": 1, "cost": 10}]},
            }
        )
        # from_dict returns self for chaining
        self.assertIs(result, g)
        self.assertIsNotNone(g.start_node)
        self.assertEqual(g.start_node.node_id, 1)

    def test_from_dict_no_start_node_raises(self):
        g = Graph()
        with self.assertRaises(AttributeError):
            g.from_dict(
                {
                    1: {"payoff": 0, "after": [{"node_id": 2, "cost": 0}]},
                    2: {"payoff": 0, "after": [{"node_id": 1, "cost": 0}]},
                }
            )

    def test_from_dict_multiple_start_nodes_raises(self):
        g = Graph()
        with self.assertRaises(AttributeError):
            g.from_dict(
                {
                    1: {"payoff": 0, "after": []},
                    2: {"payoff": 0, "after": []},
                }
            )

    def test_from_dict_dangling_after_reference_raises_descriptive_error(self):
        g = Graph()
        with self.assertRaises(AttributeError) as ctx:
            g.from_dict(
                {
                    1: {"payoff": 0, "after": []},
                    2: {"payoff": 50, "after": [{"node_id": 99, "cost": 10}]},
                }
            )
        message = str(ctx.exception)
        self.assertIn("2", message)
        self.assertIn("99", message)
        self.assertIn("after", message)


class TestGetOutcome(unittest.TestCase):
    """Single-walk simulation through the graph."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_deterministic_single_path(self):
        # 1 (start, payoff 0) -> 2 (payoff 50), edge cost 10
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 50, "after": [{"node_id": 1, "cost": 10}]},
            }
        )
        # payoff (50) - cost (10) is fixed regardless of seeding
        for _ in range(25):
            self.assertEqual(g.get_outcome(), 40)

    def test_payoffs_and_costs_net_out(self):
        # Two-step deterministic chain: 1 -> 2 (cost 5, payoff 30) -> 3 (cost 15, payoff 100)
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 30, "after": [{"node_id": 1, "cost": 5}]},
                3: {"payoff": 100, "after": [{"node_id": 2, "cost": 15}]},
            }
        )
        # total payoff 130, total cost 20 -> 110
        self.assertEqual(g.get_outcome(), 110)

    def test_get_outcome_iters_accumulates_bank(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 50, "after": [{"node_id": 1, "cost": 10}]},
            }
        )
        # each walk nets 40; 5 walks from a starting bank of 100 -> 300
        self.assertEqual(g.get_outcome(iters=5, starting_bank=100), 300)

    def test_get_outcome_node_returns_terminal_id(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 50, "after": [{"node_id": 1, "cost": 10}]},
            }
        )
        self.assertEqual(g.get_outcome_node(), 2)

    def test_get_outcome_node_picks_among_branches(self):
        # Start branches to two terminals; the reached id must be one of them.
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 10, "after": [{"node_id": 1, "cost": 0, "weight": 0.5}]},
                3: {"payoff": 20, "after": [{"node_id": 1, "cost": 0, "weight": 0.5}]},
            }
        )
        reached = {g.get_outcome_node() for _ in range(100)}
        self.assertTrue(reached.issubset({2, 3}))
        # with equal weights and 100 draws, both branches should appear
        self.assertEqual(reached, {2, 3})


class TestGetOptions(unittest.TestCase):
    """Expected-value comparison across the start node's initial choices."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def _branching_graph(self):
        # Start node 1 has two initial outcomes:
        #   -> node 2 (payoff 100, edge cost 5)
        #   -> node 3 (payoff 20, edge cost 0)
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 100, "after": [{"node_id": 1, "cost": 5}]},
                3: {"payoff": 20, "after": [{"node_id": 1, "cost": 0}]},
            }
        )
        return g

    def test_one_entry_per_initial_outcome_with_known_means(self):
        g = self._branching_graph()
        options = g.get_options(iters=50)
        self.assertEqual(set(options.keys()), {2, 3})
        # Deterministic payoffs/costs: 100 - 5 = 95 and 20 - 0 = 20
        self.assertEqual(options[2], 95.0)
        self.assertEqual(options[3], 20.0)

    def test_extended_stats_keys(self):
        g = self._branching_graph()
        options = g.get_options(iters=10, extended_stats=True)
        self.assertEqual(set(options.keys()), {2, 3})
        for stats in options.values():
            self.assertEqual(set(stats.keys()), {"mean", "max", "min", "count"})
        self.assertEqual(options[2]["count"], 10)
        self.assertEqual(options[2]["mean"], 95.0)
        self.assertEqual(options[2]["max"], 95)
        self.assertEqual(options[2]["min"], 95)

    def test_classifier_weights_receive_feature_vector(self):
        class FeatureClassifier:
            def __init__(self, selected_value):
                self.selected_value = selected_value

            def predict_proba(self, feature_vector):
                if feature_vector is None:
                    raise ValueError("feature_vector required")
                probability = float(feature_vector[0][0] == self.selected_value)
                return np.array([[1 - probability, probability]])

        g = Graph()
        start = Node(0)
        option_a = Node(1, payoff=5)
        option_b = Node(2, payoff=10)
        option_a.add_outcome(Node(3, payoff=100), classifier=FeatureClassifier(1))
        option_a.add_outcome(Node(4, payoff=-100), classifier=FeatureClassifier(0))
        option_b.add_outcome(Node(5, payoff=20), classifier=FeatureClassifier(1))
        option_b.add_outcome(Node(6, payoff=-20), classifier=FeatureClassifier(0))
        start.add_outcome(option_a, cost=5)
        start.add_outcome(option_b, cost=2)
        g.start_node = start

        feature_vector = np.array([[1]])
        options = g.get_options(iters=5, feature_vector=feature_vector)
        self.assertEqual(options, {1: 100.0, 2: 28.0})

        extended = g.get_options(iters=5, extended_stats=True, feature_vector=feature_vector)
        self.assertEqual(
            extended,
            {
                1: {"mean": 100.0, "max": 100, "min": 100, "count": 5},
                2: {"mean": 28.0, "max": 28, "min": 28, "count": 5},
            },
        )


class TestFromAdjMatrix(unittest.TestCase):
    """Construction from a numpy adjacency matrix."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_round_trip_includes_root_and_walks(self):
        # A[r, c] != 0 creates edge r -> c, so this is the chain 0 -> 1.
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        g = Graph()
        g.from_adj_matrix(A)

        # The synthetic root node (-1) becomes the start node.
        self.assertEqual(g.start_node.node_id, -1)
        node_ids = {n.node_id for n in g.node_list()}
        self.assertEqual(node_ids, {-1, 0, 1})

        # Deterministic chain -1 -> 0 -> 1 terminates at node 1.
        self.assertEqual(g.get_outcome_node(), 1)

    def test_nonzero_entry_points_row_to_col(self):
        # A[0, 1] = 1 is the only transition, so the graph is the chain -1 -> 0 -> 1.
        A = np.zeros((2, 2))
        A[0, 1] = 1.0
        g = Graph()
        g.from_adj_matrix(A)

        self.assertEqual({n.node_id for n in g.node_list()}, {-1, 0, 1})
        edges = {(e.from_node.node_id, e.to_node.node_id) for e in g.edge_list()}
        self.assertEqual(edges, {(-1, 0), (0, 1)})

    def test_non_square_matrix_raises(self):
        g = Graph()
        with self.assertRaises(ValueError):
            g.from_adj_matrix(np.zeros((2, 3)))


def _layered_spec(layers, width):
    """Build a from_dict spec: a start node followed by `layers` fully-connected layers of `width` nodes."""
    spec = {0: {"payoff": 0, "after": []}}
    previous = [0]
    node_id = 1
    for _ in range(layers):
        current = []
        for _ in range(width):
            spec[node_id] = {"payoff": 1, "after": [{"node_id": p, "cost": 1} for p in previous]}
            current.append(node_id)
            node_id += 1
        previous = current

    return spec


class TestTraversal(unittest.TestCase):
    """node_list()/edge_list() visit each node once rather than enumerating every path."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_diamond_yields_each_node_and_edge_once(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 10, "after": [{"node_id": 1, "cost": 1}]},
                3: {"payoff": 20, "after": [{"node_id": 1, "cost": 2}]},
                4: {"payoff": 30, "after": [{"node_id": 2, "cost": 3}, {"node_id": 3, "cost": 4}]},
            }
        )

        self.assertEqual({n.node_id for n in g.node_list()}, {1, 2, 3, 4})
        self.assertEqual(len(g.node_list()), 4)

        edges = sorted((e.from_node.node_id, e.to_node.node_id, e.cost) for e in g.edge_list())
        self.assertEqual(edges, [(1, 2, 1), (1, 3, 2), (2, 4, 3), (3, 4, 4)])

    def test_deep_layered_graph_is_tractable(self):
        g = Graph()
        g.from_dict(_layered_spec(layers=15, width=3))

        start = time.perf_counter()
        nodes = g.node_list()
        edges = g.edge_list()
        elapsed = time.perf_counter() - start

        self.assertEqual(len(nodes), 46)
        self.assertEqual(len(edges), 129)

        # visiting each node once takes well under a millisecond here; enumerating
        # all 3 ** 15 paths takes seconds and grows 3x per added layer
        self.assertLess(elapsed, 1.0)

    def test_get_edges_accepts_explicit_visited_set(self):
        g = Graph()
        g.from_dict(_layered_spec(layers=2, width=2))

        self.assertEqual(len(g.start_node.get_edges(set(), set())), 6)


class TestToMermaid(unittest.TestCase):
    """Export to Mermaid diagram syntax."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_mermaid_contains_expected_lines(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 50, "after": [{"node_id": 1, "cost": 10}]},
            }
        )
        mermaid = g.to_mermaid()
        self.assertIsInstance(mermaid, str)
        self.assertTrue(mermaid.startswith("graph "))

        # Start node rendered with the (()) shape.
        self.assertIn("1((Start))", mermaid)
        # Payoff node carries its payoff in the label.
        self.assertIn("Payoff: $50", mermaid)
        # Edge with a non-zero cost is labelled and points 1 -> 2.
        self.assertIn("1 -->|Cost: $10| 2", mermaid)

    def test_mermaid_orientation_argument(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 50, "after": [{"node_id": 1, "cost": 10}]},
            }
        )
        self.assertTrue(g.to_mermaid(orientation="TD").startswith("graph TD"))


if __name__ == "__main__":
    unittest.main()
