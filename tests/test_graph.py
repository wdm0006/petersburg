"""
Tests for the core Graph API: construction, simulation, options, and export.
"""

import importlib.util
import io
import os
import random
import re
import subprocess
import sys
import textwrap
import time
import unittest
from contextlib import redirect_stdout

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

    def test_numeric_matrix_dtypes_build_and_walk(self):
        for dtype in (np.int64, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                # A[r, c] != 0 creates edge r -> c, with two terminal outcomes.
                A = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=dtype)
                g = Graph()
                g.from_adj_matrix(A)

                self.assertEqual(g.start_node.node_id, -1)
                self.assertEqual({n.node_id for n in g.node_list()}, {-1, 0, 1, 2})
                self.assertEqual(
                    {(e.from_node.node_id, e.to_node.node_id) for e in g.edge_list()},
                    {(-1, 0), (0, 1), (0, 2)},
                )
                self.assertEqual(g.get_outcome(), 0)
                self.assertIn(g.get_outcome_node(), {1, 2})

    def test_non_float64_weights_are_exported_and_analyzed(self):
        for dtype in (np.int64, np.float32):
            with self.subTest(dtype=dtype):
                A = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=dtype)
                g = Graph().from_adj_matrix(A)

                self.assertEqual(g.to_mermaid().count("P: 0.50"), 2)
                sensitivity = g.analyze_sensitivity(
                    parameter_type="edge_weights", num_simulations=1
                )
                self.assertGreater(len(sensitivity["results"]), 0)

    def test_classifier_weight_uses_predict_proba_and_skips_sensitivity(self):
        class Classifier:
            def __init__(self):
                self.calls = 0

            def predict_proba(self, feature_vector):
                self.calls += 1
                return np.array([[0.0, 1.0]])

        classifier = Classifier()
        g = Graph()
        start = Node(0)
        start.add_outcome(Node(1), classifier=classifier)
        g.start_node = start

        self.assertEqual(g.get_outcome_node(np.array([[1]])), 1)
        self.assertEqual(classifier.calls, 1)
        sensitivity = g.analyze_sensitivity(parameter_type="edge_weights", num_simulations=1)
        self.assertEqual(sensitivity["results"], [])

    def test_nonzero_entry_points_row_to_col(self):
        # A[0, 1] = 1 is the only transition, so the graph is the chain -1 -> 0 -> 1.
        A = np.zeros((2, 2))
        A[0, 1] = 1.0
        g = Graph()
        g.from_adj_matrix(A)

        self.assertEqual({n.node_id for n in g.node_list()}, {-1, 0, 1})
        edges = {(e.from_node.node_id, e.to_node.node_id) for e in g.edge_list()}
        self.assertEqual(edges, {(-1, 0), (0, 1)})

    def test_nan_is_ignored_when_normalizing_and_walking(self):
        A = np.array([[0.0, 2.0, np.nan], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        g = Graph().from_adj_matrix(A)

        edge_weights = {
            (edge.from_node.node_id, edge.to_node.node_id): weight
            for node in g.node_list()
            for edge, weight in node.outcomes
        }
        self.assertEqual(edge_weights[(0, 1)], 1.0)
        self.assertTrue(np.isfinite(edge_weights[(0, 1)]))
        self.assertEqual(g.get_outcome_node(), 1)

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


class TestToTree(unittest.TestCase):
    """Export graphs as nested dictionaries."""

    def test_diamond_returns_expected_nested_dict(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 10, "after": [{"node_id": 1}]},
                3: {"payoff": 20, "after": [{"node_id": 1}]},
                4: {"payoff": 30, "after": [{"node_id": 2}, {"node_id": 3}]},
            }
        )

        self.assertEqual(g.to_tree(), {"1": {"2": {"4": None}, "3": {"4": None}}})

    def test_terminal_only_graph(self):
        g = Graph()
        g.from_dict({7: {"payoff": 5, "after": []}})

        self.assertEqual(g.to_tree(), {"7": None})

    def test_deep_layered_graph_is_tractable(self):
        g = Graph()
        g.from_dict(_layered_spec(layers=15, width=3))

        start = time.perf_counter()
        tree = g.to_tree()
        elapsed = time.perf_counter() - start

        self.assertEqual(list(tree), ["0"])
        self.assertLess(elapsed, 1.0)


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

    def test_mermaid_styles_terminal_node_with_nonzero_id(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 50, "after": [{"node_id": 1, "cost": 10}]},
            }
        )
        mermaid = g.to_mermaid()

        self.assertIn("    class 2 terminal", mermaid)
        # Node 0 does not exist here, so it must not be styled.
        self.assertNotIn("class 0 terminal", mermaid)
        # The start node still has outgoing outcomes.
        self.assertNotIn("class 1 terminal", mermaid)

    def test_mermaid_styles_every_terminal_node(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 10, "after": [{"node_id": 1}]},
                3: {"payoff": 20, "after": [{"node_id": 1}]},
                4: {"payoff": 0, "after": [{"node_id": 1}]},
                5: {"payoff": 30, "after": [{"node_id": 4}]},
            }
        )
        mermaid = g.to_mermaid()

        for terminal_id in (2, 3, 5):
            self.assertIn(f"    class {terminal_id} terminal", mermaid)
        for interior_id in (1, 4):
            self.assertNotIn(f"class {interior_id} terminal", mermaid)

    def test_mermaid_styles_node_zero_when_it_is_terminal(self):
        g = Graph()
        g.from_dict(
            {
                1: {"payoff": 0, "after": []},
                0: {"payoff": 15, "after": [{"node_id": 1}]},
            }
        )
        mermaid = g.to_mermaid()

        self.assertIn("    class 0 terminal", mermaid)


def _wide_spec(start_id=100, children=80):
    """A from_dict spec: one start node with `children` direct children, ids 0..children-1."""
    spec = {start_id: {"payoff": 0, "after": []}}
    for node_id in range(children):
        spec[node_id] = {"payoff": node_id, "after": [{"node_id": start_id, "cost": node_id}]}
    return spec


_NODE_LINE = re.compile(r"^ {4}(\S+?)(?:\(\(|\[)")
_EDGE_LINE = re.compile(r"^ {4}(\S+) -->(?:\|[^|]*\|)? (\S+)$")


def _mermaid_node_ids(mermaid):
    return [m.group(1) for m in (_NODE_LINE.match(line) for line in mermaid.splitlines()) if m]


def _mermaid_edges(mermaid):
    return [m.groups() for m in (_EDGE_LINE.match(line) for line in mermaid.splitlines()) if m]


class TestMermaidDeterminism(unittest.TestCase):
    """to_mermaid emits a stable, start-first node order and truncates intentionally."""

    START_ID = 100
    CHILDREN = 80

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def _wide_graph(self):
        return Graph().from_dict(_wide_spec(self.START_ID, self.CHILDREN))

    def test_fresh_processes_produce_identical_output(self):
        # node_list()/edge_list() are identity-hashed sets, so their iteration order
        # varies with allocation addresses: only a fresh interpreter shows the drift.
        script_source = f"""
            from petersburg.graph import Graph

            spec = {{{self.START_ID}: {{"payoff": 0, "after": []}}}}
            for node_id in range({self.CHILDREN}):
                spec[node_id] = {{
                    "payoff": node_id,
                    "after": [{{"node_id": {self.START_ID}, "cost": node_id}}],
                }}

            print(Graph().from_dict(spec).to_mermaid(max_nodes=10))
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
        self.assertIn(f"{self.START_ID}((Start))", outputs.pop())

    def test_export_ignores_traversal_iteration_order(self):
        g = self._wide_graph()
        baseline = g.to_mermaid(max_nodes=10)
        nodes = list(g.node_list())
        edges = list(g.edge_list())

        for offset in (1, 7, len(nodes) - 1):
            g.node_list = lambda nodes=nodes, offset=offset: nodes[offset:] + nodes[:offset]
            g.edge_list = lambda edges=edges, offset=offset: edges[offset:] + edges[:offset]
            self.assertEqual(g.to_mermaid(max_nodes=10), baseline)

    def test_node_order_is_start_first_then_sorted_by_id(self):
        ids = _mermaid_node_ids(self._wide_graph().to_mermaid(max_nodes=self.CHILDREN + 1))

        self.assertEqual(ids[0], str(self.START_ID))
        self.assertEqual(ids[1:], sorted(str(n) for n in range(self.CHILDREN)))

    def test_truncation_keeps_start_node(self):
        g = self._wide_graph()
        total = self.CHILDREN + 1

        for max_nodes in (1, 2, 5, 10, 40, total, total + 5):
            with self.subTest(max_nodes=max_nodes):
                ids = _mermaid_node_ids(g.to_mermaid(max_nodes=max_nodes))
                self.assertEqual(ids[0], str(self.START_ID))
                self.assertEqual(len(ids), min(max_nodes, total))

    def test_truncated_edges_stay_within_the_selected_nodes(self):
        mermaid = self._wide_graph().to_mermaid(max_nodes=10)
        included = set(_mermaid_node_ids(mermaid))
        emitted = _mermaid_edges(mermaid)

        self.assertEqual(len(emitted), 9)
        for from_id, to_id in emitted:
            self.assertIn(from_id, included)
            self.assertIn(to_id, included)

    def test_edge_order_is_sorted_by_endpoints(self):
        mermaid = self._wide_graph().to_mermaid(max_nodes=self.CHILDREN + 1)
        emitted = _mermaid_edges(mermaid)

        self.assertEqual(len(emitted), self.CHILDREN)
        self.assertEqual(emitted, sorted(emitted, key=lambda e: (e[0], e[1])))


_HAS_NETWORKX = importlib.util.find_spec("networkx") is not None


def _attribute_spec():
    """Start node 1 branching to node 2 (payoff 50, cost 10, p 0.25) and node 3 (p 0.75)."""
    return {
        1: {"payoff": 0, "after": []},
        2: {"payoff": 50, "after": [{"node_id": 1, "cost": 10, "weight": 0.25}]},
        3: {"payoff": 20, "after": [{"node_id": 1, "cost": 4, "weight": 0.75}]},
    }


@unittest.skipUnless(_HAS_NETWORKX, "networkx is not installed")
class TestToNetworkx(unittest.TestCase):
    """to_networkx exports a stable order and carries payoffs, costs, and probabilities."""

    START_ID = 100
    CHILDREN = 25

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def _wide_graph(self):
        return Graph().from_dict(_wide_spec(self.START_ID, self.CHILDREN))

    def test_fresh_processes_produce_identical_order(self):
        # node_list()/edge_list() are identity-hashed sets, so their iteration order
        # varies with allocation addresses: only a fresh interpreter shows the drift.
        script_source = f"""
            from petersburg.graph import Graph

            spec = {{{self.START_ID}: {{"payoff": 0, "after": []}}}}
            for node_id in range({self.CHILDREN}):
                spec[node_id] = {{
                    "payoff": node_id,
                    "after": [{{"node_id": {self.START_ID}, "cost": node_id}}],
                }}

            g = Graph().from_dict(spec).to_networkx()
            print(list(g.nodes()))
            print(list(g.edges()))
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
        self.assertTrue(outputs.pop().startswith(f"[{self.START_ID}, 0, 1,"))

    def test_export_ignores_traversal_iteration_order(self):
        g = self._wide_graph()
        baseline = g.to_networkx()
        nodes = list(g.node_list())
        edges = list(g.edge_list())

        for offset in (1, 7, len(nodes) - 1):
            g.node_list = lambda nodes=nodes, offset=offset: nodes[offset:] + nodes[:offset]
            g.edge_list = lambda edges=edges, offset=offset: edges[offset:] + edges[:offset]
            rotated = g.to_networkx()

            self.assertEqual(list(rotated.nodes()), list(baseline.nodes()))
            self.assertEqual(list(rotated.edges()), list(baseline.edges()))

    def test_node_order_is_start_first_then_sorted_by_id(self):
        ids = list(self._wide_graph().to_networkx().nodes())

        self.assertEqual(ids[0], self.START_ID)
        self.assertEqual(ids[1:], sorted(range(self.CHILDREN), key=str))

    def test_node_payoffs_are_exported(self):
        g = Graph().from_dict(_attribute_spec()).to_networkx()

        self.assertEqual(g.nodes[1]["payoff"], 0)
        self.assertEqual(g.nodes[2]["payoff"], 50)
        self.assertEqual(g.nodes[3]["payoff"], 20)

    def test_edges_carry_cost_and_probability(self):
        g = Graph().from_dict(_attribute_spec()).to_networkx()

        self.assertEqual(g.edges[1, 2]["cost"], 10)
        self.assertEqual(g.edges[1, 2]["probability"], 0.25)
        self.assertEqual(g.edges[1, 3]["cost"], 4)
        self.assertEqual(g.edges[1, 3]["probability"], 0.75)

    def test_weight_still_holds_the_cost(self):
        graph = Graph().from_dict(_attribute_spec())
        g = graph.to_networkx()

        self.assertEqual(g.edges[1, 2]["weight"], 10)
        self.assertEqual(g.edges[1, 3]["weight"], 4)
        for edge in graph.edge_list():
            exported = g.edges[edge.from_node.node_id, edge.to_node.node_id]
            self.assertEqual(exported["weight"], edge.cost)

    def test_classifier_weighted_edge_has_no_probability(self):
        class Classifier:
            def predict_proba(self, feature_vector):
                return np.array([[0.0, 1.0]])

        graph = Graph()
        start = Node(0)
        start.add_outcome(Node(1), cost=7, classifier=Classifier())
        graph.start_node = start

        g = graph.to_networkx()

        self.assertEqual(g.edges[0, 1]["cost"], 7)
        self.assertIsNone(g.edges[0, 1]["probability"])

    def test_parallel_edges_collapse_deterministically(self):
        # A DiGraph holds one edge per pair, so the last edge in sort order wins; the
        # point of the assertion is that which one wins no longer depends on set order.
        spec = {
            1: {"payoff": 0, "after": []},
            2: {
                "payoff": 50,
                "after": [{"node_id": 1, "cost": 5}, {"node_id": 1, "cost": 7}],
            },
        }
        graph = Graph().from_dict(spec)
        edges = list(graph.edge_list())
        g = graph.to_networkx()

        self.assertEqual(len(edges), 2)
        self.assertEqual(len(g.edges()), 1)
        self.assertEqual(g.edges[1, 2]["cost"], 7)

        graph.edge_list = lambda edges=edges: list(reversed(edges))
        self.assertEqual(graph.to_networkx().edges[1, 2]["cost"], 7)


class TestSensitivityParameterSelection(unittest.TestCase):
    """max_params selection is deterministic and reported rather than silent."""

    LEAF_IDS = list(range(1, 13))

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def _wide_graph(self, costs=None):
        """Start node 0 with twelve leaves, each reached by its own weighted edge."""
        spec = {0: {"payoff": 0, "after": []}}
        for node_id in self.LEAF_IDS:
            cost = node_id if costs is None else costs[node_id]
            spec[node_id] = {
                "payoff": 10 * node_id,
                "after": [{"node_id": 0, "cost": cost, "weight": node_id}],
            }
        return Graph().from_dict(spec)

    def _analyze(self, graph, parameter_type, **kwargs):
        return graph.analyze_sensitivity(
            parameter_type=parameter_type, num_simulations=1, perturbation=0.1, **kwargs
        )

    def test_edge_weight_selection_is_deterministic(self):
        # Candidates are sorted by (str(from_id), str(to_id), str(cost)), so the ten
        # analyzed edges are the same on every run and in every process.
        expected = sorted(f"Edge 0→{n} weight" for n in [1, 10, 11, 12, 2, 3, 4, 5, 6, 7])

        for _ in range(3):
            result = self._analyze(self._wide_graph(), "edge_weights")
            self.assertEqual(sorted(r["parameter"] for r in result["results"]), expected)

    def test_cost_selection_is_deterministic(self):
        expected = sorted(f"Edge 0→{n} cost" for n in [1, 10, 11, 12, 2, 3, 4, 5, 6, 7])

        for _ in range(3):
            result = self._analyze(self._wide_graph(), "costs")
            self.assertEqual(sorted(r["parameter"] for r in result["results"]), expected)

    def test_payoff_selection_is_deterministic(self):
        expected = sorted(f"Node {n} payoff" for n in [1, 10, 11, 12, 2, 3, 4, 5, 6, 7])

        for _ in range(3):
            result = self._analyze(self._wide_graph(), "payoffs")
            self.assertEqual(sorted(r["parameter"] for r in result["results"]), expected)

    def test_candidate_and_analyzed_counts_are_reported(self):
        for parameter_type in ("edge_weights", "costs", "payoffs"):
            result = self._analyze(self._wide_graph(), parameter_type)
            self.assertEqual(result["candidate_parameters"], 12)
            self.assertEqual(result["parameters_analyzed"], 10)
            self.assertEqual(result["parameters_analyzed"], len(result["results"]))
            self.assertEqual(result["max_params"], 10)

    def test_max_params_can_be_raised_or_lifted(self):
        raised = self._analyze(self._wide_graph(), "edge_weights", max_params=20)
        self.assertEqual(raised["parameters_analyzed"], 12)

        unlimited = self._analyze(self._wide_graph(), "edge_weights", max_params=None)
        self.assertEqual(unlimited["parameters_analyzed"], 12)

        lowered = self._analyze(self._wide_graph(), "edge_weights", max_params=3)
        self.assertEqual(lowered["candidate_parameters"], 12)
        self.assertEqual(lowered["parameters_analyzed"], 3)

    def test_ineligible_parameters_do_not_consume_the_cap(self):
        # The first four edges in sort order are free, so they are not cost candidates
        # at all and must not eat slots that eligible edges could use.
        costs = {n: (0 if n in (1, 10, 11, 12) else n) for n in self.LEAF_IDS}
        result = self._analyze(self._wide_graph(costs=costs), "costs")

        self.assertEqual(result["candidate_parameters"], 8)
        self.assertEqual(result["parameters_analyzed"], 8)
        self.assertEqual(
            sorted(r["parameter"] for r in result["results"]),
            sorted(f"Edge 0→{n} cost" for n in [2, 3, 4, 5, 6, 7, 8, 9]),
        )

    def test_identify_critical_parameters_honours_max_params(self):
        capped = self._wide_graph().identify_critical_parameters(
            num_simulations=1, perturbation=0.1, top_n=5
        )
        self.assertEqual(capped["max_params"], 10)
        self.assertEqual(capped["total_candidate_parameters"], 36)
        self.assertEqual(capped["total_parameters_analyzed"], 30)

        lifted = self._wide_graph().identify_critical_parameters(
            num_simulations=1, perturbation=0.1, top_n=5, max_params=None
        )
        self.assertEqual(lifted["total_candidate_parameters"], 36)
        self.assertEqual(lifted["total_parameters_analyzed"], 36)

    def test_report_states_when_parameters_were_excluded(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self._wide_graph().print_sensitivity_report(
                num_simulations=1, perturbation=0.1, top_n=3
            )
        capped = buffer.getvalue()

        self.assertIn("Parameters Analyzed: 30 of 36", capped)
        self.assertIn("6 parameter(s) excluded by max_params=10", capped)
        self.assertNotIn("Total Parameters Analyzed", capped)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self._wide_graph().print_sensitivity_report(
                num_simulations=1, perturbation=0.1, top_n=3, max_params=None
            )
        full = buffer.getvalue()

        self.assertIn("Parameters Analyzed: 36 of 36", full)
        self.assertNotIn("excluded by max_params", full)


class TestEdgeWeightPerturbation(unittest.TestCase):
    """The decrease arm lowers the weight, including for weights below 0.01."""

    SMALL_WEIGHT = 0.005

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def _rare_event_graph(self):
        """Start node 1 with a rare high-payoff branch and a common worthless one."""
        return Graph().from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {
                    "payoff": 1000,
                    "after": [{"node_id": 1, "cost": 0, "weight": self.SMALL_WEIGHT}],
                },
                3: {"payoff": 0, "after": [{"node_id": 1, "cost": 0, "weight": 0.995}]},
            }
        )

    def _record_rare_edge_weight(self, graph):
        """Shadow get_outcome to capture the weight installed on edge 1→2 per walk."""
        observed = []
        real_get_outcome = graph.get_outcome

        def recording_get_outcome(*args, **kwargs):
            for edge, weight in graph.start_node.outcomes:
                if edge.to_node.node_id == 2:
                    observed.append(weight)
            return real_get_outcome(*args, **kwargs)

        graph.get_outcome = recording_get_outcome
        return observed

    def test_decrease_arm_applies_the_exact_perturbed_weight(self):
        # Asserting on the weight actually installed, not on a Monte Carlo mean, so the
        # assertion is free of simulation noise. Candidates sort edge 1→2 first, and
        # num_simulations=1 gives one walk per arm.
        graph = self._rare_event_graph()
        observed = self._record_rare_edge_weight(graph)

        graph.analyze_sensitivity(
            parameter_type="edge_weights", num_simulations=1, perturbation=0.1
        )

        self.assertEqual(len(observed), 5)
        self.assertAlmostEqual(observed[0], 0.005)
        self.assertAlmostEqual(observed[1], 0.0055)
        self.assertAlmostEqual(observed[2], 0.0045)

    def test_original_weight_is_restored_after_analysis(self):
        graph = self._rare_event_graph()
        before = [(edge.to_node.node_id, weight) for edge, weight in graph.start_node.outcomes]

        graph.analyze_sensitivity(
            parameter_type="edge_weights", num_simulations=1, perturbation=0.1
        )

        after = [(edge.to_node.node_id, weight) for edge, weight in graph.start_node.outcomes]
        self.assertEqual(before, after)

    def test_out_of_range_perturbation_is_rejected(self):
        graph = self._rare_event_graph()
        for perturbation in (-0.5, 0, 1, 1.5):
            for parameter_type in ("edge_weights", "costs", "payoffs"):
                with self.assertRaises(ValueError):
                    graph.analyze_sensitivity(
                        parameter_type=parameter_type,
                        num_simulations=1,
                        perturbation=perturbation,
                    )

    def test_default_perturbation_still_runs(self):
        result = self._rare_event_graph().analyze_sensitivity(
            parameter_type="edge_weights", num_simulations=1
        )
        self.assertEqual(result["perturbation"], 0.1)
        self.assertEqual(result["parameters_analyzed"], 2)


class TestElasticitySign(unittest.TestCase):
    """Elasticity is a magnitude for every parameter type, including negative baselines."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def _chain_graph(self, first_cost, second_cost, payoff):
        """Deterministic three-node chain 1 -> 2 -> 3 with a single terminal payoff."""
        return Graph().from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 0, "after": [{"node_id": 1, "cost": first_cost}]},
                3: {"payoff": payoff, "after": [{"node_id": 2, "cost": second_cost}]},
            }
        )

    def _analyze_all(self, graph):
        # One path and fixed payoffs, so a single simulation per arm is exact.
        return graph.identify_critical_parameters(
            num_simulations=1, perturbation=0.1, top_n=100, max_params=None
        )

    def test_negative_baseline_elasticities_are_non_negative(self):
        result = self._analyze_all(self._chain_graph(100, 5, 10))

        self.assertAlmostEqual(result["baseline_ev"], -95.0)
        self.assertEqual(result["total_parameters_analyzed"], 5)
        self.assertEqual(len(result["top_parameters"]), 5)
        for param in result["top_parameters"]:
            self.assertGreaterEqual(param["elasticity"], 0, param["parameter"])

    def test_elasticity_is_sensitivity_over_absolute_baseline(self):
        result = self._analyze_all(self._chain_graph(100, 5, 10))
        baseline_ev = result["baseline_ev"]
        by_name = {param["parameter"]: param for param in result["top_parameters"]}

        cost = by_name["Edge 1→2 cost"]
        payoff = by_name["Node 3 payoff"]

        # Magnitudes first, so the shared formula cannot be satisfied by two zeros.
        self.assertAlmostEqual(cost["sensitivity"], 10.0)
        self.assertAlmostEqual(payoff["sensitivity"], 1.0)
        for param in (cost, payoff):
            self.assertAlmostEqual(param["elasticity"], param["sensitivity"] / abs(baseline_ev))

    def test_zero_baseline_yields_zero_elasticity_for_every_type(self):
        graph = self._chain_graph(10, 0, 10)

        for parameter_type in ("edge_weights", "costs", "payoffs"):
            analysis = graph.analyze_sensitivity(
                parameter_type=parameter_type, num_simulations=1, perturbation=0.1
            )
            self.assertEqual(analysis["baseline_ev"], 0)
            self.assertGreaterEqual(analysis["parameters_analyzed"], 1)
            for param in analysis["results"]:
                self.assertEqual(param["elasticity"], 0, param["parameter"])

    def test_positive_baseline_elasticity_is_unchanged(self):
        result = self._analyze_all(self._chain_graph(5, 0, 100))
        baseline_ev = result["baseline_ev"]

        self.assertAlmostEqual(baseline_ev, 95.0)
        for param in result["top_parameters"]:
            # abs() is a no-op above zero, so these are the pre-fix values.
            self.assertAlmostEqual(param["elasticity"], param["sensitivity"] / baseline_ev)


if __name__ == "__main__":
    unittest.main()
