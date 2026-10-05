import json
import subprocess
import sys
import unittest

import numpy as np

from petersburg.exceptions import ValidationError
from petersburg.graph import Graph


def _mixed_spec():
    return {
        1: {"after": []},
        2: {"payoff": 3, "after": [{"node_id": 1, "cost": 1.5, "weight": 2}]},
        3: {
            "type": "uniform",
            "min_payoff": 1,
            "max_payoff": 4,
            "after": [{"node_id": 1, "cost": 2, "weight": 1}, {"node_id": 1, "cost": 5}],
        },
        4: {"type": "gaussian", "mean": 2.5, "std": 1, "after": [{"node_id": 2}]},
        5: {"type": "lognormal", "mu": 0.5, "sigma": 0.25, "after": [{"node_id": 3}]},
        6: {"type": "powerlaw", "scale": 2, "alpha": 3, "after": [{"node_id": 4}]},
    }


def _roundtrip(g):
    return Graph().from_json(g.to_json())


class TestJsonRoundTrip(unittest.TestCase):
    def test_mixed_types_and_parallel_edges(self):
        g = Graph().from_dict(_mixed_spec())
        self.assertEqual(_roundtrip(g).to_dict(), g.to_dict())

    def test_int_ids_keep_type(self):
        g = Graph().from_dict(_mixed_spec())
        ids = [node.node_id for node in _roundtrip(g).node_list()]
        self.assertTrue(ids)
        self.assertTrue(all(type(i) is int for i in ids))

    def test_str_ids_keep_type(self):
        spec = {
            "start": {"after": []},
            "1": {"payoff": 5, "after": [{"node_id": "start", "cost": 1}]},
        }
        g = Graph().from_dict(spec)
        rebuilt = _roundtrip(g)
        self.assertEqual(rebuilt.to_dict(), g.to_dict())
        self.assertEqual({type(n.node_id) for n in rebuilt.node_list()}, {str})
        self.assertIn("1", rebuilt.to_dict())

    def test_plain_dumps_breaks_int_ids(self):
        g = Graph().from_dict(_mixed_spec())
        loaded = json.loads(json.dumps(g.to_dict()))
        self.assertNotIn(1, loaded)

    def test_adj_matrix_numpy_weights(self):
        g = Graph().from_adj_matrix(np.array([[0, 2, 1], [0, 0, 1], [0, 0, 0]]))
        with self.assertRaises(TypeError):
            json.dumps(g.to_dict())
        rebuilt = _roundtrip(g)
        self.assertEqual(rebuilt.to_dict(), g.to_dict())
        self.assertTrue(all(type(n.node_id) is int for n in rebuilt.node_list()))

    def test_document_shape_and_indent(self):
        g = Graph().from_dict(_mixed_spec())
        doc = json.loads(g.to_json(indent=2))
        self.assertEqual((doc["format"], doc["version"]), ("petersburg-graph", 1))
        self.assertEqual(doc["nodes"][0]["id"], 1)
        self.assertIn("\n", g.to_json(indent=2))
        self.assertNotIn("\n", g.to_json())


class TestJsonDeterminism(unittest.TestCase):
    def test_same_string_twice(self):
        g = Graph().from_dict(_mixed_spec())
        self.assertEqual(g.to_json(), g.to_json())

    def test_across_hash_seeds(self):
        script = "\n".join(
            [
                "from petersburg.graph import Graph",
                'spec = {"a": {"after": []}}',
                'for name in ["b", "c", "d", "e", "f"]:',
                '    spec[name] = {"payoff": 1, "after": [{"node_id": "a"}]}',
                "print(Graph().from_dict(spec).to_json())",
            ]
        )
        outputs = set()
        for seed in ("0", "1", "2"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PATH": ""},
            )
            outputs.add(result.stdout)
        self.assertEqual(len(outputs), 1)


class TestJsonErrors(unittest.TestCase):
    def test_tuple_id_rejected(self):
        g = Graph().from_dict({(0, 1): {"after": []}})
        with self.assertRaises(ValidationError):
            g.to_json()

    def test_classifier_weight_rejected(self):
        class Clf:
            def predict_proba(self, x):
                return [[0.5, 0.5]]

        g = Graph().from_dict({1: {"after": []}, 2: {"after": [{"node_id": 1}]}})
        g.start_node.outcomes[0] = (g.start_node.outcomes[0][0], Clf())
        with self.assertRaises(ValidationError):
            g.to_json()

    def test_unbuilt_graph_rejected(self):
        with self.assertRaises(ValidationError):
            Graph().to_json()

    def _doc(self, **overrides):
        doc = json.loads(Graph().from_dict(_mixed_spec()).to_json())
        doc.update(overrides)
        return json.dumps(doc)

    def test_unknown_format_and_version(self):
        for text in (
            self._doc(format="other"),
            self._doc(version=2),
            self._doc(version=True),
            self._doc(version="1"),
            json.dumps([1]),
            "not json",
        ):
            with self.subTest(text=text[:30]), self.assertRaises(ValidationError):
                Graph().from_json(text)

    def test_malformed_nodes(self):
        for nodes in ({"1": {}}, [{"type": "fixed"}], [{"id": [1], "after": []}]):
            with self.subTest(nodes=nodes), self.assertRaises(ValidationError):
                Graph().from_json(self._doc(nodes=nodes))

    def test_duplicate_ids(self):
        nodes = [{"id": 1, "after": []}, {"id": 1, "after": []}]
        with self.assertRaises(ValidationError):
            Graph().from_json(self._doc(nodes=nodes))

    def test_failed_load_preserves_start_node(self):
        g = Graph().from_dict(_mixed_spec())
        start = g.start_node
        cyclic = [
            {"id": 1, "after": []},
            {"id": 2, "after": [{"node_id": 3}]},
            {"id": 3, "after": [{"node_id": 2}]},
        ]
        for text in (self._doc(format="x"), self._doc(nodes=cyclic)):
            with self.assertRaises((ValidationError, AttributeError)):
                g.from_json(text)
            self.assertIs(g.start_node, start)


if __name__ == "__main__":
    unittest.main()
