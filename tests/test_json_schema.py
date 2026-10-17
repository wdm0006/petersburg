import copy
import json
import unittest
from importlib.resources import files

from jsonschema import Draft202012Validator, ValidationError

from petersburg.graph import Graph


def _model():
    return Graph().from_dict(
        {
            1: {"payoff": -2, "after": []},
            "1": {
                "type": "uniform",
                "min_payoff": -3,
                "max_payoff": 5,
                "after": [
                    {"node_id": 1, "cost": 2, "weight": 3},
                    {"node_id": 1, "cost": 4, "weight": 1},
                ],
            },
            "normal": {"type": "gaussian", "mean": 2, "std": 3, "after": [{"node_id": "1"}]},
            4: {"type": "lognormal", "mu": 0.5, "sigma": 0.25, "after": [{"node_id": "1"}]},
            "pareto": {"type": "powerlaw", "scale": 2, "alpha": 3, "after": [{"node_id": 4}]},
        }
    )


class TestJsonSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads(
            files("petersburg")
            .joinpath("schemas/petersburg-graph-v1.schema.json")
            .read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        cls.validator = Draft202012Validator(schema)

    def test_serializer_output(self):
        document = json.loads(_model().to_json())
        self.validator.validate(document)
        self.assertEqual(
            {node["type"] for node in document["nodes"]},
            {"fixed", "uniform", "gaussian", "lognormal", "powerlaw"},
        )
        nodes = {node["id"]: node for node in document["nodes"]}
        self.assertIn(1, nodes)
        self.assertIn("1", nodes)
        self.assertEqual(
            nodes["1"]["after"],
            [
                {"node_id": 1, "cost": 2, "weight": 3},
                {"node_id": 1, "cost": 4, "weight": 1},
            ],
        )

    def test_invalid_documents(self):
        original = json.loads(_model().to_json())
        for field, value in (("version", 2), ("version", True), ("format", "other")):
            with self.subTest(field=field, value=value):
                document = copy.deepcopy(original)
                document[field] = value
                with self.assertRaises(ValidationError):
                    self.validator.validate(document)
        for field, value in (("id", None), ("type", "unknown"), ("after", None), ("payoff", None)):
            with self.subTest(node_field=field):
                document = copy.deepcopy(original)
                if value is None:
                    del document["nodes"][0][field]
                else:
                    document["nodes"][0][field] = value
                with self.assertRaises(ValidationError):
                    self.validator.validate(document)

    def test_required_payoff_and_edge_fields(self):
        original = json.loads(_model().to_json())
        for index, node in enumerate(original["nodes"]):
            parameters = set(node) - {"id", "type", "after"}
            self.assertTrue(parameters)
            for parameter in parameters:
                with self.subTest(node_type=node["type"], parameter=parameter):
                    document = copy.deepcopy(original)
                    del document["nodes"][index][parameter]
                    with self.assertRaises(ValidationError):
                        self.validator.validate(document)
        for field in ("node_id", "cost", "weight"):
            with self.subTest(edge_field=field):
                document = copy.deepcopy(original)
                node = next(node for node in document["nodes"] if node["after"])
                del node["after"][0][field]
                with self.assertRaises(ValidationError):
                    self.validator.validate(document)
