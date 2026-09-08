"""
Tests for graph serialization hygiene: Graph.to_dict() round trips against from_dict(),
petersburg.__version__ resolves from package metadata, and plot()'s import guard points
at the [graphviz] extra when pygraphviz is missing.
"""

import builtins
import importlib.metadata
import re
from pathlib import Path

import pytest

import petersburg
from petersburg import Graph, PetersburgError, ValidationError

__author__ = "willmcginnis"


class _ClassifierWeight:
    """Stand-in for a fitted estimator attached as a transition weight."""

    def predict_proba(self, feature_vector):
        return [[0.25, 0.75]]


def _numeric_graph():
    """A seeded graph with non-default numeric weights, costs, and fixed payoffs."""
    return Graph(random_state=7).from_dict(
        {
            1: {"payoff": 0, "after": []},
            2: {"payoff": 10, "after": [{"node_id": 1, "cost": 3, "weight": 2}]},
            3: {"payoff": -4, "after": [{"node_id": 1, "cost": 0, "weight": 1}]},
            4: {
                "payoff": 25,
                "after": [
                    {"node_id": 2, "cost": 1, "weight": 3},
                    {"node_id": 3, "cost": 2, "weight": 1},
                ],
            },
        }
    )


def _distribution_graph():
    """A seeded graph exercising every node type's payoff parameters."""
    return Graph(random_state=7).from_dict(
        {
            1: {"payoff": 0, "after": []},
            2: {
                "type": "uniform",
                "min_payoff": -5,
                "max_payoff": 5,
                "after": [{"node_id": 1, "cost": 1, "weight": 1}],
            },
            3: {
                "type": "gaussian",
                "mean": 100,
                "std": 25,
                "after": [{"node_id": 1, "cost": 0, "weight": 2}],
            },
            4: {
                "type": "lognormal",
                "mu": 1.0,
                "sigma": 0.5,
                "after": [{"node_id": 3, "cost": 2, "weight": 1}],
            },
            5: {
                "type": "powerlaw",
                "scale": 2,
                "alpha": 3,
                "after": [{"node_id": 2, "cost": 0, "weight": 3}],
            },
        }
    )


def _attach_classifier_weight(g):
    """Attach a classifier weight to one of the graph's edges deterministically."""
    nodes = {node.node_id: node for node in g.start_node.get_nodes(set())}
    g.start_node.add_outcome(nodes[2], classifier=_ClassifierWeight())
    return g


def test_round_trip_numeric_weights_costs_fixed_payoffs():
    g = _numeric_graph()
    spec = g.to_dict()

    rebuilt = Graph(random_state=7).from_dict(spec)

    assert rebuilt.to_dict() == spec


def test_to_dict_emits_from_dict_format():
    spec = _numeric_graph().to_dict()

    assert spec[2]["after"] == [{"node_id": 1, "cost": 3, "weight": 2}]
    assert spec[1]["type"] == "fixed"
    assert spec[1]["payoff"] == 0


def test_round_trip_distribution_payoffs():
    g = _distribution_graph()
    spec = g.to_dict()

    rebuilt = Graph(random_state=7).from_dict(spec)

    assert rebuilt.to_dict() == spec


def test_round_trip_parallel_edges():
    g = Graph(random_state=7).from_dict(
        {
            1: {"payoff": 0, "after": []},
            2: {
                "payoff": 5,
                "after": [
                    {"node_id": 1, "cost": 1, "weight": 2},
                    {"node_id": 1, "cost": 4, "weight": 5},
                ],
            },
        }
    )
    spec = g.to_dict()

    rebuilt = Graph(random_state=7).from_dict(spec)

    assert rebuilt.to_dict() == spec
    assert sorted(edge["weight"] for edge in spec[2]["after"]) == [2, 5]


def test_to_dict_requires_built_graph():
    with pytest.raises(PetersburgError, match="from_dict"):
        Graph().to_dict()


def test_estimator_object_weights_raise_petersburg_error():
    g = Graph(random_state=7).from_dict(
        {
            1: {"payoff": 0, "after": []},
            2: {"payoff": 10, "after": [{"node_id": 1, "cost": 1, "weight": 1}]},
        }
    )
    _attach_classifier_weight(g)

    with pytest.raises(PetersburgError, match="estimator"):
        g.to_dict()


def test_validation_error_is_raised_for_classifier_weights():
    g = Graph(random_state=7).from_dict(
        {
            1: {"payoff": 0, "after": []},
            2: {"payoff": 10, "after": [{"node_id": 1, "cost": 1, "weight": 1}]},
        }
    )
    _attach_classifier_weight(g)

    # ValidationError is the precise type; the PetersburgError test above pins the hierarchy.
    with pytest.raises(ValidationError, match="cannot round-trip"):
        g.to_dict()


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    match = re.search(r'^version\s*=\s*"(.+)"', pyproject.read_text(), re.MULTILINE)
    assert match, "pyproject.toml must declare a version"

    assert petersburg.__version__ == match.group(1)


def test_version_falls_back_when_metadata_missing(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        assert importlib.reload(petersburg).__version__ == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(petersburg)


def test_plot_names_graphviz_extra_when_pygraphviz_missing(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "pygraphviz":
            raise ImportError("No module named 'pygraphviz'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.setattr(Graph, "to_networkx", lambda self: object())

    g = _numeric_graph()
    with pytest.raises(ImportError, match=r"\[graphviz\]"):
        g.plot(str(tmp_path / "graph.png"))


def test_plot_renders_when_pygraphviz_available(tmp_path):
    pytest.importorskip("pygraphviz")
    pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")

    out = tmp_path / "graph.png"
    _numeric_graph().plot(str(out))

    assert out.exists()
