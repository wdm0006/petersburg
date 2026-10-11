"""Exact analytic sensitivity of the expected value."""

import importlib.util
import unittest

import numpy as np

from petersburg import Graph, ValidationError

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

P = 0.1


def _graph(random_state=None):
    return Graph(random_state=random_state).from_dict(
        {
            1: {"payoff": 0, "after": []},
            2: {"payoff": 10, "after": [{"node_id": 1, "cost": 2, "weight": 3}]},
            3: {"payoff": 4, "after": [{"node_id": 1, "weight": 1}]},
        }
    )


def _by_name(report):
    return {r["parameter"]: r for r in report["results"]}


def _ev(w2=3.0, w3=1.0, cost=2.0, pay2=10.0, pay3=4.0):
    p = w2 / (w2 + w3)
    return p * (pay2 - cost) + (1 - p) * pay3


class TestExactSensitivityValues(unittest.TestCase):
    def test_weight_cost_payoff_match_closed_form(self):
        report = _graph().exact_sensitivity(perturbation=P)
        self.assertAlmostEqual(report["baseline_ev"], 7.0)
        rows = _by_name(report)
        cases = {
            "Edge 1→2 weight": (_ev(w2=3 * 1.1), _ev(w2=3 * 0.9), 3),
            "Edge 1→2 cost": (_ev(cost=2.2), _ev(cost=1.8), 2),
            "Node 2 payoff": (_ev(pay2=11.0), _ev(pay2=9.0), 10),
        }
        for name, (inc, dec, original) in cases.items():
            with self.subTest(name):
                row = rows[name]
                sens = (abs(inc - 7.0) + abs(dec - 7.0)) / 2
                self.assertAlmostEqual(row["increased_ev"], inc)
                self.assertAlmostEqual(row["decreased_ev"], dec)
                self.assertAlmostEqual(row["sensitivity"], sens)
                self.assertAlmostEqual(row["elasticity"], sens / 7.0)
                self.assertEqual(row["original_value"], original)

    def test_results_sorted_and_counts(self):
        report = _graph().exact_sensitivity(perturbation=P)
        sens = [r["sensitivity"] for r in report["results"]]
        self.assertEqual(sens, sorted(sens, reverse=True))
        self.assertEqual(report["parameters_analyzed"], len(report["results"]))
        self.assertEqual(len(report["results"]), 2 + 1 + 2)

    def test_parameter_types_filter(self):
        report = _graph().exact_sensitivity(parameter_types=("costs",))
        self.assertEqual([r["parameter"] for r in report["results"]], ["Edge 1→2 cost"])

    def test_sole_outgoing_weight_excluded(self):
        g = Graph().from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 5, "after": [{"node_id": 1, "weight": 2}]},
            }
        )
        report = g.exact_sensitivity(parameter_types=("edge_weights",))
        self.assertEqual(report["results"], [])

    def test_distribution_payoff_allowed(self):
        g = Graph().from_dict(
            {
                1: {"payoff": 0, "after": []},
                2: {"type": "uniform", "min_payoff": 0, "max_payoff": 8, "after": [{"node_id": 1}]},
            }
        )
        row = _by_name(g.exact_sensitivity(parameter_types=("payoffs",)))["Node 2 payoff"]
        self.assertAlmostEqual(row["increased_ev"], 4.4)
        self.assertAlmostEqual(row["decreased_ev"], 3.6)


class TestExactSensitivityDeterminism(unittest.TestCase):
    def _strip(self, report):
        return [
            (r["parameter"], r["increased_ev"], r["decreased_ev"], r["sensitivity"])
            for r in report["results"]
        ]

    def test_repeatable_and_seed_independent(self):
        first = self._strip(_graph(1).exact_sensitivity())
        self.assertEqual(first, self._strip(_graph(1).exact_sensitivity()))
        self.assertEqual(first, self._strip(_graph(99).exact_sensitivity()))
        self.assertEqual(first, self._strip(_graph().exact_sensitivity()))

    def test_rng_and_parameters_restored(self):
        g = _graph(5)
        state = g.rng.bit_generator.state
        before = g.to_dict()
        g.exact_sensitivity()
        self.assertEqual(g.rng.bit_generator.state, state)
        self.assertEqual(g.to_dict(), before)

    def test_restored_when_evaluation_raises(self):
        g = _graph(5)
        before = g.to_dict()
        original = g.expected_value
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 4:
                raise RuntimeError("boom")
            return original()

        g.expected_value = flaky
        with self.assertRaises(RuntimeError):
            g.exact_sensitivity()
        g.expected_value = original
        self.assertEqual(g.to_dict(), before)

    def test_agrees_with_monte_carlo(self):
        g = _graph(3)
        exact = _by_name(g.exact_sensitivity(parameter_types=("costs", "payoffs")))
        sim_rows = {r["parameter"]: r for r in g.analyze_sensitivity("costs", 20000, P)["results"]}
        row = sim_rows["Edge 1→2 cost"]
        self.assertAlmostEqual(
            row["sensitivity"], exact["Edge 1→2 cost"]["sensitivity"], delta=0.05
        )
        self.assertAlmostEqual(row["baseline_ev"], 7.0, delta=0.15)


class TestExactSensitivityValidation(unittest.TestCase):
    def test_invalid_perturbation(self):
        for value in (0, 1, -0.1, 1.5, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    _graph().exact_sensitivity(perturbation=value)

    def test_invalid_parameter_types(self):
        for value in (("payoff",), (), ("costs", "bogus")):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    _graph().exact_sensitivity(parameter_types=value)

    def test_classifier_weight_rejected(self):
        class Clf:
            def predict_proba(self, X):
                return np.array([[0.5, 0.5]])

        g = _graph()
        g.start_node.outcomes[0] = (g.start_node.outcomes[0][0], Clf())
        with self.assertRaises(ValidationError):
            g.exact_sensitivity()

    def test_unbuilt_graph(self):
        with self.assertRaises(ValidationError):
            Graph().exact_sensitivity()


@unittest.skipUnless(_HAS_MATPLOTLIB, "requires visualization extra")
class TestExactSensitivityPlot(unittest.TestCase):
    def test_plot_accepts_report(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        report = _graph().exact_sensitivity()
        figure = _graph().plot_sensitivity(report)
        self.assertEqual(len(figure.axes[0].patches), len(report["results"]))
        plt.close(figure)


class TestReportShapeParity(unittest.TestCase):
    def test_same_keys_as_analyze_sensitivity(self):
        g = _graph(2)
        sim = g.analyze_sensitivity("costs", num_simulations=10)
        exact = g.exact_sensitivity(parameter_types=("costs",))
        self.assertEqual(set(sim) - set(exact), {"parameter_type"})
        self.assertEqual(set(sim["results"][0]), set(exact["results"][0]))
