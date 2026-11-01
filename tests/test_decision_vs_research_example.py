"""The perfect-information example reproduces exact values from one joint table."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

from petersburg.exceptions import ValidationError

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "decision_vs_research.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("decision_vs_research_example", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDecisionVsResearchExample(unittest.TestCase):
    def setUp(self):
        self.example = _load_example()

    def test_crossing_payoffs_values_and_choices(self):
        result = self.example.compare_information()
        none, perfect = result["no_information"], result["perfect_information"]
        self.assertEqual(none["optimal_value"], 15.0)
        self.assertEqual(none["expected_value"], 15.0)
        self.assertEqual(none["policy"], {"start": "act A"})
        self.assertEqual(perfect["optimal_value"], 25.0)
        self.assertEqual(perfect["expected_value"], 25.0)
        self.assertEqual(
            perfect["policy"],
            {"observe scenario 1": "A | scenario 1", "observe scenario 2": "B | scenario 2"},
        )
        self.assertEqual(result["gross_evpi"], 10.0)
        self.assertEqual(result["net_improvement"], 5.0)

    def test_net_improvement_by_cost(self):
        for cost, net in ((0, 10.0), (10, 0.0), (15, -5.0)):
            result = self.example.compare_information(information_cost=cost)
            self.assertEqual(result["gross_evpi"], 10.0)
            self.assertEqual(result["net_improvement"], net)

    def test_dominant_action_has_zero_evpi(self):
        result = self.example.compare_information(self.example.DOMINATED_SCENARIOS)
        self.assertEqual(result["no_information"]["policy"], {"start": "act A"})
        self.assertEqual(result["gross_evpi"], 0.0)
        self.assertEqual(result["net_improvement"], -5.0)

    def test_invalid_costs_fail_descriptively(self):
        for bad in (-1, float("nan"), float("inf"), -float("inf"), True, "5", None):
            with self.assertRaisesRegex(ValidationError, "information_cost"):
                self.example.compare_information(information_cost=bad)

    def test_inconsistent_tables_rejected(self):
        with self.assertRaises(ValidationError):
            self.example.compare_information({})
        with self.assertRaises(ValidationError):
            self.example.compare_information({"s1": (0.5, {"A": 1, "B": 2}), "s2": (0.5, {"A": 1})})

    def test_builders_share_table_and_timing(self):
        ex = self.example
        none = ex.build_no_information_graph(ex.SCENARIOS)
        info = ex.build_perfect_information_graph(ex.SCENARIOS)
        # no-information: the decision is the start node, before any chance
        self.assertTrue(none.start_node.is_decision)
        self.assertFalse(info.start_node.is_decision)
        # perfect-information: one distinct decision node per scenario, after chance
        decisions = [n for n in info.node_list() if n.is_decision]
        self.assertEqual(len(decisions), len(ex.SCENARIOS))
        # unresolved graphs refuse evaluation until a policy is applied
        with self.assertRaises(ValidationError):
            none.expected_value()

    def test_forcing_same_action_across_scenarios_changes_value(self):
        # coverage check: independent per-scenario choice on the no-information side would give 25
        ex = self.example
        info = ex.build_perfect_information_graph(ex.SCENARIOS)
        same = {"observe scenario 1": "A | scenario 1", "observe scenario 2": "A | scenario 2"}
        self.assertEqual(info.apply_policy(same).expected_value(), 15.0)

    def test_script_prints_deterministic_output_and_limitation(self):
        runs = [
            subprocess.run(
                [sys.executable, str(_EXAMPLE)], capture_output=True, text=True, check=True
            ).stdout
            for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])
        for fragment in (
            "Gross EVPI: 10.00",
            "Net improvement: 5.00",
            "upper bound",
            "noisy study",
        ):
            self.assertIn(fragment, runs[0])


if __name__ == "__main__":
    unittest.main()
