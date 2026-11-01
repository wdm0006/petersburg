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
        self.assertEqual(result["value_without"], 15.0)
        self.assertEqual(result["value_with"], 25.0)
        self.assertEqual(result["policy_without"], "A")
        self.assertEqual(result["policy_with"], {("scenario 1",): "A", ("scenario 2",): "B"})
        self.assertEqual(result["gross_value"], 10.0)
        self.assertEqual(result["net_value"], 5.0)

    def test_net_value_by_cost(self):
        for cost, net in ((0, 10.0), (10, 0.0), (15, -5.0)):
            result = self.example.compare_information(information_cost=cost)
            self.assertEqual(result["gross_value"], 10.0)
            self.assertEqual(result["net_value"], net)

    def test_dominant_action_has_zero_evpi(self):
        result = self.example.compare_information(self.example.DOMINATED_SCENARIOS)
        self.assertEqual(result["policy_without"], "A")
        self.assertEqual(result["gross_value"], 0.0)
        self.assertEqual(result["net_value"], -5.0)

    def test_invalid_cost_fails_descriptively(self):
        with self.assertRaisesRegex(ValidationError, "cost"):
            self.example.compare_information(information_cost=-1)

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
