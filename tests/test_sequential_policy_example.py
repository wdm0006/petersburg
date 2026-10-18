"""The sequential-policy example matches hand calculations and leaves the model unresolved."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from petersburg.exceptions import ValidationError

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sequential_policy.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("sequential_policy_example", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSequentialPolicyExample(unittest.TestCase):
    def setUp(self):
        self.example = _load_example()
        self.graph = self.example.build_project_graph()
        self.facts = self.example.analyze(self.graph)

    def test_optimal_policy_continues_only_when_favorable(self):
        # favorable: 0.75 * 100 - 20 = 55 > 10; unfavorable: 0.25 * 100 - 20 = 5 < 10
        self.assertEqual(self.facts["optimal"]["policy"], {2: 5, 3: 8})
        self.assertEqual(self.facts["always_continue"]["policy"], {2: 5, 3: 9})

    def test_expected_values_match_hand_calculation(self):
        # optimal: 0.75 * 55 + 0.25 * 10; always continue: 0.75 * 55 + 0.25 * 5
        self.assertEqual(self.facts["optimal_value"], 43.75)
        self.assertEqual(self.facts["optimal"]["expected_value"], 43.75)
        self.assertEqual(self.facts["always_continue"]["expected_value"], 42.5)
        self.assertGreater(
            self.facts["optimal_value"], self.facts["always_continue"]["expected_value"]
        )

    def test_pmfs_and_loss_probabilities(self):
        optimal, always = self.facts["optimal"], self.facts["always_continue"]
        self.assertEqual(optimal["distribution"], {80.0: 0.5625, -20.0: 0.1875, 10.0: 0.25})
        self.assertEqual(always["distribution"], {80.0: 0.625, -20.0: 0.375})
        self.assertEqual(optimal["p_loss"], 0.1875)
        self.assertEqual(always["p_loss"], 0.375)

    def test_substituting_always_continue_fails_the_value_checks(self):
        # coverage check: the assertions above can tell the two policies apart
        optimal, always = self.facts["optimal"], self.facts["always_continue"]
        self.assertNotEqual(optimal["policy"], always["policy"])
        self.assertNotEqual(optimal["expected_value"], always["expected_value"])
        self.assertNotEqual(optimal["distribution"], always["distribution"])

    def test_unresolved_graph_is_unchanged(self):
        fresh = self.example.build_project_graph()
        self.assertEqual(self.graph.to_dict(), fresh.to_dict())
        with self.assertRaises(ValidationError):
            self.graph.expected_value()

    def test_script_runs_without_input_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(_EXAMPLE)],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertIn("optimal_policy() value: 43.75", result.stdout)
        self.assertIn("P(loss): 0.3750", result.stdout)


if __name__ == "__main__":
    unittest.main()
