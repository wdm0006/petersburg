"""The exact-analysis walkthrough prints hand-computed numbers and writes only to a temp dir."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "exact_analysis.py"
_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _load_example():
    spec = importlib.util.spec_from_file_location("exact_analysis_example", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_HAS_MATPLOTLIB, "requires the visualization extra")
class TestExactAnalysisExample(unittest.TestCase):
    def setUp(self):
        self.example = _load_example()

    def test_figures_match_hand_computation(self):
        # launch: 0.75 * 100 + 0.25 * (0 - 20) - 10 = 60; hold: 6; start weights are equal
        with tempfile.TemporaryDirectory() as directory:
            facts = self.example.analyze(self.example.build_launch_graph(), directory)
            self.assertTrue((Path(directory) / "tornado.png").is_file())
        self.assertEqual(facts["expected_value"], 33.0)
        self.assertEqual(facts["options"], {2: 60.0, 3: 6.0})
        # E[x^2] = 0.375 * 90^2 + 0.125 * 30^2 + 0.5 * 6^2 = 3168; variance 3168 - 33^2 = 2079
        self.assertAlmostEqual(facts["std"], 2079**0.5)
        self.assertEqual(facts["p_loss"], 0.125)
        self.assertEqual(facts["option_p_loss"], {2: 0.25, 3: 0.0})

    def test_sensitivity_ranking_matches_hand_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            facts = self.example.analyze(self.example.build_launch_graph(), directory)
        ranking = dict(facts["ranking"])
        # +/-25% of a payoff or cost shifts EV by 0.25 * value * P(reached)
        self.assertEqual(ranking["Node 4 payoff"], 0.25 * 100 * 0.375)
        self.assertEqual(ranking["Edge 1→2 cost"], 0.25 * 10 * 0.5)
        self.assertEqual(ranking["Node 3 payoff"], 0.25 * 6 * 0.5)
        self.assertEqual(ranking["Edge 2→5 cost"], 0.25 * 20 * 0.125)
        self.assertEqual(facts["ranking"][0][0], "Node 4 payoff")

    def test_mermaid_export_is_included(self):
        with tempfile.TemporaryDirectory() as directory:
            facts = self.example.analyze(self.example.build_launch_graph(), directory)
        self.assertTrue(facts["mermaid"].startswith("graph LR"))

    def test_script_runs_without_writing_tracked_files(self):
        before = subprocess.run(
            ["git", "status", "--porcelain"], cwd=_EXAMPLE.parent, capture_output=True, text=True
        ).stdout
        result = subprocess.run(
            [sys.executable, str(_EXAMPLE)],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Expected value: 33.00", result.stdout)
        after = subprocess.run(
            ["git", "status", "--porcelain"], cwd=_EXAMPLE.parent, capture_output=True, text=True
        ).stdout
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
