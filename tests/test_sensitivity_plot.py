"""Sensitivity charts use the report's exact magnitudes and ranking."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from petersburg import Graph, ValidationError

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _chain():
    return Graph(random_state=7).from_dict(
        {
            1: {"payoff": 8, "after": []},
            2: {"payoff": 32, "after": [{"node_id": 1, "cost": 16}]},
            3: {"payoff": 64, "after": [{"node_id": 2, "cost": 4}]},
        }
    )


class TestSensitivityPlotDependency(unittest.TestCase):
    def test_missing_dependency_names_extra(self):
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with self.assertRaisesRegex(ImportError, r"petersburg\[visualization\]"):
                _chain().plot_sensitivity()

    def test_invalid_top_n_precedes_analysis(self):
        graph = _chain()
        with patch.object(graph, "identify_critical_parameters", side_effect=AssertionError):
            for value in (0, -1, 1.5, float("nan"), None, True):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValidationError, "top_n"):
                        graph.plot_sensitivity(top_n=value)


@unittest.skipUnless(_HAS_MATPLOTLIB, "requires visualization extra")
class TestSensitivityPlot(unittest.TestCase):
    def setUp(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self.plt = plt
        self.addCleanup(plt.close, "all")
        self.graph = _chain()

    def test_combined_report_exact_values_order_title_and_save(self):
        report = self.graph.identify_critical_parameters(num_simulations=1, perturbation=0.25)
        report["top_parameters"].reverse()
        original = list(report["top_parameters"])
        with tempfile.TemporaryDirectory() as directory, patch.object(self.plt, "show") as show:
            filename = Path(directory) / "chart.png"
            figure = self.graph.plot_sensitivity(report, filename=filename, color="black")
            self.assertGreater(filename.stat().st_size, 0)
            show.assert_not_called()
        axes = figure.axes[0]
        self.assertEqual([bar.get_width() for bar in axes.patches], [16, 8, 4, 2, 1])
        self.assertEqual(
            [label.get_text() for label in axes.get_yticklabels()],
            ["Node 3 payoff", "Node 2 payoff", "Edge 1→2 cost", "Node 1 payoff", "Edge 2→3 cost"],
        )
        self.assertTrue(axes.yaxis_inverted())
        self.assertIn("baseline_ev=84.0", axes.get_title())
        self.assertIn(f"analysis_seed={report['analysis_seed']}", axes.get_title())
        self.assertEqual(report["top_parameters"], original)
        self.assertEqual(axes.patches[0].get_facecolor(), (0, 0, 0, 1))

    def test_single_type_report_and_truncation(self):
        report = self.graph.analyze_sensitivity("costs", num_simulations=1, perturbation=0.25)
        figure = self.graph.plot_sensitivity(report, top_n=1)
        self.assertEqual([bar.get_width() for bar in figure.axes[0].patches], [4])
        self.assertEqual(figure.axes[0].get_yticklabels()[0].get_text(), "Edge 1→2 cost")

    def test_omitted_report_computes_combined_report(self):
        report = self.graph.identify_critical_parameters(num_simulations=1, perturbation=0.25)
        with patch.object(
            self.graph, "identify_critical_parameters", return_value=report
        ) as analyze:
            figure = self.graph.plot_sensitivity(top_n=4)
        analyze.assert_called_once_with(top_n=4)
        self.assertEqual([bar.get_width() for bar in figure.axes[0].patches], [16, 8, 4, 2])

    def test_empty_reports_raise_without_creating_figure(self):
        before = self.plt.get_fignums()
        for report in ({"results": []}, {"top_parameters": []}, {}):
            with self.subTest(report=report):
                with self.assertRaisesRegex(ValidationError, "no parameters"):
                    self.graph.plot_sensitivity(report)
        with self.assertRaisesRegex(ValidationError, "no parameters"):
            Graph().from_dict({1: {"after": []}}).plot_sensitivity()
        self.assertEqual(self.plt.get_fignums(), before)
