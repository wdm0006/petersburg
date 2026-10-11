"""Exact analysis walkthrough: build, export, evaluate, rank assumptions, chart.

A launch-or-hold decision with fixed payoffs and numeric weights, so every number
printed here is exact and identical on every run. Needs only the `examples` extra.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from petersburg import Graph


def build_launch_graph():
    """Start -> launch (cost 10) or hold (payoff 6); a launch succeeds 3 times in 4."""
    return Graph().from_dict(
        {
            1: {"after": []},
            2: {"after": [{"node_id": 1, "cost": 10}]},
            3: {"payoff": 6, "after": [{"node_id": 1}]},
            4: {"payoff": 100, "after": [{"node_id": 2, "weight": 3}]},
            5: {"after": [{"node_id": 2, "cost": 20, "weight": 1}]},
        }
    )


def analyze(graph, directory):
    """Return the exact figures for ``graph`` and save a tornado chart in ``directory``."""
    distribution = graph.outcome_distribution()
    report = graph.exact_sensitivity(perturbation=0.25)
    figure = graph.plot_sensitivity(report, top_n=5, filename=Path(directory) / "tornado.png")
    plt.close(figure)
    return {
        "expected_value": graph.expected_value(),
        "options": graph.expected_options(),
        "std": graph.std(),
        "p_loss": sum(p for outcome, p in distribution.items() if outcome < 0),
        "option_p_loss": {
            option: sum(p for outcome, p in dist.items() if outcome < 0)
            for option, dist in graph.option_distributions().items()
        },
        "ranking": [(r["parameter"], r["sensitivity"]) for r in report["results"]],
        "mermaid": graph.to_mermaid(),
    }


def main():
    graph = build_launch_graph()
    with TemporaryDirectory() as directory:
        facts = analyze(graph, directory)
        print(facts["mermaid"])
        print(f"Expected value: {facts['expected_value']:.2f}")
        print(f"Std deviation:  {facts['std']:.2f}")
        print(f"P(loss):        {facts['p_loss']:.3f}")
        for option, value in facts["options"].items():
            print(f"Option {option}: EV {value:.2f}, P(loss) {facts['option_p_loss'][option]:.3f}")
        print("Exact sensitivity ranking (+/-25%):")
        for parameter, sensitivity in facts["ranking"]:
            print(f"  {parameter}: {sensitivity:.3f}")
        print(f"Tornado chart saved to {Path(directory) / 'tornado.png'} (temporary)")


if __name__ == "__main__":
    main()
