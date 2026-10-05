"""Create a sensitivity tornado with petersburg[visualization]; no Graphviz needed."""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from petersburg import Graph


def main():
    graph = Graph(random_state=7).from_dict(
        {
            1: {"payoff": 8, "after": []},
            2: {"payoff": 32, "after": [{"node_id": 1, "cost": 16}]},
            3: {"payoff": 64, "after": [{"node_id": 2, "cost": 4}]},
        }
    )
    report = graph.identify_critical_parameters(num_simulations=1, perturbation=0.25)
    with TemporaryDirectory() as directory:
        figure = graph.plot_sensitivity(report, filename=Path(directory) / "tornado.png")
        assert [bar.get_width() for bar in figure.axes[0].patches] == [16, 8, 4, 2, 1]
        plt.close(figure)
    print("Sensitivity tornado rendered successfully")


if __name__ == "__main__":
    main()
