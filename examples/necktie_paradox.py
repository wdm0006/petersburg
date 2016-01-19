import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('ggplot')
import pandas as pd
from petersburg import Graph

__author__ = 'willmcginnis'

if __name__ == '__main__':
    g = Graph()

    # necktie paradox
    g.from_dict({
        1: {'payoff': 0, 'after': []},
        2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 40}]},
        3: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 50}]},
        4: {'payoff': 50, 'after': [{'node_id': 2, 'cost': 0}]},
        5: {'payoff': 40, 'after': [{'node_id': 3, 'cost': 0}]},
    })

    data = []
    for iter in [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 1000000]:
        outcomes = []
        for _ in range(iter):
            outcomes.append(g.get_outcome())
        data.append([iter,
                     float(sum(outcomes))/len(outcomes),
                     min(outcomes),
                     max(outcomes)
                     ])

    df = pd.DataFrame(data, columns=['iters', 'outcome', 'min', 'max'])
    ax = df.plot(kind='line', x='iters', y='outcome')
    df.plot(ax=ax, kind='scatter', x='iters', y='min', s=100, marker='+', color='red')
    df.plot(ax=ax, kind='scatter', x='iters', y='max', s=100, marker='+', color='red')
    plt.title('Convergence of Simulation Outcome')
    plt.xlabel('Number of Iterations in Simulation')
    plt.ylabel('Outcome')
    plt.xscale('log')
    plt.grid(True, color='w', linestyle='-', linewidth=1)
    plt.gca().patch.set_facecolor('0.8')
    plt.show()