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
        2: {'payoff': 0, 'after': [{'node_id': 1}]},
        3: {'payoff': 0, 'after': [{'node_id': 1}]},
        4: {'payoff': 100, 'after': [{'node_id': 2, 'cost': 0}, {'node_id': 3, 'cost': 0}]},
        5: {'payoff': 50, 'after': [{'node_id': 3, 'cost': 0}, {'node_id': 2, 'cost': 0}]},
    })

    data_2 = []
    data_3 = []
    for iter in [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 1000000]:
        outcomes = g.get_options(iters=iter, extended_stats=True)
        data_2.append([
            outcomes[2]['count'],
            outcomes[2]['mean'],
            outcomes[2]['min'],
            outcomes[2]['max']
        ])

        data_3.append([
            outcomes[3]['count'],
            outcomes[3]['mean'],
            outcomes[3]['min'],
            outcomes[3]['max']
        ])

    df = pd.DataFrame(data_2, columns=['iters', 'outcome', 'min', 'max'])
    df2 = pd.DataFrame(data_3, columns=['iters', 'outcome', 'min', 'max'])
    ax = df.plot(kind='line', x='iters', y='outcome', label='switch', color='blue')
    df.plot(ax=ax, kind='scatter', x='iters', y='min', s=100, marker='+', color='blue')
    df.plot(ax=ax, kind='scatter', x='iters', y='max', s=100, marker='+', color='blue')

    df2.plot(ax=ax, kind='line', x='iters', y='outcome', label='don\'t switch', color='red')
    df2.plot(ax=ax, kind='scatter', x='iters', y='min', s=100, marker='+', color='red')
    df2.plot(ax=ax, kind='scatter', x='iters', y='max', s=100, marker='+', color='red')
    plt.title('Convergence of Simulation Outcome')
    plt.xlabel('Number of Iterations in Simulation')
    plt.ylabel('Outcome')
    plt.xscale('log')
    plt.grid(True, color='w', linestyle='-', linewidth=1)
    plt.gca().patch.set_facecolor('0.8')
    plt.show()