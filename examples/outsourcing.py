import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from petersburg import Graph

plt.style.use('ggplot')

__author__ = 'willmcginnis'


def simulate(c_switch):
    data = []
    in_house = 1
    third_party = in_house * 0.80
    switching = in_house * c_switch
    weights = np.linspace(0.1, 1, 100)
    for weight in weights:
        g = Graph()
        g.from_dict({
              1: {'payoff': 0, 'after': []},
              2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': in_house}]},
              3: {'payoff': 0, 'after': [{'node_id': 2, 'cost': in_house}]},
              4: {'payoff': 0, 'after': [{'node_id': 1, 'cost': third_party}]},
              5: {'payoff': 0, 'after': [{'node_id': 4, 'cost': third_party}]},
              6: {'payoff': 0, 'after': [{'node_id': 4, 'cost': switching, 'weight': weight}]},
              7: {'payoff': 0, 'after': [{'node_id': 6, 'cost': in_house}, {'node_id': 5, 'cost': 2 * switching, 'weight': weight}]},
              8: {'payoff': 0, 'after': [{'node_id': 3, 'cost': in_house}]},
              9: {'payoff': 0, 'after': [{'node_id': 5, 'cost': third_party}]},
              10: {'payoff': 0, 'after': [{'node_id': 7, 'cost': in_house}, {'node_id': 5, 'cost': 3 * switching, 'weight': weight}]},
              11: {'payoff': 0, 'after': [{'node_id': 8, 'cost': 0}]},
              12: {'payoff': 0, 'after': [{'node_id': 9, 'cost': 0}]},
              13: {'payoff': 0, 'after': [{'node_id': 10, 'cost': 0}]},
        })

        options = g.get_options(iters=1000)
        data.append([weight / (1.0 + weight), options[2] - options[4]])

    return data


def plot(c_switch):
    data = simulate(c_switch)
    df = pd.DataFrame(data, columns=['weight', 'in_house - third_party'])
    df.plot(kind='scatter', x='weight', y='in_house - third_party')
    plt.xlabel('Probability of Failure for 3rd Party Nodes')
    plt.ylabel('Strength of In-House Option')
    plt.title('In House Vs. 3rd Party For c(x)=%2.1f*M*x' % (c_switch))
    plt.grid()
    plt.show()


def get_breakeven(c_switch, threshold=0.05):
    data = simulate(c_switch)
    break_even = sorted([(x[0], abs(x[1])) for x in data], key=lambda x: x[1])
    if break_even[0][1] < threshold:
        return break_even[0][0]
    else:
        return None

if __name__ == '__main__':
    breakevens = []
    for c_switch in np.linspace(0.01, 0.99, 50):
        breakevens.append((1 - c_switch, get_breakeven(c_switch)))

    df = pd.DataFrame(breakevens, columns=['c_switch', 'breakeven probability'])
    df.plot(kind='scatter', x='c_switch', y='breakeven probability')
    plt.xlabel('Portion of Work Salvaged on Revert')
    plt.ylabel('Breakeven Probability for In-House Strength')
    plt.title('Breakeven Probabilities At Different Salvage Rates')
    plt.grid()
    plt.show()

