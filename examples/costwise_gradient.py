import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('ggplot')
import pandas as pd
from petersburg import Graph

__author__ = 'willmcginnis'


def build_graph(node_id=None):
    """
    returns a list of petersburg graph objects with a certain object perturbed
    """
    # weights
    w1, w2, w3, w4, w5, w6 = 10, 10, 10, 10, 10, 10
    w7 = random.randint(1, 10)
    w8 = random.randint(1, 10)
    w9 = random.randint(1, 10)
    w10 = random.randint(1, 10)
    w11 =random.randint(1, 10)

    # costs
    c1, c2, c3, c4, c5, c6 = 0, 0, 0, 0, 0, 0
    c7 = random.randint(1, 10)
    c8 = random.randint(1, 10)
    c9 = random.randint(1, 10)
    c10 = random.randint(1, 10)
    c11 = random.randint(1, 10)

    template = {
            1: {'payoff': 0, 'after': []},
            2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': c1, 'weight': w1}]},
            3: {'payoff': 0, 'after': [{'node_id': 1, 'cost': c2, 'weight': w2}]},
            4: {'payoff': 0, 'after': [{'node_id': 3, 'cost': c3, 'weight': w3}]},
            5: {'payoff': 0, 'after': [{'node_id': 3, 'cost': c4, 'weight': w4}]},
            6: {'payoff': 0, 'after': [{'node_id': 3, 'cost': c5, 'weight': w5}]},
            7: {'payoff': 0, 'after': [{'node_id': 4, 'cost': c6, 'weight': w6}]},
            8: {'payoff': 0, 'after': [{'node_id': 4, 'cost': c7, 'weight': w7}]},
            9: {'payoff': 0, 'after': [{'node_id': 5, 'cost': c8, 'weight': w8}]},
            10: {'payoff': 0, 'after': [{'node_id': 5, 'cost': c9, 'weight': w9}]},
            11: {'payoff': 0, 'after': [{'node_id': 6, 'cost': c10, 'weight': w10}]},
            12: {'payoff': 0, 'after': [{'node_id': 6, 'cost': c11, 'weight': w11}]},
    }

    if node_id is None:
        return [(None, Graph().from_dict(template))]

    out = []
    default = template[node_id]['after'][0]['weight']
    for x in np.linspace(0.1, default + 1, 25):
        template[node_id]['after'][0]['weight'] = x
        g = Graph()
        g.from_dict(template)
        out.append((x, g))

    return out

if __name__ == '__main__':
    # find the high opportunity edges
    data = []
    for nid in range(7, 13):
        graphs = build_graph(nid)
        means = []
        mins = []
        maxes = []
        likelyhoods = []
        for weight, graph in graphs:
            # simulate some games in our new fun graph
            outcomes = []
            for _ in range(10000):
                outcomes.append(graph.get_outcome())

            # find the distribution of end-nodes
            iters = 10000
            output_nodes = {2: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
            for _ in range(iters):
                outnid = graph.get_outcome_node()
                output_nodes[outnid] += 1

            likelyhoods.append(output_nodes[nid])
            means.append(float(sum(outcomes))/len(outcomes))
            mins.append(min(outcomes))
            maxes.append(max(outcomes))
        data.append([nid, max(means) - min(means), min(likelyhoods), max(likelyhoods), float(max(means) - min(means))/(max(likelyhoods) - min(likelyhoods)) * 100])
    df = pd.DataFrame(data, columns=['node_id', 'max_change', 'min_likelyhood', 'max_likelyhood', 'gradient'])

    # find the as-is likelyhoods
    iters = 10000
    graph = build_graph(None)[0][1]
    output_nodes = {2: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    for _ in range(iters):
        outnid = graph.get_outcome_node()
        output_nodes[outnid] += 1

    data = []
    for node_id in output_nodes.keys():
        if node_id != 2:
            data.append([node_id, float(output_nodes[node_id]) / iters])
    df2 = pd.DataFrame(data, columns=['node_id', 'frequency'])

    # plot it all
    ax = df.plot(kind='scatter', x='node_id', y='gradient', s=50)
    ax2 = ax.twinx()
    df2.plot(ax=ax2, kind='scatter', x='node_id', y='frequency', s=50, color='red')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.title('Gradient of Expected Change By Edge')
    plt.xlabel('Edge Output Node ID')
    ax.set_ylabel('Expected Gradient in Outcome Vs. Likelyhood')
    ax2.set_ylabel('Simulated Likelyhood in As-Is State')
    ax.grid(True, color='w', linestyle='-', linewidth=1)
    ax2.grid(False)
    ax.set_ylim([df['gradient'].min() * 0.8, df['gradient'].max() * 1.2])
    plt.gca().patch.set_facecolor('0.8')
    plt.tight_layout()
    plt.show()
