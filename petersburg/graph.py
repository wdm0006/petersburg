"""
.. module:: graph
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Will McGinnis <will@pedalwrencher.com>


"""

import json
import numpy as np
from petersburg import Node

__author__ = 'willmcginnis'


class Graph(object):
    """
    A graph holds a heirarchy of nodes and edges with payoffs and costs.

    Example:

    >>> from petersburg import Graph
    >>> g = Graph()
    >>> g.from_dict({
    >>>      1: {'payoff': 0, 'after': []},
    >>>      2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10}]},
    >>>      3: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10}]},
    >>>      4: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10}]},
    >>>      5: {'payoff': 0, 'after': [{'node_id': 2, 'cost': 5}, {'node_id': 3, 'cost': 10}]},
    >>>      6: {'payoff': 0, 'after': [{'node_id': 2, 'cost': 5}, {'node_id': 4, 'cost': 10}]},
    >>>      7: {'payoff': 10, 'after': [{'node_id': 5, 'cost': 0}]},
    >>>      8: {'payoff': 3, 'after': [{'node_id': 5, 'cost': 0}]},
    >>>      9: {'payoff': 10, 'after': [{'node_id': 6, 'cost': 0}]},
    >>>      10: {'payoff': 3, 'after': [{'node_id': 6, 'cost': 0}]},
    >>> })

    Which represents a decision between 3 options with differing costs and outcomes. The starting point (of which there
    can only be one, is represented by an empty list in the 'after' key of a node.

    """
    def __init__(self):
        self.start_node = None

    def from_dict(self, d):
        """
        Assembles a graph from a dictionary of nodes and their dependencies. Assumes, directed acyclic and with one
        starting node.  Each node has a payoff, each edge has a cost, and each edge has a weight which corresponds to
        likelyhood of being traversed.

        :param d:
        :return:
        """

        # here we iterate through the dictionary and instantiate all of the node objects specified. One of them (and
        # exactly one of them), should be after nothing, and is the starting node, the rest go into a node list. Edges
        # are handled later.
        node = None
        node_list = {}
        for key in d:
            if not d[key]['after']:
                if node is not None:
                    raise AttributeError('Graph cannot have more than one starting node.')

                node = Node(key, payoff=d[key].get('payoff', 0))
                node_list.update({key: node})
            else:
                node_list.update({key: Node(key, payoff=d[key].get('payoff', 0))})

        if node is None:
            raise AttributeError('Dict must contain a starting node (empty list for after key)')

        # the start node is the entry point for basically everything we will do later on. It's the only part we actually
        # need to keep around in the instance here, because all other nodes will be under it via reference once we
        # process the edges.
        self.start_node = node

        # now that we have a node list, we want to iterate through all of the other nodes, and then through the after
        # list specified for each, and add the connections that create the graph.
        for key in d:
            for edge in d[key]['after']:
                node_list[edge['node_id']].add_outcome(
                    node_list[key],
                    cost=edge.get('cost', 0),
                    weight=edge.get('weight', 1)
                )

        return self

    def from_adj_matrix(self, A, labels=None, clf_matrix=None):
        """
        Takes in a numpy adjacency matrix and forms a petersburg graph from it (of type [col -> row]).

        :param A:
        :return:
        """

        if labels is None:
            labels = [(1, 1) for _ in range(A.shape[0])]
            labels[0] = (0, 0)

        if A.shape[0] != A.shape[1]:
            raise ValueError('Adjanceny Matrix must be square')

        dict_spec = {}
        for c_idx in range(A.shape[1]):
            after = []
            for r_idx in range(A.shape[0]):
                if A[r_idx, c_idx] != 0.0 and not np.isnan(A[r_idx, c_idx]):
                    row_sum = np.sum(A[r_idx, :])
                    if row_sum != 0.0:
                        weight = A[r_idx, c_idx] / row_sum
                    else:
                        weight = 0.0
                    try:
                        clf = clf_matrix[r_idx][c_idx]
                        if clf is not None:
                            after.append({
                                'node_id': r_idx,
                                'cost': 0,
                                'weight': clf,
                                '_weight': weight,
                                '_cnt': A[r_idx, c_idx]
                            })
                        else:
                            after.append({
                                'node_id': r_idx,
                                'cost': 0,
                                'weight': weight,
                                '_weight': weight,
                                '_cnt': A[r_idx, c_idx]
                            })
                    except (IndexError, TypeError) as e:
                        after.append({
                            'node_id': r_idx,
                            'cost': 0,
                            'weight': weight,
                            '_weight': weight,
                                '_cnt': A[r_idx, c_idx]
                        })

            if len(after) > 0 or labels[c_idx][0] == 0:
                dict_spec[c_idx] = {'payoff': 0, 'after': after}

        # add in root node (super hacky)
        dict_spec[-1] = {'after': [], 'payoff': 0}
        for k in dict_spec.keys():
            if k != -1 and len(dict_spec[k].get('after', [])) == 0:
                # the weight is the sum of all _cnt values that follow this node
                weight = 0
                for node in dict_spec.keys():
                    for a in dict_spec[node].get('after', []):
                        if a.get('node_id', None) == k:
                            weight += a.get('_cnt', 0)

                dict_spec[k]['after'] = [{'node_id': -1, 'weight': weight, 'cost': 0}]

        return self.from_dict(dict_spec)

    def get_outcome(self, iters=None, ruin=False, starting_bank=0, feature_vector=None):
        """
        Starting with the starting node, the graph is walked once, and the profit is returned, run multiple times to get
        an expected value estimate.

        :return:
        """
        if iters is None:
            payoff, cost = self.start_node.get_outcome(feature_vector)
            return payoff - cost
        else:
            bank = starting_bank
            for _ in range(iters):
                payoff, cost = self.start_node.get_outcome(feature_vector)
                bank = bank + payoff - cost
                if ruin:
                    if bank <= 0:
                        return 0
            return bank

    def get_outcome_node(self, feature_vector=None):
        """
        Starting with the starting node, the graph is walked once, and the ID of the final node reached is returned

        :return:
        """

        node_id = self.start_node.get_outcome_node(feature_vector)

        return node_id

    def get_options(self, iters=100, extended_stats=False):
        """
        Starts with each of the outcomes from the starting node seperately, to get the expected values (using iters
        iterations) for each of the initial options. Returns a dictionary of node_id: expected profit pairs.

        :param iters:
        :return:
        """

        choice = {}
        for outcome in self.start_node.outcomes:
            out = []
            for _ in range(iters):
                payoff, cost = outcome[0].get_outcome()
                out.append(payoff - cost - outcome[0].cost)
            if not extended_stats:
                choice.update({outcome[0].to_node.node_id: float(sum(out))/len(out)})
            else:
                choice.update({
                    outcome[0].to_node.node_id: {
                        'mean': float(sum(out))/len(out),
                        'max': max(out),
                        'min': min(out),
                        'count': len(out),
                    }
                })
        return choice

    def to_tree(self):
        """

        :return:
        """

        return self.start_node.to_tree()

    def to_networkx(self):
        """

        :return:
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError('the to networkx function requires networkx')

        g = nx.DiGraph()

        # first make a node id: obj mapping and add nodes to the graph
        node_to_node_id = dict([(node, node.node_id) for node in self.node_list()])
        nodes = list(node_to_node_id.values())
        g.add_nodes_from(nodes)

        # now iterate through and add in our edges using that mapping
        edges = list(self.edge_list())
        for edge in edges:
            from_node_id = node_to_node_id.get(edge.from_node)
            to_node_id = node_to_node_id.get(edge.to_node)
            cost = edge.cost
            g.add_edge(from_node_id, to_node_id, weight=cost)

        return g

    def edge_list(self):
        return self.start_node.get_edges(set())

    def node_list(self):
        return self.start_node.get_nodes(set())

    def plot(self, filename):
        """
        :return:
        """

        g = self.to_networkx()
        self.graph_draw(g, filename)

    @staticmethod
    def graph_draw(g, filename):
        try:
            import networkx as nx
            import pygraphviz
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError('the plot function requires networkx and pygraphviz')

        # pure graphviz
        # A = nx.to_agraph(g)
        # A.layout(
        #     'dot',
        #     args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8'
        # )
        # A.draw(filename)

        # bastardization
        plt.figure()
        pos = nx.pygraphviz_layout(
            g,
            prog='dot',
            args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8'
        )
        nx.draw(g, pos=pos)
        # edge_labels = nx.get_edge_attributes(g, 'weight')
        # print(edge_labels)
        # nx.draw_networkx_edge_labels(g, pos, labels=edge_labels)
        plt.show()