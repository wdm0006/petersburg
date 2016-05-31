"""
.. module:: graph
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Will McGinnis <will@pedalwrencher.com>


"""

import json
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

        node = None
        node_list = {}
        for key in d:
            if d[key]['after'] == []:
                node = Node(key, payoff=d[key].get('payoff', 0))
                node_list.update({key: node})
            else:
                node_list.update({key: Node(key, payoff=d[key].get('payoff', 0))})

        if node is None:
            raise AttributeError('Dict must contain a starting node (empty list for after key)')

        self.start_node = node

        for key in d:
            for edge in d[key]['after']:
                node_list[edge['node_id']].add_outcome(node_list[key], cost=edge.get('cost', 0), weight=edge.get('weight', 1))

        return self

    def from_adj_matrix(self, A, labels=None, clf_matrix=None):
        """
        Takes in a numpy adjacency matrix and forms a petersburg graph from it (of type [col -> row]).

        :param A:
        :return:
        """

        if labels is None:
            labels = [(1, 1) for x in range(A.shape[0])]
            labels[0] = (0, 0)

        if A.shape[0] != A.shape[1]:
            raise ValueError('Adjanceny Matrix must be square')

        dict_spec = {}
        for c_idx in range(A.shape[1]):
            after = []
            for r_idx in range(A.shape[0]):
                if A[r_idx, c_idx] != 0.0:
                    try:
                        clf = clf_matrix[r_idx][c_idx]
                        if clf is not None:
                            after.append({'node_id': r_idx, 'cost': 0, 'weight': clf})
                        else:
                            after.append({'node_id': r_idx, 'cost': 0, 'weight': A[r_idx, c_idx]})
                    except (IndexError, TypeError) as e:
                        after.append({'node_id': r_idx, 'cost': 0, 'weight': A[r_idx, c_idx]})

            if len(after) > 0 or labels[c_idx][0] == 0:
                dict_spec[c_idx] = {'payoff': 0, 'after': after}

        # add in root node (super hacky)
        dict_spec[-1] = {'after': [], 'payoff': 0}
        removed = []
        for k in dict_spec.keys():
            if k != -1 and len(dict_spec[k].get('after', [])) == 0:
                removed.append(k)
        for rem in removed:
            del dict_spec[rem]
        for k in dict_spec.keys():
            for after in dict_spec[k].get('after', []):
                if after.get('node_id') in removed:
                    after['node_id'] = -1

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

    def to_dict_of_dicts(self):
        """

        :return:
        """
        return self.start_node.get_edges(set())

    def edge_list(self):
        return self.start_node.get_edges(set())

    def node_list(self):
        return self.start_node.get_nodes(set())

    def plot(self):
        """
        :return:
        """

        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError('the plot function requires networkx and matplotlib')

        return None