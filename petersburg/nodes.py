"""
.. module:: node
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Will McGinnis <will@pedalwrencher.com>


"""

from petersburg import Edge
import random

__author__ = 'willmcginnis'


class Node():
    """
    A node represents a decision point. Once reached it has some payoff (possibly negative or zero), and some model for
    probabilistically picking from a selection of outcomes (edges), or possibly having no outcomes, and being the end
    of the game.
    """
    def __init__(self, node_id, payoff=0):
        """

        :return:
        """

        self.node_id = node_id
        self.payoff = payoff
        self.outcomes = []

    def add_outcome(self, node, cost=0, weight=1):
        """
        Adds an outcome to this node

        :param node:
        :param cost:
        :return:
        """

        self.outcomes.append((Edge(self, node, cost=cost), weight))

    @staticmethod
    def weighted_choice(choices):
        total = sum(w for c, w in choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    def get_outcome(self):
        """

        :return:
        """

        if self.outcomes == []:
            return self.payoff, 0
        else:
            edge = self.weighted_choice(self.outcomes)
            payoff, cost = edge.get_outcome()
            return payoff, cost + edge.get_cost()

    def get_outcome_node(self):
        """

        :return:
        """

        if self.outcomes == []:
            return self.node_id
        else:
            edge = self.weighted_choice(self.outcomes)
            node_id = edge.get_outcome_node()
            return node_id

    def to_tree(self):
        if self.outcomes == []:
            return {self.__repr__(): self.__repr__()}
        else:
            blob = {}
            for x in self.outcomes:
                blob.update(x[0].to_node.to_tree())

            return {self.__repr__(): blob}

    def get_nodes(self, node_list):
        node_list.update({self})
        if self.outcomes != []:
            for outcome in self.outcomes:
                node_list.update(outcome[0].to_node.get_nodes(node_list))
        return node_list

    def get_edges(self, edge_list):
        if self.outcomes != []:
            for outcome in self.outcomes:
                edge_list.update({outcome[0]})
                edge_list.update(outcome[0].to_node.get_edges(edge_list))
        return edge_list

    def __str__(self):
        return 'Node %s, with payoff %s and outcomes %s' % (str(self.node_id), str(self.payoff), str(self.outcomes))

    def __repr__(self):
        return str(self.node_id)