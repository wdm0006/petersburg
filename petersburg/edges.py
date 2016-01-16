"""
.. module:: edge
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Will McGinnis <will@pedalwrencher.com>


"""

__author__ = 'willmcginnis'


class Edge(object):
    """
    An edge is simply the path from one node to another, with some cost.

    """
    def __init__(self, from_node, to_node, cost=0):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost

    def get_outcome(self):
        return self.to_node.get_outcome()

    def get_cost(self):
        return self.cost

    def __repr__(self):
        return '%s -> %s' % (self.from_node.__repr__(), self.to_node.__repr__())