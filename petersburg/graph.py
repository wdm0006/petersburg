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

    def get_outcome(self):
        """
        Starting with the starting node, the graph is walked once, and the profit is returned, run multiple times to get
        an expected value estimate.

        :return:
        """

        payoff, cost = self.start_node.get_outcome()

        return payoff - cost

    def get_options(self, iters=100):
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
            choice.update({outcome[0].to_node.node_id: float(sum(out))/len(out)})
        return choice
