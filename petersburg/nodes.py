"""
.. module:: node
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Will McGinnis <will@pedalwrencher.com>


"""

import math
from contextlib import contextmanager
from numbers import Number

import numpy as np

from petersburg import Edge

__author__ = "willmcginnis"


def is_numeric_weight(weight):
    return not hasattr(weight, "predict_proba") and isinstance(weight, Number)


def _validate_transition_weights(node_id, choices):
    """
    Check resolved transition weights before they are sampled from, and return their total.

    Weights are relative, so they need not sum to one and an individual zero is fine as long as
    some other outgoing weight is positive.

    :param node_id: id of the node the weights belong to, used in error messages
    :param choices: list of (edge, weight) pairs with classifier weights already resolved
    :return: float total of the weights
    """

    total = 0.0
    for edge, weight in choices:
        try:
            finite = math.isfinite(weight)
        except TypeError:
            finite = False
        if not finite or weight < 0:
            raise ValueError(
                f"Node {node_id} has invalid transition weight {weight!r} on the edge to node "
                f"{edge.to_node.node_id}; transition weights must be finite and non-negative"
            )
        total += weight

    if not math.isfinite(total) or total <= 0:
        raise ValueError(
            f"Node {node_id} has transition weights totalling {total!r}; at least one outgoing "
            f"weight must be positive and the total must be finite"
        )

    return total


class Node:
    """
    A node represents a decision point. Once reached it has some payoff (possibly negative or zero), and some model for
    probabilistically picking from a selection of outcomes (edges), or possibly having no outcomes, and being the end
    of the game.
    """

    def __init__(self, node_id, payoff=0, rng=None):
        """

        :return:
        """

        self.node_id = node_id
        self.payoff = payoff
        self.rng = rng
        self.outcomes = []

    def _rng(self):
        return self.rng if self.rng is not None else np.random

    def sample_payoff(self):
        """
        Returns the payoff for this node. For base Node class, this is just the fixed payoff.
        Subclasses can override this to provide stochastic payoffs.

        :return: float
        """
        return self.payoff

    def _payoff_parameters(self):
        return {"payoff": self.payoff}

    def _scale_payoff(self, factor):
        self.payoff *= factor

    @contextmanager
    def scaled_payoff(self, factor):
        """Temporarily scale sampled payoffs by a positive factor."""
        if factor <= 0:
            raise ValueError("payoff scale factor must be positive")

        original_parameters = self._payoff_parameters()
        try:
            self._scale_payoff(factor)
            yield
        finally:
            for parameter, value in original_parameters.items():
                setattr(self, parameter, value)

    def add_outcome(self, node, cost=0, weight=1, classifier=None):
        """
        Adds an outcome to this node.  Can take in a cost, a weight, and/or a classifier.  If both a weight and classifier
        are passed, then the classifier takes precesence, and must be able to predict a next node id

        :param node:
        :param cost:
        :return:
        """

        if classifier is None:
            self.outcomes.append((Edge(self, node, cost=cost), weight))
        else:
            self.outcomes.append((Edge(self, node, cost=cost), classifier))

    def get_weights(self, feature_vector=None):
        w_out = []
        for edge, w in self.outcomes:
            if is_numeric_weight(w):
                w_out.append((edge, w))
            else:
                pr = w.predict_proba(feature_vector)[0][1]
                w_out.append((edge, pr))

        return w_out

    def weighted_choice(self, feature_vector=None):
        choices = self.get_weights(feature_vector=feature_vector)
        total = _validate_transition_weights(self.node_id, choices)
        r = self._rng().uniform(0, total)
        upto = 0
        for c, w in choices:
            if w > 0 and upto + w >= r:
                return c
            upto += w
        raise AssertionError("Shouldn't get here")

    def get_outcome(self, feature_vector=None):
        """

        :return:
        """

        if self.outcomes == []:
            return self.sample_payoff(), 0
        else:
            edge = self.weighted_choice(feature_vector)
            payoff, cost = edge.get_outcome(feature_vector=feature_vector)
            return payoff + self.sample_payoff(), cost + edge.get_cost()

    def get_outcome_node(self, feature_vector=None):
        """

        :return:
        """

        if self.outcomes == []:
            return self.node_id
        else:
            edge = self.weighted_choice(feature_vector)
            node_id = edge.get_outcome_node(feature_vector=feature_vector)
            return node_id

    def to_tree(self, memo=None):
        if memo is None:
            memo = {}
        if self in memo:
            return memo[self]

        if self.outcomes == []:
            tree = {self.__repr__(): None}
        else:
            blob = {}
            for x in self.outcomes:
                blob.update(x[0].to_node.to_tree(memo))

            tree = {self.__repr__(): blob}

        memo[self] = tree
        return tree

    def get_nodes(self, node_list):
        if self in node_list:
            return node_list

        node_list.add(self)
        for outcome in self.outcomes:
            outcome[0].to_node.get_nodes(node_list)
        return node_list

    def get_edges(self, edge_list, visited=None):
        # a node can be reached by several distinct edges, so edge_list cannot
        # double as the visited marker
        if visited is None:
            visited = set()
        if self in visited:
            return edge_list

        visited.add(self)
        for outcome in self.outcomes:
            edge_list.add(outcome[0])
            outcome[0].to_node.get_edges(edge_list, visited)
        return edge_list

    def __str__(self):
        return f"Node {self.node_id}, with payoff {self.payoff} and outcomes {self.outcomes}"

    def __repr__(self):
        return str(self.node_id)


class UniformNode(Node):
    """
    A node with payoffs drawn from a continuous uniform distribution.

    Each time this node is reached, the payoff is sampled uniformly from [min_payoff, max_payoff].
    """

    def __init__(self, node_id, min_payoff, max_payoff, rng=None):
        """
        Initialize a UniformNode with a uniform distribution range.

        :param node_id: Unique identifier for this node
        :param min_payoff: Minimum payoff value (inclusive)
        :param max_payoff: Maximum payoff value (inclusive)
        """
        super().__init__(node_id, payoff=(min_payoff + max_payoff) / 2, rng=rng)
        self.min_payoff = min_payoff
        self.max_payoff = max_payoff

    def sample_payoff(self):
        """
        Sample a payoff from the uniform distribution.

        :return: float drawn from uniform[min_payoff, max_payoff]
        """
        return self._rng().uniform(self.min_payoff, self.max_payoff)

    def _payoff_parameters(self):
        return {
            "payoff": self.payoff,
            "min_payoff": self.min_payoff,
            "max_payoff": self.max_payoff,
        }

    def _scale_payoff(self, factor):
        self.payoff *= factor
        self.min_payoff *= factor
        self.max_payoff *= factor


class GaussianNode(Node):
    """
    A node with payoffs drawn from a Gaussian (normal) distribution.

    Each time this node is reached, the payoff is sampled from N(mean, std^2).
    """

    def __init__(self, node_id, mean, std, rng=None):
        """
        Initialize a GaussianNode with normal distribution parameters.

        :param node_id: Unique identifier for this node
        :param mean: Mean of the normal distribution
        :param std: Standard deviation of the normal distribution
        """
        super().__init__(node_id, payoff=mean, rng=rng)
        self.mean = mean
        self.std = std

    def sample_payoff(self):
        """
        Sample a payoff from the Gaussian distribution.

        :return: float drawn from N(mean, std^2)
        """
        return self._rng().normal(self.mean, self.std)

    def _payoff_parameters(self):
        return {"payoff": self.payoff, "mean": self.mean, "std": self.std}

    def _scale_payoff(self, factor):
        self.payoff *= factor
        self.mean *= factor
        self.std *= factor


class LogNormalNode(Node):
    """
    A node with payoffs drawn from a log-normal distribution.

    Each time this node is reached, the payoff is sampled from LogNormal(mu, sigma^2).
    The log-normal distribution is useful for modeling variables that are products of many
    independent random variables, and is always positive.
    """

    def __init__(self, node_id, mu, sigma, rng=None):
        """
        Initialize a LogNormalNode with log-normal distribution parameters.

        :param node_id: Unique identifier for this node
        :param mu: Mean of the underlying normal distribution (not the mean of the log-normal!)
        :param sigma: Standard deviation of the underlying normal distribution
        """
        super().__init__(node_id, payoff=np.exp(mu + sigma**2 / 2), rng=rng)
        self.mu = mu
        self.sigma = sigma

    def sample_payoff(self):
        """
        Sample a payoff from the log-normal distribution.

        :return: float drawn from LogNormal(mu, sigma^2)
        """
        return self._rng().lognormal(self.mu, self.sigma)

    def _payoff_parameters(self):
        return {"payoff": self.payoff, "mu": self.mu, "sigma": self.sigma}

    def _scale_payoff(self, factor):
        self.payoff *= factor
        self.mu += math.log(factor)


class PowerLawNode(Node):
    """
    A node with payoffs drawn from a power law (Pareto) distribution.

    Each time this node is reached, the payoff is sampled from a Pareto distribution.
    Power law distributions are useful for modeling phenomena with heavy tails, such as
    wealth distributions, popularity, and rare high-value events.
    """

    def __init__(self, node_id, scale, alpha, rng=None):
        """
        Initialize a PowerLawNode with Pareto distribution parameters.

        :param node_id: Unique identifier for this node
        :param scale: Scale parameter (minimum possible value)
        :param alpha: Shape parameter (controls tail heaviness, alpha > 1)
        """
        super().__init__(
            node_id,
            payoff=scale * alpha / (alpha - 1) if alpha > 1 else scale * 2,
            rng=rng,
        )
        self.scale = scale
        self.alpha = alpha

    def sample_payoff(self):
        """
        Sample a payoff from the power law distribution.

        :return: float drawn from Pareto(scale, alpha)
        """
        return (self._rng().pareto(self.alpha) + 1) * self.scale

    def _payoff_parameters(self):
        return {"payoff": self.payoff, "scale": self.scale, "alpha": self.alpha}

    def _scale_payoff(self, factor):
        self.payoff *= factor
        self.scale *= factor
