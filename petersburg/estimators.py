from sklearn.base import BaseEstimator, ClassifierMixin
from petersburg import graph
import numpy as np
import random

__author__ = 'willmcginnis'


class FrequencyEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, verbose=False):
        self._frequency_matrix = None
        self._categories = None
        self.verbose = verbose

    @property
    def _cateogry_labels(self):
        try:
            return dict(zip(range(len(self._categories)), self._categories))
        except AttributeError as e:
            return {}

    def fit(self, X, y):
        """
        In this case (for now) X is actually ignored.  Y is assumed to be multiple columns, ordered by layer in a tree.
        So the first column is a multiclass integer column of the first set of nodes, second column for the next, etc.

        The adjacency matrix constructed will assume a fully connected tree.

        :param X:
        :param y:
        :return:
        """

        # set up the frequency matrix based on unique columns present
        dims = sum([len(set(y[:, col].tolist())) for col in range(y.shape[1])])
        self._frequency_matrix = np.zeros((dims, dims))

        # set up the categories corresponding to each index
        self._categories = [(idx, a) for idx, b in enumerate([set(y[:, col].tolist()) for col in range(y.shape[1])]) for a in b]

        for ridx in range(y.shape[0]):
            for fcidx, tcidx in zip(range(0, y.shape[1] - 1), range(1, y.shape[1])):
                f = self._categories.index((fcidx, y[ridx, fcidx]))
                t = self._categories.index((tcidx, y[ridx, tcidx]))

                try:
                    self._frequency_matrix[f, t] += 1
                except IndexError as e:
                    if self.verbose:
                        print('Unknown instance found')

        return self

    def partial_fit(self, X, y):
        """
        Updates an existing fitted model with new information.

        :return:
        """

        if self._categories is None:
            if self.verbose:
                print('No existing model found so making one from scratch')

            return self.fit(X, y)

        for ridx in range(y.shape[0]):
            for fcidx, tcidx in zip(range(0, y.shape[1] - 1), range(1, y.shape[1])):
                f = self._categories.index((fcidx, y[ridx, fcidx]))
                t = self._categories.index((tcidx, y[ridx, tcidx]))

                try:
                    self._frequency_matrix[f, t] += 1
                except IndexError as e:
                    if self.verbose:
                        print('Unknown instance found')

        return self

    def predict(self, X):
        """
        Uses the observed adjacency matrix to create a petersburg graph and simulate the outcome for each entry

        :param X:
        :param y:
        :return:
        """

        g = graph.Graph()
        g.from_adj_matrix(self._frequency_matrix, self._categories)

        y_hat = np.zeros((X.shape[0], 1))
        for r_idx in range(y_hat.shape[0]):
            y_hat[r_idx, 0] = g.get_outcome_node()

        return y_hat
