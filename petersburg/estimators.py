import numpy as np
import random

__author__ = 'willmcginnis'


class FrequencyEstimator:

    def __init__(self, verbose=False):
        self._frequency_matrix = None
        self._categories = None
        self.verbose = verbose

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

        if self.verbose:
            print('Category Labels: %s' % (str(self._categories), ))

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

    def predict(self, X, y):
        """
        Uses the observed adjacency matrix to create a petersburg graph and simulate the outcome for each entry

        :param X:
        :param y:
        :return:
        """

        # TODO this whole part.

        return None

if __name__ == '__main__':
    y = np.array([[
        random.choice([0, 1, 2]),
        random.choice([0, 1, 1, 1, 2]),
        random.choice([0, 1, 2, 2, 2, 2]),
        random.choice([0, 1, 2, 2, 1, 3])
    ] for _ in range(10000)])

    clf = FrequencyEstimator(verbose=True)
    clf.fit(None, y)

    print(clf._frequency_matrix)
