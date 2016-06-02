from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from petersburg import graph
import numpy as np
from collections import Counter

__author__ = 'willmcginnis'


class FrequencyEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, verbose=False, num_simulations=10):
        self._frequency_matrix = None
        self.num_simulations = num_simulations
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
            sims = Counter([g.get_outcome_node(X[r_idx, :].reshape(1, -1)) for _ in range(self.num_simulations)])
            y_hat[r_idx, 0] = sims.most_common(1)[0][0]

        return y_hat



class MixedModeEstimator(BaseEstimator, ClassifierMixin):
    """
    Similar to the frequency estimator, but will use a classifier to predict conditional probabilities where possible
    """
    def __init__(self, verbose=False, num_simulations=10):

        self._clf = LogisticRegression
        self._clf_args = {}

        self._frequency_matrix = None
        self._clf_matrix = None

        self._categories = None

        self._min_samples = 100
        self.num_simulations = num_simulations

        self.verbose = verbose

    @property
    def _cateogry_labels(self):
        try:
            return dict(zip(range(len(self._categories)), self._categories))
        except AttributeError as e:
            return {}

    def _get_normalized_adj_matrix(self):
        """
        For each unique first index in the category labels, scale the frequency matrix (to get rough probabilities)

        :return:
        """

        # find all of the unique layers in the problem (first index of category tuples)
        row_sums = self._frequency_matrix.sum(axis=1)
        normed_matrix = self._frequency_matrix / row_sums[:, np.newaxis]

        return normed_matrix

    def _update_frequencies(self, y):
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

        return True

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """

        # first update the frequencies
        self._update_frequencies(y)

        # empty out the clf matrix
        self._clf_matrix = [[None for _ in range(len(self._cateogry_labels))] for _ in range(len(self._cateogry_labels))]

        # then for any with enough data, try to train a model
        for r_idx in range(self._frequency_matrix.shape[0]):
            for c_idx in range(self._frequency_matrix.shape[1]):
                if self._frequency_matrix[r_idx, c_idx] >= self._min_samples:
                    if self.verbose:
                        print('\nFound a sample worth modeling')
                        print('F[%s,%s]=%s' % (r_idx, c_idx, self._frequency_matrix[r_idx, c_idx]))
                        print('from label: %s' % (str(self._cateogry_labels[r_idx]), ))
                        print('to label: %s' % (str(self._cateogry_labels[c_idx]), ))

                    filter_col = self._cateogry_labels[r_idx][0]
                    filter_term = self._cateogry_labels[r_idx][1]

                    label_col = self._cateogry_labels[c_idx][0]
                    label_term = self._cateogry_labels[c_idx][1]

                    # filter down X and y to only samples which came from the from_label (index, value)
                    X_t = X[y[:, filter_col] == filter_term]
                    y_t = y[y[:, filter_col] == filter_term]

                    # filter down y to only the to_node index
                    y_t = y_t[:, label_col]

                    # create bool for if its to the correct option
                    y_t = y_t == label_term

                    try:
                        self._clf_matrix[r_idx][c_idx] = self._clf(**self._clf_args).fit(X_t, y_t)
                    except ValueError as e:
                        self._clf_matrix[r_idx][c_idx] = None

        return self

    def predict(self, X):
        """
        Uses the observed adjacency matrix to create a petersburg graph and simulate the outcome for each entry

        :param X:
        :param y:
        :return:
        """

        g = graph.Graph()

        g.from_adj_matrix(self._frequency_matrix, self._categories, clf_matrix=self._clf_matrix)

        y_hat = np.zeros((X.shape[0], 1))
        for r_idx in range(y_hat.shape[0]):
            sims = Counter([g.get_outcome_node(X[r_idx, :].reshape(1, -1)) for _ in range(self.num_simulations)])
            y_hat[r_idx, 0] = sims.most_common(1)[0][0]

        return y_hat
