from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from petersburg import graph as pg

__author__ = "willmcginnis"


def _build_categories(y):
    """
    Builds the ordered ``(layer, value)`` category list for a multi-column y.

    Values are kept in first-appearance order within each column, so a category's index is a
    function of the data alone. Iterating a ``set`` here would make the indices depend on
    ``PYTHONHASHSEED`` for string labels, and first appearance (unlike sorting) does not require
    the labels to be mutually comparable.

    :param y: multi-column outcome array, one column per layer
    :return: list of ``(layer_index, value)`` tuples
    """

    return [
        (col, value) for col in range(y.shape[1]) for value in dict.fromkeys(y[:, col].tolist())
    ]


def _terminal_label(categories, node_id):
    """
    Maps a simulated terminal node id back to the label it was fitted from.

    :param categories: the fitted ``(layer, value)`` category list
    :param node_id: node id returned by ``Graph.get_outcome_node``
    :return: the fitted label for that category
    """

    if not 0 <= node_id < len(categories):
        raise ValueError(
            f"Simulation ended on node id {node_id}, which is not a fitted category index "
            f"(expected 0 to {len(categories) - 1}). Node id -1 is the synthetic root injected "
            f"by Graph.from_adj_matrix, and means the walk reached no fitted category."
        )

    return categories[node_id][1]


class FrequencyEstimator(BaseEstimator, ClassifierMixin):
    """
    Predicts a terminal category by simulating a graph built from observed transition frequencies.

    :param random_state: Optional integer seed or ``numpy.random.Generator`` used for
        edge selection and stochastic node payoffs. Each ``predict`` call builds a fresh
        graph from it, so with an integer seed repeated ``predict`` calls on one fitted
        estimator return identical output. A ``Generator`` is shared rather than restarted,
        so its state advances from call to call.
    """

    def __init__(self, verbose=False, num_simulations=10, random_state=None):
        self._frequency_matrix = None
        self.num_simulations = num_simulations
        self._categories = None
        self.verbose = verbose
        self.random_state = random_state

    @property
    def _cateogry_labels(self):
        try:
            return dict(zip(range(len(self._categories)), self._categories))
        except AttributeError:
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

        # set up the categories corresponding to each index
        self._categories = _build_categories(y)

        # set up the frequency matrix based on unique columns present
        dims = len(self._categories)
        self._frequency_matrix = np.zeros((dims, dims))

        for ridx in range(y.shape[0]):
            for fcidx, tcidx in zip(range(0, y.shape[1] - 1), range(1, y.shape[1])):
                f = self._categories.index((fcidx, y[ridx, fcidx]))
                t = self._categories.index((tcidx, y[ridx, tcidx]))

                try:
                    self._frequency_matrix[f, t] += 1
                except IndexError:
                    if self.verbose:
                        print("Unknown instance found")

        return self

    def partial_fit(self, X, y):
        """
        Updates an existing fitted model with new information.

        :return:
        """

        if self._categories is None:
            if self.verbose:
                print("No existing model found so making one from scratch")

            return self.fit(X, y)

        for ridx in range(y.shape[0]):
            for fcidx, tcidx in zip(range(0, y.shape[1] - 1), range(1, y.shape[1])):
                f = self._categories.index((fcidx, y[ridx, fcidx]))
                t = self._categories.index((tcidx, y[ridx, tcidx]))

                try:
                    self._frequency_matrix[f, t] += 1
                except IndexError:
                    if self.verbose:
                        print("Unknown instance found")

        return self

    def predict(self, X):
        """
        Uses the observed adjacency matrix to create a petersburg graph and simulate the outcome for each entry

        Returns the fitted label of the most frequently simulated terminal category, as an
        object-dtype (n, 1) array so non-numeric labels survive.

        :param X:
        :param y:
        :return:
        """

        g = pg.Graph(random_state=self.random_state)

        g.from_adj_matrix(self._frequency_matrix, self._categories)

        y_hat = np.empty((X.shape[0], 1), dtype=object)
        for r_idx in range(y_hat.shape[0]):
            sims = Counter(
                [
                    g.get_outcome_node(X[r_idx, :].reshape(1, -1))
                    for _ in range(self.num_simulations)
                ]
            )
            y_hat[r_idx, 0] = _terminal_label(self._categories, sims.most_common(1)[0][0])

        return y_hat


class MixedModeEstimator(BaseEstimator, ClassifierMixin):
    """
    Similar to the frequency estimator, but will use a classifier to predict conditional probabilities where possible

    :param random_state: Optional integer seed or ``numpy.random.Generator`` used for
        edge selection and stochastic node payoffs. Each ``predict`` call builds a fresh
        graph from it, so with an integer seed repeated ``predict`` calls on one fitted
        estimator return identical output. A ``Generator`` is shared rather than restarted,
        so its state advances from call to call. The seed is also passed to the per-transition
        classifiers at fit time unless ``_clf_args`` already fixes it; a ``Generator`` is not,
        because scikit-learn estimators accept only ``None``, an int, or a ``RandomState``.
    """

    def __init__(self, verbose=False, num_simulations=10, random_state=None):

        self._clf = LogisticRegression
        self._clf_args = {}

        self._frequency_matrix = None
        self._clf_matrix = None

        self._categories = None

        self._min_samples = 100
        self.num_simulations = num_simulations
        self.random_state = random_state

        self.verbose = verbose

    @property
    def _cateogry_labels(self):
        try:
            return dict(zip(range(len(self._categories)), self._categories))
        except AttributeError:
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

    def _classifier_args(self):
        """
        The keyword arguments each per-transition classifier is constructed with.

        ``random_state`` is added so a fitted model is reproducible end to end, unless
        ``_clf_args`` already fixes it. A ``numpy.random.Generator`` is left out because
        scikit-learn accepts only ``None``, an int, or a ``RandomState``; it still seeds
        the simulation graph.

        :return: a new dict of classifier keyword arguments
        """

        clf_args = dict(self._clf_args)
        if "random_state" not in clf_args and not isinstance(
            self.random_state, np.random.Generator
        ):
            clf_args["random_state"] = self.random_state

        return clf_args

    def _update_frequencies(self, y):
        # set up the categories corresponding to each index
        self._categories = _build_categories(y)

        # set up the frequency matrix based on unique columns present
        dims = len(self._categories)
        self._frequency_matrix = np.zeros((dims, dims))

        for ridx in range(y.shape[0]):
            for fcidx, tcidx in zip(range(0, y.shape[1] - 1), range(1, y.shape[1])):
                f = self._categories.index((fcidx, y[ridx, fcidx]))
                t = self._categories.index((tcidx, y[ridx, tcidx]))

                try:
                    self._frequency_matrix[f, t] += 1
                except IndexError:
                    if self.verbose:
                        print("Unknown instance found")

        return True

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """

        # first update the frequencies
        self._update_frequencies(y)

        clf_args = self._classifier_args()

        # empty out the clf matrix
        self._clf_matrix = [
            [None for _ in range(len(self._cateogry_labels))]
            for _ in range(len(self._cateogry_labels))
        ]

        # then for any with enough data, try to train a model
        for r_idx in range(self._frequency_matrix.shape[0]):
            for c_idx in range(self._frequency_matrix.shape[1]):
                if self._frequency_matrix[r_idx, c_idx] >= self._min_samples:
                    if self.verbose:
                        print("\nFound a sample worth modeling")
                        print(f"F[{r_idx},{c_idx}]={self._frequency_matrix[r_idx, c_idx]}")
                        print(f"from label: {self._cateogry_labels[r_idx]}")
                        print(f"to label: {self._cateogry_labels[c_idx]}")

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
                        self._clf_matrix[r_idx][c_idx] = self._clf(**clf_args).fit(X_t, y_t)
                    except ValueError:
                        self._clf_matrix[r_idx][c_idx] = None

        return self

    def predict(self, X):
        """
        Uses the observed adjacency matrix to create a petersburg graph and simulate the outcome for each entry

        Returns the fitted label of the most frequently simulated terminal category, as an
        object-dtype (n, 1) array so non-numeric labels survive.

        :param X:
        :param y:
        :return:
        """

        g = pg.Graph(random_state=self.random_state)

        g.from_adj_matrix(self._frequency_matrix, self._categories, clf_matrix=self._clf_matrix)

        y_hat = np.empty((X.shape[0], 1), dtype=object)
        for r_idx in range(y_hat.shape[0]):
            sims = Counter(
                [
                    g.get_outcome_node(X[r_idx, :].reshape(1, -1))
                    for _ in range(self.num_simulations)
                ]
            )
            y_hat[r_idx, 0] = _terminal_label(self._categories, sims.most_common(1)[0][0])

        return y_hat
