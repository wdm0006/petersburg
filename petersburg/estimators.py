from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from petersburg import graph as pg
from petersburg.exceptions import ValidationError
from petersburg.graph import validate_sample_count

__author__ = "willmcginnis"


def _validate_path_target(estimator, y):
    """
    Coerces a training target to an array and rejects shapes that cannot encode a transition.

    A path target has one column per decision layer, so at least two columns are needed for a
    single layer-to-layer transition to exist. Checking here keeps the failure at the training
    entry point instead of surfacing it as an opaque shape error during fit, or as a synthetic
    root error once ``predict`` simulates an unusable graph.

    :param estimator: the estimator being trained, named in the error message
    :param y: array-like path target, one column per layer
    :return: the target as a ``numpy`` array
    """

    y = np.asarray(y)
    name = estimator.__class__.__name__

    if y.ndim != 2:
        raise ValidationError(
            f"{name} requires a 2D path target with one column per decision layer, "
            f"but y has {y.ndim} dimension(s)."
        )

    if y.shape[0] == 0:
        raise ValidationError(f"{name} requires at least one training path, but y has no rows.")

    if y.shape[1] < 2:
        raise ValidationError(
            f"{name} requires a path target with at least two decision layers, but y has "
            f"{y.shape[1]} column(s); no layer-to-layer transition can be observed."
        )

    return y


def _validate_feature_matrix(estimator, X):
    """
    Coerces a feature matrix to an array and rejects shapes that cannot be indexed by row.

    ``predict`` reads ``X.shape[0]`` and slices ``X[row, :]``, and ``MixedModeEstimator.fit``
    masks ``X`` with a boolean array derived from ``y``, so a list-of-lists or a 1-D input
    otherwise fails deep inside those operations with an opaque ``AttributeError``,
    ``IndexError``, or ``TypeError``.

    :param estimator: the estimator being used, named in the error message
    :param X: array-like feature matrix, one row per sample
    :return: the feature matrix as a ``numpy`` array
    """

    X = np.asarray(X)
    name = estimator.__class__.__name__

    if X.ndim != 2:
        raise ValidationError(
            f"{name} requires a 2D feature matrix with one row per sample, "
            f"but X has {X.ndim} dimension(s)."
        )

    return X


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


def _terminal_node_id(categories, node_id):
    """
    Validates a simulated terminal node id against the fitted categories and returns it.

    Shared by predict (which keeps each Counter's most common id) and predict_proba
    (which counts all of them), so both consumers reject out-of-range ids identically.

    :param categories: the fitted ``(layer, value)`` category list
    :param node_id: node id returned by ``Graph.get_outcome_node``
    :return: the validated node id
    """

    if not 0 <= node_id < len(categories):
        raise ValueError(
            f"Simulation ended on node id {node_id}, which is not a fitted category index "
            f"(expected 0 to {len(categories) - 1}). Node id -1 is the synthetic root injected "
            f"by Graph.from_adj_matrix, and means the walk reached no fitted category."
        )

    return node_id


def _terminal_label(categories, node_id):
    """
    Maps a simulated terminal node id back to the label it was fitted from.

    :param categories: the fitted ``(layer, value)`` category list
    :param node_id: node id returned by ``Graph.get_outcome_node``
    :return: the fitted label for that category
    """

    return categories[_terminal_node_id(categories, node_id)][1]


def _terminal_indices(categories):
    """
    Category indices of the final decision layer, in fitted (first-appearance) order.

    Every category of a non-final layer has an outgoing transition (the training paths
    that reached it continued on), so simulations only ever end on these ids.

    :param categories: the fitted ``(layer, value)`` category list
    :return: list of category indices whose layer is the last one
    """

    last_layer = max(layer for layer, _ in categories)

    return [idx for idx, (layer, _) in enumerate(categories) if layer == last_layer]


def _terminal_accuracy(estimator, X, y, sample_weight=None):
    """Return accuracy against the terminal column of a path target."""

    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] == 0:
        raise ValidationError("y must be a non-empty 2D path target")

    # predict returns object-dtype labels so strings survive; sklearn's metrics refuse a
    # dtype mix (binary truth vs unknown predictions), so compare in the observed
    # terminal column's dtype.
    predictions = estimator.predict(X).ravel().astype(y[:, -1].dtype)

    return accuracy_score(y[:, -1], predictions, sample_weight=sample_weight)


def _is_fitted(estimator):
    """True when an estimator has learned its categories and frequencies."""

    return (
        getattr(estimator, "_categories", None) is not None
        and getattr(estimator, "_frequency_matrix", None) is not None
    )


def _require_fitted(estimator):
    """Raise when an estimator has not learned its categories and frequencies."""

    if not _is_fitted(estimator):
        raise NotFittedError(
            f"{estimator.__class__.__name__} is not fitted. Call fit before predict or score."
        )


def _record_fitted_attributes(estimator, X):
    """
    Sets the sklearn fitted-attribute contract on a freshly fitted estimator.

    ``classes_`` holds the terminal-layer labels in first-appearance order (the order
    predict_proba's columns follow), and ``n_features_in_`` records the width of the
    fit-time feature matrix when one is supplied. Both live on fit -- never in
    ``__init__`` -- so fitted state and unfitted state are cleanly separable.

    :param estimator: the estimator that just finished fitting
    :param X: the feature matrix as passed to fit, if any
    """

    categories = estimator._categories
    estimator.classes_ = np.asarray([categories[idx][1] for idx in _terminal_indices(categories)])

    if X is not None:
        X = np.asarray(X)
        if X.ndim == 2:
            estimator.n_features_in_ = X.shape[1]


def _simulated_terminal_counters(estimator, X, clf_matrix=None):
    """
    Simulates every sample's walks and returns one Counter of terminal node ids per row.

    This is the per-sample distribution predict consumes: predict keeps only the most
    common id of each Counter, and predict_proba normalizes all of it over ``classes_``.
    One graph is built per call from the estimator's ``random_state`` and reused across
    samples, and every simulated id is validated here, so a walk that ends on the
    synthetic root or outside the fitted categories fails exactly where predict does.

    :param estimator: the fitted estimator whose graph is simulated
    :param X: validated 2D feature matrix, one row per sample
    :param clf_matrix: per-transition classifiers backing edge weights, if any
    :return: list of ``Counter`` objects, one per sample row
    """

    graph = pg.Graph(random_state=estimator.random_state)
    graph.from_adj_matrix(estimator._frequency_matrix, estimator._categories, clf_matrix=clf_matrix)

    counters = []
    for r_idx in range(X.shape[0]):
        row = X[r_idx, :].reshape(1, -1)
        counters.append(
            Counter(
                [
                    _terminal_node_id(estimator._categories, graph.get_outcome_node(row))
                    for _ in range(estimator.num_simulations)
                ]
            )
        )

    return counters


def _partial_fit_category_index(estimator, column, value):
    """Return a fitted category index or explain why an incremental update is invalid."""

    try:
        return estimator._categories.index((column, value))
    except ValueError:
        raise ValidationError(
            f"{estimator.__class__.__name__}.partial_fit received unknown category {value!r} "
            f"in column {column}; partial_fit can only update counts for categories seen "
            f"during fit."
        ) from None


class FrequencyEstimator(ClassifierMixin, BaseEstimator):
    """
    Predicts a terminal category by simulating a graph built from observed transition frequencies.

    :param random_state: Optional integer seed or ``numpy.random.Generator`` used for
        edge selection and stochastic node payoffs. Each ``predict`` call builds a fresh
        graph from it, so with an integer seed repeated ``predict`` calls on one fitted
        estimator return identical output. A ``Generator`` is shared rather than restarted,
        so its state advances from call to call.
    """

    def __init__(self, verbose=False, num_simulations=10, random_state=None):
        self.verbose = verbose
        self.num_simulations = num_simulations
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

        y = _validate_path_target(self, y)

        # set up the categories corresponding to each index
        self._categories = _build_categories(y)

        # set up the frequency matrix based on unique columns present
        dims = len(self._categories)
        self._frequency_matrix = np.zeros((dims, dims))

        for ridx in range(y.shape[0]):
            for fcidx, tcidx in zip(range(0, y.shape[1] - 1), range(1, y.shape[1])):
                f = self._categories.index((fcidx, y[ridx, fcidx]))
                t = self._categories.index((tcidx, y[ridx, tcidx]))

                self._frequency_matrix[f, t] += 1

        _record_fitted_attributes(self, X)

        return self

    def partial_fit(self, X, y, classes=None):
        """
        Updates an existing fitted model with new information.

        Accepts scikit-learn's incremental-learning ``classes`` argument for API
        compatibility; the category set is learned from the data itself, so the
        argument is accepted but not enforced.

        :return:
        """

        y = _validate_path_target(self, y)

        if not _is_fitted(self):
            if self.verbose:
                print("No existing model found so making one from scratch")

            return self.fit(X, y)

        for ridx in range(y.shape[0]):
            for fcidx, tcidx in zip(range(0, y.shape[1] - 1), range(1, y.shape[1])):
                f = _partial_fit_category_index(self, fcidx, y[ridx, fcidx])
                t = _partial_fit_category_index(self, tcidx, y[ridx, tcidx])

                self._frequency_matrix[f, t] += 1

        return self

    def predict(self, X):
        """
        Uses the observed adjacency matrix to create a petersburg graph and simulate the outcome for each entry

        Returns the fitted label of the most frequently simulated terminal category for
        each sample, as an (n,) object-dtype array so non-numeric labels survive.

        :param X:
        :param y:
        :raises ValueError: If num_simulations is not a positive integer, or X is not 2D
        :return:
        """

        _require_fitted(self)
        validate_sample_count("num_simulations", self.num_simulations)
        X = _validate_feature_matrix(self, X)

        counters = _simulated_terminal_counters(self, X)

        y_hat = np.empty(X.shape[0], dtype=object)
        for r_idx, sims in enumerate(counters):
            y_hat[r_idx] = _terminal_label(self._categories, sims.most_common(1)[0][0])

        return y_hat

    def predict_proba(self, X):
        """
        Normalizes the full per-sample terminal distribution predict consumes into
        probabilities: an (n, len(classes_)) array whose rows sum to 1, where column j
        is the probability of classes_[j].

        :param X:
        :raises ValueError: If num_simulations is not a positive integer, or X is not 2D
        :return:
        """

        _require_fitted(self)
        validate_sample_count("num_simulations", self.num_simulations)
        X = _validate_feature_matrix(self, X)

        counters = _simulated_terminal_counters(self, X)
        terminal = _terminal_indices(self._categories)

        proba = np.zeros((X.shape[0], len(terminal)), dtype=float)
        for r_idx, sims in enumerate(counters):
            for node_id, count in sims.items():
                try:
                    proba[r_idx, terminal.index(node_id)] = count
                except ValueError:
                    raise ValueError(
                        f"Simulation ended on node id {node_id}, which is not a terminal "
                        f"category; predict_proba only covers the final decision layer."
                    ) from None

        return proba / self.num_simulations

    def score(self, X, y, sample_weight=None):
        """Return the accuracy of predicted terminal labels."""

        return _terminal_accuracy(self, X, y, sample_weight=sample_weight)


class MixedModeEstimator(ClassifierMixin, BaseEstimator):
    """
    Similar to the frequency estimator, but will use a classifier to predict conditional probabilities where possible

    :param random_state: Optional integer seed or ``numpy.random.Generator`` used for
        edge selection and stochastic node payoffs. Each ``predict`` call builds a fresh
        graph from it, so with an integer seed repeated ``predict`` calls on one fitted
        estimator return identical output. A ``Generator`` is shared rather than restarted,
        so its state advances from call to call. The seed is also passed to the per-transition
        classifiers at fit time unless ``clf_args`` already fixes it; a ``Generator`` is not,
        because scikit-learn estimators accept only ``None``, an int, or a ``RandomState``.
    :param min_samples: Minimum observed transition count before that transition is
        modeled with a classifier instead of its raw frequency.
    :param clf: Unfitted scikit-learn-compatible classifier cloned once per modeled
        transition at fit time; the default ``None`` resolves to a fresh
        ``LogisticRegression()``.
    :param clf_args: Keyword arguments set on every cloned classifier before fitting it.
    """

    def __init__(
        self,
        verbose=False,
        num_simulations=10,
        random_state=None,
        min_samples=100,
        clf=None,
        clf_args=None,
    ):

        self.verbose = verbose
        self.num_simulations = num_simulations
        self.random_state = random_state
        self.min_samples = min_samples

        # Stored verbatim -- __init__ must not transform its parameters, or clone and
        # get_params would report state the caller never passed. The documented defaults
        # (a fresh LogisticRegression, an empty argument dict) resolve where they are
        # consumed: in fit and _classifier_args respectively.
        self.clf = clf
        self.clf_args = clf_args

    @property
    def _cateogry_labels(self):
        try:
            return dict(zip(range(len(self._categories)), self._categories))
        except AttributeError:
            return {}

    def _classifier_args(self):
        """
        The keyword arguments each per-transition classifier is constructed with.

        ``random_state`` is added so a fitted model is reproducible end to end, unless
        ``clf_args`` already fixes it. A ``numpy.random.Generator`` is left out because
        scikit-learn accepts only ``None``, an int, or a ``RandomState``; it still seeds
        the simulation graph.

        :return: a new dict of classifier keyword arguments
        """

        clf_args = dict(self.clf_args or {})
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

                self._frequency_matrix[f, t] += 1

        return True

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :raises ValueError: If X is not 2D, or X and y describe a different number of samples
        :return:
        """

        y = _validate_path_target(self, y)
        X = _validate_feature_matrix(self, X)

        if X.shape[0] != y.shape[0]:
            raise ValidationError(
                f"{self.__class__.__name__} requires X and y to describe the same samples, "
                f"but X has {X.shape[0]} row(s) and y has {y.shape[0]} row(s)."
            )

        # first update the frequencies
        self._update_frequencies(y)
        _record_fitted_attributes(self, X)

        clf_args = self._classifier_args()

        # empty out the clf matrix
        self._clf_matrix = [
            [None for _ in range(len(self._cateogry_labels))]
            for _ in range(len(self._cateogry_labels))
        ]

        # then for any with enough data, try to train a model
        for r_idx in range(self._frequency_matrix.shape[0]):
            for c_idx in range(self._frequency_matrix.shape[1]):
                if self._frequency_matrix[r_idx, c_idx] >= self.min_samples:
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

                    if np.unique(y_t).size > 1:
                        base_clf = self.clf if self.clf is not None else LogisticRegression()
                        transition_clf = clone(base_clf)
                        if clf_args:
                            transition_clf.set_params(**clf_args)

                        self._clf_matrix[r_idx][c_idx] = transition_clf.fit(X_t, y_t)

        return self

    def partial_fit(self, X, y, classes=None):
        """
        Updates an existing fitted model's transition frequencies with new information.

        Mirrors FrequencyEstimator.partial_fit: an estimator that has not been fitted yet
        delegates to fit, already-seen categories update their counts, and unseen
        categories are rejected. Per-transition classifiers are only trained during fit,
        when the full feature matrix behind each transition's training subset is
        available; partial_fit never sees that data again. scikit-learn's incremental
        ``classes`` argument is accepted but not enforced, mirroring
        FrequencyEstimator.partial_fit.

        :return:
        """

        y = _validate_path_target(self, y)

        if not _is_fitted(self):
            if self.verbose:
                print("No existing model found so making one from scratch")

            return self.fit(X, y)

        for ridx in range(y.shape[0]):
            for fcidx, tcidx in zip(range(0, y.shape[1] - 1), range(1, y.shape[1])):
                f = _partial_fit_category_index(self, fcidx, y[ridx, fcidx])
                t = _partial_fit_category_index(self, tcidx, y[ridx, tcidx])

                self._frequency_matrix[f, t] += 1

        return self

    def predict(self, X):
        """
        Uses the observed adjacency matrix to create a petersburg graph and simulate the outcome for each entry

        Returns the fitted label of the most frequently simulated terminal category for
        each sample, as an (n,) object-dtype array so non-numeric labels survive.

        :param X:
        :param y:
        :raises ValueError: If num_simulations is not a positive integer, or X is not 2D
        :return:
        """

        _require_fitted(self)
        validate_sample_count("num_simulations", self.num_simulations)
        X = _validate_feature_matrix(self, X)

        counters = _simulated_terminal_counters(self, X, clf_matrix=self._clf_matrix)

        y_hat = np.empty(X.shape[0], dtype=object)
        for r_idx, sims in enumerate(counters):
            y_hat[r_idx] = _terminal_label(self._categories, sims.most_common(1)[0][0])

        return y_hat

    def predict_proba(self, X):
        """
        Normalizes the full per-sample terminal distribution predict consumes into
        probabilities: an (n, len(classes_)) array whose rows sum to 1, where column j
        is the probability of classes_[j]. The simulated walks consume the trained
        classifiers' probabilities through the graph's edges.

        :param X:
        :raises ValueError: If num_simulations is not a positive integer, or X is not 2D
        :return:
        """

        _require_fitted(self)
        validate_sample_count("num_simulations", self.num_simulations)
        X = _validate_feature_matrix(self, X)

        counters = _simulated_terminal_counters(self, X, clf_matrix=self._clf_matrix)
        terminal = _terminal_indices(self._categories)

        proba = np.zeros((X.shape[0], len(terminal)), dtype=float)
        for r_idx, sims in enumerate(counters):
            for node_id, count in sims.items():
                try:
                    proba[r_idx, terminal.index(node_id)] = count
                except ValueError:
                    raise ValueError(
                        f"Simulation ended on node id {node_id}, which is not a terminal "
                        f"category; predict_proba only covers the final decision layer."
                    ) from None

        return proba / self.num_simulations

    def score(self, X, y, sample_weight=None):
        """Return the accuracy of predicted terminal labels."""

        return _terminal_accuracy(self, X, y, sample_weight=sample_weight)
