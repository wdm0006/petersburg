import json
import random
from collections import Counter

import numpy as np

from petersburg import FrequencyEstimator

__author__ = "willmcginnis"


def make_data(n_samples=10000):
    r"""
    Creates a sample hierarchical weather dataset with the graph form:


    s ---  percip -- more than 2mm
      |            |
      |            \_ less than 2mm
      \_no percip - no percip

    Where there are then two columns:

     * has_percip
     * has_more_than_2mm

    With values:

     * 0 (no percip), 1 (perceip)
     * 0 (no percip), 1 (less than 2mm), 2 (more than 2mm)

    :return:
    """

    _chance_of_percip = 0.2
    _chance_of_heavy = 0.4

    y = []
    X = []
    for _ in range(n_samples):
        if random.random() < _chance_of_percip:
            if random.random() < _chance_of_heavy:
                y.append([0, 0, 0])
                X.append([random.random() + 10 for _ in range(20)] + [random.random() * 0.2 + 1.1])
            else:
                y.append([0, 0, 1])
                X.append([random.random() + 25 for _ in range(20)] + [random.random() * 0.2 + 2.1])
        else:
            y.append([0, 1, 2])
            X.append([random.random() - 10 for _ in range(20)] + [random.random() * 0.2 - 2.0])

    return np.array(X), np.array(y)


def validate(y, truth):
    t = 0
    f = 0
    y = y.reshape(
        -1,
    ).tolist()
    for idx in range(len(truth)):
        if y[idx] == truth[idx][-1]:
            t += 1
        else:
            f += 1

    return t / (t + f)


if __name__ == "__main__":
    # train a frequency estimator
    X, y = make_data(n_samples=100000)
    clf = FrequencyEstimator(verbose=True, num_simulations=1)
    clf.fit(X, y)

    X_test, y_test = make_data(n_samples=10000)
    y_hat = clf.predict(X_test)

    # print out what we've learned from it
    print("\nCategory Labels")
    labels = clf._cateogry_labels
    print(labels)

    # predict() returns the fitted terminal label for each row, not an internal index.
    counts = Counter(
        y_hat.reshape(
            -1,
        ).tolist()
    )

    print("\nUnique Predicted Outcomes")
    print(sorted(str(label) for label in counts))

    print("\nHistogram")
    histogram = {str(label): float(count) for label, count in counts.items()}
    print(json.dumps(histogram, sort_keys=True, indent=4))

    accuracy = validate(y_hat, y_test)
    print("\nOverall Accuracy")
    print(f"{accuracy * 100.0:9.5f}%")
