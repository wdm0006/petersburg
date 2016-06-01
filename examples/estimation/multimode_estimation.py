import numpy as np
import random
import json
from petersburg import MixedModeEstimator

__author__ = 'willmcginnis'


def make_data(n_samples=10000):
    """
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
                y.append([1, 1])
                X.append([random.random() + 10 for _ in range(20)] + [random.random() * 0.2 + 1.1])
            else:
                y.append([1, 2])
                X.append([random.random() + 25 for _ in range(20)] + [random.random() * 0.2 + 2.1])
        else:
            y.append([0, 0])
            X.append([random.random() - 10 for _ in range(20)] + [random.random() * 0.2 - 2.0])

    return np.array(X), np.array(y)

if __name__ == '__main__':
    # train a frequency estimator
    X, y = make_data(n_samples=10000)
    clf = MixedModeEstimator(verbose=True)
    clf.fit(X, y)

    X_test, y_test = make_data(n_samples=1000)
    y_hat = clf.predict(X_test)

    # print out what we've learned from it
    print('\nCategory Labels')
    labels = clf._cateogry_labels
    print(labels)

    print('\nUnique Predicted Outcomes')
    outcomes = sorted([str(labels[int(x)]) for x in set(y_hat.reshape(-1, ).tolist())])
    print(outcomes)

    print('\nHistogram')
    histogram = dict(zip(outcomes, [float(x) for x in np.histogram(y_hat, bins=[1.5, 2.5, 3.5, 4.5])[0]]))
    print(json.dumps(histogram, sort_keys=True, indent=4))
