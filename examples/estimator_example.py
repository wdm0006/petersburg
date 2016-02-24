import numpy as np
import random
import json
from petersburg import FrequencyEstimator

__author__ = 'willmcginnis'

zero_count = 11
one_count = 32
two_count = 55
three_count = 50
four_count = 21
total = zero_count + one_count + two_count + three_count + four_count

# generate some data
y = np.array([[
    random.choice([0, 1, 2]),
    random.choice([0, 1, 1, 1, 2]),
    random.choice([0, 1, 2, 2, 2, 2]),
    random.choice(
        [0 for _ in range(zero_count)] +
        [1 for _ in range(one_count)] +
        [2 for _ in range(two_count)] +
        [3 for _ in range(three_count)] +
        [4 for _ in range(four_count)]
    )
] for _ in range(10000)])

# train a frequency estimator
clf = FrequencyEstimator(verbose=True)
clf.fit(None, y)
freq_matrix = clf._frequency_matrix
y_hat = clf.predict(np.zeros((10000, 10)))

# print out what we've learned from it
print('\nCategory Labels')
labels = clf._cateogry_labels
print(labels)

print('\nUnique Predicted Outcomes')
outcomes = sorted([str(labels[int(x)]) for x in set(y_hat.reshape(-1, ).tolist())])
print(outcomes)

print('\nHistogram')
histogram = dict(zip(outcomes, [float(x) for x in np.histogram(y_hat, bins=[8.5, 9.5, 10.5, 11.5, 12.5, 13.5])[0]]))
print(json.dumps(histogram, sort_keys=True, indent=4))