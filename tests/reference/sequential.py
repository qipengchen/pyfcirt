"""Independent successive-choice probabilities."""

from itertools import permutations

import numpy as np


def scalar_probabilities(utility):
    """Direct product of sequential endorsement-odds ratios, for tests only."""
    values = np.exp(np.asarray(utility) - max(utility))
    probabilities = []
    for order in permutations(range(len(values))):
        p = 1.0
        for stage, choice in enumerate(order[:-1]):
            p *= values[choice] / sum(values[i] for i in order[stage:])
        probabilities.append(p)
    return np.array(probabilities)


def scalar_odds(theta, magnitude, location, threshold):
    """Independent hyperbolic-cosine form of the binary GGUM odds."""

    def log_cosh(x):
        return abs(x) + np.log1p(np.exp(-2 * abs(x))) - np.log(2)

    values = []
    for t, a, d, tau in zip(theta, magnitude, location, threshold, strict=True):
        x = a * (t - d)
        values.append(-a * tau + log_cosh(x / 2) - log_cosh(3 * x / 2))
    return np.array(values)
