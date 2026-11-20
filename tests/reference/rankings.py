"""Independent iid Gaussian order-event probabilities and mean derivatives."""

from itertools import permutations
from math import factorial

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def normal_ranking_event(size, response):
    """Use symmetry and scalar normal order-statistic moments, without regional CDF code."""
    orders = [
        order
        for order in permutations(range(size))
        if all(item is None or order[rank] == item - 1 for rank, item in enumerate(response))
    ]
    ascending = []
    for rank in range(size):
        coefficient = factorial(size) / (factorial(rank) * factorial(size - 1 - rank))
        value = quad(
            lambda x, coefficient=coefficient, rank=rank: (
                x
                * coefficient
                * norm.pdf(x)
                * norm.cdf(x) ** rank
                * norm.sf(x) ** (size - 1 - rank)
            ),
            -np.inf,
            np.inf,
            epsabs=1e-11,
        )[0]
        ascending.append(value)
    gradient = np.zeros(size)
    for order in orders:
        for rank, item in enumerate(order):
            gradient[item] += ascending[-1 - rank] / len(orders)
    return len(orders) / factorial(size), gradient


def order_arrays(size):
    orders = np.array(list(permutations(range(size))), dtype="int64")
    contrasts = np.array(
        [
            [np.eye(size)[a] - np.eye(size)[b] for a, b in zip(row[:-1], row[1:], strict=True)]
            for row in orders
        ]
    )
    return orders, contrasts
