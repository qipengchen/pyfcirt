"""Scalar ranking laws, independent of production kernels and event membership."""

from itertools import permutations

import numpy as np
from scipy.special import logsumexp


def linear_reference(parameters, size, response, family):
    """Enumerate the defining ranking laws without production kernels or membership arrays."""
    theta, magnitude, intercept = np.asarray(parameters).reshape(3, size)
    utility = theta * magnitude * np.array([(-1) ** i for i in range(size)]) - intercept
    orders = list(permutations(range(size)))
    if family == "mgppm":
        pattern_logp = np.array(
            [
                sum((size - 1 - rank) * utility[item] for rank, item in enumerate(order))
                for order in orders
            ]
        )
        pattern_logp -= logsumexp(pattern_logp)
    else:
        pattern_logp = np.array(
            [
                sum(
                    utility[item] - logsumexp(utility[list(order[rank:])])
                    for rank, item in enumerate(order[:-1])
                )
                for order in orders
            ]
        )
    compatible = [
        logp
        for order, logp in zip(orders, pattern_logp, strict=True)
        if all(item is None or order[rank] == item - 1 for rank, item in enumerate(response))
    ]
    return logsumexp(compatible)


def ggum_reference(parameters, size, response):
    """Use GGUM hyperbolic-cosine odds and direct successive-choice products."""
    theta, magnitude, location, threshold = np.asarray(parameters).reshape(4, size)
    x = magnitude * (theta - location)
    weight = np.exp(-magnitude * threshold) * np.cosh(x / 2) / np.cosh(3 * x / 2)
    probability = 0.0
    for order in permutations(range(size)):
        if any(item is not None and order[rank] != item - 1 for rank, item in enumerate(response)):
            continue
        term = 1.0
        for rank, item in enumerate(order[:-1]):
            term *= weight[item] / sum(weight[remaining] for remaining in order[rank:])
        probability += term
    return np.log(probability)


def scalar_event_logp(utility, event, family):
    """Enumerate the defining laws, independently of production score/membership arrays."""
    orders = list(permutations(range(len(utility))))
    logits = []
    for order in orders:
        if family == "mgppm":
            logits.append(
                sum((len(order) - 1 - rank) * utility[item] for rank, item in enumerate(order))
            )
        else:
            logits.append(
                sum(
                    utility[item] - logsumexp(utility[list(order[rank:])])
                    for rank, item in enumerate(order[:-1])
                )
            )
    if family == "mgppm":
        logits = np.asarray(logits) - logsumexp(logits)
    return logsumexp(
        [
            value
            for order, value in zip(orders, logits, strict=True)
            if all(item is None or order[rank] == item - 1 for rank, item in enumerate(event))
        ]
    )
