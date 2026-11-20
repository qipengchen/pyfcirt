"""Independent scalar mathematics for MGPPM tests, not production code."""

from itertools import permutations
from math import exp, log


def category_logps(utilities, scores):
    """Evaluate normalized score-weighted probabilities using scalar arithmetic."""
    logits = [
        sum(score * utility for score, utility in zip(row, utilities, strict=True))
        for row in scores
    ]
    maximum = max(logits)
    normalizer = maximum + log(sum(exp(value - maximum) for value in logits))
    return [value - normalizer for value in logits]


def ranking_scores(size):
    """List preference orders and their per-statement K-r scores."""
    orders = list(permutations(range(1, size + 1)))
    scores = []
    for order in orders:
        row = [0] * size
        for rank, statement in enumerate(order, start=1):
            row[statement - 1] = size - rank
        scores.append(row)
    return orders, scores


def partial_logp(utilities, observed_order):
    """Sum exactly the complete orders compatible with the observed positions."""
    orders, scores = ranking_scores(len(utilities))
    complete_logps = category_logps(utilities, scores)
    selected = [
        logp
        for order, logp in zip(orders, complete_logps, strict=True)
        if all(
            choice is None or order[rank] == choice for rank, choice in enumerate(observed_order)
        )
    ]
    maximum = max(selected)
    return maximum + log(sum(exp(value - maximum) for value in selected))
