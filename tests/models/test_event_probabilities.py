"""Independent enumeration checks row-specific partial ranking events and gradients."""

from itertools import combinations, permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp

from pyfcirt.models.mgppm import MGPPMDistribution
from pyfcirt.models.sequential import GGUMRankDistribution, TwoPLRankDistribution


def partition(patterns, pairs):
    """Enumerate binary signatures independently of the production adapter."""
    signatures = [tuple(int(p.index(a) < p.index(b)) for a, b in pairs) for p in patterns]
    events = sorted(set(signatures))
    return np.array([[signature == event for signature in signatures] for event in events])


def scalar_utilities(parameters, size, family):
    theta = parameters[: 4 * size].reshape(4, size)
    magnitude, intercept = parameters[4 * size : 6 * size].reshape(2, size)
    if family != "ggum":
        return theta * magnitude * np.where(np.arange(size) % 2, -1, 1) - intercept
    threshold = parameters[6 * size :]

    def log_cosh(x):
        return abs(x) + np.log1p(np.exp(-2 * abs(x))) - np.log(2)

    return np.array(
        [
            [
                -a * tau + log_cosh(a * (t - d) / 2) - log_cosh(3 * a * (t - d) / 2)
                for t, a, d, tau in zip(row, magnitude, intercept, threshold, strict=True)
            ]
            for row in theta
        ]
    )


def scalar_patterns(utilities, patterns, family):
    """Evaluate model equations with scalar loops, independent of event kernels."""
    result = []
    for row in utilities:
        if family == "mgppm":
            logits = [
                sum((len(p) - 1 - rank) * row[s] for rank, s in enumerate(p)) for p in patterns
            ]
            result.append(np.array(logits) - logsumexp(logits))
        else:
            result.append(
                [
                    sum(row[p[j]] - logsumexp([row[s] for s in p[j:]]) for j in range(len(p) - 1))
                    for p in patterns
                ]
            )
    return np.array(result)


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("family", ["mgppm", "two_pl", "ggum"])
def test_row_specific_partial_events_and_all_parameter_gradients(size, family):
    patterns = list(permutations(range(size)))
    partitions = [
        partition(patterns, list(combinations(range(size), 2))),
        partition(patterns, [(0, 1)]),
        partition(patterns, [(i, i + 1) for i in range(size - 1)]),
    ]
    membership = np.zeros((4, max(len(p) for p in partitions), len(patterns)), bool)
    for row, values in enumerate(partitions):
        membership[row, : len(values)] = values
    membership[3] = membership[0]
    point = np.r_[
        np.linspace(-0.8, 0.9, 4 * size), np.linspace(0.7, 1.3, size), np.linspace(-1.2, 1.1, size)
    ]
    if family == "ggum":
        point = np.r_[point, np.linspace(-0.4, 0.3, size)]
    selected = np.array([len(p) - 1 for p in partitions] + [len(partitions[0]) - 1])

    def reference(values):
        complete = scalar_patterns(scalar_utilities(values, size, family), patterns, family)
        return np.array(
            [
                [logsumexp(logs[mask]) for mask in masks]
                for logs, masks in zip(complete, membership, strict=True)
            ]
        )

    def production(values):
        theta = values[: 4 * size].reshape(4, size)
        loading, offset = values[4 * size : 5 * size], values[5 * size : 6 * size]
        if family != "ggum":
            loading = loading * jnp.where(jnp.arange(size) % 2, -1, 1)
        if family == "mgppm":
            return MGPPMDistribution.kernel(
                theta,
                loading,
                offset,
                scores=size - 1 - np.argsort(patterns, axis=-1),
                membership=membership,
            )
        cls = GGUMRankDistribution if family == "ggum" else TwoPLRankDistribution
        args = (values[6 * size :],) if family == "ggum" else ()
        return cls.kernel(
            theta, loading, offset, *args, orders=np.array(patterns), membership=membership
        )

    assert_allclose(production(jnp.array(point)), reference(point), atol=2e-13)

    def objective(x):
        return production(x)[jnp.arange(4), selected].sum()

    finite = []
    for delta in np.eye(len(point)) * 1e-5:
        hi, lo = reference(point + delta), reference(point - delta)
        finite.append((hi[np.arange(4), selected] - lo[np.arange(4), selected]).sum() / 2e-5)
    assert_allclose(jax.grad(objective)(jnp.array(point)), finite, atol=2e-9, rtol=2e-8)
