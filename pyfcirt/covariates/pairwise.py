"""Proper pairwise shrinkage in orthonormal zero-sum group coordinates."""

from functools import lru_cache
from math import ceil, isfinite, lgamma, log, prod
from numbers import Integral, Real

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.random.op import RandomVariable
from scipy.linalg import helmert
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc


def group_basis(groups):
    """Map independent coordinates to zero-sum effects in group order."""
    return helmert(groups).T


def _group_contrasts(groups):
    basis = group_basis(groups)
    left, right = np.triu_indices(groups, 1)
    return basis, basis[left] - basis[right]


def _penalty(values, contrasts, cutoff, *, tensor=np):
    distances = (values[..., None, :] @ tensor.swapaxes(contrasts, -1, -2))[..., 0, :]
    return tensor.minimum(tensor.abs(distances), cutoff).sum(axis=-1)


@lru_cache(maxsize=32)
def _truncated_log_mass(groups, strength, cutoff, base_sd):
    """Integrate the penalty under the Gaussian proposal before building the RV."""
    _, contrasts = _group_contrasts(groups)

    def integrate(order, seed):
        net = qmc.Sobol(groups - 1, scramble=True, seed=seed)
        log_weight = [
            logsumexp(-strength * _penalty(base_sd * ndtri(net.random(4096)), contrasts, cutoff))
            for _ in range(2 ** (order - 12))
        ]
        return float(logsumexp(log_weight) - order * log(2))

    previous = integrate(15, 17)
    for order in (16, 17, 18, 19):
        first, second = (integrate(order, seed) for seed in (17, 29))
        if max(abs(first - previous), abs(second - first)) <= 5e-4:
            estimate = float(np.logaddexp(first, second) - log(2))
            if estimate < log(1e-3):
                raise ValueError("PairwiseL1 rejection sampling accepts fewer than 0.1% of draws")
            return estimate
        previous = first
    raise ValueError("PairwiseL1 could not resolve the truncated prior's normalizing constant")


class PairwiseL1RV(RandomVariable):
    """PyTensor event with a native batch axis and a vector of free contrasts."""

    name = "pairwise_l1"
    signature = "(g,k),(p,k),(),(),(),()->(k)"
    dtype = "floatX"
    _print_name = ("PairwiseL1", r"\operatorname{PairwiseL1}")

    @classmethod
    def rng_fn(cls, rng, basis, contrasts, strength, cutoff, base_sd, log_density, size):
        shape = () if size is None else tuple(size)
        # PyTensor broadcasts fixed inputs onto the named RV's batch axes.
        basis = basis[(0,) * (basis.ndim - 2)]
        contrasts = contrasts[(0,) * (contrasts.ndim - 2)]
        strength, cutoff, base_sd, log_density = (
            np.asarray(value).flat[0] for value in (strength, cutoff, base_sd, log_density)
        )
        groups = basis.shape[0]
        if np.isinf(cutoff):
            k = np.arange(1, groups)
            spacing = rng.exponential(1 / (strength * k * (groups - k)), size=(*shape, groups - 1))
            ordered = np.cumsum(np.concatenate((np.zeros((*shape, 1)), spacing), axis=-1), axis=-1)
            # PyMC supplies a Generator; permute each sampled label set independently.
            return rng.permuted(ordered, axis=-1) @ basis
        result = np.empty((prod(shape), groups - 1), dtype=basis.dtype)
        filled = 0
        while filled < len(result):
            request = min(ceil((len(result) - filled) * np.exp(log_density)), 16_384)
            proposals = base_sd * rng.standard_normal((request, groups - 1))
            accepted = proposals[
                np.log(rng.random(request)) < -strength * _penalty(proposals, contrasts, cutoff)
            ][: len(result) - filled]
            result[filled : filled + len(accepted)] = accepted
            filled += len(accepted)
        return result.reshape((*shape, groups - 1))


class PairwiseL1(pm.Continuous):
    """Shrink group differences on a fixed, orthonormal zero-sum contrast scale."""

    rv_op = PairwiseL1RV()

    @classmethod
    def dist(cls, groups, *, strength=1.0, cutoff=None, base_sd=1.0, **kwargs):
        """Return an unnamed prior with a vector support of length groups minus one."""
        if not isinstance(groups, Integral) or isinstance(groups, bool) or groups < 2:
            raise ValueError("PairwiseL1 requires at least two groups")
        options = {"strength": strength}
        if cutoff is not None:
            options.update(cutoff=cutoff, base_sd=base_sd)
        for name, value in options.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"PairwiseL1 {name} must be a fixed positive finite number")
        basis, contrasts = _group_contrasts(groups)
        if cutoff is None:
            # The map from zero-sum coordinates to adjacent spacings has |det| = sqrt(G).
            log_density = 0.5 * log(groups) + (groups - 1) * log(strength)
            log_density += 2 * lgamma(groups) - lgamma(groups + 1)
            cutoff, base_sd = np.inf, 0.0
        else:
            log_density = -_truncated_log_mass(groups, strength, cutoff, base_sd)
        return super().dist([basis, contrasts, strength, cutoff, base_sd, log_density], **kwargs)

    @staticmethod
    def support_point(rv, size, *parameters):
        return pt.zeros_like(rv)

    @staticmethod
    def logp(value, basis, contrasts, strength, cutoff, base_sd, log_density):
        penalty = strength * _penalty(value, contrasts, cutoff, tensor=pt)
        # Ordinary L1 has no Gaussian base; avoid dividing by zero in either branch.
        base = pm.logp(pm.Normal.dist(0, pt.switch(base_sd > 0, base_sd, 1)), value).sum(axis=-1)
        return log_density - penalty + pt.switch(base_sd > 0, base, 0.0)
