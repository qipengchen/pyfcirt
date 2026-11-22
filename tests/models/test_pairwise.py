"""Check the normalization and generation of the pairwise shrinkage prior."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.linalg import helmert

from pyfcirt.covariates.pairwise import PairwiseL1


@pytest.mark.parametrize("cutoff", [None, 0.5])
def test_two_group_pairwise_density_normalizes_and_matches_generation(cutoff):
    rv = PairwiseL1.dist(2, strength=1.2, cutoff=cutoff)
    z = pt.vector("z")
    logp = pytensor.function([z], pm.logp(rv, z), mode="JAX")

    def density(value):
        return float(np.exp(logp(np.asarray([value], dtype=z.dtype))))

    bounds = (0.0, np.inf) if cutoff is None else (0.0, cutoff / np.sqrt(2), np.inf)
    intervals = list(zip(bounds[:-1], bounds[1:], strict=True))
    integral = 2 * sum(quad(density, a, b, epsabs=1e-7)[0] for a, b in intervals)
    moment = 2 * sum(quad(lambda x: x**2 * density(x), a, b)[0] for a, b in intervals)
    draws = pm.draw(rv, draws=12_000, random_seed=32)
    assert_allclose(integral, 1, atol=5e-5)
    assert_allclose(np.mean(draws**2), moment, rtol=0.05)
    assert_allclose(draws.mean(), 0, atol=0.03)


@pytest.mark.parametrize("cutoff", [None, 0.5])
def test_named_batched_pairwise_density_is_group_permutation_invariant(cutoff):
    groups = 3
    basis = helmert(groups).T
    coords = {"trial": range(4), "contrast": range(groups - 1)}
    with pm.Model(coords=coords) as model:
        rv = PairwiseL1("offset", groups, cutoff=cutoff, dims=("trial", "contrast"))
    draws = pm.draw(rv, draws=300, random_seed=42)
    assert draws.shape == (300, 4, 2)
    assert np.isfinite(model.compile_logp(mode="JAX")(model.initial_point()))
    selected = draws[0]
    reordered = (selected @ basis.T)[:, [2, 0, 1]] @ basis
    assert_allclose(pm.logp(rv, selected).eval(), pm.logp(rv, reordered).eval(), atol=1e-6)


def test_ordinary_pairwise_spacing_rates():
    groups = 4
    strength = 1.5
    prior = PairwiseL1.dist(groups, strength=strength)
    values = pm.draw(prior, draws=12_000, random_seed=87)
    spacings = np.diff(np.sort(values @ helmert(groups), axis=-1), axis=-1)
    k = np.arange(1, groups)
    assert_allclose(spacings.mean(axis=0), 1 / (strength * k * (groups - k)), rtol=0.05)
    with pytest.raises(ValueError, match="at least two"):
        PairwiseL1.dist(1)
