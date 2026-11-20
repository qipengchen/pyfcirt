"""Model-independent regional proposals and native MH, without posterior MCMC."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import norm

from pyfcirt.inference.conditional_mode import ModeProposal, independence_step
from pyfcirt.models.thurstonian.gtum import GTUMDistribution


def distance_target(value, location=0.2):
    return -0.5 * value**2 - 0.5 * ((1.5 - jnp.abs(value - location)) / 0.3) ** 2


def proposal_case(location=0.2):
    return ModeProposal.from_target(
        partial(distance_target, location=location),
        breakpoints=jnp.atleast_1d(location),
        search_bounds=(-6.0, 6.0),
        tail_location=0.0,
        tail_scale=1.0,
    )


def test_non_fc_regional_modes_curvatures_and_proposal_normalization():
    proposal = proposal_case()
    assert_allclose(proposal.centers[1:], np.array([0.2 - 1.5, 0.2 + 1.5]) / 1.09, atol=1e-8)
    assert_allclose(proposal.scales[1:], np.sqrt(0.09 / 1.09), atol=1e-12)
    assert not proposal.invalid_regions.any()
    assert_allclose(np.exp(proposal.log_weights).sum(), 1, atol=1e-12)
    integral = quad(
        lambda x: np.sum(
            np.exp(proposal.log_weights) * norm.pdf(x, proposal.centers, proposal.scales)
        ),
        -np.inf,
        np.inf,
    )[0]
    assert_allclose(integral, 1, atol=1e-10)
    # Context changes rebuild proposal geometry; no old updated value is an input.
    assert not np.allclose(proposal.centers, proposal_case(0.8).centers)


def test_empty_intervals_are_inactive_and_invalid_regional_curvature_is_reported():
    proposal = ModeProposal.from_target(
        distance_target,
        breakpoints=jnp.array([0.2, 0.2]),
        search_bounds=(-6, 6),
        tail_location=0,
        tail_scale=1,
    )
    assert not proposal.invalid_regions.any()
    assert np.isneginf(proposal.log_weights[2])
    bad = ModeProposal.from_target(
        lambda x: -((x**2 - 1) ** 2),
        breakpoints=jnp.array([]),
        search_bounds=(-0.1, 0.1),
        tail_location=0.0,
        tail_scale=1.0,
    )
    assert bad.invalid_regions.tolist() == [True]
    assert_allclose(np.exp(bad.log_weights), [1.0, 0.0])
    assert_allclose(bad.logdensity(0.4), norm.logpdf(0.4), atol=1e-12)


@pytest.mark.parametrize("batched", [False, True])
def test_native_independence_correction_and_key_replay(batched):
    location = jnp.array([0.2, -0.4]) if batched else jnp.asarray(0.2)
    old = jnp.array([-1.0, 1.0]) if batched else jnp.asarray(-1.0)
    target = partial(distance_target, location=location)
    proposal = ModeProposal.from_target(
        target,
        breakpoints=location[..., None],
        search_bounds=(-6.0, 6.0),
        tail_location=jnp.zeros_like(location),
        tail_scale=jnp.ones_like(location),
    )
    key = jax.random.key(12)
    update = jax.jit(partial(independence_step, logdensity=target, proposal=proposal))
    actual, info = update(key, old)
    replay, replay_info = update(key, old)
    candidate = proposal.sample(jax.random.split(key)[0])
    ratio = (
        target(candidate) - target(old) + proposal.logdensity(old) - proposal.logdensity(candidate)
    )
    assert_allclose(info.acceptance_rate, np.minimum(1.0, np.exp(ratio)), atol=1e-12)
    assert_allclose(actual, jnp.where(info.is_accepted, candidate, old), atol=0)
    assert_allclose(actual, replay, atol=0)
    assert_allclose(info.is_accepted, replay_info.is_accepted)
    assert not info.invalid_proposal.any()


@pytest.mark.parametrize("invalid_density", [False, True])
def test_valid_outside_support_proposals_and_numeric_failures_are_distinct(invalid_density):
    proposal = ModeProposal(
        jnp.array([2.0]), jnp.array([0.01]), jnp.array([0.0]), jnp.array([]), jnp.array([False])
    )

    def target(x):
        return jnp.where(x < 0, -0.5 * x**2, jnp.nan if invalid_density else -jnp.inf)

    value, info = independence_step(
        jax.random.key(3), jnp.asarray(-1.0), logdensity=target, proposal=proposal
    )
    assert_allclose(value, -1.0)
    assert not info.is_accepted
    assert info.acceptance_rate == 0
    assert bool(info.invalid_proposal) == invalid_density


def test_original_executed_stan_binary_and_graded_likelihood_values():
    import json
    from pathlib import Path

    reference = json.loads(
        (Path(__file__).parent / "reference/gtum_stan_likelihood.json").read_text()
    )
    groups = {group["name"]: group for group in reference["groups"]}
    pairs = reference["conventions"]["pairs_zero_based"]
    contract = reference["numerical_contract"]
    from tests.reference.thurstonian import gtum_log_probs

    for row in reference["results"]:
        group = groups[row["group"].removesuffix("_whole")]
        original = row.get("original", row.get("whole"))
        person = row["person"]
        pair = row["pair"]
        index = np.array(pairs[pair])
        cuts = group[f"{row['kind']}_cutpoints"][pair]
        actual = GTUMDistribution.kernel(
            jnp.array(group["theta"][person])[index],
            jnp.array(group["loading"])[index],
            jnp.array(group["location"])[index],
            jnp.array(cuts),
            group["native_person_block_effect"][person],
            1.0,
        )
        production = actual[row["category"]]
        assert_allclose(
            np.exp(production),
            np.exp(original),
            atol=contract["probability_atol"],
            rtol=contract["probability_rtol"],
        )
        low, high = contract["binary_well_conditioned_probability"]
        if row["kind"] == "graded" or low <= np.exp(original) <= high:
            assert_allclose(
                production,
                original,
                atol=contract["strict_log_atol"],
                rtol=contract["strict_log_rtol"],
            )
        # The author subtracts inv_logit from one; the stable scalar law retains tails.
        utility = -np.array(group["loading"])[index] * np.abs(
            np.array(group["theta"][person])[index] - np.array(group["location"])[index]
        )
        eta = utility[0] - utility[1] + group["native_person_block_effect"][person]
        stable = gtum_log_probs(eta, cuts)[row["category"]]
        assert_allclose(production, stable, atol=contract["independent_stable_log_atol"], rtol=0)
