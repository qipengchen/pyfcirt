"""Numerical and public-workflow checks for DIF evidence."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from arviz_stats import bayes_factor
from numpy.testing import assert_allclose
from scipy.stats import norm

import pyfcirt as fc
from pyfcirt.analysis.dif import _bf_fields, _spike_evidence


def _tirt(prior, *, prior_options=None, sites=("A", "B")):
    pytest.importorskip("formulae")
    blocks = ["B1", "B2", "B3"]
    traits = [("One", "Two"), ("One", "Three"), ("Two", "Three")]
    design = pd.DataFrame(
        [
            (block, pos, f"{block}_{pos}", pair[pos - 1], key)
            for block, pair in zip(blocks, traits, strict=True)
            for pos, key in ((1, 1), (2, -1))
        ],
        columns=["block_id", "position", "statement_id", "trait", "keying"],
    )
    people = pd.DataFrame({"person_id": np.arange(12), "site": np.repeat(sites, 12 // len(sites))})
    responses = pd.DataFrame(
        {
            "person_id": people.person_id.repeat(len(blocks)).to_numpy(),
            "block_id": blocks * len(people),
            "response": [
                1 if (person >= 6 and block == "B3") or (person % 2 == 0 and block != "B3") else 0
                for person in people.person_id
                for block in blocks
            ],
        }
    )
    model = fc.Model().add_fc(
        design,
        responses,
        model=fc.TIRT(residuals="fixed"),
        format="block_long",
        response_format=None,
    )
    model.prepare()
    model.add_dif_terms(
        {
            "fc.loading_magnitude": {"anchors": ["B1_1", "B1_2", "B2_1", "B2_2"]},
            "fc.cutpoints": {"anchors": [("B1", "B1_1", "B1_2"), ("B2", "B2_1", "B2_2")]},
        },
        formula="site",
        person_data=people,
        prior=prior,
        prior_options=prior_options,
    )
    return model


def _prior_as_samples(model, *, draws=1500):
    native = model.build()
    names = ["loading", "cutpoint", "regression_coefficient"]
    values = pm.draw([native[name] for name in names], draws=draws, random_seed=930)
    dimensions = model.prepared.parameter_layouts
    variables = {
        name: (("chain", "draw", *dimensions[name].dims), draws_array[None])
        for name, draws_array in zip(names, values, strict=True)
    }
    coords = {dim: model.prepared.coords[dim] for name in names for dim in dimensions[name].dims}
    dataset = xr.Dataset(variables, coords={"chain": [0], "draw": np.arange(draws), **coords})
    return xr.DataTree.from_dict({"/posterior": model.prepared.annotate_coordinates(dataset)})


@pytest.mark.parametrize("law", ["normal", "laplace", "pairwise_l1", "truncated_l1"])
def test_bayes_factor_reuses_fitted_groups_and_prior(law):
    prior = pm.Normal.dist(0.12, 0.4) if law == "normal" else law
    model = _tirt(prior)
    samples = _prior_as_samples(model)
    model.attach_samples(samples)
    options = {"bayes_factor": True, "random_seed": 894}
    direct = model.dif_summary({"site": ("A", "B")}, **options)
    reverse = model.dif_summary({"site": ("B", "A")}, **options)
    interval = model.dif_summary({"site": ("A", "B")})
    assert direct[interval.columns].equals(interval)
    assert direct.units.tolist() == [("B3",)] * 3
    assert direct.bf_scale.tolist() == ["log_loading", "log_loading", "cutpoint"]
    assert "bf_status" not in direct.columns
    assert_allclose(direct.BF10, 1, atol=0.22)
    assert_allclose(direct.BF10 * direct.BF01, 1)
    assert_allclose(direct.BF10, reverse.BF10, rtol=0.015)
    assert_allclose(direct["mean"], -reverse["mean"])


def test_bayes_factor_rejects_a_single_comparison_from_three_groups():
    model = _tirt("laplace", sites=("A", "B", "C"))
    model.attach_samples(_prior_as_samples(model, draws=80))
    contrast = {"site": ("A", "B")}
    assert len(model.dif_summary(contrast)) == 3
    with pytest.raises(ValueError, match="two-group TIRT"):
        model.dif_summary(contrast, bayes_factor=True)


def test_continuous_bayes_factor_matches_normal_marginal_likelihood():
    rng = np.random.default_rng(930)
    # y ~ Normal(delta, 0.5), delta ~ Normal(0, 1), observed y = 0.75.
    observed, likelihood_sd, prior_sd = 0.75, 0.5, 1.0
    variance = 1 / (prior_sd**-2 + likelihood_sd**-2)
    expected = norm.pdf(observed, scale=np.hypot(prior_sd, likelihood_sd)) / norm.pdf(
        observed, scale=likelihood_sd
    )
    prior = rng.normal(0, prior_sd, size=(25_000, 3))
    posterior = rng.normal(
        variance * observed / likelihood_sd**2, np.sqrt(variance), size=(25_000, 3)
    )
    grid = (np.arange(prior.shape[0]) + 0.5) / prior.shape[0]
    prior[:, 0] = norm.ppf(grid, scale=prior_sd)
    posterior[:, 0] = norm.ppf(
        grid, loc=variance * observed / likelihood_sd**2, scale=np.sqrt(variance)
    )
    posterior[:, 1] = np.abs(posterior[:, 1]) + 1
    posterior[:, 2] = 0
    with pytest.warns(UserWarning, match="outside the posterior range"):
        result = _bf_fields(posterior, prior)
    assert_allclose(result["BF10"][0], expected, rtol=0.055)
    assert np.isnan(result["BF10"][2])
    assert "bf_status" not in result

    data = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset({"effect": (("chain", "draw"), posterior[None, :, 1])}),
            "/prior": xr.Dataset({"effect": (("chain", "draw"), prior[None, :, 1])}),
        }
    )
    with pytest.warns(UserWarning, match="outside the posterior range"):
        native = bayes_factor(data, "effect")
    for kind in ("BF10", "BF01"):
        value = (
            native["effect"][kind] if isinstance(native, dict) else native.effect.sel(bf_type=kind)
        )
        assert result[kind][1] == float(value)


def test_spike_and_slab_native_prior_and_posterior_odds():
    model = _tirt("spike_and_slab", prior_options={"inclusion_prob": 0.25, "slab_sd": 0.4})
    native = model.build()
    assert sum(rv.name.endswith("_included") for rv in native.free_RVs) == 2
    assert sum(rv.name.endswith("_slab") for rv in native.free_RVs) == 2

    samples = _prior_as_samples(model)
    coefficients = samples.posterior.regression_coefficient.values
    model.attach_samples(samples)
    result = model.dif_summary({"site": ("A", "B")}, bayes_factor=True)
    probability = np.mean(coefficients != 0, axis=(0, 1))
    assert_allclose(probability, 0.25, atol=0.035)
    assert_allclose(result.inclusion_probability, probability)
    assert_allclose(result.BF10, (probability / (1 - probability)) * 3)
    assert_allclose(result.BF10 * result.BF01, 1)

    edge = _spike_evidence(np.column_stack([np.zeros(80), np.r_[1, np.zeros(79)]]), 0.25)
    assert edge["bf_status"].tolist() == ["indicator_state_unobserved", "few_indicator_draws"]
    assert np.isnan(edge["BF10"][0])
    assert_allclose(edge["BF10"][1], 3 / 79)
