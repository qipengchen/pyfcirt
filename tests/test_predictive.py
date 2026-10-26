"""Posterior summaries and response generation using fixed posterior inputs."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyfcirt import Model
from pyfcirt.results import FitResult
from tests.test_prior_predictive import case_data


@pytest.fixture(scope="module", params=["rank", "graded"])
def fixed_posterior(request):
    built = Model(case_data(request.param)).build()
    forward = built.prior_predictive(draws=80, random_seed=19)
    # Fixed forward draws exercise calculations, not posterior inference quality.
    posterior = forward["prior"].to_dataset()
    posterior = xr.concat([posterior, posterior.assign_coords(chain=[1])], dim="chain")
    idata = xr.DataTree.from_dict({"posterior": posterior})
    return FitResult(idata=idata, built=built, sampling={})


def test_summaries_delegate_to_arviz_with_real_labels(fixed_posterior):
    result = fixed_posterior
    for role in ["item_parameters", "traits", "trait_correlation"]:
        selected = result.parameter_draws(role)
        assert set(selected.data_vars) == set(result.spec.variable_roles[role])
        pd.testing.assert_frame_equal(
            result.summary(role, kind="stats"), az.summary(selected, kind="stats")
        )
        if role == "trait_correlation":
            # Unit diagonal entries are constant, so their diagnostics are undefined.
            with pytest.warns(RuntimeWarning, match="invalid value encountered"):
                actual = result.diagnostics(role)
            with pytest.warns(RuntimeWarning, match="invalid value encountered"):
                expected = az.summary(selected, kind="diagnostics")
        else:
            actual = result.diagnostics(role)
            expected = az.summary(selected, kind="diagnostics")
        pd.testing.assert_frame_equal(actual, expected)
    assert result.parameter_draws("traits")["theta"].coords["person"].values.tolist() == list(
        result.built.inputs.person_ids
    )
    with pytest.raises(ValueError, match="parameter role"):
        result.summary("responses")


def test_posterior_prediction_uses_frozen_draws_and_preserves_input(fixed_posterior):
    result = fixed_posterior
    original = result.idata.copy(deep=True)
    prediction = result.posterior_predictive(random_seed=44, progressbar=False)
    repeated = result.posterior_predictive(random_seed=44, progressbar=False)
    xr.testing.assert_identical(result.idata, original)
    xr.testing.assert_equal(
        prediction["posterior_predictive"].to_dataset(),
        repeated["posterior_predictive"].to_dataset(),
    )
    assert prediction["posterior_predictive"].sizes["draw"] == 80
    assert prediction["posterior_predictive"].sizes["chain"] == 2
    frame = result.response_frame(prediction)
    restored = type(result.built.data).from_frames(frame, result.built.data.design)
    assert restored.summary()["n_observations"] == len(result.spec.observation_map)
    for component, group in zip(result.spec.components, result.built.inputs.groups, strict=True):
        values = prediction["posterior_predictive"][component["observed_variable"]].values
        assert np.all((values >= 0) & (values < len(group.events)))
        # All replicas equalling the observed responses would expose response reuse.
        assert np.any(values != group.response)


def test_missing_posterior_parameters_are_not_resampled(fixed_posterior):
    result = fixed_posterior
    altered = result.idata.copy(deep=True)
    altered["posterior"] = altered["posterior"].to_dataset().drop_vars("loading_magnitude")
    with pytest.raises(ValueError, match="missing required parameters"):
        FitResult(idata=altered, built=result.built, sampling={}).posterior_predictive(
            progressbar=False
        )


def test_posterior_prediction_explicitly_freezes_all_free_parameters(fixed_posterior, monkeypatch):
    import pymc as pm

    result = fixed_posterior
    called = {}

    def forward(trace, **kwargs):
        xr.testing.assert_identical(
            trace,
            result.idata["posterior"].to_dataset()[[rv.name for rv in result.built.model.free_RVs]],
        )
        called.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(pm, "sample_posterior_predictive", forward)
    assert result.posterior_predictive(random_seed=12, progressbar=False) == "sentinel"
    assert called["sample_vars"] == result.spec.variable_roles["responses"]
    assert set(called["freeze_vars"]) == {rv.name for rv in result.built.model.free_RVs}
    assert called["extend_inferencedata"] is False


def test_prediction_obeys_supplied_posterior_parameters(fixed_posterior):
    result = fixed_posterior
    fixed = result.idata.copy(deep=True)
    fixed["posterior"]["theta"].values[...] = 0
    for group in result.built.inputs.groups:
        # Ordered intercepts force a single event at machine precision. This
        # would fail if forward sampling reused observations or redrew priors.
        size = group.scores.shape[1]
        fixed["posterior"][f"{group.name}_intercept"].values[...] = np.linspace(-100, 100, size)
    chosen = FitResult(idata=fixed, built=result.built, sampling={})
    prediction = chosen.posterior_predictive(random_seed=31, progressbar=False)
    for component, group in zip(result.spec.components, result.built.inputs.groups, strict=True):
        event = (
            "e"
            if group.task == "graded"
            else [1, None, None]
            if group.task == "pick"
            else [1, None, 3]
            if group.task == "mole"
            else [1, 2, 3]
        )
        expected = component["events"].index(event)
        assert (
            prediction["posterior_predictive"][component["observed_variable"]].values == expected
        ).all()


def test_prediction_rejects_mislabeled_parameters(fixed_posterior):
    result = fixed_posterior
    altered = result.idata.copy(deep=True)
    altered["posterior"] = (
        altered["posterior"]
        .to_dataset()
        .assign_coords(person=[f"wrong{i}" for i in range(len(result.built.inputs.person_ids))])
    )
    with pytest.raises(ValueError, match="coordinates do not match"):
        FitResult(idata=altered, built=result.built, sampling={}).posterior_predictive(
            progressbar=False
        )
