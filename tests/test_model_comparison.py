"""Fixed-draw metric definitions and independent conditional likelihood checks."""

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyfcirt import FCData, FCDesign, Model, Prior
from pyfcirt.model_comparison import dic, waic
from pyfcirt.models import MGPPM
from pyfcirt.results import FitResult
from tests._reference_mgppm import category_logps, partial_logp
from tests.test_prior_predictive import case_data


def labelled_logp(values=None):
    values = np.linspace(-1.5, -0.4, 24).reshape(2, 4, 3) if values is None else np.asarray(values)
    return xr.DataArray(
        values,
        dims=("chain", "draw", "obs_id"),
        coords={
            "chain": [f"C{i}" for i in range(values.shape[0])],
            "draw": range(values.shape[1]),
            "obs_id": [f"O{i}" for i in range(values.shape[2])],
        },
        attrs={"observation_unit": "person-block", "likelihood_target": "conditional_response"},
        name="response",
    )


def test_dic_is_chainwise_total_deviance_variance():
    deviance = np.array([[2, 4, 6, 8], [20, 24, 28, 32]], dtype=float)
    logp = labelled_logp(-deviance[..., None] * np.array([0.1, 0.2, 0.2]))
    result = dic(logp)
    assert_allclose(result.Dbar_chain, [5, 26], atol=1e-12)
    assert_allclose(result.pV_chain, [10 / 3, 40 / 3], atol=1e-12)
    assert_allclose(result.DIC, 15.5 + 25 / 3, atol=1e-12)
    assert_allclose(result.pV, 25 / 3, atol=1e-12)
    assert not np.isclose(result.pV, deviance.var(ddof=1) / 2)
    assert not np.isclose(result.pV, ((-2 * logp).var("draw", ddof=1).sum("obs_id") / 2).mean())
    assert result.attrs["likelihood_target"] == "conditional_response"
    assert result.attrs["variance_ddof"] == 1


def test_waic_matches_independent_scalar_definition():
    from math import exp, log
    from statistics import pvariance

    logp = labelled_logp()
    output = waic(logp)
    expected = []
    penalty = []
    for observation in range(3):
        samples = logp.values[..., observation].ravel().tolist()
        normalizer = log(sum(exp(value) for value in samples) / len(samples))
        penalty.append(pvariance(samples))
        expected.append(normalizer - penalty[-1])
    assert_allclose(output.elpd_waic_i, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(output.p_waic_i, penalty, rtol=1e-12, atol=1e-12)
    assert_allclose(output.WAIC, -2 * sum(expected), atol=1e-12)
    assert_allclose(output.se_elpd, (3 * pvariance(expected)) ** 0.5, atol=1e-12)
    assert_allclose(output.se_WAIC, 2 * output.se_elpd, atol=1e-12)
    assert not bool(output.warning)
    assert output.attrs["variance_ddof"] == 0


def test_reordering_preserves_metrics_but_chain_reassignment_changes_dic():
    original = labelled_logp()
    permuted = original.sel(chain=["C1", "C0"], draw=[3, 1, 0, 2], obs_id=["O2", "O0", "O1"])
    assert_allclose(dic(original).DIC, dic(permuted).DIC)
    assert_allclose(waic(original).WAIC, waic(permuted).WAIC)
    xr.testing.assert_allclose(
        waic(original).elpd_waic_i,
        waic(permuted).elpd_waic_i.sel(obs_id=original.obs_id),
        rtol=1e-12,
        atol=1e-12,
    )
    regrouped = original.copy(data=original.values.swapaxes(0, 1).reshape(original.shape))
    assert not np.isclose(dic(original).DIC, dic(regrouped).DIC)
    assert_allclose(waic(original).WAIC, waic(regrouped).WAIC)
    assert_allclose(dic(original).DIC, dic(original.transpose("obs_id", "draw", "chain")).DIC)


def test_waic_extreme_finite_values_and_warning_rules():
    extreme = labelled_logp(np.full((2, 4, 3), -1000000.0))
    assert_allclose(waic(extreme).WAIC, 6000000, atol=1e-8)
    assert_allclose(dic(extreme).pV, 0)
    with pytest.warns(UserWarning, match="exceeds 0.4"):
        flagged = waic(labelled_logp(np.arange(24, dtype=float).reshape(2, 4, 3) * -0.3))
    assert bool(flagged.warning)
    with pytest.warns(UserWarning, match="Pointwise WAIC equals"):
        assert_allclose(waic(labelled_logp(np.zeros((2, 4, 3)))).WAIC, 0)
    with pytest.warns(UserWarning, match="Pointwise WAIC equals"):
        single = waic(labelled_logp(np.full((2, 4, 1), -0.2)))
    assert_allclose(single.se_WAIC, 0)
    assert not bool(single.warning)


@pytest.mark.parametrize("function", [dic, waic])
@pytest.mark.parametrize(
    "invalid",
    ["nan", "infinity", "duplicate", "missing_coordinate", "wrong_dim", "empty", "complex"],
)
def test_metrics_reject_invalid_inputs_without_dropping_observations(function, invalid):
    values = labelled_logp()
    if invalid == "nan":
        values.values[0, 0, 0] = np.nan
    elif invalid == "infinity":
        values.values[0, 0, 0] = -np.inf
    elif invalid == "duplicate":
        values = values.assign_coords(obs_id=["O", "O", "Q"])
    elif invalid == "missing_coordinate":
        values = values.drop_vars("obs_id")
    elif invalid == "wrong_dim":
        values = values.rename(obs_id="person")
    elif invalid == "empty":
        values = values.isel(draw=slice(0))
    else:
        values = values.astype(complex)
    with pytest.raises(ValueError):
        function(values)


def test_dic_requires_chainwise_variance_degrees_of_freedom():
    with pytest.raises(ValueError, match="two draws per chain"):
        dic(labelled_logp().isel(draw=[0]))
    with pytest.raises(TypeError, match="DataArray"):
        dic(np.zeros((2, 4, 3)))


def test_integer_logp_is_promoted_before_total_deviance_arithmetic():
    logp = labelled_logp(np.full((2, 4, 3), -2_000_000_000_000_000_000, dtype=np.int64))
    actual = dic(logp)
    assert_allclose(actual.Dbar, 1.2e19, rtol=1e-15)
    assert_allclose(actual.DIC, 1.2e19, rtol=1e-15)
    assert_allclose(actual.pV, 0)


@pytest.fixture(scope="module", params=["rank", "graded", "mixed_size"])
def comparison_case(request):
    if request.param == "mixed_size":
        design = FCDesign.from_mapping(
            traits={"T1": ["S1", "S3", "S5", "S7", "S9"], "T2": ["S2", "S4", "S6", "S8", "S10"]},
            blocks={
                "pair": ["S1", "S2"],
                "pick": ["S3", "S4", "S5", "S6"],
                "mole": ["S7", "S8", "S9", "S10"],
            },
            block_response_formats={"pick": "pick", "mole": "mole"},
            keying={"S2": -1},
            allow_variable_block_size=True,
        )
        data = FCData.from_frames(
            pd.DataFrame(
                {
                    "person_id": ["P"] * 3,
                    "block_id": ["pair", "pick", "mole"],
                    "response": [[2, 1], [3, None, None, None], [2, None, None, 4]],
                }
            ),
            design,
            administration=pd.DataFrame(
                {
                    "person_id": ["P", "P", "P", "Q", "Q"],
                    "block_id": ["pair", "pick", "mole", "pair", "mole"],
                    "status": ["answered", "answered", "answered", "missing", "not_administered"],
                }
            ),
        )
    else:
        data = case_data(request.param)
    family = MGPPM(
        priors={
            "loading_magnitude": Prior("LogNormal", mu=0, sigma=0.3),
            "ranking_intercept": Prior("ZeroSumNormal", sigma=0.4),
            "graded_intercept": Prior("Normal", mu=0, sigma=0.4),
        }
    )
    built = Model(data, family=family).build()
    forward = built.prior_predictive(draws=40, random_seed=72)
    posterior = forward["prior"].to_dataset()
    return FitResult(
        idata=xr.DataTree.from_dict({"posterior": posterior}), built=built, sampling={}
    )


def test_native_pointwise_logp_matches_scalar_probabilities(comparison_case):
    result = comparison_case
    original = result.idata.copy(deep=True)
    original_values = [variable.name for variable in result.built.model.value_vars]
    actual = result.log_likelihood(progressbar=False)
    xr.testing.assert_identical(result.idata, original)
    assert [variable.name for variable in result.built.model.value_vars] == original_values
    assert actual.dims == ("chain", "draw", "obs_id")
    assert actual.coords["obs_id"].values.tolist() == [
        r["obs_id"] for r in result.spec.observation_map
    ]
    assert actual.attrs["observation_unit"] == "person-block"
    posterior = result.idata["posterior"]
    expected = []
    for group in result.built.inputs.groups:
        for index, code in enumerate(group.response):
            statement = group.statement_index[group.block_index[index]]
            observed_draws = []
            for draw in range(40):
                theta = posterior["theta"].values[0, draw]
                loading = posterior["loading"].values[0, draw]
                intercept = posterior["intercept"].values[0, draw]
                utility = (
                    loading[statement]
                    * theta[
                        group.person_index[index], result.built.inputs.statement_trait[statement]
                    ]
                    - intercept[statement]
                )
                observed_draws.append(
                    category_logps(utility, group.scores)[code]
                    if group.task == "graded"
                    else partial_logp(utility, group.events[code])
                )
            expected.append(observed_draws)
    assert_allclose(actual.values[0], np.asarray(expected).T, rtol=1e-11, atol=1e-11)
    xr.testing.assert_identical(result.dic(progressbar=False), dic(actual))


def test_current_loo_is_direct_arviz_on_same_observations(comparison_case):
    result = comparison_case
    logp = result.log_likelihood(progressbar=False)
    tree = xr.DataTree.from_dict(
        {"posterior": result.idata["posterior"].to_dataset(), "log_likelihood": logp.to_dataset()}
    )
    with warnings.catch_warnings(record=True) as own:
        warnings.simplefilter("always")
        actual = result.loo(reff=1, progressbar=False)
    with warnings.catch_warnings(record=True) as native:
        warnings.simplefilter("always")
        expected = az.loo(tree, var_name="response", pointwise=True, reff=1)
    assert [(w.category, str(w.message)) for w in own] == [
        (w.category, str(w.message)) for w in native
    ]
    for name in ["elpd", "se", "p", "n_samples", "n_data_points"]:
        assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-12)
    xr.testing.assert_equal(actual.elpd_i, expected.elpd_i)
    xr.testing.assert_equal(actual.pareto_k, expected.pareto_k)


def test_likelihood_does_not_silently_relabel_posterior(comparison_case):
    result = comparison_case
    changed = result.idata.copy(deep=True)
    changed["posterior"] = (
        changed["posterior"]
        .to_dataset()
        .assign_coords(person=[f"wrong{i}" for i in range(len(result.built.inputs.person_ids))])
    )
    with pytest.raises(ValueError, match="coordinates do not match"):
        FitResult(idata=changed, built=result.built, sampling={}).log_likelihood(progressbar=False)


def test_native_likelihood_keeps_extremely_small_event_probabilities_finite(comparison_case):
    result = comparison_case
    changed = result.idata.copy(deep=True)
    size = len(result.built.inputs.trait_ids)
    changed["posterior"]["theta"].values[...] = np.linspace(-1000, 1000, size)
    logp = FitResult(idata=changed, built=result.built, sampling={}).log_likelihood(
        progressbar=False
    )
    assert np.isfinite(logp).all()
    assert float(logp.min()) < -700
