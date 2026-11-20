"""Reliability from labelled variances and explicitly supplied posterior states."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyfcirt.analysis.reliability import empirical_reliability, reliability
from tests.helpers import mixed_model


def moments():
    mean = xr.DataArray(
        [[0.0, 2.0], [1.0, 0.0], [3.0, 1.0]],
        dims=("person", "trait"),
        coords={"person": np.array([7, "7", "p"], dtype=object), "trait": ["A", "B"]},
    )
    return mean, xr.full_like(mean, 0.4)


@pytest.mark.parametrize("method", ["ratio", "subtraction"])
def test_explicit_moments_use_variance_and_align_labels(method):
    mean, variance = moments()
    actual = empirical_reliability(
        mean, variance.sel(person=np.array(["p", 7, "7"], dtype=object)).transpose(), method=method
    )
    v = mean.var("person", ddof=1)
    expected = v / (v + 0.4) if method == "ratio" else 1 - 0.4 / v
    assert_allclose(actual.reliability, expected)
    assert actual.attrs["ddof"] == 1
    assert_allclose(actual.mean_posterior_variance, 0.4)
    assert list(mean.person.values) == [7, "7", "p"]


@pytest.mark.parametrize("bad", ["negative", "nan", "boolean", "labels", "one_person"])
def test_invalid_uncertainty_rejects_without_repair(bad):
    mean, variance = moments()
    if bad == "negative":
        variance = -variance
    elif bad == "nan":
        variance[0, 0] = np.nan
    elif bad == "boolean":
        variance = variance.astype(bool)
    elif bad == "labels":
        variance = variance.assign_coords(person=["x", "y", "z"])
    else:
        mean = mean.isel(person=[0])
        variance = variance.isel(person=[0])
    with pytest.raises(ValueError):
        empirical_reliability(mean, variance)


def test_negative_subtraction_and_undefined_denominator_are_explicit():
    mean, variance = moments()
    assert (empirical_reliability(mean, variance * 100, method="subtraction").reliability < 0).all()
    with pytest.raises(ValueError, match="denominator"):
        empirical_reliability(mean * 0, variance * 0)


@pytest.mark.parametrize("inference_method", ["mcmc", "advi"])
def test_current_model_supplied_draws_and_person_selection_are_recomputed(
    tmp_path, inference_method
):
    model = mixed_model()
    constant = model.prepared.data["constant_data"].to_dataset()
    values = np.array(
        [[[[0.1, 0.3], [1.0, -0.2]], [[0.5, 0.7], [1.8, -0.8]], [[0.9, 0.2], [2.2, -0.4]]]]
    )
    theta = xr.DataArray(
        values,
        dims=("chain", "draw", "person", "trait"),
        coords={
            "chain": ["external"],
            "draw": [3, 9, 20],
            "person": constant.person,
            "trait": constant.trait,
        },
    )
    model.idata["posterior"] = xr.Dataset({"theta": theta})
    model.idata.attrs["inference_method"] = inference_method
    actual = reliability(model)
    expected = empirical_reliability(
        theta.mean(("chain", "draw")), theta.var(("chain", "draw"), ddof=1)
    )
    assert_allclose(actual.reliability, expected.reliability)
    assert bool(actual.attrs["approximate"]) == (inference_method != "mcmc")
    path = tmp_path / "reliability.nc"
    actual.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as saved:
        xr.testing.assert_equal(saved.load(), actual)
    assert_allclose(
        reliability(model, person_ids=list(reversed(constant.person.values))).reliability,
        actual.reliability,
    )
    model.idata["posterior"]["theta"] = theta * 2
    assert_allclose(reliability(model).score_variance, 4 * actual.score_variance)
    with pytest.raises(ValueError, match="duplicate or unknown"):
        reliability(model, person_ids=["unknown"])


def _fixed_information_reliability_case(*, missing=False, regression=False, one_person=False):
    import pandas as pd

    import pyfcirt as fc
    from tests.helpers import labelled_point, supplied_posterior

    data = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["item"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "item_id": ["item"] * 2,
                    "response": [None if missing == "all" else 0, None if missing else 1],
                }
            ).iloc[: 1 if one_person else 2],
            format="item_long",
        ),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(data))
    if regression:
        model.add_regressions(
            {
                "theta": fc.Regression(
                    "x", pd.DataFrame({"person_id": [1, 2], "x": [-1.0, 1.0]}), id="person_id"
                )
            }
        )
    point = labelled_point(
        model,
        {
            "theta": [[-0.5]] if one_person else [[-0.5], [0.8]],
            "covariance": [[1.0]],
            "population_mean": [[0.0]],
            "loading": [1.1],
            "item_intercept": [0.2],
            "lower_asymptote": [0.0],
            "upper_asymptote": [1.0],
        },
    )
    if regression:
        point["regression_coefficient"] = labelled_point(model, {"regression_coefficient": [0.2]})[
            "regression_coefficient"
        ]
    # Explicit fixed arrays exercise the MCMC-only gate; no sampler is executed.
    return supplied_posterior(model, point, inference_method="mcmc")


def test_information_reliability_small_nested_grid_matches_independent_scalar_formula():
    from scipy.special import expit, ndtri
    from scipy.stats import qmc

    from pyfcirt.analysis.reliability import information_reliability

    model = _fixed_information_reliability_case()
    actual = information_reliability(
        model, points=32, refine_points=64, scrambles=2, random_seed=17, chunk_size=16
    )

    def variance(theta):
        probability = expit(1.1 * np.asarray(theta) - 0.2)
        return 1 / (1 + 1.1**2 * probability * (1 - probability))

    by_scramble = []
    low = []
    for seed in [17, 18]:
        ability = ndtri(qmc.Sobol(1, scramble=True, seed=seed).random_base2(6)[:, 0])
        uncertainty = variance(ability)
        by_scramble.append(1 - uncertainty.mean())
        low.append(1 - uncertainty[:32].mean())
    by_scramble = np.array(by_scramble)
    assert_allclose(actual.marginal_by_scramble[:, 0], by_scramble, atol=1e-13, rtol=0)
    assert_allclose(actual.marginal, by_scramble.mean(), atol=1e-13, rtol=0)
    assert_allclose(actual.marginal_low, np.mean(low), atol=1e-13, rtol=0)
    assert_allclose(actual.refinement_delta, by_scramble.mean() - np.mean(low), atol=1e-13, rtol=0)
    assert_allclose(actual.scramble_sd, by_scramble.std(ddof=1), atol=1e-13, rtol=0)
    score_variance = np.var([-0.5, 0.8], ddof=1)
    error_variance = variance([-0.5, 0.8]).mean()
    assert_allclose(actual.empirical_information_variance, error_variance, atol=1e-13, rtol=0)
    assert_allclose(
        actual.empirical, score_variance / (score_variance + error_variance), atol=1e-13, rtol=0
    )
    assert "not a guaranteed error bound" in actual.attrs["integration"]
    assert actual.attrs["calibration_uncertainty"] == "excluded"


def test_information_reliability_rejects_different_actual_answered_forms():
    from pyfcirt.analysis.reliability import information_reliability

    with pytest.raises(ValueError, match="same actual answered form"):
        information_reliability(
            _fixed_information_reliability_case(missing=True),
            points=32,
            refine_points=64,
            scrambles=2,
            random_seed=17,
            chunk_size=16,
        )


def test_information_reliability_rejects_theta_regression_with_the_current_recipe():
    from pyfcirt.analysis.reliability import information_reliability

    with pytest.raises(NotImplementedError, match="covariate/group"):
        information_reliability(_fixed_information_reliability_case(regression=True))


@pytest.mark.parametrize(
    "problem", ["method", "array", "dims", "coordinate", "duplicate", "empty", "mismatch"]
)
def test_explicit_reliability_validates_axes_before_arithmetic(problem):
    mean, variance = moments()
    options = {}
    if problem == "method":
        options["method"] = "unknown"
    elif problem == "array":
        mean = mean.values
    elif problem == "dims":
        mean = mean.rename(person="respondent")
    elif problem == "coordinate":
        mean = mean.drop_vars("person")
    elif problem == "duplicate":
        mean = mean.assign_coords(person=[7, 7, "p"])
    elif problem == "empty":
        mean = mean.isel(person=[])
    else:
        variance = variance.isel(trait=0, drop=True)
    with pytest.raises(ValueError):
        empirical_reliability(mean, variance, **options)


def test_reliability_carries_uncertainty_metadata_and_rejects_conflicts():
    mean, variance = moments()
    mean = mean.isel(trait=0, drop=True)
    variance = variance.isel(trait=0, drop=True)
    mean.attrs = {"source": "external", "approximate": True}
    variance.attrs = {"uncertainty_source": "posterior variance", "source": "external"}
    result = empirical_reliability(mean, variance)
    assert result.reliability.ndim == 0
    assert result.attrs["source"] == "external" and result.attrs["approximate"]
    assert result.attrs["uncertainty_source"] == "posterior variance"
    variance.attrs["source"] = "different"
    with pytest.raises(ValueError, match="conflicting source"):
        empirical_reliability(mean, variance)


def test_model_reliability_rejects_bad_theta_and_person_selection(supplied_model):
    original = supplied_model.idata["posterior"].to_dataset()
    for bad in [
        original.drop_vars("theta"),
        original.isel(draw=[0]),
        original.assign(theta=original.theta.astype(bool)),
        original.assign(theta=xr.full_like(original.theta, np.nan)),
    ]:
        supplied_model.idata["posterior"] = bad
        with pytest.raises(ValueError, match="theta|states"):
            reliability(supplied_model)
    supplied_model.idata["posterior"] = original
    for ids in ["person", {1, "1"}, 1]:
        with pytest.raises(TypeError, match="ordered sequence"):
            reliability(supplied_model, person_ids=ids)


@pytest.mark.parametrize(
    "settings, exception, message",
    [
        ({"points": 3}, ValueError, "power of two"),
        ({"refine_points": True}, ValueError, "power of two"),
        ({"points": 8, "refine_points": 8}, ValueError, "exceed"),
        ({"scrambles": 1.5}, TypeError, "integers"),
        ({"scrambles": 1}, ValueError, "two scrambles"),
        ({"chunk_size": 0}, ValueError, "positive chunk_size"),
        ({"random_seed": True}, ValueError, "random_seed"),
    ],
)
def test_information_reliability_validates_integration_controls(
    supplied_model, settings, exception, message
):
    from pyfcirt.analysis.reliability import information_reliability

    with pytest.raises(exception, match=message):
        information_reliability(supplied_model, **settings)


def test_information_reliability_cannot_override_identified_population_location():
    from pyfcirt.analysis.reliability import information_reliability

    model = _fixed_information_reliability_case()
    posterior = model.idata["posterior"].to_dataset()
    posterior["population_mean"] = xr.full_like(posterior["population_mean"], 0.3)
    model.idata["posterior"] = posterior
    with pytest.raises(
        ValueError, match="population_mean: final values conflict with fixed entries"
    ):
        information_reliability(model, points=2, refine_points=4)


@pytest.mark.parametrize(
    "case, message", [("one_person", "at least two persons"), ("no_answers", "answered form")]
)
def test_information_reliability_requires_people_and_an_actual_answered_form(case, message):
    from pyfcirt.analysis.reliability import information_reliability

    model = _fixed_information_reliability_case(
        one_person=case == "one_person", missing="all" if case == "no_answers" else False
    )
    with pytest.raises(ValueError, match=message):
        information_reliability(model, points=2, refine_points=4)


@pytest.mark.parametrize("limitation", ["individual", "composite"])
def test_information_reliability_declines_person_effects_and_composite_rank_laws(limitation):
    import pandas as pd

    import pyfcirt as fc
    from pyfcirt.analysis.reliability import information_reliability
    from tests.helpers import supplied_posterior
    from tests.test_gtum_blackjax import gtum_model

    if limitation == "individual":
        model = gtum_model(block_effect=True)
    else:
        source = fc.FCInput(
            fc.FCDesignMapping(
                traits={"A": ["a", "c"], "B": ["b"]}, blocks={"rank": ["a", "b", "c"]}
            ),
            fc.ResponseTable(
                pd.DataFrame(
                    {
                        "person_id": [1, 2],
                        "block_id": ["rank"] * 2,
                        "response": [[1, 2, 3], [2, 1, 3]],
                    }
                ),
                format="block_long",
            ),
            model=fc.TIRT(),
        )
        model = fc.Model(fc.PreparedInputs.from_inputs(source))
    prior = fc.prior_predictive(model, draws=1, random_seed=9, store=False)
    supplied_posterior(
        model, prior["prior"].to_dataset().isel(chain=0, draw=0, drop=True), inference_method="mcmc"
    )
    with pytest.raises(NotImplementedError, match="nuisance|joint ranking"):
        information_reliability(model, points=2, refine_points=4)
    if limitation == "individual":
        coefficients = fc.coef(model, kind="stats")
        assert not any(
            "person_block_effect" in name
            for name in coefficients.index.get_level_values("parameter")
        )
