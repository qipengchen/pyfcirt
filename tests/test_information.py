"""Likelihood Fisher calculus and deterministic observation aggregation."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyfcirt.analysis.information import _aggregate_information, _batch_fisher, categorical_fisher
from pyfcirt.models.likert import DichotomousIRTDistribution


def test_logistic_fisher_outer_product_and_missing_observation_aggregation():
    loading = jnp.array([1.2, -0.7])
    point = jnp.array([0.4, -0.2])

    def kernel(theta):
        return DichotomousIRTDistribution.kernel(theta, loading, 0.3, 0.0, 1.0)

    p, score, info = categorical_fisher(kernel, point)
    assert_allclose(info, p[0] * p[1] * np.outer(loading, loading), atol=1e-13)
    assert_allclose(p @ score, 0, atol=1e-13)
    matrices = jnp.stack((info, 2 * info, 3 * info))
    by_unit, total = _aggregate_information(
        matrices, jnp.array([0, 0, 1]), jnp.array([0, 0, 1]), n_persons=3, n_units=2
    )
    assert_allclose(by_unit[0, 0], 3 * info)
    assert_allclose(by_unit[0, 1], 0)
    assert_allclose(total[2], 0)
    assert_allclose(by_unit.sum(axis=1), total)


def test_current_mixed_model_explicit_point_draws_and_stored_information_contract():
    from pyfcirt.analysis.information import information
    from tests.helpers import labelled_point, mixed_model, supplied_posterior

    model = mixed_model()
    point = labelled_point(
        model,
        {
            "theta": [[0.2, -0.4], [0.8, 0.5]],
            "covariance": [[1.0, 0.0], [0.0, 1.0]],
            "loading": [0.8, 1.1, 0.7, 1.2, 0.9, 0.8, 1.1, 1.3, 0.6, 1.2, 0.9],
            "statement_intercept": [0.0, 0.2, 0.0, -0.3, 0.1, 0.0, 0.4, -0.2, 0.3],
            "item_intercept": [0.3],
            "category_intercept": [0.0, -0.2, 0.4],
        },
    )
    actual = information(model, parameters=point, include_intermediates=True, progressbar=False)
    assert actual.block.values.tolist() == ["pair", "7", "quad"]
    assert actual.item.values.tolist() == ["pair", "ordinal"]
    assert_allclose(actual.block_information.isel(person=1, block=[1, 2]), 0)
    assert_allclose(actual.item_information.isel(person=1, item=1), 0)
    assert_allclose(actual.item_information.sel(item="pair", trait="B"), 0)
    assert_allclose(actual.item_information.sel(item="ordinal", trait="A"), 0)
    assert not actual.attrs["prior_information"]
    assert actual.attrs["parameter_source"] == "explicit"
    assert "information" in model.idata
    assert_allclose(model.idata["information/parameters"].theta, point.theta)
    assert_allclose(
        actual.test_information,
        actual.block_information.sum("block") + actual.item_information.sum("item"),
    )
    # The supplied draws are external final arrays with explicit approximate provenance.
    supplied_posterior(model, point, inference_method="advi")
    from_draw = information(model, draw=("supplied", 9), store=False, progressbar=False)
    assert_allclose(from_draw.test_information, actual.test_information)
    assert from_draw.attrs["parameter_source"] == "joint_draw"
    assert from_draw.attrs["inference_method"] == "advi"
    assert_allclose(model.idata["information"].test_information, actual.test_information)

    import pyfcirt as fc

    measurement_only = fc.Model(model.prepared)
    supplied_posterior(measurement_only, point.drop_vars("theta"), inference_method="advi")
    reordered_theta = point.theta.transpose("trait", "person").isel(person=[1, 0])
    for draw in (None, ("supplied", 9)):
        overridden = information(
            measurement_only, theta=reordered_theta, draw=draw, store=False, progressbar=False
        )
        xr.testing.assert_allclose(overridden.test_information, actual.test_information)
    with pytest.raises(ValueError, match="Missing required final posterior nodes.*theta"):
        information(measurement_only, store=False, progressbar=False)

    # An invalid public point must preserve the last successful stored result.
    malformed = point.copy(deep=True)
    malformed["theta"][0, 0] = np.nan
    with pytest.raises(ValueError):
        information(model, parameters=malformed, progressbar=False)
    assert_allclose(model.idata["information"].test_information, actual.test_information)

    invalid_covariance = point.copy(deep=True)
    invalid_covariance["covariance"].values[:] = [[1.0, 1.2], [1.2, 1.0]]
    invalid_model = mixed_model()
    with pytest.raises(ValueError, match="covariance: covariance must be positive definite"):
        supplied_posterior(invalid_model, invalid_covariance, inference_method="advi")
    assert "posterior" not in invalid_model.idata
    with pytest.raises(ValueError, match="covariance: covariance must be positive definite"):
        information(model, parameters=invalid_covariance, progressbar=False)
    assert_allclose(model.idata["information"].test_information, actual.test_information)


def test_loaded_double_precision_point_uses_float32_graph_inputs(tmp_path, request, monkeypatch):
    import jax
    import pymc as pm
    import pytensor

    import pyfcirt as fc
    from tests.helpers import mixed_model, supplied_posterior, workflow_point

    enabled = jax.config.jax_enable_x64
    request.addfinalizer(lambda: jax.config.update("jax_enable_x64", enabled))
    reference = mixed_model()
    supplied_posterior(reference, workflow_point(reference), inference_method="mcmc")
    path = tmp_path / "double-precision.nc"
    reference.idata.to_netcdf(path, engine="h5netcdf")
    with pytensor.config.change_flags(floatX="float32"):
        model = fc.Model(reference.prepared)
        with xr.open_datatree(path, engine="h5netcdf") as stored:
            model.attach_samples(stored.load())
        before = model.idata.copy(deep=True)
        point = fc.parameter_point(model)
        graph = model.build()
        assert graph["loading"].dtype == "float32"
        compile_calls = []
        compile_point = pm.compile

        def count_compilation(*args, **kwargs):
            compile_calls.append(1)
            return compile_point(*args, **kwargs)

        monkeypatch.setattr(pm, "compile", count_compilation)
        actual = fc.information(model, store=False, progressbar=False)
        converted = xr.Dataset(
            {name: value.astype(graph[name].dtype) for name, value in point.data_vars.items()}
        )
        expected = fc.information(model, parameters=converted, store=False, progressbar=False)
        assert_allclose(actual.test_information, expected.test_information, rtol=2e-5, atol=2e-6)
        changed = converted.assign(loading=converted.loading * 1.2)
        updated = fc.information(model, parameters=changed, store=False, progressbar=False)
        assert len(compile_calls) == 1
        assert not np.allclose(updated.test_information, actual.test_information)
        rebuilt = fc.Model(reference.prepared)
        independent = fc.information(rebuilt, parameters=changed, store=False, progressbar=False)
        assert len(compile_calls) == 2
        xr.testing.assert_allclose(updated, independent)
        xr.testing.assert_identical(model.idata, before)
        assert point["loading"].dtype == np.dtype("float64")
        invalid = point.copy(deep=True)
        invalid["loading"].values[0] = 1e100
        with pytest.raises(ValueError, match="loading.*graph dtype float32"):
            model._bindings.evaluate_at((graph["loading"],), invalid)
        xr.testing.assert_identical(model.idata, before)


def test_gtum_information_rejects_utility_cusps_and_projects_repeated_traits():
    from types import SimpleNamespace

    import pytest

    from pyfcirt.analysis.information import _response_fisher
    from pyfcirt.models.thurstonian.gtum import GTUMDistribution

    response = SimpleNamespace(
        kernel=GTUMDistribution.kernel,
        smooth=GTUMDistribution.smooth,
        batch=SimpleNamespace(
            name="pair", trait=jnp.array([[0, 0]]), likelihood_kind="categorical"
        ),
    )
    aligned = (
        jnp.array([[1.2, 0.7]]),
        jnp.array([[-0.4, 0.6]]),
        jnp.array([[-0.5, 0.8]]),
        jnp.zeros(1),
        jnp.ones(1),
    )
    with pytest.raises(ValueError, match="cusp"):
        _response_fisher(response, jnp.array([[-0.4, 0.2]]), aligned, n_traits=1)
    matrices, _, _ = _response_fisher(response, jnp.array([[0.2, 0.2]]), aligned, n_traits=1)

    def scalar(theta):
        utility = -np.array([1.2, 0.7]) * np.abs(theta - np.array([-0.4, 0.6]))
        eta = utility[0] - utility[1]
        cdf = 1 / (1 + np.exp(-(np.array([-0.5, 0.8]) - eta)))
        return np.diff(np.r_[0.0, cdf, 1.0])

    probability = scalar(0.2)
    derivative = (scalar(0.2 + 1e-5) - scalar(0.2 - 1e-5)) / 2e-5
    assert_allclose(matrices[0, 0, 0], np.sum(derivative**2 / probability), atol=1e-10)


def test_beta_information_matches_independent_trigamma_chain_rule():
    from scipy.special import polygamma

    from pyfcirt.analysis.information import beta_fisher

    theta = jnp.array([0.3, -0.4])
    jacobian = np.array([[2.0, 0.0], [0.0, -1.0]])
    shapes, derivative, actual = beta_fisher(lambda x: (2 * x[0] + 2, 3 - x[1]), theta)
    expected_shape = np.diag(polygamma(1, shapes)) - polygamma(1, np.sum(shapes))
    assert_allclose(derivative, jacobian, atol=0)
    assert_allclose(actual, jacobian.T @ expected_shape @ jacobian, atol=1e-13)


def test_component_information_checks_observation_axes_and_probability_laws():
    from types import SimpleNamespace

    from pyfcirt.analysis.information import _response_fisher

    response = SimpleNamespace(
        batch=SimpleNamespace(
            name="test", trait=jnp.array([[0], [0]]), likelihood_kind="categorical"
        ),
        smooth=None,
        kernel=lambda x, a: jnp.log(jnp.array([0.25, 0.25])) + 0 * x[0] + 0 * a,
    )
    theta = jnp.array([[0.2], [0.4]])
    with pytest.raises(ValueError, match="observation axis"):
        _response_fisher(response, theta, (jnp.ones(3),), n_traits=1)
    with pytest.raises(ArithmeticError, match="sum to one"):
        _response_fisher(response, theta, (1.0,), n_traits=1)
    response.batch.likelihood_kind = "beta"
    response.kernel = lambda x: jnp.array([-1.0, 2.0]) + 0 * x[0]
    with pytest.raises(FloatingPointError, match="shapes"):
        _response_fisher(response, theta, (), n_traits=1)
    response.batch.likelihood_kind = "categorical"
    response.kernel = lambda x: jnp.array([0.0, -jnp.inf]) + jnp.sqrt(x[0]) * 0
    with pytest.raises(FloatingPointError, match="derivatives"):
        _response_fisher(response, jnp.zeros((2, 1)), (), n_traits=1)


def test_information_point_rejects_conflicting_source_and_malformed_theta(supplied_model):
    import pyfcirt as fc
    from pyfcirt.analysis.information import _point

    point = fc.parameter_point(supplied_model)
    for parameters, draw, theta, exception, message in [
        (point, ("supplied", 3), None, ValueError, "mutually exclusive"),
        ({}, None, None, TypeError, "Dataset"),
        (point, None, np.zeros((2, 2)), TypeError, "labelled"),
        (point.drop_vars("theta"), None, None, ValueError, "Provide theta"),
        (point, None, point.theta.astype(bool), ValueError, "finite real"),
        (
            point,
            None,
            xr.concat(
                [point.theta, point.theta.isel(person=[0]).assign_coords(person=[99])], dim="person"
            ),
            ValueError,
            "coordinates",
        ),
        (
            point,
            None,
            point.theta.assign_coords(person=[1.0, 2.0]),
            ValueError,
            "string or integer",
        ),
        (
            point.drop_dims("person"),
            None,
            point.theta.assign_coords(person=[1, 1]),
            ValueError,
            "coordinates",
        ),
        (
            point.drop_dims("person"),
            None,
            point.theta.assign_coords(person=["x", "y"]),
            ValueError,
            "labels",
        ),
    ]:
        with pytest.raises(exception, match=message):
            _point(supplied_model, parameters, draw, theta)
    reordered = point.theta.sel(person=np.array([2, 1], dtype=object)).transpose()
    actual = _point(supplied_model, point, None, reordered)
    xr.testing.assert_identical(actual.theta, point.theta)


def test_batch_fisher_uses_runtime_parameters_and_trait_indices():
    from jax import nn

    def kernel(theta, loading):
        return nn.log_softmax(jnp.stack((jnp.zeros_like(theta[0]), loading * theta[0])))

    theta = jnp.zeros((2, 1))
    for loading, trait_index in (([2.0, 4.0], [[0], [1]]), ([6.0, 8.0], [[1], [0]])):
        probabilities, scores, matrices = _batch_fisher(
            theta,
            (jnp.asarray(loading),),
            jnp.asarray(trait_index),
            kernel=kernel,
            fisher=categorical_fisher,
            n_traits=2,
        )
        rows = np.arange(2)
        traits = np.ravel(trait_index)
        expected_scores = np.zeros((2, 2, 2))
        expected_scores[rows, :, traits] = np.asarray(loading)[:, None] * [-0.5, 0.5]
        expected_matrices = np.zeros((2, 2, 2))
        expected_matrices[rows, traits, traits] = np.square(loading) / 4
        assert_allclose(probabilities, 0.5)
        assert_allclose(scores, expected_scores)
        assert_allclose(matrices, expected_matrices)


def test_native_gaussian_rank_information_retains_refinement_diagnostics():
    import pandas as pd

    import pyfcirt as fc

    source = fc.FCInput(
        fc.FCDesignMapping(traits={"A": ["a"], "B": ["b"]}, blocks={"pair": ["a", "b"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p"], "block_id": ["pair"], "response": [[1, 2]]}),
            format="block_long",
        ),
        model=fc.TIRTRank(loading_anchors={"a": 1.0, "b": 0.8}),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    prior = fc.prior_predictive(model, draws=1, random_seed=16, store=False)
    point = prior["prior"].to_dataset().isel(chain=0, draw=0, drop=True)
    result = fc.information(model, parameters=point, store=False, progressbar=False)
    assert result.attrs["numerical_checks"]
    for diagnostic in json.loads(result.attrs["numerical_checks"]).values():
        assert diagnostic["probability_refinement_max"] == 0
        assert "refinement is not an error bound" in diagnostic["integration_method"]
    assert "block_information" in result and "item_information" not in result


def test_rank_information_short_tail_matches_full_batch(monkeypatch):
    from importlib import import_module

    import pandas as pd

    import pyfcirt as fc

    rows = 67
    source = fc.FCInput(
        fc.FCDesignMapping(traits={"A": ["a", "c"], "B": ["b"]}, blocks={"triad": ["a", "b", "c"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": np.arange(rows), "block_id": "triad", "response": [[1, 2, 3]] * rows}
            ),
            format="block_long",
        ),
        model=fc.TIRTRank(loading_anchors={"a": 1.0, "b": 0.8}),
        response_format="rank",
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    prior = fc.prior_predictive(model, draws=1, random_seed=16, store=False)
    point = prior["prior"].to_dataset().isel(chain=0, draw=0, drop=True)
    point = point.assign(
        theta=point.theta.copy(
            data=np.linspace(-0.5, 0.5, point.theta.size).reshape(point.theta.shape)
        ),
        statement_intercept=xr.zeros_like(point.statement_intercept),
    )
    module = import_module("pyfcirt.analysis.information")
    original = module._batch_fisher
    sizes = []

    def record(ability, *args, **kwargs):
        sizes.append(len(ability))
        return original(ability, *args, **kwargs)

    monkeypatch.setattr(module, "_batch_fisher", record)
    options = dict(parameters=point, include_intermediates=True, store=False, progressbar=False)
    chunked = fc.information(model, **options)
    assert sizes == [64, 3, 64, 3]
    monkeypatch.setattr(module, "_FISHER_BATCH_SIZE", rows)
    full = fc.information(model, **options)
    xr.testing.assert_identical(
        xr.Dataset(coords=dict(chunked.coords)), xr.Dataset(coords=dict(full.coords))
    )
    xr.testing.assert_allclose(chunked, full, rtol=1e-10, atol=1e-12)
    for name, diagnostic in json.loads(chunked.attrs["numerical_checks"]).items():
        assert diagnostic == pytest.approx(
            json.loads(full.attrs["numerical_checks"])[name], abs=1e-12
        )


@pytest.mark.parametrize("failure", ["probability", "score", "information", "normalization"])
def test_rank_information_refinement_checks_all_three_outputs_and_normalization(failure):
    from types import SimpleNamespace

    import jax

    from pyfcirt.analysis.information import _response_fisher

    def controlled_quadrature(theta, n_points=4096):
        slope = 100.0 if failure == "information" else 1.0
        if n_points == 16384:
            slope *= 1.004 if failure == "information" else 2 if failure == "score" else 1
        offset = 0.2 if failure == "probability" and n_points == 16384 else 0
        logs = jax.nn.log_softmax(jnp.stack((slope * theta[0] + offset, jnp.array(0.0))))
        return logs + (jnp.log(0.9) if failure == "normalization" else 0)

    response = SimpleNamespace(
        smooth=None,
        kernel=controlled_quadrature,
        batch=SimpleNamespace(name="quadrature", trait=jnp.array([[0]]), likelihood_kind="rank"),
    )
    with pytest.raises(ArithmeticError, match="refinement failed"):
        _response_fisher(response, jnp.zeros((1, 1)), (), n_traits=1)
