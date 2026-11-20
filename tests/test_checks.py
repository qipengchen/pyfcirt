"""Independent stable Pearson arithmetic and observed-event aggregation."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import pyfcirt as fc
from pyfcirt.analysis.checks import _aggregate_pearson, _event_values, _pearson_terms


def test_chi_ppp_is_the_single_public_pearson_entry_point():
    assert fc.chi_ppp.__name__ == "chi_ppp"
    assert "chi_ppp" in fc.__all__
    assert not hasattr(fc, "pearson_check")


def test_pearson_log_terms_match_categorical_identity_without_floors():
    p = np.array([[[0.1, 0.2, 0.7], [0.8, 0.15, 0.05]]])
    observed, replica = _pearson_terms(_event_values(np.log(p), [2, 0], [[1, 2]]))
    assert_allclose(np.exp(observed), [[1 / 0.7 - 1, 1 / 0.8 - 1]], atol=1e-12)
    assert_allclose(np.exp(replica), [[1 / 0.2 - 1, 1 / 0.05 - 1]], atol=1e-12)
    extreme, _ = _pearson_terms(np.array([[-1000.0, -1000.0]]))
    assert_allclose(extreme, [1000.0], atol=1e-12)
    certain, _ = _pearson_terms(np.zeros((1, 2)))
    assert np.isneginf(certain).all()
    with pytest.raises(FloatingPointError, match="zero or invalid"):
        _pearson_terms(np.array([[-np.inf, 0.0]]))


def test_aggregation_includes_empty_groups_and_equal_discrepancies():
    observed = np.log(np.array([[[1.0, 3.0, 2.0], [2.0, 4.0, 1.0]]]))
    replica = np.log(np.array([[[4.0, 1.0, 2.0], [1.0, 1.0, 1.0]]]))
    lo, lr, actual, replicated, difference, indicator, count = _aggregate_pearson(
        np.stack((observed, replica)), [0, 0, 1], 3
    )
    assert_allclose(actual, [[[4, 2, 0], [6, 1, 0]]])
    assert_allclose(replicated, [[[5, 2, 0], [2, 1, 0]]])
    assert_allclose(difference, replicated - actual)
    assert_allclose(count, [2, 1, 0])
    assert np.isneginf(np.asarray(lo)[..., -1]).all()
    assert np.isneginf(np.asarray(lr)[..., -1]).all()
    assert np.array_equal(indicator, replicated >= actual)


def test_ppp_uses_same_supplied_states_and_actual_events(supplied_model):
    predictions = fc.posterior_predictive(
        supplied_model, random_seed=8, store=False, progressbar=False
    )
    check = fc.chi_ppp(
        supplied_model,
        predictions=predictions,
        groupby="measurement",
        store=False,
        progressbar=False,
    )
    assert check.n_obs.values.tolist() == [2, 1, 1, 2, 1]
    assert check.attrs["check"] == "chi_ppp"
    assert check.attrs["statistic"] == "Pearson categorical discrepancy"
    assert_allclose(check.PPP, (check.log_D_rep >= check.log_D_obs).mean(("chain", "draw")))
    assert_allclose(check.D_diff, check.D_rep - check.D_obs, atol=1e-12)
    assert np.isnan(check.PPP_mcse).all()
    assert "posterior_predictive_checks" not in supplied_model.idata
    assert "posterior" in supplied_model.idata


def test_check_rejects_non_mcmc_and_mismatched_prediction_states(supplied_model):
    predictions = fc.posterior_predictive(
        supplied_model, random_seed=4, store=False, progressbar=False
    )
    predictions["posterior_predictive"] = (
        predictions["posterior_predictive"].to_dataset().assign_coords(draw=[0, 1])
    )
    with pytest.raises(ValueError, match="sample labels"):
        fc.chi_ppp(supplied_model, predictions=predictions, progressbar=False)
    supplied_model.idata.attrs["inference_method"] = "advi"
    with pytest.raises(NotImplementedError, match="MCMC"):
        fc.chi_ppp(supplied_model, predictions=predictions, progressbar=False)


@pytest.mark.parametrize(
    "groupby,selection,axis,labels,counts",
    [
        ("block", {"blocks": ["quad", "pair"]}, "block", ["pair", "quad"], [2, 1]),
        ("item", {"items": ["ordinal"]}, "item", ["ordinal"], [1]),
        ("measurement", {"blocks": ["pair"], "items": ["pair"]}, "unit", [0, 3], [2, 2]),
        ("overall", {"blocks": ["pair"]}, None, [], 2),
    ],
)
def test_selected_checks_reuse_saved_probabilities_and_store_frozen_subsets(
    supplied_model, monkeypatch, groupby, selection, axis, labels, counts
):
    predictions = fc.posterior_predictive(
        supplied_model, draws=1, random_seed=14, store=False, progressbar=False
    )

    def repeated_evaluation(*args, **kwargs):
        pytest.fail("A selected check recomputed already supplied probabilities")

    monkeypatch.setattr("pyfcirt.analysis.checks._native_forward", repeated_evaluation)
    monkeypatch.setattr("pyfcirt.analysis.checks._take_states", repeated_evaluation)
    model = fc.Model(supplied_model.prepared)
    model.attach_samples(supplied_model.idata)
    assert model._built is None
    result = fc.chi_ppp(
        model,
        predictions=predictions,
        groupby=groupby,
        save_discrepancy=True,
        progressbar=False,
        **selection,
    )
    assert result.n_obs.values.tolist() == counts
    if axis is not None:
        assert result[axis].values.tolist() == labels
    assert model._built is None
    discrepancy = model.idata["discrepancy"]
    design = model.prepared.constant_data
    positions = np.isin(design.obs_unit.values, result.attrs["selected_units"])
    assert discrepancy.obs.values.tolist() == design.obs.values[positions].tolist()
    assert discrepancy.sizes["obs"] == int(result.n_obs.sum())
    xr.testing.assert_identical(model.idata["posterior_predictive_checks"].to_dataset(), result)


def test_fresh_selected_check_evaluates_only_its_native_outputs(supplied_model, monkeypatch):
    import pymc as pm

    forward = pm.sample_posterior_predictive
    calls = []

    def record(*args, **kwargs):
        evaluated = forward(*args, **kwargs)
        calls.append((tuple(kwargs["var_names"]), evaluated["posterior_predictive"]))
        return evaluated

    monkeypatch.setattr(pm, "sample_posterior_predictive", record)
    result = supplied_model.chi_ppp(
        blocks=["pair"], draws=2, random_seed=14, store=False, progressbar=False
    )
    response = supplied_model._bindings.responses[0]
    ((names, evaluated),) = calls
    assert names == (response.batch.response_name, response.batch.log_probability_name)
    log_p = evaluated[names[1]].values
    replicated = evaluated[names[0]].values
    observed = np.broadcast_to(response.batch.y, replicated.shape)
    for codes, discrepancy in ((observed, result.D_obs), (replicated, result.D_rep)):
        selected = np.take_along_axis(log_p, codes[..., None], axis=-1)[..., 0]
        assert_allclose(discrepancy, np.expm1(-selected).sum(axis=-1))
    assert result.n_obs.item() == 2
    assert result.sizes["draw"] == 2
    assert result.attrs["posterior_run_id"] == supplied_model.idata.attrs["posterior_run_id"]
    assert 0 <= result.PPP.item() <= 1


@pytest.mark.parametrize("groupby", ["block", "item"])
def test_mixed_channel_checks_require_an_explicit_matching_selection(supplied_model, groupby):
    with pytest.raises(ValueError, match="explicit"):
        fc.chi_ppp(supplied_model, groupby=groupby, progressbar=False)


def test_log_aggregation_retains_extreme_comparison_and_equal_ties():
    values = _aggregate_pearson(
        np.array([[[[1000.0], [1000.0]]], [[[999.0], [1000.0]]]]),
        [0],
        1,
    )
    log_observed, log_replica, observed, replica, difference, indicator, _ = values
    assert_allclose(log_observed, [[[1000.0], [1000.0]]])
    assert_allclose(log_replica, [[[999.0], [1000.0]]])
    assert np.isposinf(observed).all() and np.isposinf(replica).all()
    assert np.isneginf(np.asarray(difference)[0, 0, 0])
    assert np.asarray(difference)[0, 1, 0] == 0
    assert np.array_equal(indicator, [[[False], [True]]])


def test_residuals_match_observed_minus_expected_in_each_native_support(supplied_model):
    predictions = fc.posterior_predictive(
        supplied_model, store=False, random_seed=12, progressbar=False
    )
    residuals = fc.residuals(supplied_model, progressbar=False)
    for response in supplied_model._bindings.responses:
        batch = response.batch
        probability = predictions["response_probabilities"][f"{batch.name}_p"].values
        labels = batch.events.labels
        numeric = all(isinstance(label, (int, float, np.integer, np.floating)) for label in labels)
        if numeric:
            scores = np.asarray(labels)
            expected = scores[batch.y] - (probability * scores).sum(axis=-1)
        else:
            expected = np.eye(len(labels))[batch.y] - probability
        dims = ("chain", "draw", batch.obs_dim)
        if not numeric:
            dims += (batch.category_dim,)
        assert_allclose(residuals[f"{batch.name}_residual"].transpose(*dims), expected, atol=1e-12)
    assert "residuals" not in supplied_model.idata


@pytest.mark.parametrize("device_arrays", [False, True])
def test_selected_predictions_grouped_by_person_and_saved_event_discrepancies(
    supplied_model, device_arrays
):
    if device_arrays:
        import jax.numpy as jnp

        supplied_model.idata["posterior"] = (
            supplied_model.idata["posterior"]
            .to_dataset()
            .map(lambda value: value.copy(data=jnp.asarray(value.values)))
        )
    predictions = fc.posterior_predictive(
        supplied_model, draws=1, random_seed=10, store=False, progressbar=False
    )
    result = fc.chi_ppp(
        supplied_model,
        predictions=predictions,
        groupby="person",
        save_discrepancy=True,
        progressbar=False,
    )
    assert result.person.values.tolist() == [1, 2]
    assert result.n_obs.values.tolist() == [5, 2]
    assert result.source_chain.item() == "supplied"
    assert result.source_draw.item() in (3, 9)
    discrepancies = supplied_model.idata["discrepancy"].to_dataset()
    assert discrepancies.sizes["obs"] == 7
    likelihood = fc.log_likelihood(supplied_model, store=False, progressbar=False)
    xr.testing.assert_identical(discrepancies.obs, likelihood.obs)
    actual_p = np.exp(likelihood.sel(draw=result.source_draw.item()).values)
    assert_allclose(
        discrepancies.log_D_obs.values[0, 0], np.log1p(-actual_p[0]) - np.log(actual_p[0])
    )
    assert "selected states" in result.attrs["PPP_mcse_method"]
    all_states = fc.posterior_predictive(supplied_model, random_seed=10, progressbar=False)
    overall = fc.chi_ppp(supplied_model, progressbar=False)
    assert overall.PPP.ndim == 0 and overall.n_obs.item() == 7
    assert (
        overall.attrs["prediction_run_id"]
        == all_states["posterior_predictive"].attrs["prediction_run_id"]
    )
    assert "discrepancy" not in supplied_model.idata


def test_prediction_operation_and_source_label_guards(supplied_model):
    from pyfcirt.analysis.checks import _prediction_states

    supplied_model.idata.attrs["posterior_run_id"] = "current"
    predictions = fc.posterior_predictive(
        supplied_model, draws=2, store=False, random_seed=5, progressbar=False
    )
    for case, message in [
        ("run", "different posterior"),
        ("missing_replica_run", "must declare posterior_run_id"),
        ("missing_replica_prediction", "must declare prediction_run_id"),
        ("missing_probability_run", "must declare posterior_run_id"),
        ("missing_probability_prediction", "must declare prediction_run_id"),
        ("unpaired", "paired"),
        ("axis", "draw dimension"),
        ("duplicate", "repeat"),
        ("unknown_source", "source labels"),
        ("probability_run", "different prediction_run_id"),
        ("probability_labels", "different sample labels"),
    ]:
        changed = predictions.copy(deep=True)
        replicas = changed["posterior_predictive"].to_dataset()
        if case == "run":
            replicas.attrs["posterior_run_id"] = "other"
        elif case == "missing_replica_run":
            del replicas.attrs["posterior_run_id"]
        elif case == "missing_replica_prediction":
            del replicas.attrs["prediction_run_id"]
        elif case == "missing_probability_run":
            del changed["response_probabilities"].attrs["posterior_run_id"]
        elif case == "missing_probability_prediction":
            del changed["response_probabilities"].attrs["prediction_run_id"]
        elif case == "unpaired":
            replicas = replicas.drop_vars("source_chain")
        elif case == "axis":
            replicas = replicas.assign_coords(source_chain="supplied")
        elif case == "duplicate":
            replicas = replicas.assign_coords(source_draw=("draw", [3, 3]))
        elif case == "unknown_source":
            replicas = replicas.assign_coords(source_draw=("draw", [3, 999]))
        elif case == "probability_run":
            changed["response_probabilities"].attrs["prediction_run_id"] = "other"
        else:
            changed["response_probabilities"] = (
                changed["response_probabilities"]
                .to_dataset()
                .assign_coords(source_draw=("draw", [8, 9]))
            )
        changed["posterior_predictive"] = replicas
        with pytest.raises(ValueError, match=message):
            _prediction_states(supplied_model, changed)
    del supplied_model.idata.attrs["posterior_run_id"]
    with pytest.raises(ValueError, match="current posterior must declare posterior_run_id"):
        _prediction_states(supplied_model, predictions)


def test_pearson_rejects_invalid_public_controls_and_replica_contracts(supplied_model):
    predictions = fc.posterior_predictive(
        supplied_model, random_seed=9, store=False, progressbar=False
    )
    for options, message, exception in [
        ({"groupby": "pair"}, "prepared pairs", ValueError),
        ({"groupby": "invalid"}, "groupby", ValueError),
        ({"store": 1}, "boolean", TypeError),
        ({"save_discrepancy": 1}, "boolean", TypeError),
        ({"predictions": predictions, "draws": 1}, "again", ValueError),
        ({"predictions": xr.Dataset()}, "posterior_predictive", TypeError),
    ]:
        with pytest.raises(exception, match=message):
            fc.chi_ppp(supplied_model, progressbar=False, **options)
    batch = supplied_model._bindings.responses[0].batch
    for problem, exception, message in [
        ("replica_axis", ValueError, "posterior_predictive: output coordinate"),
        ("replica_code", ValueError, "replicated category"),
        ("probability_axis", ValueError, "response_probabilities: output coordinate"),
        ("probability_nan", FloatingPointError, "finite"),
    ]:
        changed = predictions.copy(deep=True)
        if problem.startswith("replica"):
            values = changed["posterior_predictive"].to_dataset()
            if problem == "replica_axis":
                values = values.assign_coords({batch.obs_dim: np.arange(len(batch.y)) + 100})
            else:
                values[batch.response_name] = values[batch.response_name].astype(float)
            changed["posterior_predictive"] = values
        else:
            values = changed["response_probabilities"].to_dataset()
            if problem == "probability_axis":
                values = values.assign_coords({batch.obs_dim: np.arange(len(batch.y)) + 100})
            else:
                values[f"{batch.name}_p"].values.flat[0] = np.nan
            changed["response_probabilities"] = values
        with pytest.raises(exception, match=message):
            fc.chi_ppp(supplied_model, predictions=changed, progressbar=False)


@pytest.mark.parametrize("problem", ["reversed", "wrong", "missing"])
def test_cached_probability_category_coordinates_reject_before_gather(
    supplied_model, monkeypatch, problem
):
    predictions = fc.posterior_predictive(
        supplied_model, random_seed=9, store=False, progressbar=False
    )
    batch = supplied_model._bindings.responses[0].batch
    probabilities = predictions["response_probabilities"].to_dataset()
    count = probabilities.sizes[batch.category_dim]
    if problem == "reversed":
        probabilities = probabilities.isel({batch.category_dim: slice(None, None, -1)})
    elif problem == "wrong":
        probabilities = probabilities.assign_coords({batch.category_dim: np.arange(count) + 100})
    else:
        probabilities = probabilities.drop_vars(batch.category_dim)
    assert probabilities.sizes[batch.category_dim] == count
    predictions["response_probabilities"] = probabilities

    def unexpected_gather(*args, **kwargs):
        pytest.fail("Category coordinates must be validated before positional probability access")

    monkeypatch.setattr("pyfcirt.analysis.checks.jnp.take_along_axis", unexpected_gather)
    with pytest.raises(ValueError, match=f"output coordinate '{batch.category_dim}'"):
        fc.chi_ppp(supplied_model, predictions=predictions, store=False, progressbar=False)


@pytest.mark.parametrize("draws", [None, 1])
@pytest.mark.parametrize("include_probabilities", [False, True])
def test_prediction_provenance_and_states_survive_native_io(
    supplied_model, tmp_path, draws, include_probabilities
):
    predictions = fc.posterior_predictive(
        supplied_model,
        draws=draws,
        include_probabilities=include_probabilities,
        random_seed=8,
        store=False,
        progressbar=False,
    )
    expected = fc.chi_ppp(supplied_model, predictions=predictions, store=False, progressbar=False)
    path = tmp_path / "predictions.nc"
    predictions.to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as restored:
        result = fc.chi_ppp(supplied_model, predictions=restored, store=False, progressbar=False)
        xr.testing.assert_identical(result, expected)
        assert result.attrs["posterior_run_id"] == supplied_model.idata.attrs["posterior_run_id"]
        if draws is not None:
            for name in ("source_chain", "source_draw"):
                xr.testing.assert_identical(result[name], expected[name])


@pytest.mark.parametrize("draws", [None, 1])
def test_missing_or_underflowed_saved_probabilities_reuse_native_log_outputs(supplied_model, draws):
    predictions = fc.posterior_predictive(
        supplied_model,
        draws=draws,
        store=False,
        include_probabilities=False,
        random_seed=8,
        progressbar=False,
    )
    without = fc.chi_ppp(supplied_model, predictions=predictions, store=False, progressbar=False)
    saved = fc.posterior_predictive(
        supplied_model, draws=draws, store=False, random_seed=8, progressbar=False
    )
    cached = fc.chi_ppp(supplied_model, predictions=saved, store=False, progressbar=False)
    xr.testing.assert_allclose(without, cached)
    first = supplied_model._bindings.responses[0].batch
    saved["response_probabilities"][f"{first.name}_p"].values[...] = 0
    underflow = fc.chi_ppp(supplied_model, predictions=saved, store=False, progressbar=False)
    xr.testing.assert_allclose(without, underflow)


def test_tirt_pair_grouping_uses_statement_coordinates():
    from pyfcirt.analysis.checks import _grouping

    constant = xr.Dataset(
        {
            "obs_pair": ("obs", [0, 1, 0]),
            "pair_unit": ("pair", [7, 8]),
            "statement_label": ("statement", ["a", "b", "c"]),
            "pair_statement_1": ("pair", [0, 1]),
            "pair_statement_2": ("pair", [1, 2]),
        },
        coords={"pair": ["ab", "bc"]},
    )
    groups, count, axis, coords = _grouping(constant, "pair", np.arange(3))
    assert count == 2 and axis == "pair" and groups.tolist() == [0, 1, 0]
    assert coords["statement_1"][1].tolist() == ["a", "b"]
    assert coords["statement_2"][1].tolist() == ["b", "c"]


def test_beta_response_prediction_residual_and_information_share_native_shape_law():
    from dataclasses import replace

    import pandas as pd

    from tests.data.test_schema import graded_input
    from tests.helpers import supplied_posterior

    source = graded_input(0.4, beta=True)
    design = pd.DataFrame(
        [("graded", 1, "a", "A", 1, 1), ("graded", 2, "b", "B", -1, 1)],
        columns=["block_id", "position", "statement_id", "trait", "location_side", "keying"],
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(replace(source, design=design)))
    prior = fc.prior_predictive(model, draws=1, random_seed=3, store=False)
    point = prior["prior"].to_dataset().isel(chain=0, draw=0, drop=True)
    supplied_posterior(model, point, inference_method="mcmc")
    (response,) = model._bindings.responses
    alpha_name, beta_name = response.batch.distribution_names
    forward = model._bindings.evaluation(var_names=[alpha_name])
    assert alpha_name in forward.model.named_vars and beta_name not in forward.model.named_vars
    predictions = fc.posterior_predictive(model, random_seed=4, progressbar=False)
    assert alpha_name in forward.model.named_vars and beta_name in forward.model.named_vars
    assert "response_parameters" in predictions and "response_probabilities" not in predictions
    batch = model._bindings.responses[0].batch
    shapes = predictions["response_parameters"]
    observed = xr.DataArray(
        batch.y, dims=batch.obs_dim, coords={batch.obs_dim: shapes[batch.obs_dim]}
    )
    expected = observed - shapes[f"{batch.name}_alpha"] / (
        shapes[f"{batch.name}_alpha"] + shapes[f"{batch.name}_beta"]
    )
    actual = fc.residuals(model, progressbar=False)[f"{batch.name}_residual"]
    xr.testing.assert_allclose(actual.transpose(*expected.dims), expected)
    with pytest.raises(NotImplementedError, match="Beta"):
        fc.chi_ppp(model, progressbar=False)
    view = fc.predictive_view(predictions, block="graded")
    assert view.attrs["response_kind"] == "beta" and "data_type" not in view.attrs
    assert fc.data_summary(model, details=True).n_categories.isna().all()
    information = fc.information(
        model, parameters=point, include_intermediates=True, store=False, progressbar=False
    )
    assert information[f"{batch.name}_shape_jacobian"].sizes[f"{batch.name}_beta_parameter"] == 2
    assert np.isfinite(information.test_information).all()


def test_ordered_predictive_draws_get_native_indicator_mcse(supplied_model):
    point = supplied_model.idata["posterior"].to_dataset().isel(chain=0, draw=0, drop=True)
    supplied_model.idata["posterior"] = point.expand_dims(chain=["external"], draw=range(8))
    result = fc.chi_ppp(
        supplied_model, groupby="person", store=False, random_seed=21, progressbar=False
    )
    assert "ArviZ MCSE" in result.attrs["PPP_mcse_method"]
    assert ((result.PPP_mcse >= 0) | result.PPP_mcse.isnull()).all()


@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_response_checks_reject_generation_schedules_and_empty_actual_forms(monkeypatch):
    from dataclasses import replace

    from pyfcirt.analysis.output import pool_log_likelihood
    from tests.helpers import mixed_inputs, supplied_posterior, workflow_point

    sources = mixed_inputs()
    schedules = tuple(
        replace(
            source,
            responses=fc.ResponseTable(
                source.responses.frame.drop(columns="response"),
                format="block_schedule" if index == 0 else "item_schedule",
            ),
        )
        for index, source in enumerate(sources)
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(schedules, purpose="generate"))
    with pytest.raises(ValueError, match="actual observed responses"):
        fc.chi_ppp(model, progressbar=False)
    with pytest.raises(ValueError, match="generation schedule"):
        model.fit()
    from pyfcirt.inference.gtum import sample_gtum

    with pytest.raises(ValueError, match="generation schedule"):
        sample_gtum(model)
    supplied_posterior(model, workflow_point(model), inference_method="mcmc")
    with pytest.raises(ValueError, match="actual observed"):
        fc.log_likelihood(model, progressbar=False)

    def no_evaluation(*args, **kwargs):
        pytest.fail("Residuals evaluated a generation schedule")

    monkeypatch.setattr("pyfcirt.analysis.checks._native_forward", no_evaluation)
    with pytest.raises(ValueError, match="actual observed"):
        fc.residuals(model, progressbar=False)
    missing = tuple(
        replace(
            source,
            responses=replace(source.responses, frame=source.responses.frame.assign(response=None)),
        )
        for source in sources
    )
    empty = fc.Model(fc.PreparedInputs.from_inputs(missing))
    with pytest.raises(ValueError, match="actual observed events"):
        fc.chi_ppp(empty, progressbar=False)
    assert pool_log_likelihood(empty.prepared, xr.Dataset()) is None
    assert fc.data_summary(empty, by="item", details=True).loc["pair", "n_categories"] == 2


def test_public_pearson_keeps_tail_log_discrepancy_when_raw_values_overflow():
    from tests.test_reliability import _fixed_information_reliability_case

    model = _fixed_information_reliability_case()
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    posterior["theta"].values[..., 1, 0] = -1000
    model.idata["posterior"] = posterior
    with pytest.warns(RuntimeWarning, match="Raw Pearson discrepancy"):
        check = fc.chi_ppp(model, store=False, random_seed=4, progressbar=False)
    assert check.attrs["raw_discrepancy_overflow"]
    assert np.isposinf(check.D_obs).all()
    assert_allclose(check.log_D_obs, 1100.2, atol=1e-12)
    assert check.PPP.item() == 0
