"""Fixed-final-parameter scoring and independent native targets."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from pytensor.graph.traversal import ancestors

import pyfcirt as fc
from tests.helpers import (
    labelled_point,
    mixed_inputs,
    mixed_model,
    workflow_point,
)


def test_scoring_fixes_final_nodes_and_owns_independent_state(monkeypatch):
    inputs = mixed_inputs()
    reference = fc.Model().prepare(inputs)
    point = workflow_point(reference).drop_dims("person")
    before = reference.idata.copy(deep=True)
    calls = []

    def capture(**kwargs):
        graph = pm.modelcontext(None)
        calls.append((graph, kwargs))
        assert {rv.name for rv in graph.free_RVs} == {"theta"}
        return xr.DataTree.from_dict(
            {
                "posterior": xr.Dataset(
                    {"theta": (("chain", "draw", "person", "trait"), np.zeros((1, 2, 2, 2)))},
                    coords={
                        "chain": [0],
                        "draw": [0, 1],
                        "person": np.asarray(graph.coords["person"], dtype=object),
                        "trait": np.asarray(graph.coords["trait"], dtype=object),
                    },
                )
            }
        )

    monkeypatch.setattr(pm, "sample", capture)
    responses = {}
    for channel, data in zip(("fc", "items"), inputs, strict=True):
        frame = data.responses.frame.copy(deep=True)
        frame["person_id"] = frame.person_id.map(lambda value: 7 if value == 1 else 8)
        responses[channel] = fc.ResponseTable(frame, format=data.responses.format)
    scored = reference.score(responses, parameters=point, draws=2, tune=0)
    assert scored is not reference and scored.build() is not reference.build()
    assert calls[0][1]["draws"] == 2
    assert calls[0][1]["nuts_sampler"] == "blackjax"
    assert scored.idata.attrs["sampling_nuts_sampler"] == "blackjax"
    assert scored.prepared.data["constant_data"].person.values.tolist() == [7, 8]
    xr.testing.assert_identical(
        scored.idata["fixed_parameters"].to_dataset().reset_coords(drop=True), point
    )
    xr.testing.assert_identical(reference.idata, before)
    assert [u["response_format"] for u in scored.prepared.data["constant_data"].attrs["units"]] == [
        "rank",
        "pick",
        "mole",
        None,
        None,
    ]
    assert "cohort" not in scored.idata.attrs


def test_scoring_rejects_missing_or_raw_parameters():
    reference = mixed_model()
    with pytest.raises(ValueError, match="Missing|missing"):
        fc.score_model(
            reference,
            {"fc": mixed_inputs()[0].responses, "items": mixed_inputs()[1].responses},
            parameters=xr.Dataset(),
        )
    point = (
        workflow_point(reference)
        .drop_dims("person")
        .rename({"statement_intercept": "statement_intercept_raw"})
    )
    with pytest.raises(ValueError):
        fc.score_model(
            reference,
            {"fc": mixed_inputs()[0].responses, "items": mixed_inputs()[1].responses},
            parameters=point,
        )


def test_conditional_joint_density_and_gradient_match_scalar_target():
    import pandas as pd
    from numpy.testing import assert_allclose

    data = fc.FCInput(
        fc.FCDesignMapping(traits={"A": ["a"], "B": ["b"]}, blocks={"pair": ["a", "b"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["new"], "block_id": ["pair"], "response": [[1, 2]]}),
            format="block_long",
        ),
    )
    reference = fc.Model(fc.PreparedInputs.from_inputs(data))
    parameters = labelled_point(
        reference,
        {
            "loading": [1.0, 1.2],
            "statement_intercept": [0.0, 0.3],
            "population_mean": [[0.0, 0.0]],
            "covariance": np.eye(2),
        },
    )
    scored = fc.score_model(reference, data.responses, parameters=parameters)
    graph = scored.build()
    theta = np.array([[0.4, -0.2]])
    p = 1 / (1 + np.exp(-(theta[0, 0] - 1.2 * theta[0, 1] + 0.3)))
    expected = -0.5 * np.sum(theta**2) - np.log(2 * np.pi) + np.log(p)
    assert_allclose(graph.compile_logp(mode="JAX")({"theta": theta}), expected, atol=1e-12)
    assert_allclose(
        graph.compile_dlogp(mode="JAX")({"theta": theta}),
        [-0.4 + 1 - p, 0.2 - 1.2 * (1 - p)],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "response_format,order",
    [("rank", [2, 1, 3]), ("pick", [2, None, None]), ("mole", [1, None, None, 4])],
)
def test_partial_scoring_density_and_gradient_match_enumerated_complete_orders(
    response_format, order
):
    import pandas as pd
    from numpy.testing import assert_allclose

    from tests._reference_mgppm import partial_logp

    size = len(order)
    statements = [f"s{i}" for i in range(size)]
    data = fc.FCInput(
        fc.FCDesignMapping(
            traits={f"T{i}": [statement] for i, statement in enumerate(statements)},
            blocks={"block": statements},
        ),
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["new"], "block_id": ["block"], "response": [order]}),
            format="block_long",
        ),
        response_format=response_format,
    )
    reference = fc.Model(fc.PreparedInputs.from_inputs(data))
    loading = np.linspace(1.0, 1.3, size)
    intercept = np.linspace(0.0, 0.3, size)
    theta = np.linspace(-0.4, 0.6, size)
    parameters = labelled_point(
        reference,
        {
            "loading": loading,
            "statement_intercept": intercept,
            "population_mean": np.zeros((1, size)),
            "covariance": np.eye(size),
        },
    )
    scored = fc.score_model(reference, data.responses, parameters=parameters)

    def independent_target(values):
        utility = loading * values - intercept
        return (
            -0.5 * np.sum(values**2) - size / 2 * np.log(2 * np.pi) + partial_logp(utility, order)
        )

    graph = scored.build()
    point = {"theta": theta[None, :]}
    assert_allclose(graph.compile_logp(mode="JAX")(point), independent_target(theta), atol=1e-12)
    step = 1e-5
    expected_gradient = [
        (
            independent_target(theta + step * direction)
            - independent_target(theta - step * direction)
        )
        / (2 * step)
        for direction in np.eye(size)
    ]
    assert_allclose(graph.compile_dlogp(mode="JAX")(point), expected_gradient, atol=1e-9, rtol=1e-8)


def test_score_model_constructs_native_target_without_running_an_estimator(monkeypatch, tmp_path):
    reference = mixed_model()
    point = workflow_point(reference).drop_dims("person")
    before = reference.idata.copy(deep=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("score_model must only construct a model")

    monkeypatch.setattr(pm, "sample", forbidden)
    monkeypatch.setattr(pm, "fit", forbidden)
    monkeypatch.setattr(pm, "find_MAP", forbidden)
    inputs = mixed_inputs()
    responses = {"fc": inputs[0].responses, "items": inputs[1].responses}
    scored = fc.score_model(reference, responses, parameters=point)
    graph = scored.build()
    assert isinstance(graph, pm.Model)
    assert {rv.name for rv in graph.free_RVs} == {"theta"}
    assert graph is not reference.build()
    assert "posterior" not in scored.idata
    assert "estimates" not in scored.idata
    assert "optimizer_result" not in scored.idata
    assert scored.idata.attrs["calibration_uncertainty"] == "fixed_final_point"
    assert scored.idata.attrs["point_source"] == "explicit_parameters"
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=13)))
    xr.testing.assert_identical(reference.idata, before)
    path = tmp_path / "native-scoring-target.nc"
    scored.idata.to_netcdf(path, engine="h5netcdf")
    restored = fc.Model(scored.prepared)
    xr.testing.assert_identical(
        restored.idata["fixed_parameters"].to_dataset().reset_coords(drop=True), point
    )
    assert {rv.name for rv in restored.build().free_RVs} == {"theta"}
    xr.testing.assert_identical(
        restored.prepared.data["constant_data"].person, scored.prepared.data["constant_data"].person
    )


def test_single_person_scoring_uses_custom_theta_id_group_and_frozen_regressions(
    tmp_path, monkeypatch
):
    import pandas as pd
    import pytensor

    from pyfcirt.data import prepare

    people = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "participant": [1, 2, 3, 4],
            "school": ["A", "A", "B", "B"],
            "age": [10.0, 30.0, 20.0, 40.0],
        }
    )
    answers = pd.DataFrame(
        [
            (person, item, (person + index) % 2)
            for person in range(1, 5)
            for index, item in enumerate(["a", "b", "c"])
        ],
        columns=["person_id", "item_id", "response"],
    )
    data = fc.ItemInput(
        fc.ItemDesignMapping(traits={"T": ["a", "b", "c"]}),
        fc.ResponseTable(answers, format="item_long"),
        model=fc.DichotomousIRT(),
    )
    reference = fc.Model(
        fc.PreparedInputs.from_inputs(
            data,
            person_data=people,
            population=fc.PopulationSpec(group="school", reference_group="A"),
        ),
        marginal_correlation=True,
    )
    reference.add_regressions(
        {
            ("items", "intercept"): fc.Regression(
                "length",
                pd.DataFrame({"item_id": ["a", "b", "c"], "length": [1.0, 2.0, 4.0]}),
                id="item_id",
            ),
            "theta": fc.Regression("age", people, id="participant"),
        }
    )
    coefficient = next(
        layout
        for layout in reference.prepared.parameters
        if layout.kind == "regression_coefficient"
    )
    coefficient_values = np.zeros(coefficient.shape)
    coefficient_values[coefficient.entry_data.target_kind.values == "theta"] = 0.2
    point = labelled_point(
        reference,
        {
            "population_mean": [[0.0], [1.0]],
            "covariance": [[[1.0]], [[1.0]]],
            "regression_coefficient": coefficient_values,
            "loading": [1.0, 1.1, 1.2],
            "item_intercept": [0.0, 0.1, -0.2],
            "lower_asymptote": [0.0, 0.0, 0.0],
            "upper_asymptote": [1.0, 1.0, 1.0],
        },
    )
    point = reference.prepared.annotate_coordinates(point)
    coefficient_ids = point.regression_coefficient_id.values[coefficient_values == 0.2]
    np.testing.assert_array_equal(
        point.regression_coefficient_regression_index.sel(
            regression_coefficient_id=coefficient_ids
        ),
        [1] * len(coefficient_ids),
    )
    before = reference.idata.copy(deep=True)
    original = reference.prepared.theta_regression.targets[0]
    same_people = reference.score_model(data.responses, parameters=point).prepared
    reused = same_people.theta_regression.targets[0]
    assert np.shares_memory(original.row_index, reused.row_index)
    assert np.shares_memory(original.pool_index, reused.pool_index)
    assert np.shares_memory(
        reference.prepared.population.predictors, same_people.population.predictors
    )

    new_answers = fc.ResponseTable(
        pd.DataFrame({"person_id": ["new"], "item_id": ["a"], "response": [1]}),
        format="item_long",
    )
    # The scoring group and formula must both align through the declared custom ID.
    new_people = pd.DataFrame(
        {"participant": ["unused", "new"], "school": ["unknown", "B"], "age": [np.nan, 50.0]}
    )
    finalize, align = prepare.finalize_parameters, prepare.align_rows
    finalizations, matches = [], []

    def record_match(data, keys, columns, context):
        matches.append(columns)
        return align(data, keys, columns, context)

    monkeypatch.setattr(prepare, "align_rows", record_match)

    def record_finalization(constant):
        finalizations.append(constant)
        return finalize(constant)

    monkeypatch.setattr(prepare, "finalize_parameters", record_finalization)
    scored = fc.score_model(reference, new_answers, parameters=point, person_data=new_people)
    assert not finalizations
    assert matches == [("participant",)]
    graph = scored.build()
    assert {rv.name for rv in graph.free_RVs} == {"theta"}
    assert not any(rv in graph.free_RVs for rv in [graph["item_intercept"]])
    assert scored.prepared.data["constant_data"].person.values.tolist() == ["new"]
    np.testing.assert_array_equal(scored.prepared.data["constant_data"].person_group, [1])
    np.testing.assert_allclose(scored.prepared.data["covariates/r1"].X, [[30.0]])
    node = graph["theta"].owner
    mean, _ = node.op.dist_params(node)
    mean = pytensor.function([], mean, mode="JAX")()
    np.testing.assert_allclose(mean, [[7.0]], atol=1e-12)
    correlation = pm.draw(graph["theta_correlation_marginal"], random_seed=12)
    posterior = xr.Dataset(
        {"theta_correlation_marginal": (("trait", "trait_aux"), correlation)},
        coords={"trait": ["T"], "trait_aux": ["T"]},
    ).expand_dims(chain=[0], draw=[0])
    scored.attach_samples(xr.DataTree.from_dict({"posterior": posterior}))
    metadata = scored.idata["posterior"]["theta_correlation_marginal"].attrs
    assert metadata["N"] == 1
    assert metadata["weighting"] == "empirical_person_proportions"
    assert metadata["reference_population"] == "current analysis persons"
    fixed = scored.idata["fixed_parameters"].to_dataset()
    np.testing.assert_array_equal(fixed.regression_coefficient_id, point.regression_coefficient_id)
    np.testing.assert_array_equal(
        fixed.regression_coefficient_regression_index, point.regression_coefficient_regression_index
    )
    assert "regression_sd" not in graph.named_vars
    for name, value in fixed.data_vars.items():
        xr.testing.assert_equal(
            value.reset_coords(drop=True),
            point[name].sel({dim: value[dim] for dim in value.dims}).reset_coords(drop=True),
        )
    xr.testing.assert_identical(reference.idata, before)
    path = tmp_path / "custom-id-scoring.nc"
    scored.idata.to_netcdf(path, engine="h5netcdf")
    restored = fc.Model(scored.prepared)
    rescored = fc.score_model(restored, new_answers, person_data=new_people)
    xr.testing.assert_identical(
        rescored.prepared.data["covariates/r1"].X, scored.prepared.data["covariates/r1"].X
    )
    assert {rv.name for rv in rescored.build().free_RVs} == {"theta"}


def test_scoring_parameter_selection_prioritizes_explicit_then_fixed_then_posterior():
    from tests.helpers import supplied_posterior

    reference = mixed_model()
    responses = {
        name: data.responses for name, data in zip(("fc", "items"), mixed_inputs(), strict=True)
    }
    with pytest.raises(TypeError, match="reference must be a Model"):
        fc.score_model(reference.idata, responses)
    point = workflow_point(reference).drop_dims("person")
    supplied_posterior(reference, point, inference_method="mcmc")
    scored = fc.score_model(reference, responses)
    assert scored.idata.attrs["point_source"] == "posterior_mean"
    xr.testing.assert_allclose(
        scored.idata["fixed_parameters"].to_dataset().reset_coords(drop=True), point
    )
    reference.idata["fixed_parameters"] = point
    scored = fc.score_model(reference, responses)
    assert scored.idata.attrs["point_source"] == "fixed_parameters"
    xr.testing.assert_identical(
        scored.idata["fixed_parameters"].to_dataset().reset_coords(drop=True), point
    )
    explicit = point.copy(deep=True)
    explicit["item_intercept"].values[:] = 0.8
    scored = fc.score_model(reference, responses, parameters=explicit)
    assert scored.idata.attrs["point_source"] == "explicit_parameters"
    fixed = scored.idata["fixed_parameters"].to_dataset().reset_coords(drop=True)
    xr.testing.assert_identical(fixed, explicit)
    explicit["item_intercept"].values[:] = 5.0
    np.testing.assert_array_equal(fixed["item_intercept"], 0.8)
    assert not fixed["item_intercept"].values.flags.writeable
    rescored = fc.score_model(scored, responses, parameters=point)
    assert rescored.prepared.fixed_parameters.identical(point)
    assert scored.prepared.fixed_parameters.identical(fixed)


def test_published_fixed_group_cannot_replace_runtime_calibration():
    import pymc as pm

    reference = mixed_model()
    point = workflow_point(reference).drop_dims("person")
    responses = {
        name: data.responses for name, data in zip(("fc", "items"), mixed_inputs(), strict=True)
    }
    scored = fc.score_model(reference, responses, parameters=point)
    saved = scored.idata["fixed_parameters"].to_dataset()
    scored.idata["fixed_parameters"] = saved.assign(item_intercept=saved.item_intercept + 0.6)

    posterior = saved.expand_dims(chain=[0], draw=[0, 1])
    scored.attach_samples(
        xr.DataTree.from_dict({"posterior": posterior, "fixed_parameters": saved})
    )
    xr.testing.assert_identical(
        fc.parameter_point(scored, include_person=False).item_intercept, saved.item_intercept
    )
    np.testing.assert_array_equal(pm.draw(scored.build()["item_intercept"]), saved.item_intercept)
    replayed = fc.score_model(scored, responses)
    xr.testing.assert_identical(replayed.idata["fixed_parameters"].to_dataset(), saved)


def test_setting_reference_priors_preserves_an_existing_scoring_layout():
    import pymc as pm

    reference = mixed_model()
    responses = {"fc": mixed_inputs()[0].responses}
    scored = fc.score_model(
        reference, responses, parameters=workflow_point(reference).drop_dims("person")
    )
    original_layout = scored.prepared.constant_data
    item_owner = next(
        owner
        for owner in original_layout.attrs["owners"]
        if type(owner["specification"]) is fc.DichotomousIRT
    )
    original_prior = item_owner["specification"].priors["intercept"]

    reference.set_priors({"items.intercept": pm.Normal.dist(mu=3, sigma=1)})

    assert item_owner["specification"].priors["intercept"] is original_prior
    assert np.isfinite(scored.build().compile_logp()(scored.build().initial_point()))


def test_optional_noncentered_scoring_roundtrip_uses_final_theta_without_raw_samples(tmp_path):
    import pandas as pd
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace
    from pytensor.graph.traversal import ancestors

    from tests.helpers import supplied_posterior

    reference = fc.Model(
        fc.PreparedInputs.from_inputs(mixed_inputs(), theta_parameterization="noncentered")
    )
    reference.add_regressions(
        {
            "theta": fc.Regression(
                "age", pd.DataFrame({"person_id": [1, 2], "age": [10.0, 30.0]}), id="person_id"
            ),
            ("fc", "loading_magnitude"): fc.Regression(
                "1", pd.DataFrame({"id": list("abcdefghi")}), id="id"
            ),
        }
    )
    point = workflow_point(reference)
    coefficient = next(
        layout
        for layout in reference.prepared.parameters
        if layout.kind == "regression_coefficient"
    )
    theta_coefficients = coefficient.entry_data.target_kind.values == "theta"
    point["regression_coefficient"].values[theta_coefficients] = [0.15, -0.25]
    covariance = np.array([[1.0, 0.45], [0.45, 1.0]])
    point["covariance"].values[:] = covariance
    supplied_posterior(reference, point, inference_method="mcmc")
    assert "theta_raw" not in reference.idata["posterior"]
    reference.idata.to_netcdf(tmp_path / "optional-noncentered.nc", engine="h5netcdf")
    restored = fc.Model(reference.prepared)
    with xr.open_datatree(tmp_path / "optional-noncentered.nc", engine="h5netcdf") as stored:
        if "posterior" in stored:
            restored.attach_samples(stored.load())
    assert (
        restored.prepared.data["constant_data"].attrs["population"]["theta_parameterization"]
        == "noncentered"
    )
    assert restored.build()["theta_raw"] in restored.build().free_RVs
    xr.testing.assert_allclose(fc.parameter_point(restored).theta, point.theta)
    likelihood = fc.log_likelihood(restored, store=False, progressbar=False)
    assert likelihood.sizes["obs"] == restored.prepared.data["constant_data"].sizes["obs"]
    assert np.isfinite(likelihood).all()

    responses = {}
    for channel, data in zip(("fc", "items"), mixed_inputs(), strict=True):
        frame = data.responses.frame.copy(deep=True)
        frame["person_id"] = frame.person_id.map(lambda value: 7 if value == 1 else 8)
        responses[channel] = fc.ResponseTable(frame, format=data.responses.format)
    people = pd.DataFrame({"person_id": [7, 8], "age": [50.0, 70.0]})
    scored = fc.score_model(restored, responses, person_data=people)
    graph = scored.build()
    assert (
        scored.prepared.data["constant_data"].attrs["population"]["theta_parameterization"]
        == "noncentered"
    )
    assert {rv.name for rv in graph.free_RVs} == {"theta_raw"}
    for layout in scored.prepared.parameters:
        if layout.point_parameter:
            assert not set(graph.free_RVs).intersection(ancestors([graph[layout.kind]]))
    raw = pt.dmatrix("new_person_raw")
    final = graph_replace(graph["theta"], replace={graph["theta_raw"]: raw})
    evaluate = pytensor.function([raw], final, mode="JAX")
    values = np.array([[0.2, -0.4], [-0.3, 0.5]])
    expected_mean = np.array([[30.0], [50.0]]) @ point["regression_coefficient"].values[
        theta_coefficients
    ].reshape(1, 2)
    np.testing.assert_allclose(
        evaluate(values), expected_mean + values @ np.linalg.cholesky(covariance).T, atol=1e-12
    )
    scored.idata.to_netcdf(tmp_path / "optional-noncentered-scoring.nc", engine="h5netcdf")
    rescored = fc.Model(scored.prepared)
    with xr.open_datatree(
        tmp_path / "optional-noncentered-scoring.nc", engine="h5netcdf"
    ) as stored:
        if "posterior" in stored:
            rescored.attach_samples(stored.load())
    assert (
        rescored.prepared.data["constant_data"].attrs["population"]["theta_parameterization"]
        == "noncentered"
    )
    assert {rv.name for rv in rescored.build().free_RVs} == {"theta_raw"}


def test_new_person_gtum_scoring_preserves_native_person_effect_shape(tmp_path):
    import pandas as pd

    from tests.test_simulation import gtum_block_effect_model, gtum_fixed_point

    model = gtum_block_effect_model()
    parameters = gtum_fixed_point(model)
    responses = fc.ResponseTable(
        pd.DataFrame(
            {
                "person_id": [7, 8],
                "block_id": ["triple"] * 2,
                "statement_1": ["a"] * 2,
                "statement_2": ["b"] * 2,
                "response": [0, 1],
            }
        ),
        format="comparison_long",
    )
    scored = fc.score_model(model, responses, parameters=parameters)
    assert scored.prepared.calibration_identity == model.prepared.calibration_identity
    graph = scored.build()
    assert len(graph.free_RVs) == 2 and graph["theta"] in graph.free_RVs
    effect_ancestors = set(ancestors([graph["person_block_effect"]]))
    assert len(effect_ancestors.intersection(graph.free_RVs)) == 1
    assert graph["person_block_effect"].type.shape == (2,)
    effect = next(
        layout for layout in scored.prepared.parameters if layout.kind == "person_block_effect"
    )
    assert effect.entry_data.component.values.tolist() == ["observation", "observation"]
    reference = next(
        layout for layout in model.prepared.parameters if layout.kind == "person_block_effect"
    )
    assert np.all(
        effect.entry_data.summary_default.values == reference.entry_data.summary_default.values[0]
    )
    assert fc.coef(scored, var_names="person_block_effect").empty
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=81)))
    scored.idata.to_netcdf(tmp_path / "gtum-person-effect.nc", engine="h5netcdf")
    restored = fc.Model(scored.prepared)
    with xr.open_datatree(tmp_path / "gtum-person-effect.nc", engine="h5netcdf") as stored:
        if "posterior" in stored:
            restored.attach_samples(stored.load())
    assert restored.prepared.data["constant_data"].person.values.tolist() == [7, 8]
    assert restored.build()["person_block_effect"].type.shape == (2,)


def test_new_person_gaussian_rank_scoring_owns_fresh_augmentation(monkeypatch):
    import pandas as pd
    from pytensor.graph.traversal import ancestors

    from pyfcirt.data import prepare
    from tests.data.test_schema import pair_input

    source = pair_input(model=fc.TIRTRank(loading_anchors={"a": 1.0, "b": 0.8}))
    reference = fc.Model(fc.PreparedInputs.from_inputs(source))
    parameters = labelled_point(
        reference,
        {
            "population_mean": [[0.0, 0.0]],
            "covariance": [[1.0, 0.35], [0.35, 1.0]],
            "loading": [1.0, 0.8],
            "statement_intercept": [0.0, 0.2],
            "residual_variance": [0.5, 0.5],
        },
    )
    calibration = reference.build()
    original_nodes = dict(calibration.named_vars)
    original_observed = tuple(calibration.observed_RVs)
    before = reference.idata.copy(deep=True)
    constant = reference.prepared.data["constant_data"].copy(deep=True)
    responses = fc.ResponseTable(
        pd.DataFrame(
            {
                "person_id": [7, 8, 9],
                "block_id": ["pair"] * 3,
                "response": [[1, 2], [2, 1], [1, 2]],
            }
        ),
        format="block_long",
    )

    def unexpected_refinalization(constant):
        pytest.fail("Scoring rebuilt unchanged measurement bindings")

    monkeypatch.setattr(prepare, "finalize_parameters", unexpected_refinalization)
    scored = fc.score_model(reference, responses, parameters=parameters)
    graph = scored.build()
    response = scored._bindings.responses[0]
    auxiliary, indicator = response.augmentation
    original_auxiliary, _ = reference._bindings.responses[0].augmentation
    assert graph is not calibration and auxiliary is not original_auxiliary
    assert set(graph.free_RVs) == {graph["theta"], auxiliary}
    initial = graph.initial_point(random_seed=81)
    assert initial[graph.rvs_to_values[auxiliary].name].shape == (3, 1)
    original_initial = calibration.initial_point(random_seed=81)
    assert original_initial[calibration.rvs_to_values[original_auxiliary].name].shape == (1, 1)
    assert graph.named_vars_to_dims[auxiliary.name] == (
        response.batch.obs_dim,
        f"{response.batch.name}_difference",
    )
    assert len(graph.coords[response.batch.obs_dim]) == 3
    assert graph.coords["person"] == (7, 8, 9)
    assert graph.observed_RVs == [indicator]
    assert indicator is not original_observed[0]
    assert graph.named_vars_to_dims[indicator.name] == (response.batch.obs_dim,)
    assert not graph.potentials
    target_ancestors = set(ancestors(graph.observed_RVs))
    assert {graph["theta"], auxiliary} <= target_ancestors
    assert not set(calibration.free_RVs).intersection(target_ancestors)
    for name in parameters:
        assert not set(graph.free_RVs).intersection(ancestors([graph[name]]))
    unconditioned = fc.Model(prepare._prepare_scoring(reference.prepared, responses)).build()
    pruned = pm.do(
        unconditioned,
        {name: value.values for name, value in parameters.data_vars.items()},
        make_interventions_shared=False,
        prune_vars=True,
    )
    assert {node.name for node in pruned.free_RVs} == {node.name for node in graph.free_RVs}
    initial["theta"] = np.array([[0.1, 0.3], [-0.2, 0.2], [0.4, -0.3]])
    initial[graph.rvs_to_values[auxiliary].name] = np.array([[0.1], [0.0], [-0.2]])
    for compile_target in ("compile_logp", "compile_dlogp"):
        target, oracle = (getattr(m, compile_target)(mode="JAX") for m in (graph, pruned))
        np.testing.assert_allclose(target(initial), oracle(initial), atol=1e-12, rtol=1e-12)
    from scipy.stats import multivariate_normal, norm

    theta = initial["theta"]
    z = initial[graph.rvs_to_values[auxiliary].name][:, 0]
    slope = np.array([1.0, -0.8])
    sign = np.array([1, -1, 1])
    mean = sign * (theta @ slope - 0.2)
    delta = np.exp(z) - mean
    covariance = parameters.covariance.values
    logp = multivariate_normal.logpdf(theta, mean=[0, 0], cov=covariance).sum()
    logp += (norm.logpdf(delta) + z).sum()
    dtheta = -(theta @ np.linalg.inv(covariance)) + (sign * delta)[:, None] * slope
    dz = 1 - delta * np.exp(z)
    np.testing.assert_allclose(graph.compile_logp(mode="JAX")(initial), logp, atol=1e-12)
    derivative = graph.compile_dlogp(vars=[graph["theta"], auxiliary], mode="JAX")
    np.testing.assert_allclose(derivative(initial), np.r_[dtheta.ravel(), dz], atol=1e-12)
    (probability_name,) = response.batch.distribution_names
    forward = scored._bindings.evaluation(var_names=[probability_name])
    point = parameters.assign(theta=(("person", "trait"), theta))
    (probabilities,) = forward.evaluate_at([forward.model[probability_name]], point)
    np.testing.assert_allclose(
        probabilities[np.arange(len(mean)), response.batch.y], norm.cdf(mean), atol=1e-12
    )
    xr.testing.assert_identical(
        scored.idata["fixed_parameters"].to_dataset().reset_coords(drop=True), parameters
    )
    assert dict(calibration.named_vars) == original_nodes
    assert tuple(calibration.observed_RVs) == original_observed
    xr.testing.assert_identical(reference.idata, before)
    xr.testing.assert_identical(reference.prepared.data["constant_data"], constant)

    # A scheduled person without answers still has a score under the population law.
    empty = fc.ResponseTable(
        pd.DataFrame(columns=["person_id", "block_id", "response"]), format="block_long"
    )
    administration = pd.DataFrame(
        {"person_id": [10], "block_id": ["pair"], "status": ["not_administered"]}
    )
    prior_score = fc.score_model(
        reference, empty, parameters=parameters, administration=administration
    )
    prior_graph = prior_score.build()
    assert prior_graph.coords["person"] == (10,)
    assert prior_graph.free_RVs == [prior_graph["theta"]] and not prior_graph.observed_RVs
    log_det = np.linalg.slogdet(parameters.covariance.values)[1]
    np.testing.assert_allclose(
        prior_graph.compile_logp(mode="JAX")({"theta": np.zeros((1, 2))}),
        -np.log(2 * np.pi) - log_det / 2,
    )
