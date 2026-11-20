"""Native prediction from explicitly supplied final arrays; no inference."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import pyfcirt as fc
from tests.helpers import mixed_model


def test_native_prediction_same_cached_graph_and_stored_probabilities(supplied_model):
    model = supplied_model
    original = model.build()
    predictions = fc.posterior_predictive(model, random_seed=31, progressbar=False)
    assert model.build() is original
    forward = model._bindings.evaluation()
    assert set(predictions["response_probabilities"].data_vars) <= forward.model.named_vars.keys()
    assert "response_probabilities" in model.idata
    for response in model._bindings.responses:
        batch = response.batch
        probabilities = predictions["response_probabilities"][f"{batch.name}_p"]
        assert_allclose(probabilities.sum(batch.category_dim), 1, atol=1e-12)
        assert_allclose(probabilities.isel(draw=0), probabilities.isel(draw=1), atol=1e-12)
        assert (probabilities >= 0).all()
        replicas = predictions["posterior_predictive"][batch.response_name]
        assert int(replicas.min()) >= 0
        assert int(replicas.max()) < len(batch.events.labels)
    xr.testing.assert_identical(
        model.idata["posterior_predictive"], predictions["posterior_predictive"]
    )
    view = fc.predictive_view(predictions, block="pair")
    assert view.attrs["data_type"] == "binary"
    assert view["observed_data"].sizes["obs"] == 2


def test_selected_prediction_retains_source_labels_and_does_not_store(supplied_model):
    predictions = fc.posterior_predictive(
        supplied_model, draws=1, store=False, random_seed=47, progressbar=False
    )
    replicas = predictions["posterior_predictive"]
    assert replicas.source_chain.values.tolist() == ["supplied"]
    assert replicas.source_draw.item() in (3, 9)
    assert "posterior_predictive" not in supplied_model.idata


@pytest.mark.parametrize("store", [False, True])
def test_prediction_coordinates_are_checked_before_store_or_return(
    supplied_model, monkeypatch, store
):
    from pyfcirt.analysis import prediction

    native = prediction._native_forward
    dim = supplied_model.prepared.batches[0].obs_dim

    def invalid_labels(*args, **kwargs):
        output = native(*args, **kwargs)
        return output.assign_coords({dim: [f"wrong-{index}" for index in range(output.sizes[dim])]})

    monkeypatch.setattr(prediction, "_native_forward", invalid_labels)
    with pytest.raises(ValueError, match=f"output coordinate '{dim}'"):
        fc.posterior_predictive(supplied_model, store=store, random_seed=47, progressbar=False)
    assert "posterior_predictive" not in supplied_model.idata


@pytest.mark.parametrize("method", ["advi", "map_laplace"])
def test_non_mcmc_rejects_predictive_and_likelihood(supplied_model, method):
    supplied_model.idata.attrs["inference_method"] = method
    for operation in (fc.posterior_predictive, fc.log_likelihood):
        with pytest.raises(NotImplementedError, match="MCMC"):
            operation(supplied_model, progressbar=False)


@pytest.mark.parametrize("save_responses", [True, False])
def test_prior_predictive_is_actual_prior_and_never_posterior(save_responses):
    model = mixed_model()
    result = fc.prior_predictive(model, draws=2, random_seed=18, save_responses=save_responses)
    assert "prior" in result and "prior_predictive" in result
    assert "posterior" not in model.idata
    assert ("prior_predictive" in model.idata) == save_responses
    if save_responses:
        xr.testing.assert_identical(model.idata["prior_predictive"], result["prior_predictive"])
    assert model.idata["prior"].sizes["draw"] == 2
    assert model.idata["prior"].person.values.tolist() == [1, 2]
    assert "prior_predictive" in fc.predictive_view(result, item="ordinal")


def test_covariate_ancestors_do_not_change_forward_likelihood_given_final_nodes(
    supplied_model, monkeypatch
):
    import pymc as pm

    from tests.helpers import mixed_inputs, supplied_posterior, workflow_point

    model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs()))
    model.add_regressions(
        {
            "theta": fc.Regression(
                "age", pd.DataFrame({"person": [1, 2], "age": [20, 40]}), id="person"
            ),
            ("fc", "loading_magnitude"): fc.Regression(
                "length",
                pd.DataFrame({"statement": list("abcdefghi"), "length": range(9)}),
                id="statement",
            ),
        }
    )
    point = workflow_point(model)
    point["regression_coefficient"].values[:] = 0.1
    supplied_posterior(model, point, inference_method="mcmc")
    extracted = fc.parameter_point(model, include_person=False)
    xr.testing.assert_equal(
        extracted["regression_coefficient"].reset_coords(drop=True), point["regression_coefficient"]
    )
    assert "theta" not in extracted
    assert set(extracted) <= set(point)
    # These are supplied final arrays, not estimates or a calibration convergence claim.
    assert not any(name.endswith("_raw") for name in model.idata["posterior"].data_vars)
    expected = fc.log_likelihood(supplied_model, progressbar=False)
    actual = fc.log_likelihood(model, progressbar=False)
    assert_allclose(actual, expected, atol=1e-12)
    for reference in (model, supplied_model):
        fc.posterior_predictive(reference, random_seed=31, progressbar=False)
    xr.testing.assert_allclose(
        model.idata["response_probabilities"].to_dataset(),
        supplied_model.idata["response_probabilities"].to_dataset(),
    )
    posterior = model.idata["posterior"].to_dataset()
    assert {rv.name for rv in model.build().free_RVs} - posterior.data_vars.keys()

    def sample(**options):
        assert not options["idata_kwargs"]["log_likelihood"]
        return xr.DataTree.from_dict({"posterior": posterior})

    monkeypatch.setattr(pm, "sample", sample)
    fitted = model.fit(overwrite=True, progressbar=False, idata_kwargs={"log_likelihood": True})
    xr.testing.assert_identical(fitted["log_likelihood"]["response"], expected)


@pytest.mark.parametrize("draws", [0, -1, True, 1.5])
def test_prior_draw_count_rejected_before_graph(draws, monkeypatch):
    model = mixed_model()
    monkeypatch.setattr(model, "build", lambda: pytest.fail("invalid input built graph"))
    with pytest.raises(ValueError, match="positive integer"):
        fc.prior_predictive(model, draws=draws)


def test_log_likelihood_matches_direct_native_event_logp(supplied_model):
    model = supplied_model
    actual = fc.log_likelihood(model, progressbar=False)
    assert actual.dims == ("chain", "draw", "obs")
    assert actual.attrs["observation_unit"] == "prepared actual events"
    np.testing.assert_array_equal(actual.obs, model.prepared.constant_data.obs)
    for name, value in model.prepared.output_coords.items():
        if name.startswith("obs_"):
            assert name in actual.coords
            np.testing.assert_array_equal(actual[name], value)
    assert actual.obs_person_id.values.tolist() == [1, 2, 1, 1, 1, 2, 1]
    assert actual.obs_unit_index.values.tolist() == [0, 0, 1, 2, 3, 3, 4]
    assert actual.obs_block_index.values.tolist() == [0, 0, 1, 2, -1, -1, -1]
    assert actual.obs_item_index.values.tolist() == [-1, -1, -1, -1, 0, 0, 1]
    assert actual.obs_response_format.values.tolist() == [
        "rank",
        "rank",
        "pick",
        "mole",
        "",
        "",
        "",
    ]
    assert_allclose(actual.isel(draw=0), actual.isel(draw=1))
    predictions = fc.posterior_predictive(model, store=False, random_seed=1, progressbar=False)
    expected, positions = [], []
    for response in model._bindings.responses:
        b = response.batch
        probs = predictions["response_probabilities"][f"{b.name}_p"].values
        expected.append(
            np.log(
                np.take_along_axis(
                    probs, np.broadcast_to(b.y, probs.shape[:-1])[..., None], axis=-1
                )[..., 0]
            )
        )
        positions.extend(range(b.obs_slice.start, b.obs_slice.stop))
    assert_allclose(
        actual, np.concatenate(expected, axis=-1)[..., np.argsort(positions)], atol=1e-12
    )
    assert fc.log_likelihood(model, store=False, progressbar=False).identical(actual)


@pytest.mark.parametrize(
    "family,link", [("TIRT", "beta_probit"), ("TIRT", "beta_logit"), ("GTUM", "beta_logit")]
)
def test_pooled_likelihood_keeps_joint_comparison_and_beta_units_without_augmentation(family, link):
    from itertools import combinations

    import pandas as pd
    import pymc as pm
    from scipy.stats import beta

    from tests.helpers import supplied_posterior

    groups = {"joint": [11, 12, 13], "comparison": [21, 22, 23], "continuous": [31, 32]}
    design = pd.DataFrame(
        [
            (block, position, statement, ("A", "B", "C")[position - 1], 1)
            for block, statements in groups.items()
            for position, statement in enumerate(statements, 1)
        ],
        columns=["block_id", "position", "statement_id", "trait", "location_side"],
    )
    design["keying"] = 1
    answers = pd.DataFrame(
        [
            (101, block, first, second, 0.4 if block == "continuous" else 1)
            for block, statements in groups.items()
            for first, second in combinations(statements, 2)
        ],
        columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
        dtype=object,
    )
    options = {"link": link, "priors": {"response_precision": pm.Gamma.dist(alpha=3, beta=1)}}
    if family == "TIRT":
        options["loading_anchors"] = {31: 1.0, 32: 1.1}
    source = fc.FCInput(
        design,
        fc.ResponseTable(answers, format="comparison_long"),
        blocks=(
            fc.BlockSpec("joint", model=fc.TIRTRank(), response_format="rank"),
            fc.BlockSpec("comparison", model=fc.TIRT(), response_format=None),
            fc.BlockSpec(
                "continuous",
                model=getattr(fc, family)(**options),
                response_format=None,
                scale=fc.UnitIntervalScale(),
            ),
        ),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    # Native prior values serve only as explicitly supplied final-array test inputs.
    point = (
        fc.prior_predictive(model, draws=1, random_seed=79, store=False)["prior"]
        .to_dataset()
        .isel(chain=0, draw=0, drop=True)
    )
    supplied_posterior(model, point, inference_method="mcmc")
    predictions = fc.posterior_predictive(model, random_seed=80, progressbar=False)
    actual = fc.log_likelihood(model, progressbar=False)
    assert actual.sizes["obs"] == 5
    assert actual.obs_unit_index.values.tolist() == [0, 1, 1, 1, 2]
    assert actual.obs_block_index.values.tolist() == [0, 1, 1, 1, 2]
    assert actual.obs_item_index.values.tolist() == [-1] * 5
    assert actual.obs_pair_index.values[0] == -1
    assert (actual.obs_pair_index.values[1:] >= 0).all()
    for name, value in model.prepared.output_coords.items():
        if name.startswith("obs_"):
            np.testing.assert_array_equal(actual[name], value)
    design = predictions["design"]
    np.testing.assert_array_equal(design.obs, actual.obs)
    np.testing.assert_array_equal(design.obs_unit, actual.obs_unit_index)
    np.testing.assert_array_equal(design.obs_batch, actual.obs_batch_index)
    assert any(response.augmentation for response in model._bindings.responses)
    for response in model._bindings.responses:
        batch = response.batch
        if batch.likelihood_kind == "beta":
            expected = beta.logpdf(
                batch.y,
                predictions["response_parameters"][f"{batch.name}_alpha"],
                predictions["response_parameters"][f"{batch.name}_beta"],
            )
        else:
            probability = predictions["response_probabilities"][f"{batch.name}_p"].values
            expected = np.log(
                np.take_along_axis(
                    probability,
                    np.broadcast_to(batch.y, probability.shape[:-1])[..., None],
                    axis=-1,
                )[..., 0]
            )
        assert_allclose(actual.isel(obs=batch.obs_slice), expected, atol=2e-10, rtol=2e-10)
        np.testing.assert_array_equal(predictions["observed_data"][batch.response_name], batch.y)
    for block, dtype in (("joint", "i"), ("comparison", "i"), ("continuous", "f")):
        view = fc.predictive_view(predictions, block=block)
        values = view["observed_data"]["response"]
        assert values.dtype.kind == dtype
        xr.testing.assert_equal(values.obs, view["posterior_predictive"].obs)


def test_native_partial_likelihood_retains_true_obs_labels_and_reuses_cached_batch(
    monkeypatch,
):
    import pandas as pd
    import pymc as pm

    from tests.data.test_schema import pair_input
    from tests.helpers import supplied_posterior

    forced_choice = pair_input(model=fc.TIRTRank(loading_anchors={"a": 1.0, "b": 1.1}))
    items = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["item"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p"], "item_id": ["item"], "response": [1]}),
            format="item_long",
        ),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs((forced_choice, items)))
    point = (
        fc.prior_predictive(model, draws=1, random_seed=53, store=False)["prior"]
        .to_dataset()
        .isel(chain=0, draw=0, drop=True)
    )
    supplied_posterior(model, point, inference_method="mcmc")
    (ordinary,) = [response for response in model._bindings.responses if not response.augmentation]
    (rank,) = [response for response in model._bindings.responses if response.augmentation]
    assert rank.augmentation[1] in model.build().observed_RVs
    batch = ordinary.batch
    posterior = model.idata["posterior"].to_dataset()

    def sample(**kwargs):
        assert kwargs["idata_kwargs"]["log_likelihood"] is False
        return xr.DataTree.from_dict({"posterior": posterior})

    monkeypatch.setattr(pm, "sample", sample)
    model.fit(draws=2, overwrite=True, progressbar=False, idata_kwargs={"log_likelihood": True})
    partial = model.idata["log_likelihood"]["response"]
    assert rank.batch.log_likelihood_name not in model._bindings.evaluation().model
    assert partial.sizes["obs"] == 1 and not partial.attrs["complete_observations"]
    np.testing.assert_array_equal(partial.obs_unit_index, [1])
    np.testing.assert_array_equal(partial.obs_block_index, [-1])
    np.testing.assert_array_equal(partial.obs_item_index, [0])
    from pyfcirt.analysis import prediction

    evaluate = prediction._response_log_densities

    def rank_only(bindings, posterior, batches, *, progressbar):
        assert len(batches) == 1 and batches[0] is rank.batch
        return evaluate(bindings, posterior, batches, progressbar=progressbar)

    monkeypatch.setattr(prediction, "_response_log_densities", rank_only)
    full = fc.log_likelihood(model, store=False, progressbar=False)
    assert full.attrs["complete_observations"]
    assert rank.batch.log_likelihood_name in model._bindings.evaluation().model
    xr.testing.assert_allclose(partial, full.isel(obs=batch.obs_slice))
    assert model.idata["log_likelihood"]["response"].identical(partial)
    stored = fc.log_likelihood(model, store=True, progressbar=False)
    xr.testing.assert_identical(stored, full)
    xr.testing.assert_identical(stored, model.idata["log_likelihood"]["response"])
    xr.testing.assert_identical(fc.log_likelihood(model, store=False, progressbar=False), stored)


def test_parameter_point_averages_final_covariance_and_selects_labelled_joint_draw(supplied_model):
    posterior = supplied_model.idata["posterior"].to_dataset().copy(deep=True)
    posterior["covariance"].loc[{"draw": 9}] = [[1.0, 0.4], [0.4, 1.0]]
    posterior["optimizer_raw"] = (("chain", "draw"), [[123.0, -99.0]])
    supplied_model.idata["posterior"] = posterior
    point = fc.parameter_point(supplied_model, include_person=False)
    assert_allclose(point["covariance"], [[1.0, 0.2], [0.2, 1.0]], atol=1e-12)
    assert "theta" not in point and "optimizer_raw" not in point
    joint = fc.parameter_point(supplied_model, draw=("supplied", 9))
    assert_allclose(joint["covariance"], [[1.0, 0.4], [0.4, 1.0]], atol=1e-12)
    assert "chain" not in joint.dims and "draw" not in joint.dims
    assert_allclose(joint.theta, posterior.theta.sel(chain="supplied", draw=9))


def test_long_posterior_mean_keeps_structural_anchors_exact_for_new_person_scoring():
    import pandas as pd

    from tests.helpers import supplied_posterior

    answers = pd.DataFrame(
        {
            "person_id": [1],
            "block_id": ["pair"],
            "statement_1": ["a"],
            "statement_2": ["b"],
            "response": [1],
        }
    )
    inputs = fc.FCInput(
        fc.FCDesignMapping(traits={"A": ["a", "b"]}, blocks={"pair": ["a", "b"]}),
        fc.ResponseTable(answers, format="comparison_long"),
        blocks=(fc.BlockSpec("pair", model=fc.TIRT(loading_anchors={"a": 1.1, "b": 0.7})),),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs))
    point = xr.Dataset(
        {
            layout.kind: xr.DataArray(
                np.where(np.isfinite(layout.fixed), layout.fixed, 0.2),
                dims=layout.dims,
                coords={dim: model.prepared.coords[dim] for dim in layout.dims},
            )
            for layout in model.prepared.parameters
        }
    )
    point["theta"] = xr.DataArray(
        [[0.2]],
        dims=("person", "trait"),
        coords={dim: model.prepared.coords[dim] for dim in ("person", "trait")},
    )
    supplied_posterior(model, point, inference_method="mcmc", chains=range(4), draws=range(1000))
    new_answers = {"fc": fc.ResponseTable(answers.assign(person_id=7), format="comparison_long")}
    extracted = fc.parameter_point(model, include_person=False)
    np.testing.assert_array_equal(extracted.loading, point.loading)
    scored = fc.score_model(model, new_answers, parameters=extracted)
    np.testing.assert_array_equal(scored.idata["fixed_parameters"].loading, point.loading)
    assert scored.prepared.constant_data.person.values.tolist() == [7]

    # Invalid externally assigned draws cannot be hidden by restoring known constants.
    posterior = model.idata["posterior"].to_dataset().copy(deep=True)
    posterior["loading"][0, 0, 0] += 0.1
    model.idata["posterior"] = posterior
    with pytest.raises(ValueError, match="structural|fixed"):
        fc.score_model(
            model, new_answers, parameters=fc.parameter_point(model, include_person=False)
        )


def test_native_forward_final_theta_preserves_typed_person_coordinates(supplied_model):
    from pyfcirt.analysis.prediction import _native_forward

    result = _native_forward(
        supplied_model._bindings,
        supplied_model.idata["posterior"].to_dataset(),
        ["theta"],
        progressbar=False,
    )
    assert result.person.values.tolist() == [1, 2]
    assert_allclose(result.theta, supplied_model.idata["posterior"].theta, atol=1e-12)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_predictive_seed_rejects_before_model_access(seed):
    with pytest.raises(ValueError, match="random_seed"):
        fc.posterior_predictive(None, random_seed=seed)


@pytest.mark.parametrize("operation", [fc.posterior_predictive, fc.prior_predictive])
def test_prediction_boolean_controls_are_strict(operation):
    with pytest.raises(TypeError, match="boolean"):
        operation(None, store=1)


def test_prediction_requires_genuine_nonempty_sample_axes():
    from pyfcirt.analysis.output import posterior_samples

    cases = [
        xr.DataTree(),
        xr.DataTree.from_dict({"posterior": xr.Dataset({"theta": ("draw", [1.0])})}),
        xr.DataTree.from_dict({"posterior": xr.Dataset(coords={"chain": [0], "draw": []})}),
    ]
    for idata in cases:
        with pytest.raises(ValueError):
            posterior_samples(idata)


@pytest.mark.parametrize("draws", [0, -1, True, 1.5, 7])
def test_state_selection_rejects_invalid_or_unavailable_counts(draws):
    from pyfcirt.analysis.prediction import _select_states

    posterior = xr.Dataset(coords={"chain": ["a", "b"], "draw": [3, 8, 11]})
    with pytest.raises(ValueError, match="draws"):
        _select_states(posterior, draws, 1)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_balanced_selection_uses_unique_joint_labels_and_all_seed_words(backend):
    from pyfcirt.analysis.prediction import _select_states

    values = np.arange(15).reshape(3, 5)
    if backend == "jax":
        import jax.numpy as jnp

        values = jnp.asarray(values)
    posterior = xr.Dataset(
        {"value": (("chain", "draw"), values)},
        coords={"chain": ["a", "b", "c"], "draw": [2, 5, 9, 12, 20]},
    )
    chosen = _select_states(posterior, 8, 2**70 + 7)
    assert isinstance(chosen.value.data, type(values))
    xr.testing.assert_identical(chosen, _select_states(posterior, 8, 2**70 + 7))
    pairs = list(zip(chosen.source_chain.values, chosen.source_draw.values, strict=True))
    assert len(set(pairs)) == 8
    assert sorted(np.unique(chosen.source_chain, return_counts=True)[1]) == [2, 3, 3]
    for i, (chain, draw) in enumerate(pairs):
        assert chosen.value[0, i] == posterior.value.sel(chain=chain, draw=draw)
    assert _select_states(posterior, None, None) is posterior
    assert _select_states(posterior, 1, None).sizes["draw"] == 1


def test_final_point_validation_and_joint_draw_selection(supplied_model):
    from pyfcirt.analysis.prediction import _native_forward

    for draw in [3, ("supplied",), ("missing", 3)]:
        with pytest.raises((ValueError, KeyError), match="draw|chain|index|not all"):
            fc.parameter_point(supplied_model, draw=draw)
    original = supplied_model.idata["posterior"].to_dataset()
    with pytest.raises(TypeError, match="include_person"):
        fc.parameter_point(supplied_model, include_person=1)
    for bad in [
        original.drop_vars("theta"),
        original.assign(theta=original.theta.isel(trait=0, drop=True)),
        original.assign(theta=original.theta.astype(bool)),
        original.assign(theta=original.theta.astype(complex)),
        original.assign(theta=original.theta.astype(object)),
        original.assign_coords(person=["wrong", "people"]),
    ]:
        supplied_model.idata["posterior"] = bad
        with pytest.raises(ValueError, match="final|Missing"):
            fc.parameter_point(supplied_model)
    supplied_model.idata["posterior"] = original
    empty = _native_forward(supplied_model._bindings, original, [])
    assert not empty.data_vars and empty.sizes == {"chain": 1, "draw": 2}
    with pytest.raises(ValueError, match="without sample"):
        supplied_model._bindings.evaluate_at([], original)
    with pytest.raises(ValueError, match="Missing required"):
        _native_forward(
            supplied_model._bindings,
            original.drop_vars("theta"),
            ["theta"],
            progressbar=False,
        )


def test_prediction_without_probabilities_replaces_cached_probability_outputs(supplied_model):
    fc.posterior_predictive(supplied_model, random_seed=2, progressbar=False)
    result = fc.posterior_predictive(
        supplied_model, random_seed=3, include_probabilities=False, progressbar=False
    )
    assert set(result.children) == {
        "posterior_predictive",
        "observed_data",
        "design",
        "response_support",
    }
    assert "response_probabilities" not in supplied_model.idata
    assert fc.predictive_view(result, block="pair")["observed_data"].sizes["obs"] == 2
    assert fc.predictive_view(result, item="pair")["observed_data"].sizes["obs"] == 2
    with pytest.raises(ValueError, match="Unknown"):
        fc.predictive_view(result, block="missing")
    for options in ({}, {"block": 7, "item": 7}):
        with pytest.raises(ValueError, match="exactly one"):
            fc.predictive_view(result, **options)


def test_predictive_view_validates_tree_support_and_actual_observations(supplied_model):
    with pytest.raises(TypeError, match="DataTree"):
        fc.predictive_view(xr.Dataset(), block="pair")
    with pytest.raises(ValueError, match="no predictive"):
        fc.predictive_view(xr.DataTree(), block="pair")
    result = fc.posterior_predictive(supplied_model, random_seed=9, store=False, progressbar=False)
    ordinal = fc.predictive_view(result, item="ordinal")
    assert ordinal.attrs["data_type"] == "ordinal"
    categorical = fc.predictive_view(result, block="quad")
    assert categorical.attrs["data_type"] == "categorical"
    del result["observed_data"]
    with pytest.raises(ValueError, match="no actual observed"):
        fc.predictive_view(result, block="pair")


def test_detached_prediction_preserves_recorded_obs_after_netcdf_roundtrip(
    supplied_model, tmp_path
):
    pytest.importorskip("h5netcdf")
    result = fc.posterior_predictive(
        supplied_model, random_seed=47, store=False, include_probabilities=False, progressbar=False
    )
    expected = fc.predictive_view(result, block="pair")
    assert expected["observed_data"].response.dtype.kind == "i"
    path = tmp_path / "detached_predictions.nc"
    result.to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as saved:
        actual = fc.predictive_view(saved, block="pair")
        xr.testing.assert_identical(actual, expected)


def test_partial_ranking_with_different_pairs_has_distinct_predictive_support(tmp_path):
    pytest.importorskip("h5netcdf")
    source = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a", "b"], "B": ["c"]}, blocks={"triplet": ["a", "b", "c"]}
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["first", "second"],
                    "block_id": ["triplet", "triplet"],
                    "statement_1": ["a", "a"],
                    "statement_2": ["b", "c"],
                    "response": [0, 1],
                }
            ),
            format="comparison_long",
        ),
        model="TwoPLRank",
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    result = fc.prior_predictive(model, draws=2, random_seed=17, store=False)
    with pytest.raises(ValueError, match="consistent support"):
        fc.predictive_view(result, block="triplet")
    path = tmp_path / "predictions.nc"
    result.to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as saved:
        with pytest.raises(ValueError, match="consistent support"):
            fc.predictive_view(saved, block="triplet")


def test_parameter_point_reads_fixed_global_parameters_without_reconstructing_draws(supplied_model):
    from tests.helpers import mixed_inputs, supplied_posterior

    point = fc.parameter_point(supplied_model)
    fixed = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs(), fixed_parameters=point))
    supplied_posterior(fixed, point[["theta"]], inference_method="mcmc")
    actual = fc.parameter_point(fixed)
    for name in point:
        xr.testing.assert_allclose(actual[name], point[name])
    information = fc.information(fixed, parameters=actual, store=False, progressbar=False)
    assert np.isfinite(information.test_information).all()


def test_fixed_graph_roots_cannot_be_overridden_by_extra_posterior_values(supplied_model):
    from pyfcirt.analysis.prediction import _native_forward
    from tests.helpers import mixed_inputs, supplied_posterior

    point = fc.parameter_point(supplied_model)
    fixed = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs(), fixed_parameters=point))
    supplied_posterior(fixed, point[["theta"]], inference_method="mcmc")
    expected = fc.posterior_predictive(fixed, random_seed=14, store=False, progressbar=False)
    posterior = fixed.idata["posterior"].to_dataset()
    loading = _native_forward(
        fixed._bindings,
        posterior.drop_vars("theta"),
        ["loading"],
        draws=1,
        random_seed=14,
        progressbar=False,
    )
    assert loading.sizes["draw"] == 1
    assert loading.source_draw.item() in posterior.draw.values
    xr.testing.assert_allclose(
        loading.loading.isel(chain=0, draw=0, drop=True), point.loading.reset_coords(drop=True)
    )
    posterior["loading"] = (
        point["loading"].expand_dims(chain=posterior.chain, draw=posterior.draw) * 10
    )
    fixed.idata["posterior"] = posterior
    actual = fc.posterior_predictive(fixed, random_seed=14, store=False, progressbar=False)
    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("fixed_analysis", [False, True])
def test_attachment_rejects_conflicting_fixed_entries_atomically(supplied_model, fixed_analysis):
    from tests.helpers import mixed_inputs

    point = fc.parameter_point(supplied_model)
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            mixed_inputs(), **({"fixed_parameters": point} if fixed_analysis else {})
        )
    )
    before = model.idata.copy(deep=True)
    posterior = supplied_model.idata["posterior"].to_dataset().copy(deep=True)
    # Identified statement references remain fixed in ordinary and conditioned analyses.
    posterior["statement_intercept"][..., 0] = 4.0
    with pytest.raises(ValueError, match="fixed|conflict"):
        model.attach_samples(xr.DataTree.from_dict({"posterior": posterior}))
    xr.testing.assert_identical(model.idata, before)
    assert model._built is None


def test_attachment_checks_full_calibration_conditions_as_well_as_structural_entries(
    supplied_model,
):
    from tests.helpers import mixed_inputs

    point = fc.parameter_point(supplied_model)
    model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs(), fixed_parameters=point))
    posterior = point.expand_dims(chain=[0], draw=[0, 1])
    conflict = posterior.copy(deep=True)
    conflict["loading"][..., 1] += 0.2
    with pytest.raises(ValueError, match="conflict with fixed calibration values"):
        model.attach_samples(xr.DataTree.from_dict({"posterior": conflict}))
    assert "posterior" not in model.idata
    model.attach_samples(xr.DataTree.from_dict({"posterior": posterior}))
    xr.testing.assert_equal(model.idata["posterior"].to_dataset(), posterior)
    assert model._built is None


@pytest.mark.parametrize("complete_cache", [False, True])
def test_log_likelihood_reuses_shuffled_complete_batches(
    supplied_model, monkeypatch, complete_cache
):
    from pyfcirt.analysis import prediction

    model = supplied_model
    expected = fc.log_likelihood(model, store=False, progressbar=False)
    positions = np.arange(expected.sizes["obs"])[::-1]
    incomplete = next(
        batch
        for batch in model.prepared.batches
        if batch.obs_slice.stop - batch.obs_slice.start > 1
    )
    if not complete_cache:
        positions = positions[positions != incomplete.obs_slice.stop - 1]
    partial = expected.isel(obs=positions).copy(deep=True)
    if not complete_cache:
        # A row from an incomplete batch must be replaced together with that batch.
        partial.loc[{"obs": expected.obs.values[incomplete.obs_slice.start]}] = -100
    partial.attrs["complete_observations"] = 0
    model.idata["log_likelihood"] = partial.to_dataset()
    evaluate = prediction._response_log_densities
    requested = []

    def record(bindings, posterior, batches, *, progressbar):
        requested.extend(batches)
        return evaluate(bindings, posterior, batches, progressbar=progressbar)

    monkeypatch.setattr(prediction, "_response_log_densities", record)
    result = fc.log_likelihood(model, store=False, progressbar=False)
    xr.testing.assert_identical(result, expected)
    assert len(requested) == (0 if complete_cache else 1)
    if requested:
        assert requested[0] is incomplete
    xr.testing.assert_identical(model.idata["log_likelihood"].response, partial)


@pytest.mark.parametrize("include_raw", [False, True])
def test_log_likelihood_uses_final_nodes_with_extra_raw_and_partial_cache(
    supplied_model, include_raw
):
    model = supplied_model
    expected = fc.log_likelihood(model, store=False, progressbar=False)
    if include_raw:
        posterior = model.idata["posterior"].to_dataset()
        for rv in model.build().free_RVs:
            if rv.name not in posterior:
                # Extra sampler fields, including LKJ raw coordinates, are not inputs.
                posterior[rv.name] = (("chain", "draw"), np.ones((1, 2)))
        model.idata["posterior"] = posterior
    model.idata["log_likelihood"] = xr.Dataset(
        {"response": (("chain", "draw", "obs"), np.zeros((1, 2, 1)))}
    )
    result = fc.log_likelihood(model, store=False, progressbar=False)
    assert result.sizes["obs"] == 7 and result.attrs["complete_observations"]
    xr.testing.assert_allclose(result, expected)
    assert model.idata["log_likelihood"].sizes["obs"] == 1
