"""Native result persistence and explicit sample attachment; no posterior MCMC."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pyfcirt as fc
from tests.helpers import labelled_point


def inputs(model="TIRT", people=(101, 102)):
    return fc.FCInput(
        fc.FCDesignMapping(
            traits={"a": ["s1", "s3"], "b": ["s2"]},
            blocks={"B1": ["s1", "s2", "s3"]},
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": list(people),
                    "block_id": ["B1"] * len(people),
                    "response": [(1, 2, 3)] * len(people),
                }
            ),
            format="block_long",
        ),
        model=model,
    )


def samples(model):
    """Explicit labelled final values, not a fitted posterior or convergence claim."""

    graph = model.build()
    values = {
        "theta": [[0.1, -0.2], [-0.1, 0.4]],
        "population_mean": [[0.0, 0.0]],
        "covariance": np.eye(2),
        "loading": [1.0, 1.2, 0.8],
        "residual_variance": [0.5, 0.6, 0.7],
        "statement_effect": [0.0, 0.1, -0.1],
        "cutpoint": [0.0, 0.0, 0.0],
    }
    posterior = xr.Dataset(
        {
            name: xr.DataArray(
                value,
                dims=graph.named_vars_to_dims[name],
                coords={dim: list(graph.coords[dim]) for dim in graph.named_vars_to_dims[name]},
            )
            for name, value in values.items()
        }
    ).expand_dims(chain=[0, 1], draw=np.arange(6))
    return xr.DataTree.from_dict(
        {
            "posterior": model.prepared.annotate_coordinates(posterior),
            "sample_stats": xr.Dataset(
                {"diverging": (("chain", "draw"), np.zeros((2, 6), dtype=bool))},
                coords={"chain": [0, 1], "draw": np.arange(6)},
            ),
        }
    )


def test_preparation_and_results_have_separate_ownership():
    prepared = fc.PreparedInputs.from_inputs(inputs())
    assert isinstance(prepared, fc.PreparedInputs)
    model = fc.Model(prepared)
    assert model.prepared is prepared
    assert set(model.idata.children) == {
        "design",
        "parameter_data",
        "parameter_uses",
        "categories",
        "response_support",
    }
    assert model.build() is model.build()
    assert fc.data_coverage(prepared).equals(fc.data_coverage(model))
    assert not hasattr(fc, "save") and not hasattr(fc, "load")
    with pytest.raises(TypeError, match="FCInput or ItemInput"):
        fc.Model(xr.DataTree())


@pytest.mark.parametrize("people", [(101, "102"), (True, 102), (None, 102)])
def test_invalid_identifier_columns_fail_during_preparation(people):
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(inputs(people=people))


def test_ids_can_differ_between_entity_domains():
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    assert model.prepared.coords["person"].tolist() == [101, 102]
    assert model.prepared.coords["block"].tolist() == ["B1"]


@pytest.mark.parametrize("fc_integer", [False, True])
@pytest.mark.parametrize("item_first", [False, True])
def test_native_fulltree_keeps_typed_entity_domains_and_global_references(
    tmp_path, fc_integer, item_first
):

    pytest.importorskip("h5netcdf")
    statements = [11, 12, 13] if fc_integer else ["s1", "s2", "s3"]
    block, item = (17, "ordinal") if fc_integer else ("B1", 29)
    fc_source = replace(
        inputs(),
        design=fc.FCDesignMapping(
            traits={"a": [statements[0], statements[2]], "b": [statements[1]]},
            blocks={block: statements},
        ),
        responses=fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": [101, 102],
                    "block_id": [block, block],
                    "response": [(1, 2, 3), (2, 1, 3)],
                }
            ),
            format="block_long",
        ),
        statement_features=pd.DataFrame({"statement_id": statements, "note": ["a", "b", "c"]}),
    )
    item_source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"a": [item]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [101, 102], "item_id": [item, item], "response": [0, 2]}),
            format="item_long",
        ),
        model=fc.GPCM(),
        categories=(0, 1, 2),
    )
    sources = (item_source, fc_source) if item_first else (fc_source, item_source)
    prepared = fc.PreparedInputs.from_inputs(sources)
    model = fc.Model(prepared)
    values = {}
    for layout in prepared.parameters:
        value = np.where(np.isfinite(layout.fixed), layout.fixed, 0.0)
        if layout.kind == "loading":
            value = np.where(np.isfinite(layout.fixed), layout.fixed, 1.0)
        elif layout.kind == "residual_variance":
            value = np.where(np.isfinite(layout.fixed), layout.fixed, 0.7)
        values[layout.kind] = xr.DataArray(
            value, dims=layout.dims, coords={dim: prepared.coords[dim] for dim in layout.dims}
        )
    values["theta"] = xr.DataArray(
        np.zeros((2, 2)),
        dims=("person", "trait"),
        coords={dim: prepared.coords[dim] for dim in ("person", "trait")},
    )
    posterior = xr.Dataset(values).expand_dims(chain=[0], draw=[0, 1])
    supplied = xr.DataTree.from_dict({"posterior": prepared.annotate_coordinates(posterior)})
    model.attach_samples(supplied)
    # Sentinels exercise output identity publication, not likelihood mathematics.
    likelihood = xr.Dataset(
        {"response": (("chain", "draw", "obs"), np.zeros((1, 2, len(prepared.coords["obs"]))))},
        coords={"chain": [0], "draw": [0, 1], "obs": prepared.coords["obs"]},
    )
    model._publish_outputs({"log_likelihood": likelihood})
    target = tmp_path / "typed-domains.nc"
    model.idata.to_netcdf(target, engine="h5netcdf")
    with xr.open_datatree(target, engine="h5netcdf") as stored:
        loaded = stored.load()
    design = loaded["design"].to_dataset()
    expected_statement = np.arange(1 if item_first else 0, 4 if item_first else 3)
    np.testing.assert_array_equal(design.statement, expected_statement)
    assert design.statement_label.values.tolist() == statements
    assert design.block.item() == block and type(design.block.item()) is type(block)
    assert design.item.item() == item and type(design.item.item()) is type(item)
    assert all(
        isinstance(label, (int, np.integer) if fc_integer else str)
        for label in design.statement_label.values
    )
    assert "unit_id" not in design and "item_statement" not in design
    assert not any(
        name.startswith(("statement_effect_", "statement_feature_")) for name in design.variables
    )
    fc_unit, item_unit = (1, 0) if item_first else (0, 1)
    assert design.unit_block.values[fc_unit] == 0 and design.unit_item.values[fc_unit] == -1
    assert design.unit_item.values[item_unit] == 0 and design.unit_block.values[item_unit] == -1
    np.testing.assert_array_equal(design.pair_unit, [fc_unit] * 3)
    assert design.statement_label.sel(statement=design.pair_statement_1).values.tolist() == [
        statements[0],
        statements[0],
        statements[1],
    ]
    assert design.statement_label.sel(statement=design.pair_statement_2).values.tolist() == [
        statements[1],
        statements[2],
        statements[2],
    ]
    for layout in prepared.parameters:
        entries = loaded[f"parameter_data/{layout.kind}"].to_dataset()
        references = entries.statement_index.values
        assert np.isin(references[references >= 0], design.statement.values).all()
    references = loaded["parameter_data/cutpoint"].pair_index.values
    assert references.tolist() == [0, 1, 2]
    assert np.isin(loaded["parameter_uses"].unit_index.values, design.unit.values).all()
    pooled = loaded["log_likelihood"].to_dataset()
    assert "obs_unit_id" not in pooled.coords
    np.testing.assert_array_equal(
        pooled.obs_block_index, design.unit_block.values[pooled.obs_unit_index.values]
    )
    np.testing.assert_array_equal(
        pooled.obs_item_index, design.unit_item.values[pooled.obs_unit_index.values]
    )
    assert set(pooled.obs_block_index.values) == {-1, 0}
    assert set(pooled.obs_item_index.values) == {-1, 0}
    restored = fc.Model(fc.PreparedInputs.from_inputs(sources))
    restored.attach_samples(loaded)
    xr.testing.assert_equal(
        restored.idata["posterior"].to_dataset(), loaded["posterior"].to_dataset()
    )
    assert model._built is None and restored._built is None


def test_native_roundtrip_requires_explicit_rebuild_and_attachment(tmp_path):
    original = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    original.attach_samples(samples(original))
    path = tmp_path / "samples.nc"
    original.idata.to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as stored:
        idata = stored.load()
    assert set(idata.children) == {
        "posterior",
        "sample_stats",
        "design",
        "parameter_data",
        "parameter_uses",
        "categories",
        "response_support",
    }
    assert "loading" in idata["parameter_data"]
    assert "loading_parameter_set_index" in idata["posterior"].coords
    assert idata["parameter_uses"].entry_index.size > 0
    restored = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    restored.attach_samples(idata)
    xr.testing.assert_equal(original.idata, restored.idata)
    assert restored.summary() == original.summary()
    pd.testing.assert_frame_equal(fc.coef(original, kind="stats"), fc.coef(restored, kind="stats"))
    assert not any(name.endswith("_raw") for name in idata["posterior"].data_vars)


@pytest.mark.parametrize(
    "problem",
    [
        "person",
        "type",
        "float",
        "dimension",
        "sample_stats",
        "duplicate_unindexed_sample_label",
        "empty",
        "parameter_identity",
        "legacy_name",
        "missing_parameter",
        "parameter_as_coordinate",
    ],
)
def test_attachment_rejects_mismatches_atomically(problem):
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    idata = samples(model)
    if problem == "person":
        idata["posterior"] = idata["posterior"].to_dataset().sel(person=[102, 101])
    elif problem == "type":
        idata["posterior"] = idata["posterior"].to_dataset().assign_coords(person=["101", "102"])
    elif problem == "float":
        idata["posterior"] = idata["posterior"].to_dataset().assign_coords(person=[101.0, 102.0])
    elif problem == "dimension":
        posterior = idata["posterior"].to_dataset()
        posterior["theta"] = posterior.theta.transpose("chain", "draw", "trait", "person")
        idata["posterior"] = posterior
    elif problem == "sample_stats":
        idata["sample_stats"] = (
            idata["sample_stats"].to_dataset().assign_coords(draw=np.arange(6) + 1)
        )
    elif problem == "duplicate_unindexed_sample_label":
        for group in ("posterior", "sample_stats"):
            idata[group] = (
                idata[group].to_dataset().assign_coords(chain=[0.5, 0.5]).drop_indexes("chain")
            )
    elif problem == "empty":
        idata["posterior"] = idata["posterior"].to_dataset().isel(draw=[])
    elif problem == "parameter_identity":
        idata["posterior"] = (
            idata["posterior"].to_dataset().drop_vars("loading_parameter_set_index")
        )
    elif problem == "legacy_name":
        idata["posterior"] = (
            idata["posterior"].to_dataset().rename({"loading": "o0::loading_magnitude"})
        )
    elif problem == "missing_parameter":
        idata["posterior"] = idata["posterior"].to_dataset().drop_vars("loading")
    else:
        idata["posterior"] = idata["posterior"].to_dataset().set_coords("loading")
    with pytest.raises(ValueError):
        model.attach_samples(idata)
    assert "posterior" not in model.idata


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("theta", np.nan),
        ("theta", np.inf),
        ("theta", -np.inf),
        ("loading", np.inf),
        ("retained_raw", np.nan),
    ],
)
def test_attachment_rejects_nonfinite_draws_and_allows_retry(variable, value):
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    valid = samples(model)
    valid["posterior"]["retained_raw"] = xr.zeros_like(valid["posterior"]["theta"])
    invalid = valid.copy(deep=True)
    draws = invalid["posterior"][variable].values.copy()
    index = (0, 0, 1) if variable == "loading" else (0, 0, 0, 0)
    draws[index] = value
    invalid["posterior"][variable] = invalid["posterior"][variable].copy(data=draws)
    before, graph, prepared = model.idata, model.build(), model.prepared

    with pytest.raises(ValueError, match=rf"{variable}: .*finite"):
        model.attach_samples(invalid)

    assert model.idata is before and "posterior" not in model.idata
    assert model.prepared is prepared and model.build() is graph
    model.attach_samples(valid)
    xr.testing.assert_identical(model.idata.posterior, valid.posterior)
    xr.testing.assert_identical(model.idata.sample_stats, valid.sample_stats)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("residual_variance", [0.5, 0.6, -10.0], "positive"),
        ("covariance", [[1.0, 0.3], [0.0, 1.0]], "symmetric"),
        ("covariance", [[1.0, 1.2], [1.2, 1.0]], "positive definite"),
    ],
)
def test_attachment_validates_prepared_support_in_each_state(name, value, message):
    model = fc.Model(inputs())
    valid = samples(model)
    invalid = valid.copy(deep=True)
    # Keep the other states valid so validation cannot pass by checking a mean.
    invalid.posterior[name].values[1, 4] = value
    before = model.idata

    with pytest.raises(ValueError, match=message):
        model.attach_samples(invalid)

    assert model.idata is before
    model.attach_samples(valid)
    likelihood = fc.log_likelihood(model, store=False, progressbar=False)
    assert np.isfinite(likelihood).all()


def test_attachment_checks_linked_asymptotes_within_each_draw():
    model = fc.Model(fc.load_example("single_binary", model=fc.DichotomousIRT(kind="4PL")).data)
    shape = model.prepared.parameter_layouts["lower_asymptote"].shape
    point = labelled_point(
        model, {"lower_asymptote": np.full(shape, 0.1), "upper_asymptote": np.full(shape, 0.3)}
    )
    posterior = point.expand_dims(chain=[2], draw=[3, 9]).copy(deep=True)
    posterior.upper_asymptote.values[0, 1] = 0.9
    posterior = model.prepared.annotate_coordinates(posterior)
    valid = xr.DataTree.from_dict({"posterior": posterior})
    invalid = valid.copy(deep=True)
    invalid.posterior.lower_asymptote.values[0, 0] = 0.4

    with pytest.raises(ValueError, match="lower asymptotes must be below upper"):
        model.attach_samples(invalid)

    model.attach_samples(valid)
    xr.testing.assert_identical(model.idata.posterior, valid.posterior)


@pytest.mark.parametrize("indexed", [True, False])
def test_attachment_uses_original_labels_for_joint_state_selection(indexed):
    model = fc.Model(inputs())
    trace = samples(model)
    axes = dict(chain=[0.5, 1.5], draw=np.arange(6) + 0.25)
    for group in ("posterior", "sample_stats"):
        dataset = trace[group].to_dataset().assign_coords(axes).copy(deep=True)
        trace[group] = dataset if indexed else dataset.drop_indexes(["chain", "draw"])
    trace.posterior.theta.values[1, 4] += 0.7
    model.attach_samples(trace)

    draw = (1.5, 4.25)
    expected = trace.posterior.theta.sel(chain=draw[0], draw=draw[1], drop=True)
    actual = fc.parameter_point(model, draw=draw)
    xr.testing.assert_identical(
        actual.theta.reset_coords(drop=True), expected.reset_coords(drop=True)
    )
    with pytest.raises(ValueError, match="one posterior state"):
        fc.parameter_point(model, draw=([1.5], 4.25))


def test_attachment_locks_declarations_and_does_not_import_saved_configuration():
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    idata = samples(model)
    idata["constant_data"] = xr.Dataset(attrs={"ignored": "not model declarations"})
    model.attach_samples(idata)
    assert "constant_data" not in model.idata
    with pytest.raises(ValueError, match="already"):
        model.attach_samples(idata)
    with pytest.raises(ValueError, match="declarations"):
        model.set_priors({"fc.loading_magnitude": object()})


def test_explicit_summary_samples_are_read_only():
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    idata = samples(model)
    result = fc.coef(model, idata=idata, kind="stats")
    assert not result.empty
    assert "posterior" not in model.idata
    traits = fc.trait_summary(model, idata=idata, kind="stats")
    assert "trait_correlation" in traits


def test_rebuilt_model_scores_new_people_with_fixed_final_parameters():
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    model.attach_samples(samples(model))
    response = fc.ResponseTable(
        pd.DataFrame(
            {
                "person_id": [999],
                "block_id": ["B1"],
                "response": [(2, 1, 3)],
            }
        ),
        format="block_long",
    )
    scored = fc.score_model(model, response)
    assert scored.prepared.coords["person"].tolist() == [999]
    assert "fixed_parameters" in scored.idata
    assert "theta" in scored.build().named_vars
    assert "theta" not in scored.idata["fixed_parameters"]


def test_runtime_covariate_encoder_is_reused_for_scoring(monkeypatch):
    from tests.helpers import replay_design

    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    people = pd.DataFrame({"person_id": [101, 102], "age": [20.0, 40.0]})
    model.add_regressions("theta ~ age", person_data=people)
    plan = model.prepared.regressions[0]
    assert "encoder" in plan.design.attrs
    monkeypatch.setattr(
        "pyfcirt.covariates.design._compile",
        lambda *a, **k: pytest.fail("training formula recompiled"),
    )
    keys = pd.DataFrame({"person_id": [999]})
    new = pd.DataFrame({"person_id": [999], "age": [30.0]})
    matrix = replay_design(plan.design, new, keys)
    assert matrix.sizes["row"] == 1
    assert matrix.term.values.tolist() == plan.design.term.values.tolist()


def test_native_prediction_and_likelihood_outputs_can_be_saved(tmp_path):
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    model.attach_samples(samples(model))
    predictions = fc.posterior_predictive(model, progressbar=False)
    assert "response_probabilities" in predictions
    likelihood = fc.log_likelihood(model, progressbar=False)
    assert likelihood.dims == ("chain", "draw", "obs")
    path = tmp_path / "results.nc"
    model.idata.to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as stored:
        view = fc.predictive_view(stored.load(), block="B1")
    assert "posterior_predictive" in view


@pytest.mark.parametrize("family", ["MGPPM", "TwoPLRank", "GGUMRank", "TIRT", "TIRTRank", "GTUM"])
def test_fc_families_build_from_native_runtime_declarations(family):

    source = inputs(model=family)
    design = pd.DataFrame(
        {
            "block_id": ["B1"] * 3,
            "position": [1, 2, 3],
            "statement_id": ["s1", "s2", "s3"],
            "trait": ["a", "b", "a"],
            "keying": [1, 1, 1],
            "location_side": [1, -1, -1],
        }
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(replace(source, design=design)))
    assert model._bindings.prepared is model.prepared
    assert "theta" in model.build().named_vars
    assert set(model.idata.children) == {
        "design",
        "parameter_data",
        "parameter_uses",
        "categories",
        "response_support",
    }


@pytest.mark.parametrize(
    "family",
    [fc.DichotomousIRT(kind=kind) for kind in ("1PL", "2PL", "3PL", "4PL")]
    + [fc.GPCM(), fc.GGUM()],
)
def test_likert_families_and_mixed_models_keep_runtime_layouts(family):
    item = fc.ItemInput(
        pd.DataFrame({"item_id": ["I1"], "trait": ["a"], "location_side": [1]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [101, 102], "item_id": ["I1", "I1"], "response": [0, 1]}),
            format="item_long",
        ),
        model=family,
    )
    model = fc.Model(fc.PreparedInputs.from_inputs((inputs(), item)))
    assert "theta" in model.build().named_vars
    assert set(fc.data_coverage(model, by="item").index) == {"I1"}


def test_item_prior_updates_do_not_restore_a_serialized_specification():
    import pymc as pm

    from pyfcirt.models.priors import same_prior

    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    prior = pm.LogNormal.dist(mu=0.1, sigma=0.3)
    model.set_priors({"fc.loading_magnitude": prior})
    (loading,) = model.prepared.blocks_by_kind["loading"]
    assert same_prior(loading.priors["loading_magnitude"], prior)
    assert "mu=0.1" in model.summary()
    model.build()


def test_loading_regression_builds_from_runtime_templates():
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    model.add_regressions(
        "fc.loading_magnitude ~ length",
        statement_data=pd.DataFrame(
            {
                "statement_id": ["s1", "s2", "s3"],
                "length": [1.0, 2.0, 4.0],
            }
        ),
    )
    assert len(model.prepared.regressions) == 1
    assert "regression_coefficient" in model.build().named_vars


def test_native_results_include_checks_and_information_without_attribute_codec(tmp_path):
    import jax

    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    model.attach_samples(samples(model))
    with jax.enable_x64():
        predictions = fc.posterior_predictive(model, progressbar=False)
        fc.chi_ppp(model, predictions=predictions)
        fc.information(model, progressbar=False)
    model.idata.to_netcdf(tmp_path / "analysis.nc", engine="h5netcdf")


def test_native_sampler_output_does_not_become_preparation(monkeypatch):
    import pymc as pm

    model = fc.Model(fc.PreparedInputs.from_inputs(inputs()))
    supplied = samples(model)
    supplied["constant_data"] = xr.Dataset({"backend_data": 1})
    monkeypatch.setattr(pm, "sample", lambda **kwargs: supplied)
    result = model.fit(draws=6, tune=0, chains=2, progressbar=False)
    assert "constant_data" not in result
    assert "posterior" in result and "sample_stats" in result
    assert result.attrs["scoring_batch_identity"] == model.prepared.scoring_batch_identity
    assert model.prepared.constant_data.sizes["person"] == 2


@pytest.mark.parametrize("family", [fc.GPCM(), fc.GGUM()])
def test_integer_item_categories_roundtrip_attach_summary_and_scoring(tmp_path, family):
    source = fc.ItemInput(
        pd.DataFrame({"item_id": ["i"], "trait": ["a"], "location_side": [1]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1, 2], "item_id": ["i", "i"], "response": [1, 3]}),
            format="item_long",
        ),
        model=family,
        categories=(1, 2, 3),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    prior = fc.prior_predictive(model, draws=6, random_seed=19, store=False)
    # Supply real native prior draws to exercise the result contract, not inference.
    idata = xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
    path = tmp_path / "items.nc"
    idata.to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as stored:
        loaded = stored.load()
    explicit = fc.coef(model, idata=loaded, kind="stats")
    model.attach_samples(loaded)
    pd.testing.assert_frame_equal(explicit, fc.coef(model, kind="stats"))
    point = fc.parameter_point(model)
    scored = fc.score_model(model, source.responses, parameters=point)
    assert scored.prepared.constant_data.m0_category.values.tolist() == [1, 2, 3]
    assert np.isfinite(fc.information(model, store=False).test_information).all()
