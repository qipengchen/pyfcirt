"""Frozen designs, actual generation schedules and final parameter boundaries."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import pyfcirt as fc
from pyfcirt.data.encoding import generated_rows
from pyfcirt.data.prepare import (
    _prepare_generated,
    _prepare_generation,
    _validate_parameters,
)
from tests.data.test_schema import fixed_tirt_design, graded_input, pair_input
from tests.helpers import (
    generated_pair_frame,
    mixed_inputs,
    mixed_model,
    runtime_snapshot,
    workflow_point,
)


def test_canonical_one_pl_has_one_loading_and_many_item_uses_without_inactive_q_entries():
    source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["a", "b"], "B": ["c"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": [101, 101, 101], "item_id": ["a", "b", "c"], "response": [1, 0, 1]}
            ),
            format="item_long",
        ),
        model=fc.DichotomousIRT(kind="1PL"),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    loading = next(layout for layout in prepared.parameters if layout.kind == "loading")
    assert loading.dims == ("loading_id",) and loading.shape == (1,)
    assert loading.free_flat.tolist() == [0] and not loading.fixed_flat.size
    uses = prepared.parameter_uses
    rows = uses.where(uses.kind == "loading", drop=True)
    assert rows.entry_index.values.tolist() == [0, 0, 0]
    assert set(prepared.constant_data.unit_id.values[rows.unit_index.values.astype(int)]) == {
        "a",
        "b",
        "c",
    }
    assert set(rows.trait_index.values.tolist()) == {0, 1}
    assert (rows.entry_index.values >= 0).all()
    assert "loading_id" in prepared.output_coords["loading_parameter_set_index"].dims
    assert "loading_item_index" not in prepared.output_coords
    assert "loading_trait_index" not in prepared.output_coords
    xr.testing.assert_equal(prepared.result_metadata["parameter_uses"], prepared.parameter_uses)


@pytest.mark.parametrize("family", ["MGPPM", "TwoPLRank", "GGUMRank", "TIRTRank"])
@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("task", ["rank", "pick", "mole"])
def test_canonical_ranking_catalogue_retains_supported_sizes_tasks_and_typed_origins(
    family, size, task
):
    from itertools import combinations

    statements = list(range(11, 11 + size))
    design = pd.DataFrame(
        {
            "block_id": ["block"] * size,
            "position": range(1, size + 1),
            "statement_id": statements,
            "trait": ["A" if index % 2 == 0 else "B" for index in range(size)],
            "location_side": [1 if index % 2 == 0 else -1 for index in range(size)],
            "keying": [1] * size,
        }
    )
    pairs = [
        (first, second)
        for first, second in combinations(range(size), 2)
        if task == "rank"
        or task == "pick"
        and first == 0
        or task == "mole"
        and (first == 0 or second == size - 1)
    ]
    answers = pd.DataFrame(
        [(101, "block", statements[first], statements[second], 1) for first, second in pairs],
        columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
    )
    options = (
        {"loading_anchors": {statements[0]: 1.0, statements[1]: 1.1}}
        if family == "TIRTRank"
        else {}
    )
    prepared = fc.PreparedInputs.from_inputs(
        fc.FCInput(
            design,
            fc.ResponseTable(answers, format="comparison_long"),
            model=getattr(fc, family)(**options),
            response_format=task,
        )
    )
    assert len({layout.kind for layout in prepared.parameters}) == len(prepared.parameters)
    assert all("::" not in layout.kind for layout in prepared.parameters)
    loading = next(layout for layout in prepared.parameters if layout.kind == "loading")
    assert loading.shape == (size,)
    assert loading.entry_data.statement_index.values.tolist() == list(range(size))
    assert prepared.constant_data.statement_label.values.tolist() == statements
    assert prepared.constant_data.sizes["obs"] == 1
    coords = prepared.output_coords
    assert coords["obs_person_id"].values.tolist() == [101]
    assert coords["obs_unit_index"].values.tolist() == [0]
    assert coords["obs_block_index"].values.tolist() == [0]
    assert coords["obs_item_index"].values.tolist() == [-1]
    assert coords["obs_model"].values.tolist() == [family]
    assert coords["obs_response_format"].values.tolist() == [task]
    assert coords["obs_pair_index"].values.tolist() == [-1]
    assert coords["obs_input_event_index"].dims == ("obs",)


def test_canonical_loading_catalogue_partitions_heterogeneous_native_priors_once():
    source, items = mixed_inputs()
    source = replace(
        source,
        blocks=(
            fc.BlockSpec(
                "pair", model=fc.MGPPM(priors={"loading_magnitude": pm.HalfNormal.dist(sigma=1)})
            ),
            fc.BlockSpec(
                "7",
                model=fc.TwoPLRank(
                    priors={"loading_magnitude": pm.LogNormal.dist(mu=0.1, sigma=0.2)}
                ),
                response_format="pick",
            ),
            *source.blocks[1:],
        ),
    )
    prepared = fc.PreparedInputs.from_inputs((source, items))
    loading = [layout for layout in prepared.parameters if layout.kind == "loading"]
    assert len(loading) == 1 and loading[0].shape == (11,)
    assert prepared.parameter_layouts["loading"] is loading[0]
    assert tuple(prepared.parameter_layouts) == tuple(layout.kind for layout in prepared.parameters)
    blocks = [block for block in prepared.blocks if block.kind == "loading"]
    assert len(blocks) >= 3
    assert len({type(block.priors["loading_magnitude"].owner.op) for block in blocks}) >= 2
    active = np.concatenate([block.entry_index.ravel() for block in blocks])
    assert set(active[active >= 0]) == set(range(11))
    assert set(prepared.parameter_uses.kind.values) >= {"loading", "statement_intercept"}


@pytest.mark.parametrize(
    "family,link", [("TIRT", "beta_probit"), ("TIRT", "beta_logit"), ("GTUM", "beta_logit")]
)
@pytest.mark.parametrize("size", [2, 3, 4])
def test_canonical_beta_block_has_pair_offsets_and_one_shared_precision(family, link, size):
    from itertools import combinations

    statements = list(range(11, 11 + size))
    design = pd.DataFrame(
        {
            "block_id": ["block"] * size,
            "position": range(1, size + 1),
            "statement_id": statements,
            "trait": ["A" if i % 2 == 0 else "B" for i in range(size)],
            "location_side": [1 if i % 2 == 0 else -1 for i in range(size)],
            "keying": [1] * size,
        }
    )
    pairs = list(combinations(statements, 2))
    answers = pd.DataFrame(
        [(101, "block", first, second, 0.4) for first, second in pairs],
        columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
    )
    options = {"link": link, "priors": {"response_precision": pm.Gamma.dist(alpha=8, beta=0.2)}}
    if family == "TIRT":
        options["loading_anchors"] = {statements[0]: 0.25, statements[1]: 0.3}
    prepared = fc.PreparedInputs.from_inputs(
        fc.FCInput(
            design,
            fc.ResponseTable(answers, format="comparison_long"),
            model=getattr(fc, family)(**options),
            response_format=None,
            scale=fc.UnitIntervalScale(),
        )
    )
    layouts = {layout.kind: layout for layout in prepared.parameters}
    assert layouts["comparison_offset"].shape == (len(pairs),)
    assert layouts["response_precision"].shape == (1,)
    assert "cutpoint" not in layouts
    assert prepared.constant_data.sizes["obs"] == len(pairs)
    np.testing.assert_array_equal(prepared.output_coords["obs_pair_index"], np.arange(len(pairs)))
    assert prepared.output_coords["obs_block_index"].values.tolist() == [0] * len(pairs)
    uses = prepared.parameter_uses.where(
        prepared.parameter_uses.kind == "response_precision", drop=True
    )
    assert np.unique(uses.entry_index.values).tolist() == [0]


@pytest.mark.parametrize("family", ["TIRT", "GTUM"])
@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("category_count", [3, 4, 5])
@pytest.mark.parametrize("reverse_catalog", [False, True])
def test_canonical_graded_block_keeps_pair_and_cutpoint_positions(
    family, size, category_count, reverse_catalog, monkeypatch
):
    from itertools import combinations

    from pyfcirt.data import constraints

    if reverse_catalog:
        # Change order at its owner; encoding must follow, not infer another order.
        monkeypatch.setattr(
            constraints,
            "combinations",
            lambda values, count: reversed(list(combinations(values, count))),
        )
    statements = [11, 37, 83, 109][:size]
    design = pd.DataFrame(
        {
            "block_id": ["block"] * size,
            "position": range(1, size + 1),
            "statement_id": statements,
            "trait": ["A" if i % 2 == 0 else "B" for i in range(size)],
            "location_side": [1 if i % 2 == 0 else -1 for i in range(size)],
            "keying": [1] * size,
        }
    )
    pairs = list(combinations(statements, 2))
    answers = pd.DataFrame(
        [(101, "block", first, second, 1) for first, second in pairs],
        columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
    )
    if reverse_catalog:
        answers[["statement_1", "statement_2"]] = answers[["statement_2", "statement_1"]].to_numpy()
    options = (
        {"loading_anchors": {statements[0]: 1.0, statements[1]: 1.1}} if family == "TIRT" else {}
    )
    prepared = fc.PreparedInputs.from_inputs(
        fc.FCInput(
            design,
            fc.ResponseTable(answers, format="comparison_long"),
            model=getattr(fc, family)(**options),
            response_format=None,
            scale=fc.CategoryScale(tuple(range(category_count))),
        )
    )
    cuts = next(layout for layout in prepared.parameters if layout.kind == "cutpoint")
    assert cuts.shape == (len(pairs) * (category_count - 1),)
    np.testing.assert_array_equal(
        cuts.entry_data.pair_index, np.repeat(np.arange(len(pairs)), category_count - 1)
    )
    np.testing.assert_array_equal(
        cuts.entry_data.cutpoint_position, np.tile(np.arange(category_count - 1), len(pairs))
    )
    assert prepared.constant_data.sizes["obs"] == len(pairs)
    (batch,) = prepared.batches
    assert batch.events.labels == tuple(range(category_count))
    expected_response = category_count - 2 if reverse_catalog else 1
    np.testing.assert_array_equal(batch.y, np.full(len(pairs), expected_response))
    slot = next(slot for slot in batch.slots if slot.kind == "cutpoint")
    pair_index = cuts.entry_data.pair_index.values[slot.entry_index]
    for member in (0, 1):
        np.testing.assert_array_equal(
            prepared.constant_data[f"pair_statement_{member + 1}"].values[pair_index],
            np.broadcast_to(batch.statement[:, member, None], pair_index.shape),
        )
    np.testing.assert_array_equal(
        cuts.entry_data.cutpoint_position.values[slot.entry_index],
        np.broadcast_to(np.arange(category_count - 1), slot.entry_index.shape),
    )


@pytest.fixture(scope="module")
def reference():
    model = mixed_model()
    return model, workflow_point(model)


def test_public_identities_leave_construction_policies_in_metadata(reference):
    model, point = reference
    prepared = model.prepared
    samples = point.expand_dims(chain=[0], draw=[0, 1])
    annotated = prepared.annotate_coordinates(samples)
    for name in samples.data_vars:
        assert annotated[name].variable.identical(samples[name].variable)
    assert {name for name in annotated.coords if name.startswith("loading_")} == {
        "loading_id",
        "loading_parameter_set_index",
        "loading_statement_index",
        "loading_item_index",
        "loading_trait_index",
    }
    assert not any(name.startswith("covariance_") for name in annotated.coords)
    assert annotated.covariance.dims == ("chain", "draw", "trait", "trait_aux")
    entries = prepared.result_metadata["parameter_data/loading"]
    assert {"binding_id", "local_position", "domain", "fixed_value", "ordered"} <= set(entries)
    for kind in (layout.kind for layout in prepared.parameters):
        assert not any(
            f"{kind}_{field}" in annotated.coords
            for field in ("binding_id", "local_position", "fixed_for_scoring", "upper_limit")
        )


def test_presentation_fields_are_independent_of_frozen_measurement_identity():
    prepared = fc.PreparedInputs.from_inputs(mixed_inputs())
    original = fc.PreparedInputs(prepared.data)
    published = prepared.result_metadata
    published["design"]["display_label"] = "Example analysis"
    published["categories"]["display_label"] = "Category legend"
    assert prepared.calibration_identity == original.calibration_identity
    assert prepared.scoring_batch_identity == original.scoring_batch_identity


def test_annotation_selects_parameter_and_observation_identities_on_their_output_axes(reference):
    model, point = reference
    prepared = model.prepared
    full = point[["loading"]].assign(
        response=("obs", np.arange(prepared.constant_data.sizes["obs"]))
    )
    full = full.assign_coords(obs=prepared.constant_data.obs)
    expected = prepared.annotate_coordinates(full).isel(loading_id=[4, 1], obs=[3, 0])
    subset = full.isel(loading_id=[4, 1], obs=[3, 0])
    xr.testing.assert_identical(prepared.annotate_coordinates(subset), expected)
    with pytest.raises(ValueError, match="missing.*loading_id"):
        prepared.annotate_coordinates(subset.drop_vars("loading_id"))
    bad = subset.assign_coords(obs=["unknown", "labels"])
    with pytest.raises(ValueError, match="frozen design"):
        prepared.annotate_coordinates(bad)


@pytest.mark.parametrize("operation", ["scoring", "simulation", "information"])
def test_explicit_point_conflicting_identities_fail_at_the_data_boundary(reference, operation):
    model, point = reference
    before = model.idata.copy(deep=True)
    supplied = model.prepared.annotate_coordinates(point)
    statement = supplied.loading_statement_index.values.copy()
    statement[0] += 1
    supplied = supplied.assign_coords(loading_statement_index=("loading_id", statement))
    with pytest.raises(ValueError, match="loading_statement_index: coordinate conflicts"):
        if operation == "scoring":
            responses = {
                name: data.responses
                for name, data in zip(("fc", "items"), mixed_inputs(), strict=True)
            }
            fc.score_model(model, responses, parameters=supplied.drop_dims("person"))
        elif operation == "simulation":
            fc.simulate(model, source="fixed", parameters=supplied, random_seed=37)
        else:
            fc.information(model, parameters=supplied, store=False)
    xr.testing.assert_identical(model.idata, before)


def test_reordered_point_identities_align_by_primary_labels_before_validation(reference):
    model, point = reference
    supplied = model.prepared.annotate_coordinates(point[["loading"]]).isel(
        loading_id=slice(None, None, -1)
    )
    checked = _validate_parameters(model.prepared, supplied)
    xr.testing.assert_identical(checked, point[["loading"]])


def test_scoring_rejects_unknown_primary_point_labels_before_rebinding(reference):
    model, point = reference
    supplied = point.drop_dims("person").assign_coords(loading_id=np.arange(100, 111))
    responses = {
        name: data.responses for name, data in zip(("fc", "items"), mixed_inputs(), strict=True)
    }
    with pytest.raises(ValueError, match="labels must match the frozen design"):
        fc.score_model(model, responses, parameters=supplied)


@pytest.mark.parametrize("model", [fc.TIRT(), fc.TIRTRank()])
@pytest.mark.parametrize("connected", [False, True])
def test_unobserved_statement_warning_preserves_the_design_and_references(model, connected):
    import warnings

    design = pd.DataFrame(
        dict(
            block_id=["b"] * 3,
            position=[1, 2, 3],
            statement_id=["S1", "S2", "S3"],
            trait=["A", "B", "C"],
            keying=[1] * 3,
        )
    )
    pairs = [("S2", "S3")] + ([("S1", "S2")] if connected else [])
    answers = pd.DataFrame(
        [("p", "b", first, second, 1) for first, second in pairs],
        columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        state = fc.PreparedInputs.from_inputs(
            fc.FCInput(design, fc.ResponseTable(answers, format="comparison_long"), model=model)
        )
    notices = [w for w in recorded if "Statements without observed comparisons" in str(w.message)]
    assert bool(notices) == (not connected)
    if notices:
        assert "['S1']" in str(notices[0].message)
        assert "Remove unused entries" in str(notices[0].message)
    layout = next(layout for layout in state.parameters if layout.kind == "residual_variance")
    np.testing.assert_array_equal(layout.fixed_flat, [0])
    np.testing.assert_array_equal(layout.fixed, [0.5, np.nan, np.nan])


def test_explicit_references_do_not_hide_an_unobserved_statement():
    import warnings

    design = pd.DataFrame(
        dict(
            block_id=["b"] * 3,
            position=[1, 2, 3],
            statement_id=["S1", "S2", "S3"],
            trait=["A", "B", "C"],
            keying=[1] * 3,
            fix_var=[np.nan, 0.5, np.nan],
        )
    )
    answers = pd.DataFrame(
        dict(person_id=["p"], block_id=["b"], statement_1=["S2"], statement_2=["S3"], response=[1])
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fc.PreparedInputs.from_inputs(
            fc.FCInput(
                design,
                fc.ResponseTable(answers, format="comparison_long"),
                model=fc.TIRTRank(reference_statements=("S2",)),
            )
        )
    assert any(
        "Statements without observed comparisons: ['S1']" in str(w.message) for w in recorded
    )


def test_a_statement_observed_in_another_block_is_not_reported_for_deletion():
    import warnings

    design = pd.DataFrame(
        dict(
            block_id=["b1"] * 3 + ["b2"] * 3,
            position=[1, 2, 3] * 2,
            statement_id=["S1", "S2", "S3", "S1", "S4", "S5"],
            trait=["A", "B", "C", "A", "B", "C"],
            keying=[1] * 6,
        )
    )
    answers = pd.DataFrame(
        [("p", "b1", "S2", "S3", 1), ("p", "b2", "S1", "S4", 1), ("p", "b2", "S4", "S5", 1)],
        columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fc.PreparedInputs.from_inputs(
            fc.FCInput(
                design,
                fc.ResponseTable(answers, format="comparison_long"),
                model=fc.TIRTRank(),
            )
        )
    assert not any("Statements without observed comparisons" in str(w.message) for w in recorded)


@pytest.mark.parametrize(
    "people,error", [("p", TypeError), ([], ValueError), ([1, 1], ValueError), ([True], ValueError)]
)
def test_generation_person_axis_requires_unique_explicit_labels(reference, people, error):
    with pytest.raises(error):
        fc.generation_schedule(reference[0], people)


def test_complete_mixed_schedule_roundtrip_preserves_frozen_design(reference):
    model, point = reference
    before = model.idata.copy(deep=True)
    schedule = fc.generation_schedule(model, [7, 8]).isel(schedule=slice(None, None, -1))
    parameters = _validate_parameters(model.prepared, point.drop_dims("person"), scoring=True)
    state = _prepare_generation(model.prepared, schedule, fixed_parameters=parameters)
    xr.testing.assert_identical(state.data["fixed_parameters"].to_dataset(), parameters)
    assert state.data.attrs["purpose"] == "generate"
    assert state.data["constant_data"].person.values.tolist() == [7, 8]
    assert state.data["constant_data"].sizes["obs"] == 10
    assert state.data["observed_data"].data_vars.keys() >= {"response_status"}
    values = xr.Dataset(
        {
            b["response_name"]: (
                b["obs_dim"],
                np.zeros(state.data["constant_data"].sizes[b["obs_dim"]], dtype=int),
            )
            for b in state.data["constant_data"].attrs["batches"]
        }
    )
    result = _prepare_generated(state, values)
    assert result.data["constant_data"].sizes["obs"] == 10
    for prepared in (state, result):
        observed = prepared.observed_data
        parents = observed.response_administration_index.values
        np.testing.assert_array_equal(
            observed.administration_person.values[parents], observed.response_person
        )
        np.testing.assert_array_equal(
            observed.administration_unit.values[parents], observed.response_unit
        )
        constant = prepared.constant_data
        np.testing.assert_array_equal(
            parents[constant.observation_source_response_row_index.values],
            constant.obs_input_event_index.values[constant.observation_source_obs_index.values],
        )
    assert "parameters" not in result.data
    xr.testing.assert_identical(model.idata, before)
    item = next(b for b in state.constant_data.attrs["batches"] if b["channel"] == "items")
    with pytest.raises(ValueError, match="Generated values are missing"):
        _prepare_generated(state, values.drop_vars(item["response_name"]))


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("same_people", [False, True])
def test_generation_schedule_detaches_borrowed_state(regression, same_people):
    model = mixed_model()
    if regression:
        model.add_regressions(
            "theta ~ x", person_data=pd.DataFrame({"person_id": [1, 2], "x": [1.0, 3.0]})
        )
    constant = model.prepared.constant_data
    before = runtime_snapshot(model.prepared)
    people = [1, 2] if same_people else [7, 8]
    person_data = (
        pd.DataFrame({"person_id": people, "x": [5.0, 9.0]})
        if regression and not same_people
        else None
    )
    schedule = fc.generation_schedule(model, people, person_data=person_data)
    recipe = constant.attrs["population"]["recipe"]
    assert schedule.attrs["population_recipe"] is not recipe
    assert schedule.attrs["population_recipe"]["groups"] is not recipe["groups"]
    for name in ("unit", "unit_id", "unit_channel", "theta_predictor", "person_group"):
        assert not np.shares_memory(schedule[name].values, constant[name].values)
    if regression:
        assert not np.shares_memory(
            schedule.predictors.values, model.prepared.data["covariates/r0"].X.values
        )
        schedule.predictors.values[:] = 99.0
    schedule.person_group.values[:] = -1
    schedule.unit_id.values[0] = "changed"
    schedule.unit_channel.values[0] = "changed"
    schedule.attrs["population_recipe"]["groups"].append("changed")
    xr.testing.assert_identical(model.prepared.data, before)

    drifted = fc.generation_schedule(model, people, person_data=person_data)
    drifted.attrs["population_recipe"]["groups"].append("changed")
    with pytest.raises(ValueError, match="frozen population recipe"):
        _prepare_generation(model.prepared, drifted)
    xr.testing.assert_identical(model.prepared.data, before)


def test_mixed_native_and_comparison_schedules_preserve_full_event_support():
    source = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": list("acegik"), "B": list("bdfhjl")},
            blocks={
                "rank": list("abc"),
                "pick": list("def"),
                "mole": list("ghij"),
                "pair": list("kl"),
            },
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p"] * 4,
                    "block_id": ["rank", "pick", "mole", "pair"],
                    "response": [[2, 1, 3], [1, None, None], [1, None, None, 4], 0],
                }
            ),
            format="block_long",
        ),
        blocks=(
            fc.BlockSpec("pick", response_format="pick"),
            fc.BlockSpec("mole", response_format="mole"),
            fc.BlockSpec(
                "pair",
                model=fc.TIRT(loading_anchors={"k": 1, "l": 1}),
                response_format=None,
                scale=fc.CategoryScale((0, 1)),
            ),
        ),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    state = _prepare_generation(model.prepared, fc.generation_schedule(model, ["new", "new2"]))
    constant = state.data["constant_data"]
    expected_support = {"rank": 6, "pick": 3, "mole": 12, "pair": 2}
    values = {}
    for batch in constant.attrs["batches"]:
        unit = constant.unit_id.values[batch["unit_index"]]
        assert len(batch["events"]["labels"]) == expected_support[unit]
        if unit != "pair":
            assert "pairs" not in batch["events"]
        values[batch["response_name"]] = (batch["obs_dim"], np.zeros(2, dtype=int))
    decoded = generated_pair_frame(state, xr.Dataset(values)).rename(
        columns={"unit_id": "block_id"}
    )
    expected = {
        ("rank", "a", "b", 1),
        ("rank", "a", "c", 1),
        ("rank", "b", "c", 1),
        ("pick", "d", "e", 1),
        ("pick", "d", "f", 1),
        ("mole", "g", "h", 1),
        ("mole", "g", "i", 1),
        ("mole", "g", "j", 1),
        ("mole", "h", "i", 0),
        ("mole", "h", "j", 0),
        ("pair", "k", "l", 0),
    }
    for person in ("new", "new2"):
        actual = decoded.loc[
            decoded.person_id == person, ["block_id", "statement_1", "statement_2", "response"]
        ]
        assert set(actual.itertuples(index=False, name=None)) == expected


@pytest.mark.parametrize(
    "problem,match",
    [
        ("schema", "current generation_schedule"),
        ("missing_status", "index arrays"),
        ("status_dimension", "row dimension"),
        ("units", "unit coordinates"),
        ("missing_pair", "both statement"),
        ("pair_dimension", "statement indices"),
        ("person_index", "invalid person"),
        ("unit_index", "invalid person"),
        ("pair_type", "must be integers"),
        ("item_pair", "Item schedules"),
        ("same_statement", "distinct unit members"),
        ("duplicate", "duplicate events"),
        ("recipe", "frozen population recipe"),
        ("predictors", "invalid frozen population"),
        ("status", "status must be"),
    ],
)
def test_generation_schedule_rejects_structural_drift(reference, problem, match):
    model, _ = reference
    schedule = fc.generation_schedule(model, [7, 8]).copy(deep=True)
    if problem == "schema":
        schedule.attrs["schema_version"] = 0
    elif problem == "missing_status":
        schedule = schedule.drop_vars("schedule_status")
    elif problem == "status_dimension":
        schedule["schedule_status"] = "other", schedule.schedule_status.values
    elif problem == "units":
        schedule = schedule.assign_coords(unit_id=("unit", ["bad"] * 5))
    elif problem == "missing_pair":
        schedule = schedule.drop_vars("schedule_statement_1")
    elif problem == "pair_dimension":
        schedule["schedule_statement_1"] = "other", schedule.schedule_statement_1.values
    elif problem in {"person_index", "unit_index"}:
        schedule[f"schedule_{problem.removesuffix('_index')}"].values[0] = 99
    elif problem == "pair_type":
        schedule["schedule_statement_1"] = schedule.schedule_statement_1.astype(float)
    elif problem == "item_pair":
        schedule.schedule_statement_1.values[3] = 0
        schedule.schedule_statement_2.values[3] = 1
    elif problem == "same_statement":
        schedule.schedule_statement_1.values[0] = 0
        schedule.schedule_statement_2.values[0] = 0
    elif problem == "duplicate":
        schedule = schedule.isel(schedule=[0, 0, *range(1, 10)])
    elif problem == "recipe":
        schedule.attrs["population_recipe"] = {}
    elif problem == "predictors":
        schedule.person_group.values[0] = -1
    else:
        schedule.schedule_status.values[0] = "answered"
    with pytest.raises(ValueError, match=match):
        _prepare_generation(model.prepared, schedule)


@pytest.mark.parametrize(
    "administration",
    [
        3,
        {"other": pd.DataFrame()},
        pd.DataFrame({"channel": ["other"]}),
        pd.DataFrame({"person_id": ["p"]}),
    ],
)
def test_scoring_administration_requires_known_channels(reference, administration):
    model, point = reference
    with pytest.raises((TypeError, ValueError)):
        fc.score_model(
            model,
            {"items": mixed_inputs()[1].responses},
            parameters=point.drop_dims("person"),
            administration=administration,
        )


@pytest.mark.parametrize(
    "responses", [3, {"other": 1}, {"items": pd.DataFrame()}, {}, mixed_inputs()[0].responses]
)
def test_new_responses_require_declared_channels_and_people(reference, responses):
    model, point = reference
    with pytest.raises((TypeError, ValueError)):
        fc.score_model(model, responses, parameters=point.drop_dims("person"))


def test_scoring_one_channel_marks_the_other_not_administered(reference):
    model, point = reference
    answers = mixed_inputs()[1].responses
    answers = replace(
        answers,
        frame=answers.frame.assign(person_id=answers.frame.person_id.map({1: "203", 2: "101"})),
    )
    admin = answers.frame[["person_id", "item_id"]].assign(status="answered", channel="items")
    design = model.prepared.constant_data
    item_units = np.flatnonzero(design.unit_channel.values == "items")
    units = item_units[design.get_index("item").get_indexer(admin.item_id)]
    people = pd.Index(["203", "101"], dtype=object)
    indexed = xr.Dataset(
        {
            "administration_person": ("administration", people.get_indexer(admin.person_id)),
            "administration_unit": ("administration", units),
            "administration_status": ("administration", admin.status.to_numpy()),
        },
        coords={"person": people.to_numpy()},
    )
    scored, indexed_scored = (
        fc.score_model(
            model, {"items": answers}, parameters=point.drop_dims("person"), administration=rows
        )
        for rows in (admin, indexed)
    )
    observed = scored.prepared.observed_data
    xr.testing.assert_identical(indexed_scored.prepared.observed_data, observed)
    assert indexed_scored.prepared.scoring_batch_identity == scored.prepared.scoring_batch_identity
    assert indexed_scored.prepared.constant_data.person.values.tolist() == ["203", "101"]
    assert (observed.administration_status.values == "not_administered").sum() == 6
    assert indexed_scored.prepared.constant_data.sizes["obs"] == 3
    block = design.block.values[0]
    missing = pd.DataFrame({"person_id": ["203"], "block_id": [block], "status": ["missing"]})
    labelled = {"items": admin.drop(columns="channel"), "fc": missing}
    scored = fc.score_model(
        model, {"items": answers}, parameters=point.drop_dims("person"), administration=labelled
    )
    statuses = scored.prepared.observed_data.administration_status.values
    assert (statuses == "missing").sum() == 1
    assert (statuses == "not_administered").sum() == 5
    with pytest.raises(ValueError, match="has no response/schedule"):
        fc.score_model(
            model,
            {"items": answers},
            parameters=point.drop_dims("person"),
            administration=labelled | {"fc": missing.assign(status="answered")},
        )
    with pytest.raises(ValueError, match="absent from administration"):
        fc.score_model(
            model,
            {"fc": mixed_inputs()[0].responses, "items": answers},
            parameters=point.drop_dims("person"),
            administration=indexed,
        )


@pytest.mark.parametrize(
    "problem",
    [
        "dims",
        "missing_coords",
        "duplicate_coords",
        "short_coords",
        "float_coords",
        "nonreal",
        "asymmetric",
    ],
)
def test_final_parameter_arrays_require_exact_labelled_domains(reference, problem):
    model, point = reference
    name = "covariance" if problem == "asymmetric" else "loading"
    value = point[name].copy(deep=True)
    if problem == "dims":
        value = value.rename(loading_id="wrong")
    elif problem == "missing_coords":
        value = value.drop_vars("loading_id")
    elif problem == "duplicate_coords":
        value = value.assign_coords(loading_id=["a"] * 11)
    elif problem == "short_coords":
        value = value.isel(loading_id=slice(1, None))
    elif problem == "float_coords":
        value = value.assign_coords(loading_id=value.loading_id.astype(float))
    elif problem == "nonreal":
        value = value.astype(complex)
    else:
        value.values[0, 1] = 0.1
    with pytest.raises(ValueError):
        _validate_parameters(model.prepared, value.to_dataset(name=name))


def test_final_parameter_axes_can_be_reordered_and_transposed(reference):
    model, point = reference
    value = (
        point[["covariance"]].isel(trait=[1, 0], trait_aux=[1, 0]).transpose("trait_aux", "trait")
    )
    checked = _validate_parameters(model.prepared, value)
    xr.testing.assert_identical(checked[["covariance"]], point[["covariance"]])
    with pytest.raises(TypeError):
        fc.PreparedInputs.from_inputs(mixed_inputs(), population={})


@pytest.mark.parametrize(
    "model",
    [
        fc.MGPPM(reference_statements=("unknown",)),
        fc.MGPPM(reference_statements=("a", "b")),
        fc.TIRT(loading_anchors={"unknown": 1.0}),
        fc.TIRT(loading_anchors={"a": -1.0}),
        fc.TIRT(loading_anchors={"a": float("inf")}),
    ],
)
def test_identification_declarations_must_refer_to_valid_statements(model):
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(pair_input(model=model))


def test_same_trait_tirt_anchor_and_theta_residual_scale_rules_are_explicit():
    source = pair_input(model=fc.TIRT())
    source = replace(source, design={"traits": {"A": ["a", "b"]}, "blocks": {"pair": ["a", "b"]}})
    prepared = fc.PreparedInputs.from_inputs(source)
    layouts = {layout.kind: layout for layout in prepared.parameters}
    np.testing.assert_array_equal(layouts["loading"].fixed, [1.0, np.nan])
    np.testing.assert_array_equal(layouts["residual_variance"].fixed, [0.5, 0.5])
    for model in (fc.MGPPM(), fc.TIRTRank()):
        with pytest.raises(ValueError, match="loading anchor"):
            fc.PreparedInputs.from_inputs(replace(source, model=model))
    fixed = fc.PreparedInputs.from_inputs(replace(source, model=fc.TIRT(residuals="fixed")))
    assert next(
        layout for layout in fixed.parameters if layout.kind == "residual_variance"
    ).fixed.tolist() == [0.5, 0.5]


def test_fixed_templates_keep_final_coordinates_and_readonly_index_views():
    prepared = fc.PreparedInputs.from_inputs(mixed_inputs())
    constant = prepared.constant_data
    for layout in prepared.parameters:
        template = constant[f"{layout.kind}_fixed_value"]
        assert template.dims == layout.dims
        assert template.shape == layout.shape
        np.testing.assert_array_equal(layout.free_flat, np.flatnonzero(np.isnan(template)))
        np.testing.assert_array_equal(layout.fixed_flat, np.flatnonzero(~np.isnan(template)))
        np.testing.assert_array_equal(
            layout.fixed_value, template.values.ravel()[layout.fixed_flat]
        )
        assert np.shares_memory(layout.fixed, template.values)
        assert not layout.fixed.flags.writeable
        assert not layout.fixed_value.flags.writeable
        assert layout.fixed_source.shape == layout.fixed_flat.shape
        assert np.all(layout.fixed_source != "")
        assert not layout.fixed_source.flags.writeable


def test_mixed_binary_graph_uses_canonical_bounds_without_projecting_fixed_views(monkeypatch):
    items = ["a", "b", "c", "d"]
    source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": items}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1] * 4, "item_id": items, "response": [0, 1, 1, 0]}),
            format="item_long",
        ),
        model=fc.DichotomousIRT(kind="2PL"),
        items=[fc.ItemOverride(item, model=fc.DichotomousIRT(kind="4PL")) for item in ("b", "d")],
    )
    model = fc.Model(source)

    def unexpected_projection(*args):
        pytest.fail("Native construction projected a local fixed-value view")

    with monkeypatch.context() as patch:
        patch.setattr("pyfcirt.data.coordinates._gather_entries", unexpected_projection)
        model.build()

    lower, upper = (
        model.prepared.parameter_layouts[name].entry_data
        for name in ("lower_asymptote", "upper_asymptote")
    )
    blocks = model.prepared.blocks_by_kind["upper_asymptote"]
    (block,) = (block for block in blocks if len(block.free_flat))
    lower_kind, indices, limit = block.bounds
    assert (lower_kind, limit) == ("lower_asymptote", 1.0)
    np.testing.assert_array_equal(
        lower.item_index.values[indices], upper.item_index.values[block.entry_index]
    )
    np.testing.assert_array_equal(lower.item_index.values[indices], [1, 3])
    assert not indices.flags.writeable


@pytest.mark.parametrize(
    "spec",
    [
        *(fc.DichotomousIRT(kind=kind) for kind in ("1PL", "2PL", "3PL", "4PL")),
        fc.GPCM(),
        fc.GGUM(),
    ],
)
def test_item_fixed_provenance_distinguishes_design_structure_and_model(spec):
    design = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "trait": ["A", "B"],
            "location_side": [1, -1],
            "fix_loading": [0.7, np.nan],
        }
    )
    prepared = fc.PreparedInputs.from_inputs(
        fc.ItemInput(
            design,
            fc.ResponseTable(
                pd.DataFrame({"person_id": [1, 1], "item_id": ["a", "b"], "response": [0, 1]}),
                format="item_long",
            ),
            model=spec,
            categories=(0, 1) if isinstance(spec, fc.DichotomousIRT) else (0, 1, 2),
        )
    )
    for layout in prepared.parameters:
        assert layout.fixed_source.shape == layout.fixed_flat.shape
        assert np.all(layout.fixed_source != "")
        assert "inactive" not in layout.fixed_source
        if layout.kind == "loading":
            sources = dict(zip(layout.fixed_flat, layout.fixed_source, strict=True))
            assert sources[0] == "design"
            assert layout.shape == (
                (1,) if isinstance(spec, fc.DichotomousIRT) and spec.kind == "1PL" else (2,)
            )
        elif layout.kind == "location":
            assert not layout.fixed_source.size
        elif layout.kind in {"lower_asymptote", "upper_asymptote"}:
            assert np.all(layout.fixed_source == "model")
        elif layout.kind in {"category_intercept", "category_threshold"}:
            assert layout.fixed_source.tolist() == ["reference", "reference"]


@pytest.mark.parametrize(
    "source",
    [
        replace(graded_input("equal"), model=fc.TwoPLRank()),
        replace(graded_input("equal"), model=fc.GGUMRank()),
        replace(graded_input("equal"), model=fc.TIRTRank()),
        replace(graded_input("equal"), model=fc.TIRT(link="beta_logit")),
        replace(graded_input(0.4, beta=True), model=fc.GTUM()),
        replace(
            pair_input(),
            design=fixed_tirt_design(),
            model=fc.MGPPM(),
            responses=fc.ResponseTable(
                pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "response": [[1, 2]]}),
                format="block_long",
            ),
        ),
    ],
)
def test_model_support_is_checked_even_without_graph_construction(source):
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(source)


@pytest.mark.parametrize(
    "source,key,values",
    [
        (
            replace(graded_input("equal"), model=fc.TIRT(loading_anchors={"a": 1.0, "b": 1.2})),
            "cutpoints",
            [[0.2, 0.1]],
        ),
        (
            replace(
                graded_input(0.4, beta=True),
                design="[traits]\nA~ a\nB~ b\n[blocks]\ngraded ~ a b\nside +~ a\nside -~ b",
            ),
            "response_precision",
            0.0,
        ),
        (
            replace(
                pair_input(),
                design="[traits]\nA~ a c\nB~ b\n[blocks]\npair ~ a b c",
                model=fc.TIRT(),
                responses=fc.ResponseTable(
                    pd.DataFrame(
                        {"person_id": ["p"], "block_id": ["pair"], "response": [[1, 2, 3]]}
                    ),
                    format="block_long",
                ),
            ),
            "residual_variance",
            [0.5, -0.1, 0.5],
        ),
        (
            replace(
                mixed_inputs()[1],
                items=(),
                design={"traits": {"A": ["pair"]}},
                model=fc.DichotomousIRT(kind="4PL"),
                responses=replace(
                    mixed_inputs()[1].responses, frame=mixed_inputs()[1].responses.frame.iloc[:2]
                ),
            ),
            "lower_asymptote",
            [-0.1],
        ),
        (
            replace(
                mixed_inputs()[1],
                items=(),
                design={"traits": {"A": ["pair"]}},
                model=fc.DichotomousIRT(kind="4PL"),
                responses=replace(
                    mixed_inputs()[1].responses, frame=mixed_inputs()[1].responses.frame.iloc[:2]
                ),
            ),
            "upper_asymptote",
            [1.1],
        ),
    ],
)
def test_fixed_parameter_mathematical_domains_are_enforced(source, key, values):
    prepared = fc.PreparedInputs.from_inputs(source)
    constant = prepared.constant_data
    kind = "cutpoint" if key == "cutpoints" else key
    layout = next(layout for layout in prepared.parameters if layout.kind == kind)
    dims = layout.dims
    value = xr.DataArray(
        np.asarray(values).reshape(layout.shape),
        dims=dims,
        coords={dim: constant[dim] for dim in dims},
    )
    with pytest.raises(ValueError):
        _validate_parameters(prepared, value.to_dataset(name=layout.kind))


def test_final_loading_values_retain_a_user_prior_that_relaxes_orientation():
    source = fc.load_example(
        "single_binary",
        model=fc.DichotomousIRT(priors={"loading_magnitude": pm.Normal.dist()}),
        n_persons=1,
    ).data
    with pytest.warns(UserWarning, match="relaxes the default orientation"):
        prepared = fc.PreparedInputs.from_inputs(source)
    layout = next(p for p in prepared.parameters if p.kind == "loading")
    values = np.where(np.isnan(layout.fixed), -0.7, layout.fixed)
    point = xr.DataArray(
        values, dims=layout.dims, coords={dim: prepared.coords[dim] for dim in layout.dims}
    ).to_dataset(name=layout.kind)
    xr.testing.assert_equal(_validate_parameters(prepared, point)[[layout.kind]], point)


def test_unfolding_orientation_and_zero_baseline_are_enforced_with_free_steps():
    source = fc.load_example("single_unfolding", model="GGUM").data
    prepared = fc.PreparedInputs.from_inputs(source)
    constant = prepared.constant_data
    location = next(layout for layout in prepared.parameters if layout.kind == "location")
    threshold = next(
        layout for layout in prepared.parameters if layout.kind == "category_threshold"
    )
    first_category_entries = np.flatnonzero(threshold.entry_data.item_index.values == 0)
    valid_thresholds = np.where(np.isnan(threshold.fixed), 0.2, threshold.fixed)
    for layout, values in (
        (location, [[-1.0], [-1.0], [1.0], [-1.0], [1.0]]),
        (location, [[1.0], [1.0], [1.0], [-1.0], [1.0]]),
        (threshold, [0.1, 0.3, 0.5]),
        (threshold, [0.0, float("nan"), -0.4]),
    ):
        dims = layout.dims
        actual = np.asarray(values).reshape(-1)
        if layout.kind == "category_threshold":
            actual = valid_thresholds.copy()
            actual[first_category_entries] = values
        value = xr.DataArray(
            actual.reshape(layout.shape),
            dims=dims,
            coords={dim: constant[dim] for dim in dims},
        )
        with pytest.raises(ValueError):
            _validate_parameters(prepared, value.to_dataset(name=layout.kind))
    for values in ([0.0, 0.3, 0.5], [0.0, -0.2, -0.4], [0.0, 0.3, -0.4]):
        actual = valid_thresholds.copy()
        actual[first_category_entries] = values
        point = xr.DataArray(
            actual, dims=threshold.dims, coords={dim: constant[dim] for dim in threshold.dims}
        ).to_dataset(name=threshold.kind)
        xr.testing.assert_identical(_validate_parameters(prepared, point)[[threshold.kind]], point)


def test_mixed_rank_and_graded_shared_statements_require_compatible_priors():
    source = fc.FCInput(
        "[traits]\nA~ a c\nB~ b\n[blocks]\nrank ~ a b\ngraded ~ b c",
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p", "p"],
                    "block_id": ["rank", "graded"],
                    "response": [[1, 2], "equal"],
                }
            ),
            format="block_long",
        ),
        blocks=(
            fc.BlockSpec(
                "graded", response_format=None, scale=fc.CategoryScale(("low", "equal", "high"))
            ),
        ),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    (intercept,) = prepared.blocks_by_kind["statement_intercept"]
    assert intercept.prior_role_index.tolist() == [
        "ranking_intercept",
        "graded_intercept",
    ]
    with pytest.raises(ValueError, match="matching intercept priors"):
        fc.PreparedInputs.from_inputs(
            replace(
                source,
                model=fc.MGPPM(priors={"ranking_intercept": pm.Normal.dist(mu=0, sigma=1)}),
            )
        )


@pytest.mark.parametrize("model", [fc.DichotomousIRT(kind="1PL"), fc.DichotomousIRT(kind="2PL")])
def test_group_mean_directions_are_observed_by_item_measurements(model):
    source = fc.ItemInput(
        {"traits": {"A": ["a"], "B": ["b"]}},
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p", "q"], "item_id": ["a", "b"], "response": [1, 0]}),
            format="item_long",
        ),
        model=model,
    )
    state = fc.PreparedInputs.from_inputs(
        source,
        population=fc.PopulationSpec(group="cohort", reference_group="first"),
        person_data=pd.DataFrame({"person_id": ["p", "q"], "cohort": ["first", "second"]}),
    )
    assert state.data["constant_data"].population_group.values.tolist() == ["first", "second"]


@pytest.mark.parametrize("traits", [1, 2])
@pytest.mark.parametrize("grouped,scale", [(False, "unit"), (True, "unit"), (True, "group")])
def test_population_templates_keep_reference_labels_and_scoring_conditions(traits, grouped, scale):
    source = fc.ItemInput(
        {"traits": {f"T{i}": [f"I{i}"] for i in range(traits)}},
        fc.ResponseTable(
            pd.DataFrame(
                [(person, f"I{i}", i % 2) for person in ("p", "q") for i in range(traits)],
                columns=["person_id", "item_id", "response"],
            ),
            format="item_long",
        ),
    )
    prepared = fc.PreparedInputs.from_inputs(
        source,
        population=fc.PopulationSpec(
            group="site" if grouped else None,
            reference_group="reference" if grouped else None,
            theta_residual_scale=scale,
        ),
        person_data=pd.DataFrame({"person_id": ["p", "q"], "site": ["other", "reference"]}),
    )
    layouts = {layout.kind: layout for layout in prepared.parameters}
    matrix_shape = (2, traits, traits) if grouped else (traits, traits)
    correlation = np.full(matrix_shape, np.nan)
    covariance = np.full(matrix_shape, np.nan)
    diagonal = np.arange(traits)
    correlation[..., diagonal, diagonal] = 1.0
    if grouped and scale == "group":
        covariance[1, diagonal, diagonal] = 1.0
    else:
        covariance[..., diagonal, diagonal] = 1.0
    means = np.zeros((2 if grouped else 1, traits))
    sd = np.ones_like(means)
    if grouped:
        means[0] = np.nan
        if scale == "group":
            sd[0] = np.nan
    for key, expected in (
        ("trait_correlation", correlation),
        ("population_mean", means),
        ("population_sd", sd),
        ("covariance", covariance),
    ):
        np.testing.assert_array_equal(layouts[key].fixed, expected)
        np.testing.assert_array_equal(layouts[key].fixed_source, "identification")
        assert layouts[key].fixed_for_scoring == (key in {"population_mean", "covariance"})
    assert (
        next(block for block in prepared.population.blocks if block.kind == "covariance").prior_role
        is None
    )
    assert set(prepared.population.specification.priors) == (
        ({"trait_correlation"} if traits > 1 else set())
        | ({"population_mean"} if grouped else set())
        | ({"population_sd"} if grouped and scale == "group" else set())
    )


def test_binary_category_count_is_checked_for_an_unobserved_item():
    source = fc.ItemInput(
        {"traits": {"A": ["a"]}},
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p"], "item_id": ["a"], "response": [None]}),
            format="item_long",
        ),
        categories=(0, 1, 2),
    )
    with pytest.raises(ValueError, match="complete item categories"):
        fc.PreparedInputs.from_inputs(source)


def test_one_pl_estimates_one_shared_loading_and_preserves_inactive_zeros():
    from pytensor.graph.traversal import ancestors

    design = pd.DataFrame({"item_id": ["a", "b", "c"], "trait": ["A", "A", "B"]})
    answers = pd.DataFrame(
        {"person_id": ["p"] * 3, "item_id": ["a", "b", "c"], "response": [1, 0, 1]}
    )
    inputs = fc.ItemInput(
        design,
        fc.ResponseTable(answers, format="item_long"),
        model=fc.DichotomousIRT(kind="1PL"),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs))
    model.set_priors({"items.loading_magnitude": pm.LogNormal.dist(mu=0.1, sigma=0.2)})
    native = model.build()
    raw = [rv for rv in native.free_RVs if rv in ancestors([native["loading"]])]
    assert len(raw) == 1 and raw[0].type.shape == ()
    draws = pm.draw(native["loading"], draws=4, random_seed=24)
    assert draws.shape == (4, 1) and (draws > 0).all()
    assert np.std(draws[:, 0]) > 0
    for batch in model.prepared.batches:
        slot = next(slot for slot in batch.slots if slot.kind == "loading")
        assert slot.entry_index.ravel().tolist() == [0]
    (block,) = model.prepared.blocks_by_kind["loading"]
    assert block.entry_index.ravel().tolist().count(-1) == 3
    assert block.entry_index.ravel().tolist().count(0) == 3
    invalid = xr.Dataset(
        {"loading": ("loading_id", [0.8, 0.9])},
        coords={"loading_id": [0, 1]},
    )
    with pytest.raises(ValueError, match="coordinate"):
        _validate_parameters(model.prepared, invalid)
    for common_value in (0.0, -0.5, 0.8):
        point = xr.Dataset({"loading": ("loading_id", [common_value])}, coords={"loading_id": [0]})
        validated = _validate_parameters(model.prepared, point)
        xr.testing.assert_equal(validated[["loading"]], point)
    with pytest.raises(ValueError, match="common loading"):
        model.add_regressions(
            "items.loading_magnitude ~ feature", item_data=design.assign(feature=[0, 1, 2])
        )
    fixed = design.assign(fix_loading=[np.nan, np.nan, 0.7])
    prepared = fc.PreparedInputs.from_inputs(replace(inputs, design=fixed))
    loading = next(layout for layout in prepared.parameters if layout.kind == "loading")
    np.testing.assert_array_equal(loading.entry_data.local_position, [5])
    common = fc.Model(prepared).build()
    assert not any(rv in ancestors([common["loading"]]) for rv in common.free_RVs)
    np.testing.assert_allclose(pm.draw(common["loading"]), [0.7])
    with pytest.raises(ValueError, match="common fix_loading"):
        fc.PreparedInputs.from_inputs(
            replace(inputs, design=design.assign(fix_loading=[0.7, 0.8, np.nan]))
        )
    with pytest.raises(ValueError, match="positive"):
        fc.PreparedInputs.from_inputs(replace(inputs, design=design.assign(fix_loading=0.0)))


def test_fixed_one_pl_owner_does_not_block_other_item_loading_regressions():
    design = pd.DataFrame(
        {"item_id": ["a", "b", "c"], "trait": ["A"] * 3, "fix_loading": [0.7, np.nan, np.nan]}
    )
    responses = fc.ResponseTable(
        pd.DataFrame({"person_id": ["p"] * 3, "item_id": ["a", "b", "c"], "response": [1, 0, 1]}),
        format="item_long",
    )
    inputs = fc.ItemInput(
        design,
        responses,
        model="DichotomousIRT",
        items=(fc.ItemOverride("a", model=fc.DichotomousIRT(kind="1PL")),),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs))
    model.add_regressions(
        "items.loading_magnitude ~ feature",
        item_data=pd.DataFrame({"item_id": ["a", "b", "c"], "feature": [0, 1, 2]}),
    )
    assert model.prepared.regressions
    assert "regression_coefficient" in model.build().named_vars


def test_shared_one_pl_items_expose_an_unobserved_trait_contrast():
    source = fc.ItemInput(
        {"traits": {"A": ["shared"], "B": ["shared"]}},
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": ["p", "q"], "item_id": ["shared", "shared"], "response": [1, 0]}
            ),
            format="item_long",
        ),
        model=fc.DichotomousIRT(kind="1PL"),
    )
    with pytest.raises(ValueError, match="likelihood-blind"):
        fc.PreparedInputs.from_inputs(
            source,
            population=fc.PopulationSpec(group="cohort"),
            person_data=pd.DataFrame({"person_id": ["p", "q"], "cohort": ["first", "second"]}),
        )


@pytest.mark.parametrize(
    "anchors,identified",
    [({"a": 1.0, "b": 1.0}, False), ({"a": 1.0, "b": 2.0}, True), ({"a": 1.0}, True)],
)
def test_tirt_group_means_depend_on_fixed_pair_loading_directions(anchors, identified):
    source = pair_input(model=fc.TIRT(loading_anchors=anchors))
    source = replace(source, design={"traits": {"A": ["a", "b"]}, "blocks": {"pair": ["a", "b"]}})
    options = dict(
        population=fc.PopulationSpec(group="cohort"),
        person_data=pd.DataFrame({"person_id": ["p"], "cohort": ["first"]}),
    )
    # Two declared groups are needed to free a population mean.
    source = replace(
        source,
        responses=replace(
            source.responses,
            frame=pd.concat(
                [source.responses.frame, source.responses.frame.assign(person_id="q")],
                ignore_index=True,
            ),
        ),
    )
    options["person_data"] = pd.DataFrame({"person_id": ["p", "q"], "cohort": ["first", "second"]})
    if identified:
        prepared = fc.PreparedInputs.from_inputs(source, **options)
        assert prepared.parameter_layouts["population_mean"].free_flat.tolist() == [1]
    else:
        with pytest.raises(ValueError, match="likelihood-blind"):
            fc.PreparedInputs.from_inputs(source, **options)


@pytest.mark.parametrize("model", [fc.TIRT(), fc.GTUM(block_effect=True)])
@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_comparison_generation_schedules_keep_all_pairs_and_fresh_person_layouts(model):
    source = fc.FCInput(
        "[traits]\nA~ a c\nB~ b\n[blocks]\ntriple ~ a b c\nside +~ a b c",
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p"],
                    "block_id": ["triple"],
                    "statement_1": ["a"],
                    "statement_2": ["b"],
                    "response": [1],
                }
            ),
            format="comparison_long",
        ),
        model=model,
        response_format=None,
        scale=fc.CategoryScale((0, 1, 2)),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    schedule = fc.generation_schedule(model, [7, 8])
    assert schedule.sizes["schedule"] == 6
    np.testing.assert_array_equal(schedule.schedule_person, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(schedule.schedule_statement_1, [0, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(schedule.schedule_statement_2, [1, 2, 2, 1, 2, 2])
    state = _prepare_generation(model.prepared, schedule)
    assert state.data["constant_data"].sizes["obs"] == 6
    assert len(state.data["constant_data"].attrs["batches"]) == 1
    if isinstance(source.model, fc.GTUM):
        (effect,) = state.blocks_by_kind["person_block_effect"]
        assert effect.free_flat.size == 2
        assert tuple(effect.prior_shapes["person_block_effect"]) == (2,)
        assert np.shares_memory(
            state.parameter_layouts["loading"].fixed,
            model.prepared.parameter_layouts["loading"].fixed,
        )
        xr.testing.assert_identical(
            state.parameter_uses.where(
                state.parameter_uses.kind != "person_block_effect", drop=True
            ),
            model.prepared.parameter_uses.where(
                model.prepared.parameter_uses.kind != "person_block_effect", drop=True
            ),
        )
        relabelled = _prepare_generation(model.prepared, fc.generation_schedule(model, [7]))
        (original,) = model.prepared.blocks_by_kind["person_block_effect"]
        (reused,) = relabelled.blocks_by_kind["person_block_effect"]
        assert relabelled.constant_data.person.values.tolist() == [7]
        assert model.prepared.constant_data.person.values.tolist() == ["p"]
        assert np.shares_memory(reused.entry_index, original.entry_index)
    reversed_schedule = schedule.copy(deep=True)
    reversed_schedule.schedule_statement_1.values[0] = 1
    reversed_schedule.schedule_statement_2.values[0] = 0
    reversed_state = _prepare_generation(model.prepared, reversed_schedule)
    rows = reversed_state.data["observed_data"]
    assert rows.response_statement_1.values[0] == 0
    assert rows.response_statement_2.values[0] == 1
    assert rows.response_source_statement_1.values[0] == 1
    assert rows.response_source_statement_2.values[0] == 0
    broken = schedule.copy(deep=True)
    broken.schedule_statement_1.values[0] = -1
    broken.schedule_statement_2.values[0] = -1
    with pytest.raises(ValueError, match="require scheduled statement pairs"):
        _prepare_generation(model.prepared, broken)


@pytest.mark.parametrize(
    "problem", ["missing", "extra_dimension", "short", "wrong_label", "outside_support"]
)
def test_generated_response_values_must_match_frozen_event_arrays(reference, problem):
    state = reference[0].prepared
    values = xr.Dataset(
        {
            batch["response_name"]: (
                batch["obs_dim"],
                np.zeros(state.data["constant_data"].sizes[batch["obs_dim"]], dtype=int),
            )
            for batch in state.data["constant_data"].attrs["batches"]
        }
    )
    first = state.data["constant_data"].attrs["batches"][0]["response_name"]
    if problem == "missing":
        values = values.drop_vars(first)
    elif problem == "extra_dimension":
        values[first] = values[first].expand_dims(replicate=[0])
    elif problem == "short":
        (dimension,) = values[first].dims
        values = values.isel({dimension: slice(0, 0)})
    elif problem == "wrong_label":
        (dimension,) = values[first].dims
        values = values.assign_coords({dimension: np.arange(values.sizes[dimension])})
    else:
        values[first].values[0] = 99
    with pytest.raises(ValueError):
        generated_rows(state, values)


def test_decoded_pick_does_not_answer_an_explicitly_refused_comparison():
    source = fc.FCInput(
        "[traits]\nA ~ a c\nB ~ b\n[blocks]\ntriplet ~ a b c",
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p"] * 3,
                    "block_id": ["triplet"] * 3,
                    "statement_1": ["a", "a", "c"],
                    "statement_2": ["b", "c", "b"],
                    "response": [1, 1, None],
                    "status": ["answered", "answered", "refused"],
                }
            ),
            format="comparison_long",
        ),
        model=fc.MGPPM(),
        response_format="pick",
    )
    state = fc.PreparedInputs.from_inputs(source)
    (batch,) = state.batches
    code = batch.events.labels.index((2, None, None))
    result = _prepare_generated(state, xr.Dataset({batch.response_name: (batch.obs_dim, [code])}))
    observed = result.observed_data
    design = result.constant_data
    assert sorted(observed.response_status.values) == ["answered", "refused"]
    assert design.sizes["obs"] == 1 and design.sizes["observation_source"] == 2
    (encoded,) = result.batches
    assert encoded.events.labels[encoded.y[0]] == (0,)
    refused = int(np.flatnonzero(observed.response_status.values == "refused")[0])
    for side, label in ((1, "c"), (2, "b")):
        position = observed[f"response_source_statement_{side}"].values[refused]
        assert design.statement_label.values[position] == label


@pytest.mark.parametrize("continuous", [False, True])
def test_generated_rows_keep_canonical_values_and_original_pair_direction(continuous):
    source = replace(
        graded_input(0.25 if continuous else "high", beta=continuous),
        design="[traits]\nA~ a\nB~ b\n[blocks]\ngraded ~ a b\nside +~ a\nside -~ b",
    )
    table = pd.DataFrame(
        {
            "person_id": ["p"],
            "block_id": ["graded"],
            "statement_1": ["b"],
            "statement_2": ["a"],
            "response": [0.25 if continuous else "high"],
        }
    )
    state = fc.PreparedInputs.from_inputs(
        replace(source, responses=fc.ResponseTable(table, format="comparison_long"))
    )
    batch = state.data["constant_data"].attrs["batches"][0]
    raw = [0.25] if continuous else [0]
    decoded = generated_pair_frame(
        state, xr.Dataset({batch["response_name"]: (batch["obs_dim"], raw)})
    )
    assert decoded.statement_1.tolist() == ["a"]
    assert decoded.statement_2.tolist() == ["b"]
    assert decoded.source_statement_1.tolist() == ["b"]
    assert decoded.source_statement_2.tolist() == ["a"]
    assert decoded.response.tolist() == [0.75 if continuous else "low"]
    if continuous:
        for raw in ([0.0], [1.0], [np.nan], [np.nextafter(0.0, 1.0)]):
            with pytest.raises(ValueError, match="outside"):
                generated_rows(
                    state,
                    xr.Dataset({batch["response_name"]: (batch["obs_dim"], raw)}),
                )


@pytest.mark.parametrize("lower,upper", [(0.5, 0.5), (0.7, 0.2), (0.1, 0.9), (0.0, 1.0)])
def test_fixed_asymptotes_must_form_an_ordered_probability_interval(lower, upper):
    source = fc.load_example("single_binary", model=fc.DichotomousIRT(kind="4PL")).data
    prepared = fc.PreparedInputs.from_inputs(source)
    constant = prepared.constant_data
    layouts = prepared.parameters
    values = {}
    for key, value in (("lower_asymptote", lower), ("upper_asymptote", upper)):
        layout = next(layout for layout in layouts if layout.kind == key)
        dims = layout.dims
        values[layout.kind] = xr.DataArray(
            np.full(layout.shape, value), dims=dims, coords={dim: constant[dim] for dim in dims}
        )
    point = xr.Dataset(values)
    if lower >= upper:
        with pytest.raises(ValueError, match="below upper"):
            _validate_parameters(prepared, point)
    else:
        xr.testing.assert_equal(_validate_parameters(prepared, point)[list(point)], point)
    # A genuinely partial point has no linked endpoint to check, but a fixed
    # calibration value is available to the same check as an explicitly supplied endpoint.
    for name in point:
        partial = point[[name]]
        xr.testing.assert_equal(_validate_parameters(prepared, partial)[[name]], partial)
        fixed = point.drop_vars(name)
        if lower >= upper:
            with pytest.raises(ValueError, match="below upper"):
                _validate_parameters(prepared, partial, fixed_parameters=fixed)
        else:
            merged = _validate_parameters(prepared, partial, fixed_parameters=fixed)
            xr.testing.assert_equal(merged[list(point)], point)


def test_shared_statements_keep_each_blocks_presentation_order():
    model = fc.Model()
    model.add_fc(
        "[traits]\nA ~ a\nB ~ b\n[blocks]\nforward ~ a b\nbackward ~ b a",
        pd.DataFrame(
            {
                "person_id": ["p", "p"],
                "block_id": ["forward", "backward"],
                "statement_1": ["a", "a"],
                "statement_2": ["b", "b"],
                "response": [1, 1],
            }
        ),
        format="comparison_long",
        model=fc.TIRT(loading_anchors={"a": 1, "b": 1}),
        response_format=None,
    )
    model.prepare()
    state = model.prepared
    assert state.constant_data.fc_statement.values.tolist() == [[0, 1], [1, 0]]
    observed = state.observed_data
    assert observed.response_statement_1.values.tolist() == [0, 1]
    assert observed.response_source_statement_1.values.tolist() == [0, 0]
    values = {
        batch["response_name"]: observed[batch["response_name"]]
        for batch in state.constant_data.attrs["batches"]
    }
    decoded = generated_pair_frame(state, xr.Dataset(values))
    assert decoded[["statement_1", "statement_2", "response"]].values.tolist() == [
        ["a", "b", 1],
        ["b", "a", 0],
    ]
    assert decoded[["source_statement_1", "source_statement_2"]].values.tolist() == [
        ["a", "b"],
        ["a", "b"],
    ]


@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_partial_comparison_schedule_retains_only_its_actual_edge():
    source = pair_input()
    source = replace(
        source,
        design="[traits]\nA~ a c\nB~ b\n[blocks]\npair ~ a b c",
        responses=fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p"],
                    "block_id": ["pair"],
                    "statement_1": ["c"],
                    "statement_2": ["a"],
                    "response": [0],
                }
            ),
            format="comparison_long",
        ),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    existing = model.prepared
    batch = model.prepared.data["constant_data"].attrs["batches"][0]
    decoded = generated_pair_frame(
        existing, xr.Dataset({batch["response_name"]: (batch["obs_dim"], [1])})
    )
    assert list(
        decoded[["statement_1", "statement_2", "response"]].itertuples(index=False, name=None)
    ) == [("a", "c", 1)]
    assert decoded[["source_statement_1", "source_statement_2"]].values.tolist() == [["c", "a"]]
    schedule = fc.generation_schedule(model, ["new"])
    schedule.schedule_statement_1.values[0] = 2
    schedule.schedule_statement_2.values[0] = 0
    state = _prepare_generation(model.prepared, schedule)
    batch = state.data["constant_data"].attrs["batches"][0]
    assert len(batch["events"]["labels"]) == 2
    answers = generated_pair_frame(
        state, xr.Dataset({batch["response_name"]: (batch["obs_dim"], [0])})
    )
    assert list(
        answers[["statement_1", "statement_2", "response"]].itertuples(index=False, name=None)
    ) == [("a", "c", 0)]
    assert answers[["source_statement_1", "source_statement_2"]].values.tolist() == [["c", "a"]]
    mixed = fc.generation_schedule(model, ["new", "other"])
    mixed.schedule_statement_1.values[0] = 0
    mixed.schedule_statement_2.values[0] = 2
    with pytest.raises(ValueError, match="cannot mix block events"):
        _prepare_generation(model.prepared, mixed)


def test_disjoint_graded_statements_keep_their_own_intercept_prior():
    source = fc.FCInput(
        "[traits]\nA~ a c\nB~ b d\n[blocks]\nrank ~ a b\ngraded ~ c d",
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": ["p", "p"], "block_id": ["rank", "graded"], "response": [[1, 2], 1]}
            ),
            format="block_long",
        ),
        model=fc.MGPPM(priors={"ranking_intercept": pm.Normal.dist(mu=0, sigma=1)}),
        blocks=(fc.BlockSpec("graded", response_format=None, scale=fc.CategoryScale((0, 1, 2))),),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    intercept = prepared.blocks_by_kind["statement_intercept"][0]
    assert prepared.parameter_layouts["statement_intercept"].fixed_flat.tolist() == [0]
    assert intercept.prior_role_index.tolist() == [
        "ranking_intercept",
        "graded_intercept",
        "graded_intercept",
    ]


def test_tirt_rank_supports_heterogeneous_fc_with_component_identification():
    source = fc.FCInput(
        pd.DataFrame(
            {
                "block_id": ["rank", "rank", "other", "other"],
                "position": [1, 2, 1, 2],
                "statement_id": list("abcd"),
                "trait": ["A"] * 4,
                "keying": [1] * 4,
                "fix_loading": [np.nan, np.nan, 1.0, np.nan],
            }
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p", "p"],
                    "block_id": ["rank", "other"],
                    "response": [[1, 2], [2, 1]],
                }
            ),
            format="block_long",
        ),
        model=fc.TIRTRank(loading_anchors={"a": 1}),
        blocks=(fc.BlockSpec("other", model=fc.MGPPM()),),
    )
    constant = fc.PreparedInputs.from_inputs(source).data["constant_data"]
    assert [type(owner["specification"]).__name__ for owner in constant.attrs["owners"]] == [
        "TIRTRank",
        "MGPPM",
    ]
    with pytest.raises(ValueError, match="loading anchor"):
        fc.PreparedInputs.from_inputs(replace(source, model=fc.TIRTRank()))


@pytest.mark.parametrize("channel", ["fc", "items"])
def test_unfolding_designs_require_explicit_location_sides(channel):
    source = (
        pair_input(model=fc.GTUM())
        if channel == "fc"
        else fc.ItemInput(
            {"traits": {"A": ["a"]}},
            fc.ResponseTable(
                pd.DataFrame({"person_id": ["p"], "item_id": ["a"], "response": [1]}),
                format="item_long",
            ),
            model=fc.GGUM(),
        )
    )
    with pytest.raises(ValueError, match="requires location_side"):
        fc.PreparedInputs.from_inputs(source)


@pytest.mark.parametrize(
    "side,relaxed", [(0, False), (1, False), (-1, False), (1, True), (-1, True)]
)
def test_unfolding_location_support_preserves_active_and_inactive_sides(side, relaxed):
    from contextlib import nullcontext

    from pyfcirt.models.specification import IdentificationWarning

    source = fc.load_example("single_unfolding", model="GGUM").data
    role = "location_positive" if side > 0 else "location_negative" if side < 0 else "location_free"
    source = replace(source, design=source.design.assign(location_side=side))
    if relaxed:
        source = replace(source, model=fc.GGUM(priors={role: pm.Normal.dist()}))
    expected_warning = (
        pytest.warns(IdentificationWarning, match="orientation")
        if side == 0 or relaxed
        else nullcontext()
    )
    with expected_warning:
        prepared = fc.PreparedInputs.from_inputs(source)
    (location,) = prepared.blocks_by_kind["location"]
    assert role in location.priors
    entries = prepared.parameter_layouts["location"].entry_data
    assert bool(entries.orientation.any()) is (side != 0)
    assert bool(entries.strict_positive.all()) is (side > 0 and not relaxed)
    assert bool(entries.strict_negative.all()) is (side < 0 and not relaxed)


@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_new_people_reject_incompatible_native_person_block_effect_array():
    source = fc.FCInput(
        "[traits]\nA ~ a c\nB ~ b\n[blocks]\ntriple ~ a b c\nside + ~ a b c",
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p", "q"],
                    "block_id": ["triple", "triple"],
                    "statement_1": ["a", "a"],
                    "statement_2": ["b", "b"],
                    "response": [0, 1],
                }
            ),
            format="comparison_long",
        ),
        model=fc.GTUM(
            block_effect=True,
            priors={"person_block_effect": pm.Normal.dist(mu=[0.1, 0.2], sigma=1)},
        ),
        response_format=None,
        scale=fc.CategoryScale((0, 1, 2)),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    before = model.idata.copy(deep=True)
    schedule = fc.generation_schedule(model, [1, 2, 3])
    with pytest.raises(ValueError, match="shape|broadcast"):
        _prepare_generation(model.prepared, schedule)
    xr.testing.assert_identical(model.idata, before)
