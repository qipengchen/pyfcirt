"""One generated dataset retains truth, response_format, missingness and independent state."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pyfcirt as fc
from tests.helpers import mixed_model, workflow_point


def test_simulate_without_inputs_uses_default_balanced_fc_design():
    result = fc.simulate(random_seed=42)
    design = result.idata["design"].to_dataset()
    assert design.sizes["person"] == 100
    assert design.fc_statement.shape == (12, 3)
    assert design.sizes["trait"] == 3
    traits = design.statement_trait.values[design.fc_statement.values]
    np.testing.assert_array_equal(np.sort(traits, axis=1), np.tile(np.arange(3), (12, 1)))
    assert design.unit_model.values.tolist() == ["MGPPM"] * 12
    assert len(fc.response_data(result)) == 1200
    assert "posterior" not in result.idata and "generation" in result.idata["parameters"]


def test_automatic_design_balances_incomplete_blocks_and_replays_seed():
    options = dict(n_persons=4, n_traits=5, n_blocks=7, block_size=3, response_format="mole")
    first = fc.simulate(**options, random_seed=57)
    second = fc.simulate(**options, random_seed=57)
    xr.testing.assert_identical(first.idata, second.idata)
    design = first.idata["design"].to_dataset()
    traits = design.statement_trait.values[design.fc_statement.values]
    assert all(len(set(row)) == 3 for row in traits)
    np.testing.assert_array_equal(np.sort(np.bincount(traits.ravel())), [4, 4, 4, 4, 5])
    assert len(fc.response_data(first)) == 28


@pytest.mark.parametrize("spec", ["TIRT", fc.TIRTRank(), "GTUM", fc.GGUMRank(), "TwoPLRank"])
def test_automatic_pairs_use_model_requirements_and_native_response_laws(spec):
    result = fc.simulate(
        model=spec, n_persons=3, n_traits=2, n_blocks=2, block_size=2, random_seed=12
    )
    name = spec if isinstance(spec, str) else type(spec).__name__
    design = result.idata["design"].to_dataset()
    assert design.unit_model.values.tolist() == [name, name]
    assert len(fc.response_data(result)) == 6
    if name in {"TIRT", "GTUM"}:
        assert design.pair.size == 2
        assert set(fc.response_data(result).category) <= {0, 1}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"n_persons": True}, "positive integers"),
        ({"n_persons": 0}, "positive integers"),
        ({"block_size": 1}, "block_size"),
        ({"n_traits": 7, "n_blocks": 2}, "n_traits"),
        ({"n_traits": 2, "block_size": 3}, "block_size"),
        ({"source": "posterior"}, "source='prior'"),
    ],
)
def test_automatic_simulation_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        fc.simulate(**options)


@pytest.mark.parametrize("json_input", [False, True])
def test_tirt_fixed_mapping_uses_canonical_values_and_axes(tmp_path, json_input):
    design = "[traits]\nagency ~ A C\ncommunion ~ B\n[blocks]\nB1 ~ A B C"
    schedule = pd.DataFrame(
        {
            "person_id": ["P1"] * 3 + ["P2"] * 3,
            "block_id": "B1",
            "statement_1": ["A", "A", "B"] * 2,
            "statement_2": ["B", "C", "C"] * 2,
        }
    )
    template = fc.Model().prepare(
        fc.FCInput(
            design,
            fc.ResponseTable(schedule, format="comparison_schedule"),
            model=fc.TIRT(residuals="fixed", loading_anchors={"A": 1.0, "B": 1.0}),
            response_format=None,
        ),
        purpose="generate",
    )
    values = {
        "loading": [1.0, 1.0, 1.2],
        "statement_effect": [0.1, 0.0, -0.1],
        "residual_variance": [0.5, 0.5, 0.5],
        "cutpoint": [0.1, -0.2, 0.3],
        "population_mean": [[0.0, 0.0]],
        "covariance": [[1.0, 0.25], [0.25, 1.0]],
        "theta": [[0.4, -0.3], [-0.2, 0.6]],
    }
    if json_input:
        path = tmp_path / "parameters.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        with path.open(encoding="utf-8") as stream:
            parameters = json.load(stream)
    else:
        parameters = {name: np.asarray(value) for name, value in values.items()}
    result = template.simulate(source="fixed", parameters=parameters, random_seed=42)
    truth = result.idata["parameters/generation"].to_dataset()
    for name, value in values.items():
        np.testing.assert_array_equal(truth[name], value)
        for dim in truth[name].dims:
            np.testing.assert_array_equal(truth[dim], template.prepared.coords[dim])
    responses = fc.response_data(result)
    assert len(responses) == 6
    assert set(responses.pair_label) == {"A > B", "A > C", "B > C"}


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("unknown", [0], "Unknown final simulation parameters"),
        ("theta", [0], "theta: expected shape"),
        ("theta", None, "Provide theta"),
        ("statement_intercept", [1] * 9, "structural constraints"),
    ],
)
def test_fixed_mapping_rejects_invalid_parameters_atomically(name, value, message):
    model = mixed_model()
    point = workflow_point(model)
    parameters = {key: point[key].values for key in point.data_vars}
    if value is None:
        parameters.pop(name)
    else:
        parameters[name] = value
    before = model.idata
    with pytest.raises(ValueError, match=message):
        model.simulate(source="fixed", parameters=parameters, random_seed=10)
    assert model.idata is before


def test_fixed_mapping_uses_existing_design_for_new_people():
    model = mixed_model()
    point = workflow_point(model).drop_dims("person")
    parameters = {name: value.values.tolist() for name, value in point.data_vars.items()}
    schedule = model.generation_schedule(["new1", "new2", "new3"])
    generated = model.simulate(
        source="fixed", parameters=parameters, administration=schedule, random_seed=42
    )
    truth = generated.idata["parameters/generation"].to_dataset()
    assert truth.theta.person.values.tolist() == ["new1", "new2", "new3"]
    xr.testing.assert_allclose(truth[list(point)].reset_coords(drop=True), point)
    with pytest.raises(ValueError, match="Automatic design options"):
        model.simulate(n_persons=3)


def test_fixed_generation_is_reproducible_and_never_a_posterior():
    model = mixed_model()
    point = workflow_point(model)
    before = model.idata.copy(deep=True)
    generated = fc.simulate(model, source="fixed", parameters=point, random_seed=19)
    repeated = fc.simulate(model, source="fixed", parameters=point, random_seed=19)
    xr.testing.assert_identical(
        generated.prepared.data["observed_data"], repeated.prepared.data["observed_data"]
    )
    xr.testing.assert_identical(model.idata, before)
    assert "posterior" not in generated.idata and "fixed_parameters" not in generated.idata
    assert "generation" in generated.idata["parameters"]
    assert generated.idata["parameters/generation"].attrs["source"] == "fixed"
    assert [
        u["response_format"] for u in generated.prepared.data["constant_data"].attrs["units"]
    ] == [
        "rank",
        "pick",
        "mole",
        None,
        None,
    ]
    assert (
        generated.prepared.data["constant_data"].sizes["obs"]
        == model.prepared.data["constant_data"].sizes["obs"]
    )
    assert generated.build() is not model.build()


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("parameter", ["loading", "covariance"])
def test_fixed_generation_rejects_changing_existing_calibration_conditions(
    scheduled, parameter, monkeypatch
):
    from pyfcirt.data import prepare
    from tests.helpers import mixed_inputs

    point = workflow_point(mixed_model())
    fixed = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs(), fixed_parameters=point))
    conflict = point.copy(deep=True)
    if parameter == "covariance":
        conflict[parameter].values[0, 1] = conflict[parameter].values[1, 0] = 0.2
    else:
        conflict[parameter] *= 2
    schedule = fc.generation_schedule(fixed, person_ids=["new"]) if scheduled else None

    def forbidden(*args, **kwargs):
        pytest.fail("Conflicting calibration values reached generation preparation")

    monkeypatch.setattr(prepare, "_prepare_generation", forbidden)
    with pytest.raises(ValueError, match="conflict"):
        fc.simulate(
            fixed, source="fixed", parameters=conflict, administration=schedule, random_seed=19
        )


def test_simulation_preserves_recorded_nonresponses_and_joint_origins():
    source = fc.FCInput(
        "[traits]\nA ~ a b c\n[blocks]\ntriplet ~ a b c",
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p1", "p1", "p2", "p2"],
                    "block_id": ["triplet"] * 4,
                    "statement_1": ["a", "c", "a", "c"],
                    "statement_2": ["b", "a", "b", "a"],
                    "response": [1, None, 0, None],
                    "status": ["answered", "refused", "answered", "missing"],
                }
            ),
            format="comparison_long",
        ),
        model=fc.TIRTRank(loading_anchors={"a": 1}),
    )
    model = fc.Model(source)
    generated = model.simulate(random_seed=5)
    observed = generated.prepared.observed_data
    design = generated.prepared.constant_data
    assert sorted(observed.response_status.values) == ["answered", "answered", "missing", "refused"]
    unanswered = observed.isel(response_row=observed.response_status.isin(["refused", "missing"]))
    for side, label in ((1, "c"), (2, "a")):
        positions = unanswered[f"response_source_statement_{side}"].values
        assert design.statement_label.values[positions].tolist() == [label, label]
    assert design.sizes["obs"] == 2
    assert np.bincount(design.observation_source_obs_index.values).tolist() == [2, 2]
    assert design.sizes["observation_source"] == observed.sizes["response_row"]
    assert len(generated.build().observed_RVs) == len(model.build().observed_RVs)


def test_prior_simulation_records_and_validates_actual_prior_truth(monkeypatch):
    from pyfcirt.models.assembly import ModelGraph

    model = mixed_model()
    generated = fc.simulate(model, random_seed=29)
    assert generated.idata["parameters/generation"].attrs["source"] == "prior"
    assert "posterior" not in generated.idata
    assert np.isfinite(generated.idata["parameters/generation"].theta).all()

    draw = ModelGraph.prior_draws

    def nonfinite_truth(self, **options):
        sample = draw(self, **options)
        return sample.assign(theta=xr.full_like(sample.theta, np.nan))

    monkeypatch.setattr(ModelGraph, "prior_draws", nonfinite_truth)
    with pytest.raises(ValueError, match="theta: fixed values must be finite"):
        fc.simulate(model, random_seed=29)


@pytest.mark.parametrize(
    "options",
    [{"source": "unknown"}, {"source": "fixed"}, {"parameters": xr.Dataset()}, {"draw": (0, 0)}],
)
def test_invalid_generation_sources_reject(options):
    with pytest.raises(ValueError):
        fc.simulate(mixed_model(), **options)


def test_fixed_generation_validates_existing_people_theta(supplied_model):
    point = fc.parameter_point(supplied_model)
    for bad in [
        point.drop_vars("theta"),
        point.assign(theta=point.theta.isel(trait=0, drop=True)),
        point.assign_coords(person=["missing", "people"]),
        point.assign(theta=xr.full_like(point.theta, np.nan)),
    ]:
        with pytest.raises(ValueError, match="theta"):
            fc.simulate(supplied_model, source="fixed", parameters=bad, random_seed=1)
    with pytest.raises(TypeError, match="Model"):
        fc.simulate(supplied_model.idata)


@pytest.mark.parametrize("coordinate_type", ["object", "integer", "string", "float"])
@pytest.mark.parametrize("explicit", [False, True])
def test_posterior_generation_uses_one_actual_joint_state_and_records_labels(
    supplied_model, tmp_path, coordinate_type, explicit
):
    source = supplied_model.idata["posterior"].to_dataset().copy(deep=True)
    if coordinate_type == "integer":
        source = source.assign_coords(chain=np.array([2]), draw=np.array([3, 9]))
    elif coordinate_type == "string":
        source = source.assign_coords(chain=np.array(["chain"]), draw=np.array(["early", "late"]))
    elif coordinate_type == "float":
        source = source.assign_coords(chain=[0.5], draw=[0.25, 0.75])
    last = source.draw.values[-1]
    source["theta"].loc[{"draw": last}] = source.theta.sel(draw=last) + 1.5
    supplied_model.idata["posterior"] = source
    before = supplied_model.idata.copy(deep=True)
    draw = (source.chain.values[0], last) if explicit else None
    generated = fc.simulate(supplied_model, source="posterior", draw=draw, random_seed=41)
    truth = generated.idata["parameters/generation"].to_dataset()
    selected = tuple(truth[name].item() for name in ("source_chain", "source_draw"))
    assert all(truth[name].ndim == 0 for name in ("source_chain", "source_draw"))
    assert "selected_draw" not in truth.attrs
    if draw is not None:
        assert selected == draw
    xr.testing.assert_allclose(
        truth.theta.reset_coords(drop=True),
        source.theta.sel(chain=selected[0], draw=selected[1], drop=True),
    )
    assert truth.attrs["source_inference_method"] == "mcmc"
    assert "posterior" not in generated.idata
    xr.testing.assert_identical(supplied_model.idata, before)
    path = tmp_path / "generated.nc"
    generated.idata.to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as stored:
        xr.testing.assert_equal(stored["parameters/generation"].to_dataset(), truth)


@pytest.mark.parametrize("source", ["prior", "fixed", "posterior", "conditioned_prior"])
def test_generation_schedule_draws_new_people_and_preserves_global_conditions(
    supplied_model, source, monkeypatch
):
    from pyfcirt.data import prepare
    from tests.helpers import mixed_inputs, runtime_snapshot

    replace_people = prepare._replace_people
    replacements = []

    def counted_replace(*args, **kwargs):
        replacements.append(1)
        return replace_people(*args, **kwargs)

    monkeypatch.setattr(prepare, "_replace_people", counted_replace)

    if source == "conditioned_prior":
        point = fc.parameter_point(supplied_model, include_person=False)
        supplied_model = fc.Model(
            fc.PreparedInputs.from_inputs(mixed_inputs(), fixed_parameters=point)
        )
    schedule = fc.generation_schedule(supplied_model, person_ids=["new", "new2"])
    before = runtime_snapshot(supplied_model.prepared)
    previous_results = supplied_model.idata.copy(deep=True)
    options = {"source": "prior" if source == "conditioned_prior" else source}
    if source == "fixed":
        options["parameters"] = fc.parameter_point(supplied_model, include_person=False)
    model_prepare = fc.Model.prepare
    publications = []

    def record_prepare(model, *args, **kwargs):
        publications.append(model)
        return model_prepare(model, *args, **kwargs)

    monkeypatch.setattr(fc.Model, "prepare", record_prepare)
    generated = fc.simulate(supplied_model, administration=schedule, random_seed=71, **options)
    assert publications == [generated]
    assert len(replacements) == 1
    assert schedule.person_group.values.flags.writeable
    schedule.person_group.values[:] = -1
    assert np.all(generated.prepared.constant_data.person_group.values >= 0)
    xr.testing.assert_identical(supplied_model.prepared.data, before)
    xr.testing.assert_identical(supplied_model.idata, previous_results)
    truth = generated.idata["parameters/generation"]
    assert truth.person.values.tolist() == ["new", "new2"]
    assert truth.attrs["person_parameters"] == "fresh population draw"
    assert ("source_chain" in truth.coords) is (source == "posterior")
    assert ("source_draw" in truth.coords) is (source == "posterior")
    assert "fixed_parameters" not in generated.idata
    if source != "prior":
        assert truth["loading"].values.tolist() == [1.0] * 11
    assert generated.prepared.data.attrs["purpose"] == "observed"


def gtum_block_effect_model():
    source = fc.FCInput(
        "[traits]\nA ~ a c\nB ~ b\n[blocks]\ntriple ~ a b c\nside + ~ a b c",
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p"],
                    "block_id": ["triple"],
                    "statement_1": ["a"],
                    "statement_2": ["b"],
                    "response": [0],
                }
            ),
            format="comparison_long",
        ),
        model=fc.GTUM(block_effect=True),
        response_format=None,
        scale=fc.CategoryScale((0, 1, 2)),
    )
    return fc.Model(fc.PreparedInputs.from_inputs(source))


def gtum_fixed_point(model):
    from tests.helpers import labelled_point

    return labelled_point(
        model,
        {
            "population_mean": [[0.0, 0.0]],
            "covariance": np.eye(2),
            "loading": [1.0, 1.0, 1.0],
            "location": [1.0, 1.0, 1.0],
            "cutpoint": [-0.5, 0.5] * 3,
            "block_effect_scale": [0.4],
        },
    )


def test_fixed_new_person_gtum_simulation_preserves_person_prior_shape():
    model = gtum_block_effect_model()
    point = gtum_fixed_point(model)
    schedule = fc.generation_schedule(model, [7, 8])
    before = model.idata.copy(deep=True)
    generated = fc.simulate(
        model, source="fixed", parameters=point, administration=schedule, random_seed=82
    )
    truth = generated.idata["parameters/generation"]
    assert truth["person_block_effect"].shape == (2,)
    assert truth.person.values.tolist() == [7, 8]
    assert np.isfinite(truth["person_block_effect"]).all()
    (effect,) = generated.prepared.blocks_by_kind["person_block_effect"]
    assert tuple(effect.prior_shapes["person_block_effect"]) == (2,)
    layout = next(
        layout for layout in generated.prepared.parameters if layout.kind == "person_block_effect"
    )
    assert layout.entry_data.component.values.tolist() == ["observation", "observation"]
    reference = next(
        layout for layout in model.prepared.parameters if layout.kind == "person_block_effect"
    )
    assert np.all(
        layout.entry_data.summary_default.values == reference.entry_data.summary_default.values[0]
    )
    assert fc.coef(generated, var_names="person_block_effect").empty
    xr.testing.assert_identical(model.idata, before)
