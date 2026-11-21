"""Bundled raw examples preserve original assets under explicit model choices."""

import gzip
import hashlib
import json
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

import pyfcirt as fc


def test_ranking_resource_preserves_response_cells_and_independent_copies():
    first = fc.load_example("ranking", model="MGPPM")
    second = fc.load_example("ranking", model="MGPPM")
    assert isinstance(first, fc.ExampleData) and isinstance(first.data, fc.FCInput)
    assert first.data.responses.format == "block_long"
    assert first.data.responses.frame.response.tolist() == [[1, 2], [2, 1], [2, 1], [1, 2]]
    assert first.data.design.trait.tolist() == ["T1", "T2", "T1", "T2"]
    assert first.truth is None
    first.data.responses.frame.iloc[0].response[0] = 9
    assert second.data.responses.frame.iloc[0].response == [1, 2]


def test_mixed_asset_is_one_graded_comparison_input_and_requires_compatible_model():
    example = fc.load_example("mixed", model=fc.TIRT())
    assert isinstance(example.data, fc.FCInput)
    assert example.data.responses.format == "comparison_long"
    assert example.data.response_format is None
    assert len(example.data.responses.frame) == 15
    assert example.data.responses.frame.response.tolist() == [1] * 15
    assert example.data.design.location_side.tolist() == [-1, 1, -1] * 5
    assert {b.response_format for b in example.data.blocks} == {None}
    state = fc.PreparedInputs.from_inputs(example.data)
    assert state.data["constant_data"].sizes["block"] == 5
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(fc.load_example("mixed", model="MGPPM").data)


@pytest.mark.parametrize(
    "name,model,format",
    [
        ("mgppm_study", "MGPPM", "block_long"),
        ("two_pl_rank_study", "TwoPLRank", "block_long"),
        ("ggum_rank_study", "GGUMRank", "block_long"),
        ("tirt_study", "TIRT", "comparison_long"),
        ("gtum_study", "GTUM", "comparison_long"),
    ],
)
def test_study_selection_limits_people_and_keeps_complete_design(name, model, format):
    example = fc.load_example(name, model=model, n_persons=2)
    assert example.data.responses.format == format
    assert example.data.responses.frame.person_id.nunique() == 2
    assert example.data.design.block_id.nunique() == 24
    assert example.metadata["selection"]["n_persons"] == 2


def test_resource_checksums_match_installed_bytes():
    root = files("pyfcirt").joinpath("_example_data", "studies")
    manifest = json.loads(root.joinpath("checksums.json").read_text())
    for name, expected in manifest.items():
        content = root.joinpath(name).read_bytes()
        assert len(content) == expected["bytes"]
        assert hashlib.sha256(content).hexdigest() == expected["sha256"]
        if name.endswith(".json.gz"):
            payload = json.loads(gzip.decompress(content))
            assert payload["format"] == "pyfcirt-example" and payload["version"] == 1
            for source in payload["inputs"]:
                if source["kind"] == "fc":
                    assert all(
                        block["task"] in {"rank", "pick", "mole", None}
                        for block in source["blocks"]
                    )


def test_cohort_filenames_record_the_number_of_people():
    root = files("pyfcirt").joinpath("_example_data")
    for parent in (root, root.joinpath("studies")):
        for case in parent.iterdir():
            if not case.is_dir() or case.name == "studies":
                continue
            for resource in case.iterdir():
                if not resource.name.endswith(".json.gz"):
                    continue
                payload = json.loads(gzip.decompress(resource.read_bytes()))
                people = set()
                for entry in payload["inputs"]:
                    for table in (entry["responses"], entry["administration"]):
                        if table is not None:
                            column = table["columns"].index("person_id")
                            people.update(row[column] for row in table["rows"])
                assert resource.name == f"n{len(people)}.json.gz", case.name


@pytest.mark.parametrize("name", ["unknown", "../ranking", None])
def test_unknown_examples_report_available_names(name):
    with pytest.raises(ValueError, match="Available examples"):
        fc.load_example(name, model="MGPPM")


def test_resources_load_outside_checkout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert fc.load_example("graded", model="TIRT").data.responses.frame.response.tolist() == [5]


def test_bundled_fc3_supplies_complete_mole_inputs_and_desirability(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    example = fc.load_example("hexaco_fc3", model="TwoPLRank")
    inputs = example.data
    assert inputs.response_format == "mole"
    assert inputs.responses.format == "block_long"
    assert inputs.responses.frame.person_id.nunique() == 200
    assert inputs.design.block_id.nunique() == 20
    assert len(inputs.responses.frame) == 4000
    assert inputs.responses.frame.response.iloc[0] == [3, None, 1]
    assert len(inputs.statement_features) == 60
    assert inputs.statement_features.statement_id.tolist() == inputs.design.statement_id.tolist()
    assert np.isfinite(inputs.statement_features.desirability).all()
    assert example.truth is None


def test_grouped_mgppm_bundle_keeps_rankings_groups_and_truth_aligned(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    example = fc.load_example("mgppm_groups", model="MGPPM", include_truth=True)
    inputs = example.data
    responses = inputs.responses.frame
    assert inputs.response_format == "rank"
    assert inputs.responses.format == "block_long"
    assert len(responses) == 1440
    assert inputs.design.block_id.nunique() == 12
    assert example.person_data.group.value_counts().to_dict() == {"A": 60, "B": 60}
    assert responses.person_id.unique().tolist() == example.person_data.person_id.tolist()
    np.testing.assert_allclose(
        example.truth.population_mean.sel(population_group="B"), [0.6, 0, 0, 0]
    )

    subset = fc.load_example(
        "mgppm_groups", model="MGPPM", n_persons=10, random_seed=42, include_truth=True
    )
    people = subset.data.responses.frame.person_id.unique().tolist()
    assert subset.person_data.person_id.tolist() == people
    assert subset.truth.person.values.tolist() == people
    assert len(people) == 10


def test_statement_features_are_numeric_without_coercing_identifiers():
    import numpy as np

    example = fc.load_example("custom_tirt", model=fc.TIRT())
    features = example.data.statement_features
    assert features.likert_loading.dtype.kind == "f"
    assert np.isfinite(np.log(features.likert_loading)).all()
    assert features.statement_id.dtype.kind == "O"
    assert all(isinstance(value, str) for value in features.statement_id)


@pytest.mark.parametrize("n_persons", [0, -1, True, 1.5])
def test_selection_requires_a_positive_integer(n_persons):
    with pytest.raises(ValueError, match="positive integer"):
        fc.load_example("ranking", model="MGPPM", n_persons=n_persons)


@pytest.mark.parametrize("name,count", [("ranking", 99), ("simulated_mgppm", 999999)])
def test_selection_cannot_exceed_stored_cohort(name, count):
    with pytest.raises(ValueError, match="at most"):
        fc.load_example(name, model="MGPPM", n_persons=count)


def test_original_study_truth_is_selected_without_becoming_fixed_parameters():
    example = fc.load_example(
        "simulated_mgppm", model={"fc": fc.MGPPM()}, n_persons=2, include_truth=True
    )
    assert example.truth is not None
    assert example.truth.attrs["pyfcirt_truth_source"] == "original_generation"
    assert example.truth.attrs["pyfcirt_current_fixed_parameters"] is False
    assert example.metadata["truth_available"]
    assert "original_generation" in example.metadata
    assert 2 in example.truth.sizes.values()
    default = fc.load_example("simulated_mgppm", model="MGPPM")
    assert default.metadata["selection"]["n_persons"] > 2


@pytest.mark.parametrize(
    "name,model",
    [
        ("single_binary", "DichotomousIRT"),
        ("single_pcm", "GPCM"),
        ("single_unfolding", "GGUM"),
        ("beta_comparisons", "GTUM"),
    ],
)
def test_item_and_continuous_resources_keep_declared_scales(name, model):
    example = fc.load_example(name, model=model, include_truth=True)
    if isinstance(example.data, fc.ItemInput):
        assert example.data.items
        assert example.data.items[0].categories == example.data.categories
        fc.PreparedInputs.from_inputs(example.data)
    else:
        assert isinstance(example.data.blocks[0].scale, fc.UnitIntervalScale)


@pytest.mark.parametrize(
    "name,model,error",
    [
        ("ranking", {"items": "GPCM"}, ValueError),
        ("joint_likert", {"fc": "MGPPM", "items": "GPCM"}, ValueError),
    ],
)
def test_model_mapping_must_match_actual_channels(name, model, error):
    with pytest.raises(error):
        fc.load_example(name, model=model)


def test_truth_option_is_boolean_and_empirical_assets_have_no_truth():
    with pytest.raises(TypeError, match="boolean"):
        fc.load_example("ranking", model="MGPPM", include_truth=1)
    example = fc.load_example("joint_likert", model={"items": "GPCM"})
    assert isinstance(example.data, fc.ItemInput)
    empirical = fc.load_example("custom_tirt", model="TIRT", include_truth=True, n_persons=2)
    assert empirical.truth is None
    assert fc.load_example("single_binary", model="DichotomousIRT").data.administration is not None


@pytest.mark.parametrize(
    "name,field,rows_per_person",
    [("custom_tirt", "person_data", 1), ("jrt_tirt", "response_times", 9)],
)
def test_companions_reuse_selected_people_and_preserve_source_rows(name, field, rows_per_person):
    declaration = fc.TIRT(link="logit", residuals="fixed")
    example = fc.load_example(name, model=declaration, n_persons=2, include_truth=True)
    selected = example.data.responses.frame.person_id.drop_duplicates().tolist()
    assert example.data.model is declaration
    assert selected == (["P001", "P002"] if name == "custom_tirt" else ["J001", "J002"])
    assert example.data.design.block_id.nunique() == 9
    assert example.truth is None
    path = files("pyfcirt").joinpath("_example_data", example.metadata["companion_tables"][field])
    with path.open("r", encoding="utf-8", newline="") as stream:
        source = pd.read_csv(stream, dtype={"person_id": str, "block_id": str})
    expected = source.loc[source.person_id.isin(selected)].reset_index(drop=True)
    actual = getattr(example, field)
    pd.testing.assert_frame_equal(actual, expected)
    assert len(actual) == 2 * rows_per_person
    assert set(actual.person_id) == set(selected)
    assert actual.person_id.map(type).eq(str).all()
    assert (
        example.metadata["companion_sha256"][field] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert (example.person_data is None) == (field != "person_data")
    assert (example.response_times is None) == (field != "response_times")


def test_companion_tables_and_metadata_are_independent_and_loading_does_not_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Loading an example must not prepare or fit a model")

    monkeypatch.setattr(fc.PreparedInputs, "from_inputs", forbidden)
    monkeypatch.setattr(fc.Model, "fit", forbidden)
    first = fc.load_example("custom_tirt", model="TIRT", n_persons=2)
    second = fc.load_example("custom_tirt", model="TIRT", n_persons=2)
    first.person_data.loc[0, "practice_hours"] = -999
    first.metadata["companion_tables"]["person_data"] = "changed.csv"
    assert second.person_data.loc[0, "practice_hours"] > 0
    assert second.metadata["companion_tables"]["person_data"] == "custom_tirt/people.csv"
    first_times = fc.load_example("jrt_tirt", model="TIRT", n_persons=2)
    second_times = fc.load_example("jrt_tirt", model="TIRT", n_persons=2)
    first_times.response_times.loc[0, "seconds"] = -999
    assert second_times.response_times.loc[0, "seconds"] > 0


@pytest.mark.parametrize(
    "name,model", [("ranking", "MGPPM"), ("single_pcm", "GPCM"), ("simulated_tirt", "TIRT")]
)
def test_resources_without_companions_return_none(name, model):
    example = fc.load_example(name, model=model)
    assert example.person_data is None and example.response_times is None


def test_mixed_measurement_bundle_retains_aligned_channels_and_declared_scales():
    example = fc.load_example(
        "mixed_measurement", model={"fc": "MGPPM", "items": "GPCM"}, n_persons=3
    )
    choices, ratings = example.data
    assert isinstance(choices, fc.FCInput) and isinstance(ratings, fc.ItemInput)
    assert len(choices.responses.frame) == len(ratings.responses.frame) == 12
    assert set(choices.responses.frame.person_id) == set(ratings.responses.frame.person_id)
    assert choices.design.statement_id.nunique() == 12
    assert choices.design.block_id.nunique() == 4
    assert next(item for item in ratings.items if item.item_id == "I4").categories == (0, 1)
    assert example.metadata["synthetic"] is True
    assert example.person_data is None and example.response_times is None


def test_companion_csv_ids_are_literal_strings(monkeypatch, tmp_path):
    from pyfcirt.data import examples

    root = files("pyfcirt").joinpath("_example_data")
    payload = json.loads(gzip.decompress(root.joinpath("ranking", "n2.json.gz").read_bytes()))
    source_ids = list(dict.fromkeys(row[0] for row in payload["inputs"][0]["responses"]["rows"]))
    ids = dict(zip(source_ids, ["001", "NA"], strict=True))
    for row in payload["inputs"][0]["responses"]["rows"]:
        row[0] = ids[row[0]]
    payload["metadata"]["companion_tables"] = {"person_data": "literal/people.csv"}
    (tmp_path / "_example_data" / "studies").mkdir(parents=True)
    example_dir = tmp_path / "_example_data" / "literal"
    example_dir.mkdir()
    (example_dir / "n2.json.gz").write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))
    (example_dir / "people.csv").write_text("person_id,value\nNA,2.5\n001,1.5\n", encoding="utf-8")
    monkeypatch.setattr(examples, "files", lambda package: tmp_path)
    example = fc.load_example("literal", model="MGPPM", n_persons=2)
    assert example.person_data.person_id.tolist() == ["NA", "001"]
    assert example.data.responses.frame.person_id.drop_duplicates().tolist() == ["001", "NA"]
    assert example.person_data.value.tolist() == [2.5, 1.5]


def test_teaching_covariates_have_reproducible_provenance_and_current_declarations():
    pytest.importorskip("formulae")
    example = fc.load_example("custom_tirt", model=fc.TIRT(), n_persons=80)
    people = example.person_data.assign(
        course_format=lambda frame: pd.Categorical(
            frame.course_format, categories=["in_person", "online"]
        )
    )
    assert len(people) == 80
    note_path = files("pyfcirt").joinpath("_example_data", example.metadata["generation_resource"])
    note = json.loads(note_path.read_text(encoding="utf-8"))
    category = note["teaching_person_covariates"]["course_format"]
    complete = fc.load_example("custom_tirt", model="TIRT")
    ordered = sorted(complete.person_data.person_id)
    labels = np.random.default_rng(category["random_seed"]).choice(category["levels"], len(ordered))
    expected = dict(zip(ordered, labels, strict=True))
    assert people.course_format.tolist() == people.person_id.map(expected).tolist()
    assert not category["response_generator_uses_this_covariate"]
    statements = example.data.statement_features.assign(
        log_likert_loading=lambda frame: np.log(frame.likert_loading)
    )
    prepared = fc.PreparedInputs.from_inputs(example.data, theta_parameterization="noncentered")
    baseline = fc.Model(prepared)
    explanatory = fc.Model(prepared)
    explanatory.add_regressions(
        "theta ~ practice_hours + course_format\nfc.loading_magnitude ~ log_likert_loading",
        person_data=people,
        statement_data=statements,
    )
    assert "theta ~ practice_hours + course_format" in explanatory.summary(details=True)
    assert "Regression: loading ~ log_likert_loading" in explanatory.summary(details=True)
    assert explanatory.prepared.regressions[0].design.term.values.tolist() == [
        "practice_hours",
        "course_format[online]",
    ]
    assert "practice_hours" not in baseline.summary(details=True)
    assert "posterior" not in baseline.idata and "posterior" not in explanatory.idata


@pytest.mark.parametrize(
    "name,field", [("custom_tirt", "person_data"), ("jrt_tirt", "response_times")]
)
def test_seeded_selection_is_reproducible_and_shared_by_companions(name, field):
    complete = fc.load_example(name, model="TIRT")
    source_people = complete.data.responses.frame.person_id.drop_duplicates()
    expected = set(source_people.sample(n=7, replace=False, random_state=42))
    first = fc.load_example(name, model="TIRT", n_persons=7, random_seed=42)
    repeat = fc.load_example(name, model="TIRT", n_persons=7, random_seed=42)
    different = fc.load_example(name, model="TIRT", n_persons=7, random_seed=43)
    default = fc.load_example(name, model="TIRT", n_persons=7)
    selected = first.data.responses.frame.person_id.drop_duplicates().tolist()
    assert set(selected) == expected
    assert selected == [person for person in source_people if person in expected]
    assert set(getattr(first, field).person_id) == expected
    pd.testing.assert_frame_equal(first.data.responses.frame, repeat.data.responses.frame)
    pd.testing.assert_frame_equal(getattr(first, field), getattr(repeat, field))
    assert set(different.data.responses.frame.person_id) != expected
    assert (
        default.data.responses.frame.person_id.drop_duplicates().tolist()
        == source_people[:7].tolist()
    )
    assert default.metadata["selection"]["random_seed"] is None
    assert first.metadata["selection"]["random_seed"] == 42
    assert "without replacement" in first.metadata["selection"]["policy"]
    pd.testing.assert_frame_equal(first.data.design, complete.data.design)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "42"])
def test_random_seed_requires_a_native_integer_seed(seed):
    with pytest.raises(ValueError, match="random_seed must be an integer"):
        fc.load_example("custom_tirt", model="TIRT", n_persons=2, random_seed=seed)


def test_random_seed_requires_explicit_person_count_and_keeps_truth_aligned():
    with pytest.raises(ValueError, match="requires an explicit n_persons"):
        fc.load_example("custom_tirt", model="TIRT", random_seed=42)
    example = fc.load_example(
        "simulated_tirt", model="TIRT", n_persons=7, random_seed=0, include_truth=True
    )
    selected = example.data.responses.frame.person_id.drop_duplicates().tolist()
    complete = fc.load_example("simulated_tirt", model="TIRT", n_persons=500)
    expected = complete.data.responses.frame.person_id.drop_duplicates().sample(
        n=7, replace=False, random_state=0
    )
    assert set(selected) == set(expected)
    assert example.metadata["selection"]["stored_n_persons"] == 500
    assert example.truth.person.values.tolist() == selected
    assert example.metadata["selection"]["random_seed"] == 0
