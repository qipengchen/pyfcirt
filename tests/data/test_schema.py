"""Design and response contracts, including structural versus genuine missingness."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyfcirt import FCData, FCDesign


def response(value, block="B1", person="P1"):
    return pd.DataFrame({"person_id": [person], "block_id": [block], "response": [value]})


def test_mapping_and_frame_are_equivalent(design, design_frame):
    mapped = FCDesign.from_mapping(
        traits={"T1": ["S1", "S4"], "T2": ["S2", "S5"], "T3": ["S3", "S6"]},
        blocks={"B1": ["S1", "S2", "S3"], "B2": ["S4", "S5", "S6"]},
        keying={"S2": -1, "S6": -1},
    )
    assert_frame_equal(mapped.to_frame(), design_frame)
    assert mapped.summary() == design.summary()


def test_design_views_and_nested_inputs_are_copied(design_frame):
    design_frame["metadata"] = [["original"] for _ in range(len(design_frame))]
    design = FCDesign.from_frame(design_frame)
    design_frame.at[0, "metadata"].append("input mutation")
    exported = design.to_frame()
    exported.at[0, "metadata"].append("view mutation")
    assert design.to_frame().at[0, "metadata"] == ["original"]
    design.block_sizes["B1"] = 99
    design.response_formats["B1"] = "graded"
    assert design.block_sizes["B1"] == 3
    assert design.response_formats["B1"] == "rank"


@pytest.mark.parametrize(
    "field,value",
    [
        ("position", 0),
        ("position", 2),
        ("position", 1.5),
        ("position", True),
        ("keying", 0),
        ("keying", True),
        ("trait", None),
        ("statement_id", ""),
        ("block_id", False),
    ],
)
def test_invalid_design_fields_fail(design_frame, field, value):
    frame = design_frame.astype({field: object})
    frame.at[0, field] = value
    with pytest.raises(ValueError):
        FCDesign.from_frame(frame)


def test_unknown_columns_are_not_used_as_membership(design_frame):
    design_frame["block_size"] = 999
    design = FCDesign.from_frame(design_frame)
    assert design.block_size == 3
    assert "block_size" in design.to_frame()


def test_different_sizes_require_explicit_opt_in(design_frame):
    frame = design_frame.iloc[:-1]
    with pytest.raises(ValueError, match="allow_variable_block_size"):
        FCDesign.from_frame(frame)
    design = FCDesign.from_frame(frame, allow_variable_block_size=True)
    assert design.block_sizes == {"B1": 3, "B2": 2}
    assert design.block_size is None
    assert FCDesign.from_frame(design_frame, allow_variable_block_size=True).block_size == 3


def test_shared_statement_metadata_must_agree(design_frame):
    design_frame.at[3, "statement_id"] = "S1"
    design = FCDesign.from_frame(design_frame)
    assert design.summary()["reused_statement_ids"] == ("S1",)
    design_frame.at[3, "trait"] = "T3"
    with pytest.raises(ValueError, match="S1"):
        FCDesign.from_frame(design_frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_format": "unknown"},
        {"block_response_formats": {"typo": "pick"}},
        {"block_response_formats": {"B1": "unknown"}},
        {"response_format": "graded", "graded_scale": {"categories": [1, 2]}},
    ],
)
def test_invalid_tasks_fail(design_frame, kwargs):
    with pytest.raises(ValueError):
        FCDesign.from_frame(design_frame, **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        {"categories": [1, 1]},
        {"categories": [1]},
        {"higher_prefers": "guess"},
        {"higher_prefers": None},
        {"labels": ["one"]},
        {"categoriez": [1, 2]},
    ],
)
def test_invalid_graded_scales_fail(graded_design, change):
    scale = graded_design.graded_scale
    scale.update(change)
    with pytest.raises(ValueError):
        FCDesign.from_frame(graded_design.to_frame(), response_format="graded", graded_scale=scale)


def test_graded_scale_is_required(graded_design):
    with pytest.raises(ValueError, match="explicit graded_scale"):
        FCDesign.from_frame(graded_design.to_frame(), response_format="graded")


def test_mapping_unknown_and_unused_statements_are_visible():
    with pytest.raises(ValueError, match="S9"):
        FCDesign.from_mapping(traits={"T1": ["S1"]}, blocks={"B1": ["S1", "S9"]})
    with pytest.raises(ValueError, match="multiple trait"):
        FCDesign.from_mapping(traits={"T1": ["S1"], "T2": ["S1"]}, blocks={})
    design = FCDesign.from_mapping(
        traits={"T1": ["S1", "unused"], "T2": ["S2"]},
        blocks={"B1": ["S1", "S2"]},
    )
    assert design.summary()["unused_statement_ids"] == ("unused",)


def test_features_are_aligned_by_id_and_copied(design_frame):
    features = pd.DataFrame({"statement_id": ["S2", "S1"], "likert": [0.4, 0.8]})
    design = FCDesign.from_frame(design_frame, statement_features=features)
    features.at[0, "likert"] = 99
    assert design.statement_features.set_index("statement_id").at["S2", "likert"] == 0.4
    features.at[0, "statement_id"] = "unknown"
    with pytest.raises(ValueError, match="Unknown statement_features"):
        FCDesign.from_frame(design_frame, statement_features=features)


def test_responses_are_copied_in_both_directions(design):
    frame = response([2, 3, 1])
    data = FCData.from_frames(frame, design)
    frame.at[0, "response"][0] = 1
    data.to_frame().at[0, "response"].append(9)
    data.original_responses.at[0, "response"].append(9)
    assert data.to_frame().at[0, "response"] == [2, 3, 1]
    assert data.original_responses.at[0, "response"] == [2, 3, 1]


@pytest.mark.parametrize(
    "value",
    [
        [1, 1, 3],
        [0, 2, 3],
        [1, 2, 4],
        [1, 2],
        [1, None, 3],
        [True, 2, 3],
        [1.2, 2, 3],
        "123",
        "__import__('os')",
        [[1], 2, 3],
        np.array([[1, 2, 3]]),
    ],
)
def test_invalid_rank_responses_fail_with_person_and_block(design, value):
    with pytest.raises(ValueError, match="P1.*B1"):
        FCData.from_frames(response(value), design)


@pytest.mark.parametrize("value", [[1, 2, 3], (1, 2, 3), np.array([1, 2, 3]), "[1,2,3]"])
def test_supported_order_containers(design, value):
    assert FCData.from_frames(response(value), design).to_frame().at[0, "response"] == [1, 2, 3]


@pytest.mark.parametrize(
    "task,valid,invalid",
    [
        ("pick", [2, None, None], [None, 2, None]),
        ("mole", [2, None, 1], [2, 1, None]),
    ],
)
def test_task_controls_na_slots(design_frame, task, valid, invalid):
    design = FCDesign.from_frame(design_frame, response_format=task)
    data = FCData.from_frames(response(valid), design)
    assert data.to_frame().at[0, "response"] == valid
    assert len(data.to_rank_frame()) == 3
    with pytest.raises(ValueError, match="NA positions"):
        FCData.from_frames(response(invalid), design)


@pytest.mark.parametrize("raw", [None, [None, None, None], [pd.NA, np.nan, None]])
def test_wholly_missing_requires_explicit_status(design, raw):
    with pytest.raises(ValueError, match="explicit missing status"):
        FCData.from_frames(response(raw), design)
    administration = pd.DataFrame(
        {
            "person_id": ["P1", "P2"],
            "block_id": ["B1", "B2"],
            "status": ["missing", "not_administered"],
        }
    )
    data = FCData.from_frames(response(raw), design, administration=administration)
    assert data.person_ids == ("P1", "P2")
    assert data.summary()["n_observations"] == 0
    assert data.to_frame(answered_only=True).empty
    assert data.to_rank_frame().empty
    assert data.to_frame().at[0, "response"] == [None, None, None]


@pytest.mark.parametrize("status", ["missing", "not_administered"])
def test_missing_status_cannot_contain_known_choices(design, status):
    administration = pd.DataFrame({"person_id": ["P1"], "block_id": ["B1"], "status": [status]})
    with pytest.raises(ValueError, match="known choices"):
        FCData.from_frames(response([1, 2, 3]), design, administration=administration)


def test_answered_status_needs_a_response(design):
    administration = pd.DataFrame(
        {
            "person_id": ["P1", "P2"],
            "block_id": ["B1", "B1"],
            "status": ["answered", "answered"],
        }
    )
    with pytest.raises(ValueError, match="no response"):
        FCData.from_frames(response([1, 2, 3]), design, administration=administration)


def test_response_pair_must_exist_in_administration(design):
    administration = pd.DataFrame({"person_id": ["P2"], "block_id": ["B1"], "status": ["missing"]})
    with pytest.raises(ValueError, match="absent from administration"):
        FCData.from_frames(response([1, 2, 3]), design, administration=administration)


def test_duplicates_and_unknown_blocks_fail(design):
    with pytest.raises(ValueError, match="Duplicate"):
        FCData.from_frames(pd.concat([response([1, 2, 3])] * 2), design)
    with pytest.raises(ValueError, match="Unknown response block"):
        FCData.from_frames(response([1, 2, 3], block="unknown"), design)


def test_graded_categories_are_not_inferred_from_observed_values(graded_design):
    data = FCData.from_frames(response(3), graded_design)
    assert data.design.graded_scale["categories"] == (1, 2, 3, 4, 5)
    assert data.to_frame().at[0, "response"] == 3
    assert data.to_rank_frame().empty
    assert len(data.to_graded_frame()) == 1
    assert data.summary()["n_observations"] == 1


@pytest.mark.parametrize("value", [0, 6, True, [1, 2], "5", 1.5])
def test_invalid_graded_categories_fail(graded_design, value):
    with pytest.raises(ValueError):
        FCData.from_frames(response(value), graded_design)


def test_graded_numeric_column_can_contain_real_missingness(graded_design):
    frame = pd.DataFrame(
        {"person_id": ["P1", "P2"], "block_id": ["B1", "B1"], "response": [5, np.nan]}
    )
    administration = pd.DataFrame(
        {"person_id": ["P1", "P2"], "block_id": ["B1", "B1"], "status": ["answered", "missing"]}
    )
    data = FCData.from_frames(frame, graded_design, administration=administration)
    assert data.to_frame()["response"].tolist() == [5, None]
    assert data.summary() == {"n_persons": 2, "n_observations": 1, "n_response_rows": 2}


def test_mixed_tasks_have_separate_views(design_frame):
    frame = design_frame.iloc[:-1]
    design = FCDesign.from_frame(
        frame,
        allow_variable_block_size=True,
        block_response_formats={"B2": "graded"},
        graded_scale={"categories": ["low", "middle", "high"], "higher_prefers": "position_1"},
    )
    data = FCData.from_frames(
        pd.DataFrame(
            {"person_id": ["P1", "P1"], "block_id": ["B1", "B2"], "response": [[2, 3, 1], "high"]}
        ),
        design,
    )
    assert len(data.to_rank_frame()) == 3
    assert len(data.to_graded_frame()) == 1
    assert data.summary()["n_observations"] == 2


def test_empty_responses_retain_administered_people(design):
    frame = pd.DataFrame(columns=["person_id", "block_id", "response"])
    administration = pd.DataFrame({"person_id": ["P1"], "block_id": ["B1"], "status": ["missing"]})
    data = FCData.from_frames(frame, design, administration=administration)
    assert data.person_ids == ("P1",)
    assert data.summary()["n_observations"] == 0
