"""Event partitions, label alignment and mixed-size preparation without sampling."""

import json
from math import factorial

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from pyfcirt import FCData, FCDesign
from pyfcirt.data._encoding import compatible_patterns, ranking_events
from pyfcirt.data._indexing import observation_id
from pyfcirt.models.mgppm._data import prepare_data


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("task", ["rank", "pick", "mole"])
def test_event_support_partitions_all_complete_patterns(size, task):
    patterns = np.asarray(ranking_events(size, "rank"))
    events = ranking_events(size, task)
    membership = compatible_patterns(patterns, events)
    assert len(patterns) == factorial(size)
    assert_array_equal(membership.sum(axis=0), np.ones(factorial(size), dtype=int))
    expected = {"rank": 1, "pick": factorial(size - 1), "mole": factorial(size - 2)}[task]
    assert_array_equal(membership.sum(axis=1), np.full(len(events), expected))


def test_observation_ids_preserve_types_and_escape_delimiters():
    pairs = [(1, "B"), ("1", "B"), ("a:b", "c"), ("a", "b:c"), ("张三", 'B"1')]
    encoded = [observation_id(*pair) for pair in pairs]
    assert len(set(encoded)) == len(pairs)
    assert [json.loads(value) for value in encoded] == [list(pair) for pair in pairs]


def test_two_sizes_and_three_tasks_keep_exact_observation_units():
    design = FCDesign.from_mapping(
        traits={
            "T1": ["S1", "S4", "S6"],
            "T2": ["S2", "S5", "S7"],
            "T3": ["S3", "S8"],
            "T4": ["S9"],
        },
        blocks={"B3": ["S1", "S2", "S3"], "B2": ["S4", "S5"], "B4": ["S6", "S7", "S8", "S9"]},
        block_response_formats={"B2": "pick", "B4": "mole"},
        allow_variable_block_size=True,
    )
    responses = pd.DataFrame(
        {
            "person_id": ["P", "P", "P"],
            "block_id": ["B4", "B2", "B3"],
            "response": [[2, None, None, 4], [2, None], [2, 3, 1]],
        }
    )
    inputs = prepare_data(FCData.from_frames(responses, design))
    assert len(inputs.groups) == 3
    assert sum(len(g.response) for g in inputs.groups) == 3
    assert set(oid for g in inputs.groups for oid in g.obs_ids) == {
        observation_id("P", block) for block in ["B2", "B3", "B4"]
    }
    for group in inputs.groups:
        for block_index, block in enumerate(group.block_ids):
            frame = design.to_frame()
            expected = frame.loc[frame["block_id"] == block, "statement_id"].tolist()
            actual = [inputs.statement_ids[i] for i in group.statement_index[block_index]]
            assert actual == expected
        assert group.scores.shape[1] == design.block_sizes[group.block_ids[0]]
    mole = next(g for g in inputs.groups if g.task == "mole")
    assert mole.membership[mole.response[0]].sum() == 2


def test_shared_membership_is_indexed_once(design_frame):
    design_frame.at[3, "statement_id"] = "S1"
    design = FCDesign.from_frame(design_frame)
    responses = pd.DataFrame(
        {"person_id": [7, 7], "block_id": ["B1", "B2"], "response": [[1, 2, 3], [2, 1, 3]]}
    )
    inputs = prepare_data(FCData.from_frames(responses, design))
    assert len(inputs.statement_ids) == 5
    assert inputs.groups[0].statement_index[0, 0] == inputs.groups[0].statement_index[1, 0]


def test_ranking_score_columns_refer_to_statements(design):
    responses = pd.DataFrame({"person_id": ["P1"], "block_id": ["B1"], "response": [[2, 3, 1]]})
    group = prepare_data(FCData.from_frames(responses, design)).groups[0]
    assert_array_equal(group.scores[group.response[0]], [0, 2, 1])


def test_graded_orientation_and_unobserved_categories(graded_design):
    responses = pd.DataFrame({"person_id": ["P1"], "block_id": ["B1"], "response": [5]})
    group = prepare_data(FCData.from_frames(responses, graded_design)).groups[0]
    assert group.response.tolist() == [4]
    assert group.scores.shape == (5, 2)
    assert_array_equal(group.scores[4], [2, 0])
    scale = graded_design.graded_scale
    scale["higher_prefers"] = "position_2"
    reversed_design = FCDesign.from_frame(
        graded_design.to_frame(),
        response_format="graded",
        graded_scale=scale,
    )
    reversed_group = prepare_data(FCData.from_frames(responses, reversed_design)).groups[0]
    assert_array_equal(reversed_group.scores[4], [0, 2])


def test_no_answers_retains_people_and_design_groups(design):
    responses = pd.DataFrame(columns=["person_id", "block_id", "response"])
    administration = pd.DataFrame({"person_id": ["P1"], "block_id": ["B1"], "status": ["missing"]})
    inputs = prepare_data(FCData.from_frames(responses, design, administration=administration))
    assert inputs.person_ids == ("P1",)
    assert inputs.groups[0].response.shape == (0,)
    assert len(inputs.groups[0].block_ids) == 2


def test_data_acceptance_is_not_a_model_capability_claim():
    design = FCDesign.from_mapping(
        traits={f"T{i}": [f"S{i}"] for i in range(10)},
        blocks={"B": [f"S{i}" for i in range(10)]},
    )
    responses = pd.DataFrame(
        {"person_id": ["P"], "block_id": ["B"], "response": [list(range(10, 0, -1))]}
    )
    data = FCData.from_frames(responses, design)
    assert data.to_rank_frame()["rank"].tolist() == list(range(10, 0, -1))
    with pytest.raises(ValueError, match="currently supports sizes"):
        prepare_data(data)
