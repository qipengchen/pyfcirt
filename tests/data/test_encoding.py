"""Independent checks of selection-order and statement-rank meanings."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyfcirt import FCData, FCDesign


def test_selection_order_roundtrip(design):
    responses = pd.DataFrame({"person_id": ["P1"], "block_id": ["B1"], "response": [[2, 3, 1]]})
    data = FCData.from_frames(responses, design)
    ranks = data.to_rank_frame()
    assert ranks["statement_id"].tolist() == ["S1", "S2", "S3"]
    assert ranks["rank"].tolist() == [3, 1, 2]
    restored = FCData.from_frames(ranks.iloc[::-1], design, encoding="statement_rank")
    assert_frame_equal(restored.to_frame(), data.to_frame())
    assert restored.input_encoding == "statement_rank"


@pytest.mark.parametrize(
    "task,order,ranks",
    [
        ("pick", [2, None, None], [None, 1, None]),
        ("mole", [2, None, 1], [3, 1, None]),
    ],
)
def test_structural_na_survives_long_format(design_frame, task, order, ranks):
    design = FCDesign.from_frame(design_frame, response_format=task)
    long = pd.DataFrame(
        {
            "person_id": ["P1"] * 3,
            "block_id": ["B1"] * 3,
            "statement_id": ["S1", "S2", "S3"],
            "rank": pd.array(ranks, dtype="Int64"),
        }
    )
    data = FCData.from_frames(long, design, encoding="statement_rank")
    assert data.to_frame().at[0, "response"] == order
    assert_frame_equal(data.to_rank_frame(), long)
    assert len(data.to_rank_frame()) == 3


@pytest.mark.parametrize("ranks", [[1, 1, 3], [0, 2, 3], [1, 2, 4], [1, 2, 2.5]])
def test_invalid_long_ranks_fail(design, ranks):
    long = pd.DataFrame(
        {
            "person_id": ["P1"] * 3,
            "block_id": ["B1"] * 3,
            "statement_id": ["S1", "S2", "S3"],
            "rank": ranks,
        }
    )
    with pytest.raises(ValueError, match="P1.*B1"):
        FCData.from_frames(long, design, encoding="statement_rank")


def test_sparse_statement_input_is_not_silently_completed(design):
    long = pd.DataFrame(
        {"person_id": ["P1"], "block_id": ["B1"], "statement_id": ["S2"], "rank": [1]}
    )
    with pytest.raises(ValueError, match="every statement once"):
        FCData.from_frames(long, design, encoding="statement_rank")


def test_graded_cannot_be_represented_as_statement_ranks(graded_design):
    long = pd.DataFrame(
        {
            "person_id": ["P1", "P1"],
            "block_id": ["B1", "B1"],
            "statement_id": ["S1", "S2"],
            "rank": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="graded categories cannot"):
        FCData.from_frames(long, graded_design, encoding="statement_rank")


def test_design_row_permutation_does_not_change_local_positions(design_frame):
    responses = pd.DataFrame({"person_id": ["P1"], "block_id": ["B1"], "response": [[2, 3, 1]]})
    a = FCData.from_frames(responses, FCDesign.from_frame(design_frame))
    b = FCData.from_frames(responses, FCDesign.from_frame(design_frame.iloc[::-1]))
    assert_frame_equal(a.to_rank_frame(), b.to_rank_frame())


def test_four_statement_mole_keeps_unobserved_internal_order():
    design = FCDesign.from_mapping(
        traits={f"T{i}": [f"S{i}"] for i in range(1, 5)},
        blocks={"B1": ["S1", "S2", "S3", "S4"]},
        response_format="mole",
    )
    data = FCData.from_frames(
        pd.DataFrame({"person_id": ["P1"], "block_id": ["B1"], "response": [[2, None, None, 4]]}),
        design,
    )
    assert data.to_rank_frame()["rank"].isna().tolist() == [True, False, True, False]
    assert data.summary()["n_observations"] == 1
