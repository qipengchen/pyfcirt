"""Shared statements have one final parameter and one reference per component."""

import numpy as np
import pandas as pd
import pytest

import pyfcirt as fc


def shared_input(spec=None):
    return fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a", "c"], "B": ["b", "d"]},
            blocks={"left": ["a", "b"], "right": ["b", "c"], "last": ["c", "d"]},
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p"] * 3,
                    "block_id": ["left", "right", "last"],
                    "response": [[1, 2], [2, 1], [1, 2]],
                }
            ),
            format="block_long",
        ),
        model=fc.MGPPM() if spec is None else spec,
    )


def test_shared_statement_owner_and_reference_are_frozen_once():
    state = fc.PreparedInputs.from_inputs(shared_input())
    c = state.data["constant_data"]
    assert len(c.attrs["owners"]) == 1
    owner = c.attrs["owners"][0]
    assert c.sizes["statement"] == 4
    offset = next(v for v in owner["blocks"] if v["key"] == "intercept")
    np.testing.assert_array_equal(c[offset["fixed_flat"]], [0])
    np.testing.assert_array_equal(
        next(p.fixed for p in state.parameters if p.kind == "statement_intercept"),
        [0.0, np.nan, np.nan, np.nan],
    )
    slots = [
        next(v for v in m["slots"] if v["key"] == "intercept") for m in c.attrs["measurements"]
    ]
    left, right = [c[s["entry_index"]].values for s in slots[:2]]
    assert left[1] == right[0]
    assert {block.binding_id for block in state.blocks} >= {
        block["binding_id"] for block in owner["blocks"]
    }
    model = fc.Model(state)
    assert model.build() is model.build()
    assert "statement_intercept" in model.build().named_vars


def test_ranking_reference_components_do_not_include_graded_bridges():
    from dataclasses import replace

    source = shared_input()
    answers = source.responses.frame.copy()
    answers.at[1, "response"] = 1
    source = replace(
        source,
        responses=fc.ResponseTable(answers, format="block_long"),
        blocks=(fc.BlockSpec("right", response_format=None, scale=fc.CategoryScale((0, 1, 2))),),
    )
    constant = fc.PreparedInputs.from_inputs(source).constant_data
    owner = constant.attrs["owners"][0]
    # The graded bridge joins parameter ownership but not ranking-offset invariance.
    np.testing.assert_array_equal(np.unique(constant[owner["component"]]), [0])
    np.testing.assert_array_equal(
        constant["statement_intercept_fixed_value"], [0.0, np.nan, 0.0, np.nan]
    )


def test_declared_reference_replaces_default_without_zero_sum_centering():
    c = fc.PreparedInputs.from_inputs(shared_input(fc.MGPPM(reference_statements=("c",)))).data[
        "constant_data"
    ]
    layout = c.attrs["owners"][0]["blocks"][1]
    labels = c[c.attrs["owners"][0]["statement_indices"]].values
    assert c.statement_label.values[labels[c[layout["fixed_flat"]].values]].tolist() == ["c"]
    assert c[layout["free_flat"]].size == 3


@pytest.mark.parametrize("references", [("a", "c"), ("unknown",)])
def test_incompatible_or_unknown_reference_is_not_silently_repaired(references):
    with pytest.raises(ValueError, match="reference"):
        fc.PreparedInputs.from_inputs(shared_input(fc.MGPPM(reference_statements=references)))


@pytest.mark.parametrize("kind", ["anchors", "references"])
def test_distinct_block_identification_can_share_one_parameter_owner(kind):
    first, second = (
        (
            fc.TIRT(loading_anchors={"a": 1.0, "b": 1.2}),
            fc.TIRT(loading_anchors={"c": 1.4, "d": 1.6}),
        )
        if kind == "anchors"
        else (fc.MGPPM(reference_statements=("a",)), fc.MGPPM(reference_statements=("c",)))
    )
    source = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a", "c"], "B": ["b", "d"]},
            blocks={"left": ["a", "b"], "right": ["c", "d"]},
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p", "p"],
                    "block_id": ["left", "right"],
                    "response": [[1, 2], [2, 1]],
                }
            ),
            format="block_long",
        ),
        blocks=(fc.BlockSpec("left", model=first), fc.BlockSpec("right", model=second)),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    assert len(prepared.constant_data.attrs["owners"]) == 1
    if kind == "anchors":
        loading = prepared.parameter_layouts["loading"]
        np.testing.assert_allclose(loading.fixed_value, [1.0, 1.2, 1.4, 1.6])
    else:
        intercept = prepared.parameter_layouts["statement_intercept"]
        np.testing.assert_array_equal(intercept.fixed_flat, [0, 2])
