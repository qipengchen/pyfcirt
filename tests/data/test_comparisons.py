"""The collection response_format survives all supported table encodings."""

from dataclasses import replace

import pandas as pd
import pytest

import pyfcirt as fc
from tests.helpers import generated_pair_frame, mixed_inputs


@pytest.mark.parametrize("channel", ["fc", "items"])
@pytest.mark.parametrize("person_column", [None, "person"])
def test_wide_and_long_tables_encode_identical_events(channel, person_column):
    import xarray as xr

    source = input_case() if channel == "fc" else mixed_inputs()[1]
    if channel == "items":
        source = replace(source, items=(), design={"traits": {"A": [7]}})
        long = pd.DataFrame({"person_id": [1, 2], "item_id": [7, 7], "response": [0, 1]})
        wide = pd.DataFrame({7: [0, 1]}, index=pd.Index([1, 2], dtype=object))
        long_format, wide_format = "item_long", "item_wide"
    else:
        long = pd.DataFrame(
            {"person_id": [1, 2], "block_id": ["b", "b"], "response": [[1, 2, 3], [3, 1, 2]]}
        )
        wide = pd.DataFrame({"b": [[1, 2, 3], [3, 1, 2]]}, index=pd.Index([1, 2], dtype=object))
        long_format, wide_format = "block_long", "block_wide"
    if person_column:
        wide[person_column] = wide.index
    expected = fc.PreparedInputs.from_inputs(
        replace(source, responses=fc.ResponseTable(long, format=long_format))
    )
    actual = fc.PreparedInputs.from_inputs(
        replace(
            source,
            responses=fc.ResponseTable(wide, format=wide_format, person_id_column=person_column),
        )
    )
    xr.testing.assert_identical(actual.data["observed_data"], expected.data["observed_data"])
    xr.testing.assert_equal(actual.data["constant_data"], expected.data["constant_data"])


@pytest.mark.parametrize(
    "frame,options",
    [
        (pd.DataFrame({"b": [[1, 2, 3]]}), {"person_id_column": "absent"}),
        (pd.DataFrame({"b": [[1, 2, 3], [1, 2, 3]]}, index=["p", "p"]), {}),
        (pd.DataFrame(index=["p"]), {}),
        (pd.DataFrame([[1, 2]], columns=["b", "b"]), {}),
        (
            pd.DataFrame({"b": [[1, 2, 3]]}),
            {"comparison_columns": (fc.ComparisonColumn("b", "b", "a", "c"),)},
        ),
    ],
)
def test_invalid_wide_axes_and_declarations_fail(frame, options):
    with pytest.raises((ValueError, TypeError)):
        fc.PreparedInputs.from_inputs(
            replace(input_case(), responses=fc.ResponseTable(frame, format="block_wide", **options))
        )


@pytest.mark.parametrize(
    "columns",
    [
        (),
        ("opaque",),
        (fc.ComparisonColumn("other", "b", "a", "c"),),
        (fc.ComparisonColumn("opaque", "b", "a", "c"),) * 2,
    ],
)
def test_wide_comparisons_require_a_bijective_column_mapping(columns):
    wide = pd.DataFrame({"opaque": [1]}, index=["p"])
    with pytest.raises((TypeError, ValueError)):
        fc.PreparedInputs.from_inputs(
            replace(
                input_case(),
                responses=fc.ResponseTable(
                    wide, format="comparison_wide", comparison_columns=columns
                ),
            )
        )


@pytest.mark.parametrize(
    "response_format,rank", [("rank", [3, 1, 2]), ("pick", [None, 1, None]), ("mole", [3, 1, None])]
)
def test_statement_ranks_preserve_only_the_declared_response_format(response_format, rank):
    source = input_case(response_format)
    rows = pd.DataFrame(
        {
            "person_id": ["p"] * 3,
            "block_id": ["b"] * 3,
            "statement_id": ["a", "b", "c"],
            "rank": rank,
            "status": ["answered"] * 3,
        }
    )
    state = fc.PreparedInputs.from_inputs(
        replace(source, responses=fc.ResponseTable(rows, format="statement_rank"))
    )
    assert state.data["constant_data"].sizes["obs"] == 1
    assert (
        state.data["constant_data"].attrs["batches"][0]["events"]["response_format"]
        == response_format
    )


@pytest.mark.parametrize(
    "change",
    [
        {"statement_id": ["a", "a", "c"]},
        {"statement_id": ["a", "b", "unknown"]},
        {"rank": [1, 1, 3]},
        {"rank": [0, 2, 3]},
        {"rank": [1, 2, 4]},
        {"status": ["answered", "missing", "answered"]},
    ],
)
def test_invalid_statement_ranks_fail_before_encoding(change):
    rows = pd.DataFrame(
        {
            "person_id": ["p"] * 3,
            "block_id": ["b"] * 3,
            "statement_id": ["a", "b", "c"],
            "rank": [1, 2, 3],
        }
    ).assign(**change)
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(
            replace(input_case(), responses=fc.ResponseTable(rows, format="statement_rank"))
        )


@pytest.mark.parametrize("status", ["missing", "refused", "not_administered"])
@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_unanswered_statement_ranks_remain_absent_events(status):
    rows = pd.DataFrame(
        {
            "person_id": ["p"] * 3,
            "block_id": ["b"] * 3,
            "statement_id": ["a", "b", "c"],
            "rank": [None] * 3,
            "status": [status] * 3,
        }
    )
    state = fc.PreparedInputs.from_inputs(
        replace(input_case(), responses=fc.ResponseTable(rows, format="statement_rank"))
    )
    assert state.data["constant_data"].sizes["obs"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"statement_2": ["a", "c", "c"]},
        {"statement_2": ["unknown", "c", "c"]},
        {"response": [None, None, None], "status": ["missing", "refused", "missing"]},
        {"response": [1, 1, 1], "status": ["missing"] * 3},
        {"status": ["unknown"] * 3},
    ],
)
def test_invalid_comparison_members_and_statuses_are_errors(change):
    source = input_case(format="comparison_long")
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(
            replace(
                source,
                responses=replace(source.responses, frame=source.responses.frame.assign(**change)),
            )
        )


@pytest.mark.parametrize(
    "administration",
    [
        pd.DataFrame({"person_id": ["p"], "block_id": ["unknown"], "status": ["answered"]}),
        pd.DataFrame(
            {"person_id": ["p", "p"], "block_id": ["b", "b"], "status": ["answered", "answered"]}
        ),
        pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "status": ["invalid"]}),
        pd.DataFrame({"person_id": ["absent"], "block_id": ["b"], "status": ["answered"]}),
        pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "status": ["missing"]}),
    ],
)
def test_administration_must_agree_with_recorded_answers(administration):
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(replace(input_case(), administration=administration))


def test_active_administration_requires_a_recorded_event():
    source = input_case()
    admin = pd.DataFrame(
        {"person_id": ["p", "absent"], "block_id": ["b", "b"], "status": ["answered", "answered"]}
    )
    with pytest.raises(ValueError, match="has no response"):
        fc.PreparedInputs.from_inputs(replace(source, administration=admin))


def test_recorded_status_must_match_nonresponse_administration():
    source = input_case()
    source.responses.frame.at[0, "response"] = None
    source.responses.frame["status"] = "refused"
    admin = pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "status": ["missing"]})
    with pytest.raises(ValueError, match="conflicts with administration"):
        fc.PreparedInputs.from_inputs(replace(source, administration=admin))


def test_response_format_options_are_specific_to_channel_and_purpose():
    source = input_case()
    for table in (
        replace(source.responses, format="item_long"),
        replace(source.responses, person_id_column="person_id"),
        replace(source.responses, format="block_schedule"),
    ):
        with pytest.raises(ValueError):
            fc.PreparedInputs.from_inputs(replace(source, responses=table))
    with pytest.raises(ValueError, match="cannot contain response"):
        fc.PreparedInputs.from_inputs(
            replace(source, responses=replace(source.responses, format="block_schedule")),
            purpose="generate",
        )


def test_generation_schedules_can_mix_missing_and_active_units():
    source = mixed_inputs()[0]
    rows = source.responses.frame.drop(columns="response").copy()
    rows["status"] = ["scheduled", "missing", "refused", "not_administered"]
    state = fc.PreparedInputs.from_inputs(
        replace(source, responses=fc.ResponseTable(rows, format="block_schedule")),
        purpose="generate",
    )
    assert state.data["constant_data"].sizes["obs"] == 1
    assert state.data["observed_data"].response_status.values.tolist() == rows.status.tolist()
    assert not any(name.endswith("_response") for name in state.data["observed_data"].data_vars)


@pytest.mark.parametrize(
    "response_format,response", [("pick", [3, None, None]), ("mole", [3, None, 1])]
)
def test_nonleading_choices_keep_the_correct_known_pair_directions(response_format, response):
    from pyfcirt.data.encoding import _known_pairs

    source = input_case(response_format)
    source.responses.frame.at[0, "response"] = response
    state = fc.PreparedInputs.from_inputs(source)
    batch = state.data["constant_data"].attrs["batches"][0]
    code = state.data["observed_data"][batch["response_name"]].item()
    assert batch["events"]["labels"][code] == tuple(response)
    expected = (
        [(0, 2, 0), (1, 2, 0)] if response_format == "pick" else [(0, 1, 0), (0, 2, 0), (1, 2, 0)]
    )
    assert list(_known_pairs(response, response_format)) == expected


def input_case(response_format="rank", format="block_long"):
    design = fc.FCDesignMapping(traits={"A": ["a", "c"], "B": ["b"]}, blocks={"b": ["a", "b", "c"]})
    if format == "block_long":
        response = {"rank": [1, 2, 3], "pick": [1, None, None], "mole": [1, None, 3]}[
            response_format
        ]
        table = pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "response": [response]})
    else:
        pairs = [("a", "b"), ("a", "c"), ("b", "c")]
        if response_format == "pick":
            pairs = pairs[:2]
        table = pd.DataFrame(
            [("p", "b", a, b, 1) for a, b in pairs],
            columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
        )
    return fc.FCInput(
        design, fc.ResponseTable(table, format=format), response_format=response_format
    )


@pytest.mark.parametrize("response_format", ["rank", "pick", "mole"])
def test_block_and_comparison_encodings_keep_the_original_response_format(response_format):
    states = [
        fc.PreparedInputs.from_inputs(input_case(response_format, f))
        for f in ("block_long", "comparison_long")
    ]
    for state in states:
        c = state.data["constant_data"]
        assert c.attrs["units"][0]["response_format"] == response_format
        assert all(b["events"]["response_format"] == response_format for b in c.attrs["batches"])
        assert fc.data_summary(state).index[0].endswith(" / " + response_format)
    assert (
        states[0].data["constant_data"].sizes["obs"] == states[1].data["constant_data"].sizes["obs"]
    )


def test_comparison_cycles_and_duplicates_are_rejected():
    source = input_case(format="comparison_long")
    frame = source.responses.frame
    bad = frame.copy()
    bad.loc[1, "response"] = 0
    for table in (bad, pd.concat([frame, frame.iloc[:1]])):
        with pytest.raises(ValueError):
            fc.PreparedInputs.from_inputs(
                replace(source, responses=replace(source.responses, frame=table))
            )


@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_comparison_wide_has_explicit_column_meanings():
    source = input_case(format="comparison_long")
    wide = pd.DataFrame({"opaque": [1]}, index=pd.Index(["p"], name="person_id"))
    declaration = fc.ResponseTable(
        wide,
        format="comparison_wide",
        comparison_columns=(fc.ComparisonColumn("opaque", "b", "a", "c"),),
    )
    state = fc.PreparedInputs.from_inputs(replace(source, responses=declaration))
    assert state.data["constant_data"].sizes["obs"] == 1
    assert state.data["constant_data"].attrs["units"][0]["response_format"] == "rank"


def test_missing_administration_does_not_fabricate_events():
    source = mixed_inputs()[0]
    administration = pd.DataFrame({"person_id": [99], "block_id": ["pair"], "status": ["missing"]})
    administration = pd.concat(
        [
            source.responses.frame[["person_id", "block_id"]].assign(status="answered"),
            administration,
        ],
        ignore_index=True,
    )
    state = fc.PreparedInputs.from_inputs(replace(source, administration=administration))
    summary = fc.data_summary(state, by="block", details=True)
    assert summary.loc["pair", "n_missing"] == 1
    assert summary.loc["pair", "n_obs"] == 2


@pytest.mark.parametrize("model", [fc.TIRT(), fc.GTUM()])
def test_triplet_intensity_comparisons_keep_all_rated_pairs_and_declared_support(model):
    import xarray as xr

    source = input_case(format="comparison_long")
    design = "[traits]\nA ~ a c\nB ~ b\n[blocks]\nb ~ a b c\nside + ~ a b c"
    frame = source.responses.frame.assign(response=[0, 1, 2])
    source = replace(
        source,
        design=design,
        model=model,
        response_format=None,
        scale=fc.CategoryScale((0, 1, 2)),
        responses=replace(source.responses, frame=frame),
    )
    state = fc.PreparedInputs.from_inputs(source)
    constant = state.data["constant_data"]
    batch = constant.attrs["batches"][0]
    assert constant.sizes["obs"] == 3
    assert constant.attrs["units"][0]["response_format"] is None
    assert batch["events"] == {"response_format": None, "labels": (0, 1, 2)}
    # Public pair checks must select the same named comparisons for both families.
    from pyfcirt.analysis.checks import _grouping

    groups, count, dimension, coords = _grouping(constant, "pair", range(3))
    assert groups.tolist() == [0, 1, 2]
    assert (count, dimension) == (3, "pair")
    assert list(zip(coords["statement_1"][1], coords["statement_2"][1], strict=True)) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    values = xr.Dataset(
        {batch["response_name"]: state.data["observed_data"][batch["response_name"]]}
    )
    decoded = generated_pair_frame(state, values)
    assert list(
        decoded[["statement_1", "statement_2", "response"]].itertuples(index=False, name=None)
    ) == [("a", "b", 0), ("a", "c", 1), ("b", "c", 2)]
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(
            replace(
                source,
                responses=fc.ResponseTable(
                    pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "response": [1]}),
                    format="block_long",
                ),
            )
        )


@pytest.mark.parametrize("model", [fc.TIRT(), fc.GTUM()])
@pytest.mark.parametrize("response_format", ["rank", "pick", "mole"])
def test_comparison_likelihoods_keep_original_response_format_after_pair_projection(
    model, response_format
):
    source = input_case(response_format)
    first = source.responses.frame.response.iloc[0]
    reverse = {
        "rank": [3, 2, 1],
        "pick": [3, None, None],
        "mole": [3, None, 1],
    }[response_format]
    responses = replace(
        source.responses,
        frame=pd.DataFrame(
            {"person_id": ["p", "q", "r"], "block_id": "b", "response": [first, reverse, first]}
        ),
    )
    design = "[traits]\nA ~ a c\nB ~ b\n[blocks]\nb ~ a b c\nside + ~ a b c"
    state = fc.PreparedInputs.from_inputs(
        replace(source, design=design, model=model, responses=responses)
    )
    constant = state.data["constant_data"]
    assert constant.attrs["units"][0]["response_format"] == response_format
    assert all(
        batch["events"]["response_format"] == response_format for batch in constant.attrs["batches"]
    )
    count = 2 if response_format == "pick" else 3
    assert constant.sizes["obs"] == 3 * count
    expected_pairs = [0, 1, 1, 2, 0, 1] if response_format == "pick" else [0, 1, 2] * 3
    assert constant.obs_pair.values.tolist() == expected_pairs
    assert constant.observation_source_obs_index.values.tolist() == list(range(3 * count))
    assert (
        constant.observation_source_response_row_index.values.tolist()
        == [0] * count + [1] * count + [2] * count
    )
    (batch,) = state.batches
    assert batch.y.tolist() == [1] * count + [0] * count + [1] * count


def test_pair_factors_keep_exact_rows_for_shuffled_comparisons_with_a_missing_answer():
    import numpy as np

    source = input_case(format="comparison_long")
    frame = source.responses.frame.iloc[[2, 0, 1]].copy()
    frame.loc[0, "response"] = None
    state = fc.PreparedInputs.from_inputs(
        replace(source, responses=replace(source.responses, frame=frame), model=fc.TIRT())
    )
    design = state.data["constant_data"]
    observed = state.data["observed_data"]
    source_rows = design.observation_source_response_row_index.values
    np.testing.assert_array_equal(design.observation_source_obs_index, [0, 1])
    np.testing.assert_array_equal(
        source_rows, np.flatnonzero(observed.response_status.values == "answered")
    )
    (batch,) = state.batches
    assert batch.obs_slice == slice(0, 2)
    assert np.shares_memory(batch.person, design.obs_person.values)
    np.testing.assert_array_equal(
        design.obs_input_event_index.values[batch.obs_slice],
        observed.response_administration_index.values[source_rows],
    )
    np.testing.assert_array_equal(
        batch.statement,
        np.column_stack(
            [
                observed.response_statement_1.values[source_rows],
                observed.response_statement_2.values[source_rows],
            ]
        ),
    )


@pytest.mark.parametrize("format", ["comparison_long", "block_long", "comparison_wide"])
def test_graded_mgppm_stays_an_ordinary_category_law(format):
    import numpy as np

    design = pd.DataFrame(
        dict(
            block_id=["g", "g"],
            position=[1, 2],
            statement_id=["a", "b"],
            trait=["A", "B"],
            keying=[1, -1],
        )
    )
    people = pd.Index(["p1", "p2", "p", "q"], dtype=object)
    frame = pd.DataFrame(dict(person_id=people, block_id="g", response=[0, 1, 2, 3]))
    options = {}
    if format == "comparison_long":
        frame = frame.assign(statement_1="a", statement_2="b")
    elif format == "comparison_wide":
        frame = pd.DataFrame({"opaque": [0, 1, 2, 3]}, index=people)
        options["comparison_columns"] = (fc.ComparisonColumn("opaque", "g", "a", "b"),)
    state = fc.PreparedInputs.from_inputs(
        fc.FCInput(
            design,
            fc.ResponseTable(frame, format=format, **options),
            model=fc.MGPPM(),
            response_format=None,
            scale=fc.CategoryScale((0, 1, 2, 3)),
        )
    )
    constant = state.data["constant_data"]
    batch = constant.attrs["batches"][0]
    assert batch["events"]["response_format"] is None
    assert "orders" not in batch["events"]
    np.testing.assert_array_equal(state.data["observed_data"][batch["response_name"]], [0, 1, 2, 3])
    np.testing.assert_array_equal(
        constant[batch["events"]["scores"]], [[0, 2], [0, 1], [1, 0], [2, 0]]
    )


@pytest.mark.parametrize(
    ("response_format", "response"), [("rank", [1, 2, 3]), ("mole", [1, None, 3])]
)
def test_equal_ranking_catalogs_share_frozen_geometry(response_format, response):
    import numpy as np

    design = pd.DataFrame(
        dict(
            block_id=["a"] * 3 + ["b"] * 3,
            position=[1, 2, 3] * 2,
            statement_id=list("abcdef"),
            trait=["A", "B", "C"] * 2,
            keying=[1] * 6,
            fix_var=[0.5] * 6,
        )
    )
    responses = fc.ResponseTable(
        pd.DataFrame(dict(person_id=[1, 1], block_id=["a", "b"], response=[response, response])),
        format="block_long",
    )
    state = fc.PreparedInputs.from_inputs(
        fc.FCInput(design, responses, model=fc.TIRTRank(), response_format=response_format)
    )
    constant = state.data["constant_data"]
    first, second = constant.attrs["batches"]
    for field in ("orders", "scores", "membership", "contrasts"):
        if field == "membership" and response_format == "rank":
            assert field not in first["events"] and field not in second["events"]
            continue
        left, right = (
            constant[first["events"][field]].values,
            constant[second["events"][field]].values,
        )
        np.testing.assert_array_equal(left, right)
        assert np.shares_memory(left, right)
        assert not left.flags.writeable and not right.flags.writeable
        detached = constant[first["events"][field]].copy(deep=True).values
        assert detached.flags.writeable and not np.shares_memory(left, detached)


@pytest.mark.parametrize("person_column", [None, "person"])
def test_wide_integer_category_cells_survive_other_columns_with_missing_values(person_column):
    import numpy as np
    import xarray as xr

    large = 2**53
    people = pd.Index([1, 2], dtype=object)
    wide = pd.DataFrame({"large": [large + 1, large], "other": [np.nan, 1.0]}, index=people)
    if person_column:
        wide[person_column] = people
    long = pd.DataFrame(
        [(1, "large", large + 1), (1, "other", None), (2, "large", large), (2, "other", 1)],
        columns=["person_id", "item_id", "response"],
        dtype=object,
    )
    options = dict(
        design={"traits": {"A": ["large", "other"]}},
        items=(fc.ItemOverride("large", categories=(large, large + 1)),),
    )
    expected = fc.PreparedInputs.from_inputs(
        fc.ItemInput(responses=fc.ResponseTable(long, format="item_long"), **options)
    )
    actual = fc.PreparedInputs.from_inputs(
        fc.ItemInput(
            responses=fc.ResponseTable(wide, format="item_wide", person_id_column=person_column),
            **options,
        )
    )
    xr.testing.assert_identical(actual.data["observed_data"], expected.data["observed_data"])


@pytest.mark.parametrize(
    "blocks, statements",
    [(("a", "b"), tuple("abcdef")), ((17, 3), tuple(range(2**60, 2**60 + 6)))],
    ids=["string-ids", "integer-ids"],
)
def test_shuffled_reversed_rank_comparisons_decode_to_the_original_typed_rows(blocks, statements):
    from itertools import combinations

    import xarray as xr

    design = pd.DataFrame(
        dict(
            block_id=[blocks[0]] * 3 + [blocks[1]] * 3,
            position=[1, 2, 3] * 2,
            statement_id=statements,
            trait=["A", "B", "C"] * 2,
            keying=[1] * 6,
            fix_var=[0.5] * 6,
        )
    )
    rows = []
    for person in (1, 2):
        for block, members in zip(blocks, (statements[:3], statements[3:]), strict=True):
            for index, (first, second) in enumerate(combinations(members, 2)):
                response = int(person == 1)
                if index % 2:
                    first, second, response = second, first, 1 - response
                rows.append(
                    dict(
                        person_id=person,
                        block_id=block,
                        statement_1=first,
                        statement_2=second,
                        response=response,
                    )
                )
    frame = pd.DataFrame(rows, dtype=object).sample(frac=1, random_state=23)
    state = fc.PreparedInputs.from_inputs(
        fc.FCInput(design, fc.ResponseTable(frame, format="comparison_long"), model=fc.TIRTRank())
    )
    constant = state.data["constant_data"]
    assert constant.block.values.tolist() == list(blocks)
    assert constant.statement_label.values.tolist() == list(statements)
    samples = xr.Dataset(
        {
            batch["response_name"]: state.data["observed_data"][batch["response_name"]]
            for batch in constant.attrs["batches"]
        }
    )
    decoded = generated_pair_frame(state, samples).assign(block_id=lambda rows: rows.unit_id)
    reversed_pair = decoded.source_statement_1 != decoded.statement_1
    decoded.loc[reversed_pair, "response"] = 1 - decoded.loc[reversed_pair, "response"]
    decoded = decoded.drop(columns=["statement_1", "statement_2"]).rename(
        columns={"source_statement_1": "statement_1", "source_statement_2": "statement_2"}
    )

    def semantic_rows(table):
        return sorted(
            (
                type(row.person_id).__name__,
                repr(row.person_id),
                row.block_id,
                row.statement_1,
                row.statement_2,
                int(row.response),
            )
            for row in table.itertuples(index=False)
        )

    assert semantic_rows(decoded) == semantic_rows(frame)
