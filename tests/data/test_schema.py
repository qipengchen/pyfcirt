"""Current declaration and frozen-state contracts."""

from dataclasses import replace

import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import pyfcirt as fc
from tests.helpers import assert_same_preparation, mixed_inputs, mixed_model


def pair_input(**options):
    return fc.FCInput(
        {"traits": {"A": ["a"], "B": ["b"]}, "blocks": {"pair": ["a", "b"]}},
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p"], "block_id": ["pair"], "response": [[1, 2]]}),
            format="block_long",
        ),
        **options,
    )


@pytest.mark.parametrize(
    "design",
    [
        3,
        {"traits": {"A": ["a"]}},
        {"traits": {}, "blocks": {"pair": ["a", "b"]}},
        {"traits": {"A": []}, "blocks": {"pair": ["a", "b"]}},
        {"traits": {"A": "a"}, "blocks": {"pair": ["a", "b"]}},
        {"traits": {"A": [["a"]]}, "blocks": {"pair": ["a", "b"]}},
        {"traits": {"A": ["a"], "B": ["a"]}, "blocks": {"pair": ["a", "b"]}},
        {"traits": {"A": ["a", "b"]}, "blocks": {}},
        {"traits": {"A": ["a", "b"]}, "blocks": {"pair": ["a"]}},
        {"traits": {"A": ["a"]}, "blocks": {"pair": ["a", "b"]}},
        {"traits": {"A": ["a", "b"]}, "blocks": {"pair": ["a", "b"]}, "keying": []},
        {"traits": {"A": ["a", "b"]}, "blocks": {"pair": ["a", "b"]}, "keying": {"c": -1}},
    ],
)
def test_design_mapping_errors_are_reported_before_layout(design):
    with pytest.raises((TypeError, ValueError)):
        fc.PreparedInputs.from_inputs(replace(pair_input(), design=design))


@pytest.mark.parametrize(
    "change",
    [
        {"position": [1, 1]},
        {"position": [1.5, 2]},
        {"keying": [1, 0]},
        {"statement_id": ["a", "a"]},
        {"location_side": [2, 0]},
        {"fix_loading": [True, 1.0]},
        {"block_type": ["MGPPM", "MGPPM"]},
        {"model": [fc.MGPPM(), fc.TwoPLRank()]},
    ],
)
def test_dataframe_design_errors_have_no_automatic_repair(change):
    design = fixed_tirt_design().drop(columns=["fix_loading", "fix_var"]).assign(**change)
    source = pair_input()
    responses = source.responses.frame.assign(block_id="b")
    with pytest.raises((TypeError, ValueError)):
        fc.PreparedInputs.from_inputs(
            replace(source, design=design, responses=replace(source.responses, frame=responses))
        )


@pytest.mark.parametrize(
    "options",
    [
        {"model": "unknown"},
        {"model": fc.GPCM()},
        {"blocks": ("pair",)},
        {"blocks": (fc.BlockSpec("other"),)},
        {"blocks": (fc.BlockSpec("pair"), fc.BlockSpec("pair"))},
        {"response_format": "graded"},
        {"scale": 3},
        {"scale": fc.CategoryScale((0, 1), higher_prefers="neither")},
        {"scale": fc.CategoryScale((0, 1), labels=("only one",))},
        {"scale": fc.CategoryScale((0, 1), labels=("", "high"))},
        {"scale": fc.CategoryScale((0, 1), higher_prefers="second")},
        {"scale": fc.UnitIntervalScale()},
    ],
)
def test_unit_declaration_errors_are_checked_without_model_build(options):
    with pytest.raises((TypeError, ValueError)):
        fc.PreparedInputs.from_inputs(pair_input(**options))


def test_ranking_binary_scale_and_labelled_graded_scale_are_frozen():
    state = fc.PreparedInputs.from_inputs(
        pair_input(scale=fc.CategoryScale((0, 1), labels=("no", "yes")))
    )
    assert state.data["constant_data"].attrs["units"][0]["scale"]["labels"] == ["no", "yes"]
    source = graded_input("equal")
    state = fc.PreparedInputs.from_inputs(
        replace(source, scale=fc.CategoryScale(("low", "equal", "high"), labels=("L", "E", "H")))
    )
    assert state.data["constant_data"].attrs["units"][0]["scale"]["labels"] == ["L", "E", "H"]


@pytest.mark.parametrize(
    "response",
    [[], [1], [1, None], [1, 1], [0, 2], [1, 3], [True, 2], 1, [[1, 2]], "[1.5, 2]"],
)
def test_invalid_rank_events_are_rejected_as_recorded(response):
    source = pair_input()
    source.responses.frame.at[0, "response"] = response
    with pytest.raises((TypeError, ValueError)):
        fc.PreparedInputs.from_inputs(source)


@pytest.mark.parametrize("value", ["not numeric", True, float("inf"), [1]])
def test_statement_feature_values_must_be_supported_scalars(value):
    source = pair_input()
    features = pd.DataFrame({"statement_id": ["a"], "feature": [value]})
    if isinstance(value, (str, bool)):
        state = fc.PreparedInputs.from_inputs(replace(source, statement_features=features))
        values = state.data["constant_data"].statement_feature_0.values
        assert values[0] == value and pd.isna(values[1])
    else:
        with pytest.raises(ValueError):
            fc.PreparedInputs.from_inputs(replace(source, statement_features=features))


def test_statement_features_preserve_scalars_and_distinguish_absent_values():
    import numpy as np

    source = mixed_inputs()[0]
    features = pd.DataFrame(
        {
            "statement_id": list("abcdefghi"),
            7: [np.float32(2), np.int64(3), None, pd.NA, np.nan, "4", np.bool_(True), 0.0, -1.0],
        },
        dtype=object,
    )
    state = fc.PreparedInputs.from_inputs(replace(source, statement_features=features))
    assert state.data["constant_data"].statement_feature_0.values.tolist() == [
        2.0,
        3,
        None,
        None,
        None,
        "4",
        True,
        0.0,
        -1.0,
    ]
    for bad in (
        features.rename(columns={7: 1.5}),
        pd.concat([features, features.iloc[:1]]),
        features.assign(statement_id="unknown"),
    ):
        with pytest.raises(ValueError):
            fc.PreparedInputs.from_inputs(replace(source, statement_features=bad))


def test_shared_design_values_must_be_consistent_across_blocks():
    source = pair_input()
    design = fixed_tirt_design().drop(columns=["fix_var"]).assign(block_id="pair")
    second = design.assign(block_id="other")
    for column, value in (
        ("trait", "C"),
        ("keying", -1),
        ("text", "different"),
        ("fix_loading", 2.0),
    ):
        first = design.copy()
        other = second.copy()
        if column == "text":
            first[column] = "original"
            other[column] = "original"
        other.loc[0, column] = value
        with pytest.raises(ValueError, match="Conflicting|conflicting"):
            fc.PreparedInputs.from_inputs(replace(source, design=pd.concat([first, other])))
    with pytest.raises(ValueError, match="at least one unit"):
        fc.PreparedInputs.from_inputs(replace(source, design=design.iloc[:0]))
    items = mixed_inputs()[1]
    duplicate = pd.DataFrame({"item_id": [7, 7], "trait": ["A", "A"]})
    with pytest.raises(ValueError, match="Duplicate item/trait"):
        fc.PreparedInputs.from_inputs(replace(items, design=duplicate, items=()))


def test_item_fixed_loadings_reject_nonreal_values_and_one_pl_conflicts():
    items = mixed_inputs()[1]
    for value in (True, "1", -1.0, float("inf")):
        design = pd.DataFrame({"item_id": [7], "trait": ["A"], "fix_loading": [value]})
        source = replace(
            items,
            design=design,
            items=(),
            model=fc.DichotomousIRT(kind="1PL"),
            responses=replace(items.responses, frame=items.responses.frame.iloc[:2]),
        )
        with pytest.raises(ValueError):
            fc.PreparedInputs.from_inputs(source)


@pytest.mark.parametrize(
    "text",
    [
        "A:",
        '[traits]\nA~ "a"\nB~ b\n[blocks]\npair ~ a b',
        "A: a",
        "[traits]\nA~ a\nB~ b\n[blocks]\npair ~ a b\npair other ~ [MGPPM]",
        "A B: a\npair = a b",
    ],
)
def test_design_grammar_rejects_empty_quoted_and_unknown_declarations(text):
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(replace(pair_input(), design=text))


def test_item_sides_can_be_declared_for_only_some_items():
    source = mixed_inputs()[1]
    state = fc.PreparedInputs.from_inputs(
        replace(
            source,
            design="[traits]\nA~ 7 ordinal\nside +~ 7",
            items=(),
            responses=replace(
                source.responses,
                frame=pd.DataFrame({"person_id": ["p"], "item_id": ["7"], "response": [1]}),
            ),
        )
    )
    assert state.data["constant_data"].item_location_side.values[0, 0] == 1


def test_mixed_catalog_is_typed_complete_and_detached():
    inputs = mixed_inputs()
    state = fc.PreparedInputs.from_inputs(inputs)
    frozen = state.data.copy(deep=True)
    inputs[0].responses.frame.at[0, "response"][0] = 2
    xr.testing.assert_identical(state.data, frozen)
    c = state.data["constant_data"]
    assert c.person.values.tolist() == [1, 2]
    assert c.block.values.tolist() == ["pair", "7", "quad"]
    assert c.item.values.tolist() == ["pair", "ordinal"]
    assert [u["response_format"] for u in c.attrs["units"]] == ["rank", "pick", "mole", None, None]
    assert all(
        batch["events"]["response_format"] is None
        for batch in c.attrs["batches"]
        if batch["channel"] == "items"
    )
    assert "posterior" not in state.data
    assert fc.Model(state).prepared is state


@pytest.mark.parametrize(
    "option,value", [("purpose", "unknown"), ("theta_parameterization", "raw")]
)
def test_bad_preparation_options(option, value):
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(mixed_inputs(), **{option: value})


def test_declarations_and_channels_must_be_explicit():
    source = mixed_inputs()[0]
    for data in ([], (), source.responses, (source, source)):
        with pytest.raises((TypeError, ValueError)):
            fc.PreparedInputs.from_inputs(data)
    with pytest.raises(ValueError, match="response_format"):
        fc.PreparedInputs.from_inputs(replace(source, response_format="comparison"))
    with pytest.raises(ValueError, match="format"):
        fc.PreparedInputs.from_inputs(
            replace(source, responses=replace(source.responses, format="guess"))
        )


@pytest.mark.parametrize("identifier", [True, "", "  ", None, 1.5])
def test_person_ids_are_explicit_labels(identifier):
    source = mixed_inputs()[1]
    frame = source.responses.frame.astype({"person_id": object}).copy()
    frame.loc[0, "person_id"] = identifier
    with pytest.raises((ValueError, TypeError)):
        fc.PreparedInputs.from_inputs(
            replace(source, responses=replace(source.responses, frame=frame))
        )


def test_declared_unobserved_categories_are_preserved():
    state = fc.PreparedInputs.from_inputs(mixed_inputs()[1])
    units = state.data["constant_data"].attrs["units"]
    assert units[1]["scale"]["categories"] == [0, 1, 2]
    source = mixed_inputs()[1]
    bad = source.responses.frame.copy()
    bad.loc[2, "response"] = 999
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(
            replace(source, responses=replace(source.responses, frame=bad))
        )


def test_duplicate_answers_and_unknown_units_are_not_repaired():
    source = mixed_inputs()[1]
    for frame in (
        pd.concat([source.responses.frame, source.responses.frame.iloc[:1]]),
        source.responses.frame.assign(item_id="unknown"),
    ):
        with pytest.raises(ValueError):
            fc.PreparedInputs.from_inputs(
                replace(source, responses=replace(source.responses, frame=frame))
            )


def graded_input(response, *, beta=False):
    design = fc.FCDesignMapping(traits={"A": ["a"], "B": ["b"]}, blocks={"graded": ["a", "b"]})
    scale = (
        fc.UnitIntervalScale(higher_prefers="second")
        if beta
        else fc.CategoryScale(("low", "equal", "high"), higher_prefers="second")
    )
    model = (
        fc.GTUM(link="beta_logit", priors={"response_precision": pm.Gamma.dist(alpha=2, beta=1)})
        if beta
        else fc.MGPPM()
    )
    return fc.FCInput(
        design,
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p"], "block_id": ["graded"], "response": [response]}),
            format="block_long",
        ),
        model=model,
        response_format=None,
        scale=scale,
    )


@pytest.mark.parametrize("response", [0.0, 1.0, -0.1, 1.1])
def test_beta_data_rejects_endpoints_and_outside_support(response):
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(graded_input(response, beta=True))


def test_beta_reversed_pair_uses_the_same_unit_interval_support():
    source = replace(
        graded_input(0.25, beta=True),
        model=fc.TIRT(
            link="beta_logit",
            loading_anchors={"a": 1.0, "b": 1.0},
            priors={"response_precision": pm.Gamma.dist(alpha=2, beta=1)},
        ),
    )
    frame = source.responses.frame.assign(statement_1="a", statement_2="b")
    forward = fc.ResponseTable(frame, format="comparison_long")
    reversed_frame = frame.assign(statement_1="b", statement_2="a", response=0.75)
    reverse = replace(forward, frame=reversed_frame)
    expected = fc.PreparedInputs.from_inputs(replace(source, responses=forward))
    actual = fc.PreparedInputs.from_inputs(replace(source, responses=reverse))
    assert actual.batches[0].y.tolist() == expected.batches[0].y.tolist()

    # The smallest positive float rounds to an excluded endpoint when reflected.
    rounded = replace(reverse, frame=reversed_frame.assign(response=float.fromhex("0x1p-1074")))
    with pytest.raises(ValueError, match="reversed response cannot be represented"):
        fc.PreparedInputs.from_inputs(replace(source, responses=rounded))


def test_beta_scalar_precision_and_offset_have_canonical_entries(tmp_path):
    pytest.importorskip("h5netcdf")
    source = replace(
        graded_input(0.5, beta=True),
        model=fc.TIRT(
            link="beta_logit",
            loading_anchors={"a": 1.0, "b": 1.0},
            priors={"response_precision": pm.Gamma.dist(alpha=2, beta=1)},
        ),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    parameters = {layout.kind: layout for layout in prepared.parameters}
    assert parameters["response_precision"].shape == (1,)
    assert parameters["comparison_offset"].shape == (1,)
    assert "cutpoint" not in parameters
    precision = next(block for block in prepared.blocks if block.key == "response_precision")
    assert precision.shape == ()
    assert precision.entry_index.shape == ()
    assert parameters["response_precision"].entry_data.unit_index.item() == 0
    assert prepared.output_coords["obs_population_group_index"].values.tolist() == [0]
    metadata = prepared.result_metadata
    support = metadata["response_support/b0"]
    assert support.attrs["response_kind"] == "unit_interval"
    assert support.sizes["b0_category"] == 0
    assert metadata["categories"].sizes["category_entry"] == 0
    target = tmp_path / "beta-metadata.nc"
    xr.DataTree.from_dict(metadata).to_netcdf(target, engine="h5netcdf")
    with xr.open_datatree(target, engine="h5netcdf") as saved:
        xr.testing.assert_identical(saved["response_support/b0"].to_dataset(), support)
        assert saved["categories"].sizes["category_entry"] == 0


def test_constraint_metadata_associates_ordered_rows_and_bounds_and_roundtrips(tmp_path):
    import numpy as np

    pytest.importorskip("h5netcdf")
    fc_source = replace(
        pair_input(model=fc.TIRT(loading_anchors={"a": 1.0, "b": 1.0})),
        responses=fc.ResponseTable(
            pd.DataFrame({"person_id": ["p"], "block_id": ["pair"], "response": [2]}),
            format="block_long",
        ),
        response_format=None,
        scale=fc.CategoryScale((0, 1, 2, 3)),
    )
    item_source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["x", "y"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p", "p"], "item_id": ["x", "y"], "response": [0, 1]}),
            format="item_long",
        ),
        model=fc.DichotomousIRT(kind="4PL"),
    )
    prepared = fc.PreparedInputs.from_inputs((fc_source, item_source))
    parameters = {layout.kind: layout for layout in prepared.parameters}
    cutpoint = parameters["cutpoint"].entry_data
    assert cutpoint.ordered.values.tolist() == [True] * 3
    assert cutpoint.ordered_count.values.tolist() == [3] * 3
    assert cutpoint.ordered_group_index.values.tolist() == [0] * 3
    upper = parameters["upper_asymptote"].entry_data
    assert upper.lower_kind.values.tolist() == ["lower_asymptote"] * 2
    np.testing.assert_array_equal(upper.lower_entry_index, [0, 1])
    np.testing.assert_array_equal(upper.upper_limit, [1, 1])
    target = tmp_path / "metadata.nc"
    xr.DataTree.from_dict(prepared.result_metadata).to_netcdf(target, engine="h5netcdf")
    with xr.open_datatree(target, engine="h5netcdf") as saved:
        xr.testing.assert_identical(saved["parameter_data/cutpoint"].to_dataset(), cutpoint)
        xr.testing.assert_identical(saved["parameter_data/upper_asymptote"].to_dataset(), upper)
        design = saved["design"].to_dataset()
        assert design.block.values.tolist() == ["pair"]
        assert design.item.values.tolist() == ["x", "y"]
        assert design.unit_block.values.tolist() == [0, -1, -1]
        assert design.unit_item.values.tolist() == [-1, 0, 1]
        assert "unit_id" not in design


def test_graded_scale_direction_and_complete_support_are_explicit():
    state = fc.PreparedInputs.from_inputs(graded_input("equal"))
    unit = state.data["constant_data"].attrs["units"][0]
    assert unit["scale"]["categories"] == ["low", "equal", "high"]
    assert unit["scale"]["higher_prefers"] == "second"
    assert fc.data_coverage(state, details=True).iloc[0].n_categories == 3
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(graded_input("unknown"))


def test_reversed_graded_scale_uses_declared_order_for_density_and_scores():
    import jax.numpy as jnp
    import numpy as np
    from scipy.special import log_softmax

    from pyfcirt.analysis.information import categorical_fisher
    from tests.helpers import labelled_point

    model = fc.Model(fc.PreparedInputs.from_inputs(graded_input("high")))
    (response,) = model._bindings.responses
    theta = np.array([0.7, -0.3])
    loading = np.array([0.9, 1.2])
    intercept = np.array([0.0, 0.2])
    point = labelled_point(
        model, {"theta": [theta], "loading": loading, "statement_intercept": intercept}
    )
    (logp,) = model._bindings.evaluate_at(
        (pm.logp(response.distribution, response.batch.y),), point
    )
    # The lowest label favours the first statement when higher prefers the second.
    weights = np.array([[1, 0], [0, 0], [0, 1]])
    logs = log_softmax(weights @ (loading * theta - intercept))
    expected = np.exp(logs)
    expected_scores = (weights - expected @ weights) * loading
    probabilities, scores, _ = categorical_fisher(
        lambda ability: response.kernel(ability, loading, intercept), jnp.asarray(theta)
    )
    np.testing.assert_allclose(logp, [logs[2]], atol=1e-12)
    np.testing.assert_allclose(probabilities, expected, atol=1e-12)
    np.testing.assert_allclose(scores, expected_scores, atol=1e-12)


@pytest.mark.parametrize(
    "categories",
    [
        (0, 0),
        (0,),
        (False, True),
        ("low", "mid", "high"),
        (0, "neutral", 2),
        (2, 1, 0),
        (0.0, 1.0),
        (1, 3, 9),
        (0, 2),
    ],
)
def test_invalid_declared_item_support(categories):
    source = mixed_inputs()[1]
    with pytest.raises((ValueError, TypeError)):
        fc.PreparedInputs.from_inputs(replace(source, categories=categories))


def test_supplied_final_arrays_preserve_mixed_coordinate_types():
    from tests.helpers import labelled_point, supplied_posterior

    model = mixed_model()
    point = labelled_point(model, {"theta": [[0.1, 0.2], [0.3, 0.4]]})
    assert point.person.values.tolist() == [1, 2]
    assert [type(v) for v in point.person.values] == [int, int]
    supplied_posterior(model, point, inference_method="mcmc", chains=(7, "7"), draws=(3, 9))
    assert model.idata["posterior"].chain.values.tolist() == [7, "7"]


def test_documented_fc_design_string_matches_explicit_dataframe():
    design = """
    # The complete example in docs/data.md, including intentionally free sides.
    [traits]
    agency ~ A C E
    communion ~ B D F
    [blocks]
    B1 ~ A B
    B2 ~ C D
    B3 ~ E F
    sign - ~ C F
    side + ~ A C
    side - ~ B D
    side 0 ~ E F
    """
    explicit = pd.DataFrame(
        [
            ("B1", 1, "A", "agency", 1, 1),
            ("B1", 2, "B", "communion", 1, -1),
            ("B2", 1, "C", "agency", -1, 1),
            ("B2", 2, "D", "communion", 1, -1),
            ("B3", 1, "E", "agency", 1, 0),
            ("B3", 2, "F", "communion", -1, 0),
        ],
        columns=["block_id", "position", "statement_id", "trait", "keying", "location_side"],
    )
    responses = fc.ResponseTable(
        pd.DataFrame(
            {
                "person_id": ["p"] * 3,
                "block_id": ["B1", "B2", "B3"],
                "response": [[1, 2], [2, 1], [1, 2]],
            }
        ),
        format="block_long",
    )
    assert_same_preparation(
        fc.PreparedInputs.from_inputs(fc.FCInput(design, responses)),
        fc.PreparedInputs.from_inputs(fc.FCInput(explicit, responses)),
    )


def test_fc_range_and_block_model_declarations_match_mapping_overrides():
    design = """
    [traits]
    first ~ S01-S02
    second ~ S03-S04
    [blocks]
    B1 ~ S01 S03
    B2 ~ S02 S04
    * ~ [TwoPLRank]
    """
    mapping = fc.FCDesignMapping(
        traits={"first": ["S01", "S02"], "second": ["S03", "S04"]},
        blocks={"B1": ["S01", "S03"], "B2": ["S02", "S04"]},
    )
    responses = fc.ResponseTable(
        pd.DataFrame(
            {"person_id": ["p", "p"], "block_id": ["B1", "B2"], "response": [[1, 2], [2, 1]]}
        ),
        format="block_long",
    )
    expected = fc.PreparedInputs.from_inputs(fc.FCInput(mapping, responses, model=fc.TwoPLRank()))
    assert_same_preparation(fc.PreparedInputs.from_inputs(fc.FCInput(design, responses)), expected)


def test_item_design_string_ranges_shared_loadings_and_sides_match_dataframe():
    design = """
    [traits]
    agency ~ I01-I03
    communion ~ I03
    side + ~ I01 I03
    side - ~ I02
    """
    explicit = pd.DataFrame(
        {
            "item_id": ["I01", "I02", "I03", "I03"],
            "trait": ["agency", "agency", "agency", "communion"],
            "location_side": [1, -1, 1, 1],
        }
    )
    responses = fc.ResponseTable(
        pd.DataFrame(
            {"person_id": ["p"] * 3, "item_id": ["I01", "I02", "I03"], "response": [0, 1, 2]}
        ),
        format="item_long",
    )
    options = dict(model=fc.GGUM(), categories=(0, 1, 2))
    assert_same_preparation(
        fc.PreparedInputs.from_inputs(fc.ItemInput(design, responses, **options)),
        fc.PreparedInputs.from_inputs(fc.ItemInput(explicit, responses, **options)),
    )


@pytest.mark.parametrize(
    "text",
    [
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a b\nsign +~ a\nsign -~ a",
        "[traits]\nA~ a\nB~ a b\n[blocks]\nb ~ a b",
        "[traits]\nA~ a\nA~ b\n[blocks]\nb ~ a b",
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a b\nb ~ a b",
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a b\nside 2~ a",
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a b\nside +~ unknown",
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a unknown",
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a b\nb ~ [MGPPM]\nb ~ [TwoPLRank]",
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a b\nb ~ [MGPPM TwoPLRank]",
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a b\nmalformed",
        "[traits]\nA~ a\nB~ b\n[blocks]\nb ~ a b\nextra ~ [: a]",
        "[traits]\nA~ S03-S01\nB~ b\n[blocks]\nb ~ S03 b",
        "[traits]\nA~ S01-T02\nB~ b\n[blocks]\nb ~ S01 b",
        "[traits]\nA~ S01-S2\nB~ b\n[blocks]\nb ~ S01 b",
    ],
)
def test_fc_design_string_conflicts_and_malformed_declarations_are_errors(text):
    responses = fc.ResponseTable(
        pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "response": [[1, 2]]}),
        format="block_long",
    )
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(fc.FCInput(text, responses))


@pytest.mark.parametrize(
    "text",
    [
        "A = i",
        "[traits]\nA~ i\nA~ j",
        "[traits]\nA~ i\nside +~ unknown",
        "[traits]\nA~ i\nside +~ i\nside -~ i",
        "[traits]\nA~ i\nside 2~ i",
    ],
)
def test_item_design_string_conflicts_are_errors(text):
    responses = fc.ResponseTable(
        pd.DataFrame({"person_id": ["p"], "item_id": ["i"], "response": [1]}), format="item_long"
    )
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(fc.ItemInput(text, responses))


def test_csv_json_preference_orders_match_in_memory_orders(tmp_path):
    import json

    source = mixed_inputs()[0]
    source.responses.frame["person_id"] = source.responses.frame.person_id.map(str)
    # CSV has one identifier type; use a distinct explicit label for the second person.
    source.responses.frame["person_id"] = source.responses.frame.person_id.astype(str)
    source.responses.frame.loc[3, "person_id"] = "second"
    source.responses.frame["block_id"] = source.responses.frame.block_id.map(str)
    design = fc.FCDesignMapping(
        traits=source.design.traits,
        blocks={"pair": ["a", "b"], "triple": ["c", "d", "e"], "quad": ["f", "g", "h", "i"]},
        keying=source.design.keying,
    )
    source.responses.frame["block_id"] = source.responses.frame.block_id.replace({"7": "triple"})
    source = replace(
        source,
        design=design,
        blocks=(
            fc.BlockSpec("triple", response_format="pick"),
            fc.BlockSpec("quad", response_format="mole"),
        ),
    )
    encoded = source.responses.frame.copy()
    encoded["response"] = encoded.response.map(json.dumps)
    path = tmp_path / "data.csv"
    encoded.to_csv(path, index=False)
    loaded = pd.read_csv(path, dtype={"person_id": str, "block_id": str})
    assert loaded.response.iloc[2] == "[1, null, null, 4]"
    actual = fc.PreparedInputs.from_inputs(
        replace(source, responses=fc.ResponseTable(loaded, format="block_long"))
    )
    assert_same_preparation(actual, fc.PreparedInputs.from_inputs(source))
    loaded.loc[0, "response"] = "[1, broken]"
    with pytest.raises(ValueError, match="JSON arrays"):
        fc.PreparedInputs.from_inputs(
            replace(source, responses=fc.ResponseTable(loaded, format="block_long"))
        )


def fixed_tirt_design():
    return pd.DataFrame(
        {
            "block_id": ["b", "b"],
            "position": [1, 2],
            "statement_id": ["a", "c"],
            "trait": ["A", "B"],
            "keying": [1, -1],
            "fix_loading": [1.0, 1.2],
            "fix_var": [0.5, 0.8],
        }
    )


def test_dataframe_fixed_loading_and_residual_declarations_are_frozen():
    import numpy as np

    design = fixed_tirt_design()
    responses = fc.ResponseTable(
        pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "response": [[1, 2]]}),
        format="block_long",
    )
    state = fc.PreparedInputs.from_inputs(fc.FCInput(design, responses, model=fc.TIRT()))
    layouts = {layout.kind: layout for layout in state.parameters}
    for key, expected in (("loading", [1.0, 1.2]), ("residual_variance", [0.5, 0.8])):
        layout = layouts[key]
        assert len(layout.free_flat) == 0
        np.testing.assert_array_equal(layout.fixed.ravel(), expected)
    with pytest.raises(ValueError, match="conflicting loading anchors"):
        fc.PreparedInputs.from_inputs(
            fc.FCInput(design, responses, model=fc.TIRT(loading_anchors={"a": 2.0}))
        )


@pytest.mark.parametrize(
    "column,value",
    [("fix_loading", -1.0), ("fix_loading", float("inf")), ("fix_var", 0.0), ("fix_var", -1.0)],
)
def test_invalid_dataframe_fixed_parameter_domains_fail_before_graph(column, value):
    design = fixed_tirt_design()
    design.loc[0, column] = value
    responses = fc.ResponseTable(
        pd.DataFrame({"person_id": ["p"], "block_id": ["b"], "response": [[1, 2]]}),
        format="block_long",
    )
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(fc.FCInput(design, responses, model=fc.TIRT()))


def test_item_dataframe_fixed_active_loadings_are_frozen():
    import numpy as np

    design = pd.DataFrame(
        {"item_id": ["shared", "shared"], "trait": ["A", "B"], "fix_loading": [1.0, 0.7]}
    )
    responses = fc.ResponseTable(
        pd.DataFrame({"person_id": ["p"], "item_id": ["shared"], "response": [1]}),
        format="item_long",
    )
    state = fc.PreparedInputs.from_inputs(fc.ItemInput(design, responses))
    loading = next(layout for layout in state.parameters if layout.kind == "loading")
    assert len(loading.free_flat) == 0
    np.testing.assert_array_equal(loading.fixed.ravel(), [1.0, 0.7])


def test_two_trait_tirt_pair_error_names_documented_anchor_inputs():
    design = fc.FCDesignMapping(
        traits={"agency": ["left"], "communion": ["right"]}, blocks={"pair": ["left", "right"]}
    )
    responses = fc.ResponseTable(
        pd.DataFrame({"person_id": ["person"], "block_id": ["pair"], "response": [[1, 2]]}),
        format="block_long",
    )
    with pytest.raises(ValueError, match="loading_anchors=.*fix_loading"):
        fc.PreparedInputs.from_inputs(fc.FCInput(design, responses, model=fc.TIRT()))
    point = fc.TIRT(loading_anchors={"left": 1.0, "right": 1.0})
    prepared = fc.PreparedInputs.from_inputs(fc.FCInput(design, responses, model=point))
    assert fc.data_coverage(prepared).index.tolist() == ["TIRT / rank"]


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"pair": {}},
        {"pair": ["MGPPM"]},
        (fc.BlockSpec("pair"),),
        (fc.BlockSpec("pair", response_format=...),),
    ],
)
def test_omitted_block_fields_inherit_wildcard_response_format(override):
    source = pair_input(model=fc.TwoPLRank(), response_format="mole")
    design = "[traits]\nA ~ a\nB ~ b\n[blocks]\n* ~ [MGPPM, pick]\npair ~ a b"
    responses = replace(source.responses, frame=source.responses.frame.assign(response=[[1, None]]))
    state = fc.PreparedInputs.from_inputs(
        replace(source, design=design, responses=responses, blocks=override)
    )
    unit = state.data["constant_data"].attrs["units"][0]
    assert unit["response_format"] == "pick"
    assert state.constant_data.unit_model.item() == "MGPPM"
    assert fc.BlockSpec("pair").response_format is Ellipsis


@pytest.mark.parametrize(
    "override",
    [
        {
            "pair": {
                "response_format": None,
                "scale": fc.CategoryScale(("low", "equal", "high"), higher_prefers="second"),
            }
        },
        {"pair": ["MGPPM", None]},
        (fc.BlockSpec("pair", response_format=None),),
    ],
)
def test_explicit_none_clears_inherited_response_format_fieldwise(override):
    source = graded_input("equal")
    source = replace(
        source,
        design=("[traits]\nA ~ a\nB ~ b\n[blocks]\n* ~ [MGPPM, mole]\npair ~ a b [MGPPM, pick]"),
        responses=replace(source.responses, frame=source.responses.frame.assign(block_id="pair")),
        response_format="rank",
    )
    state = fc.PreparedInputs.from_inputs(replace(source, blocks=override))
    unit = state.data["constant_data"].attrs["units"][0]
    assert unit["response_format"] is None
    assert unit["scale"]["categories"] == ["low", "equal", "high"]
    assert state.data["constant_data"].attrs["batches"][0]["events"]["response_format"] is None


@pytest.mark.parametrize(
    "override",
    [{"pair": ["MGPPM"]}, {"pair": {"model": fc.MGPPM()}}, (fc.BlockSpec("pair", model="MGPPM"),)],
)
def test_external_model_inline_response_format_and_wildcard_defaults_resolve_fieldwise(override):
    source = pair_input(model=fc.TwoPLRank(), response_format="mole")
    design = "[traits]\nA ~ a\nB ~ b\n[blocks]\n* ~ [MGPPM, rank]\npair ~ a b [TwoPLRank, pick]"
    responses = replace(source.responses, frame=source.responses.frame.assign(response=[[1, None]]))
    actual = fc.PreparedInputs.from_inputs(
        replace(source, design=design, responses=responses, blocks=override)
    )
    expected = fc.PreparedInputs.from_inputs(
        replace(pair_input(), responses=responses, response_format="pick")
    )
    assert_same_preparation(actual, expected)


@pytest.mark.parametrize(
    "model_text,model",
    [
        (
            'TIRT(link="logit", loading_anchors={"a": 1.0, "b": 1.2})',
            fc.TIRT(link="logit", loading_anchors={"a": 1.0, "b": 1.2}),
        ),
        (
            'TIRT(link="probit", loading_anchors={"a": 1.0, "b": 1.2})',
            fc.TIRT(link="probit", loading_anchors={"a": 1.0, "b": 1.2}),
        ),
    ],
)
def test_literal_constructor_string_mapping_and_dataframe_have_one_resolution(model_text, model):
    source = pair_input(model=model, response_format=None, scale=fc.CategoryScale((0, 1, 2)))
    responses = replace(source.responses, frame=source.responses.frame.assign(response=[1]))
    text = f"[traits]\nA ~ a\nB ~ b\n[blocks]\n* ~ [{model_text}, None]\npair ~ a b"
    frame = pd.DataFrame(
        {
            "block_id": ["pair"] * 2,
            "position": [1, 2],
            "statement_id": ["a", "b"],
            "trait": ["A", "B"],
            "keying": [1, 1],
        }
    )
    expected = fc.PreparedInputs.from_inputs(replace(source, responses=responses))
    for design in (text, frame):
        assert_same_preparation(
            fc.PreparedInputs.from_inputs(replace(source, design=design, responses=responses)),
            expected,
        )


@pytest.mark.parametrize(
    "options",
    [
        "[unknown, rank]",
        "[fc.MGPPM(), rank]",
        "[MGPPM(1), rank]",
        "[MGPPM(priors=make_priors()), rank]",
        "[MGPPM(**{}), rank]",
        "[MGPPM(priors={}, priors={}), rank]",
        "[MGPPM(unknown=True), rank]",
        '[TIRT(loading_anchors={"a": 1, "a": 2}), rank]',
        "[MGPPM, graded]",
        "[MGPPM, item]",
        "[MGPPM, comparison]",
    ],
)
def test_block_options_reject_unknown_calls_nonliteral_and_duplicate_keywords(options):
    text = f"[traits]\nA ~ a\nB ~ b\n[blocks]\npair ~ a b {options}"
    with pytest.raises((TypeError, ValueError), match="[Ll]ine|pair"):
        fc.PreparedInputs.from_inputs(replace(pair_input(), design=text))


@pytest.mark.parametrize(
    "extra",
    [
        "pair ~ a b",
        "* ~ [MGPPM, rank]\n* ~ [MGPPM, pick]",
        "side + ~ a a",
        "sign - ~ a a",
        "other ~ a a",
    ],
)
def test_duplicate_declaration_tokens_are_not_silently_overwritten(extra):
    text = f"[traits]\nA ~ a\nB ~ b\n[blocks]\npair ~ a b\n{extra}"
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(replace(pair_input(), design=text))


@pytest.mark.parametrize(
    "override",
    [
        {"pair": {"unknown": 1}},
        {"absent": ["MGPPM"]},
        {"pair": []},
        {"pair": ["MGPPM", "rank", None]},
    ],
)
def test_external_mapping_override_errors_name_the_block(override):
    with pytest.raises((TypeError, ValueError), match="pair|absent"):
        fc.PreparedInputs.from_inputs(pair_input(blocks=override))


@pytest.mark.parametrize("response", [0, 1])
def test_absent_response_format_defaults_to_complete_binary_scale(response):
    source = pair_input(response_format=None)
    frame = source.responses.frame.assign(response=[response])
    source = replace(source, responses=replace(source.responses, frame=frame))
    actual = fc.PreparedInputs.from_inputs(source)
    explicit = fc.PreparedInputs.from_inputs(replace(source, scale=fc.CategoryScale((0, 1))))
    assert_same_preparation(actual, explicit)


@pytest.mark.parametrize("response", [0.5, 2])
def test_absent_response_format_does_not_infer_a_different_scale_from_values(response):
    source = pair_input(response_format=None)
    frame = source.responses.frame.assign(response=[response])
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(
            replace(source, responses=replace(source.responses, frame=frame))
        )


def test_beta_comparisons_still_require_an_explicit_scale():
    source = graded_input(0.5, beta=True)
    with pytest.raises(ValueError, match="scale"):
        fc.PreparedInputs.from_inputs(replace(source, scale=None))


@pytest.mark.parametrize(
    "text",
    [
        "A: a\nB: b\npair = a b",
        "[traits]\nA: a\nB ~ b\n[blocks]\npair ~ a b",
        "[traits]\nA ~ a\nB ~ b\n[blocks]\npair = a b",
    ],
)
def test_replaced_colon_equals_syntax_is_rejected(text):
    with pytest.raises(ValueError, match="Line"):
        fc.PreparedInputs.from_inputs(replace(pair_input(), design=text))


@pytest.mark.parametrize("inline", ["[MGPPM]", "[MGPPM, pick]"])
def test_inline_model_only_inherits_wildcard_response_format(inline):
    source = pair_input(model=fc.TwoPLRank(), response_format="mole")
    text = f"[traits]\nA ~ a\nB ~ b\n[blocks]\n* ~ [TwoPLRank, pick]\npair ~ a b {inline}"
    responses = replace(source.responses, frame=source.responses.frame.assign(response=[[1, None]]))
    actual = fc.PreparedInputs.from_inputs(replace(source, design=text, responses=responses))
    expected = fc.PreparedInputs.from_inputs(
        replace(pair_input(), responses=responses, response_format="pick")
    )
    assert_same_preparation(actual, expected)


@pytest.mark.parametrize("response", ["0", "low", True, 0.5])
def test_item_responses_require_integer_codes_from_declared_support(response):
    source = mixed_inputs()[1]
    frame = source.responses.frame.astype({"response": object}).copy()
    frame.loc[2, "response"] = response
    with pytest.raises(ValueError):
        fc.PreparedInputs.from_inputs(
            replace(source, responses=replace(source.responses, frame=frame))
        )


def test_item_overrides_also_reject_text_category_declarations():
    source = mixed_inputs()[1]
    with pytest.raises(ValueError, match="consecutive integers in increasing order"):
        fc.PreparedInputs.from_inputs(
            replace(
                source,
                items=(
                    fc.ItemOverride("ordinal", model=fc.GPCM(), categories=("low", "mid", "high")),
                ),
            )
        )


@pytest.mark.parametrize("model", [fc.GPCM(), fc.GGUM()])
@pytest.mark.parametrize("categories", [(0, 1, 2), (1, 2, 3), (-2, -1, 0), (2**63 - 1, 2**63)])
def test_item_categories_allow_consecutive_integer_support_from_any_origin(
    model, categories, tmp_path
):
    pytest.importorskip("h5netcdf")
    source = fc.ItemInput(
        pd.DataFrame({"item_id": ["i"], "trait": ["A"], "location_side": [1]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1], "item_id": ["i"], "response": [categories[0]]}),
            format="item_long",
        ),
        model=model,
        categories=categories,
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    assert prepared.constant_data.attrs["units"][0]["scale"]["categories"] == list(categories)
    metadata = prepared.result_metadata
    category = metadata["categories"]
    count = len(categories)
    assert category.unit_index.values.tolist() == [0] * count
    assert category.position.values.tolist() == list(range(count))
    assert category.value_is_integer.values.tolist() == [True] * count
    support = metadata["response_support/b0"]
    for record in (category, support):
        decoded = [
            int(text) if text else int(integer)
            for integer, text in zip(
                record.value_integer.values, record.value_text.values, strict=True
            )
        ]
        assert decoded == list(categories)
    if categories[-1] == 2**63:
        assert category.value_integer.values.tolist() == [2**63 - 1, 0]
        assert category.value_text.values.tolist() == ["", str(2**63)]
    assert support.sizes["b0_category"] == count
    assert prepared.observed_data.b0_response.values.tolist() == [0]
    target = tmp_path / "item-support.nc"
    xr.DataTree.from_dict(metadata).to_netcdf(target, engine="h5netcdf")
    with xr.open_datatree(target, engine="h5netcdf") as saved:
        xr.testing.assert_identical(saved["categories"].to_dataset(), category)
        xr.testing.assert_identical(saved["response_support/b0"].to_dataset(), support)
        assert saved["design"].unit_scale_kind.values.tolist() == ["category"]


@pytest.mark.parametrize("response", [0, None])
def test_item_observation_source_columns_exist_even_without_answered_events(response):
    source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["i"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1], "item_id": ["i"], "response": [response]}),
            format="item_long",
        ),
        model=fc.GPCM(),
        categories=(0, 1, 2),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    for field in ("response_statement_1", "response_statement_2"):
        variable = prepared.observed_data.variables[field]
        assert variable.dims == ("response_row",)
        assert variable.dtype.kind == "i"
        assert variable.values.tolist() == [-1]
    assert prepared.constant_data.sizes["obs"] == (1 if response is not None else 0)
    assert prepared.output_coords["obs_block_index"].values.tolist() == (
        [-1] if response is not None else []
    )


def test_saved_category_codes_distinguish_integer_and_text_and_keep_unobserved(tmp_path):
    pytest.importorskip("h5netcdf")
    source = replace(
        graded_input("1"),
        responses=fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p", "q"],
                    "block_id": ["graded", "graded"],
                    "response": pd.Series([1, "1"], dtype=object),
                }
            ),
            format="block_long",
        ),
        scale=fc.CategoryScale((1, "1", "unobserved"), higher_prefers="second"),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    metadata = prepared.result_metadata
    category = metadata["categories"]
    assert category.value_is_integer.values.tolist() == [True, False, False]
    assert category.value_integer.values.tolist() == [1, 0, 0]
    assert category.value_text.values.tolist() == ["", "1", "unobserved"]
    assert category.position.values.tolist() == [0, 1, 2]
    assert prepared.observed_data.b0_response.values.tolist() == [0, 1]
    target = tmp_path / "typed-categories.nc"
    xr.DataTree.from_dict(metadata).to_netcdf(target, engine="h5netcdf")
    with xr.open_datatree(target, engine="h5netcdf") as saved:
        codes = saved["categories"].to_dataset()
        decoded = [
            int(integer) if known else str(text)
            for known, integer, text in zip(
                codes.value_is_integer.values,
                codes.value_integer.values,
                codes.value_text.values,
                strict=True,
            )
        ]
        assert decoded == [1, "1", "unobserved"]
        assert [type(value) for value in decoded] == [int, str, str]
        xr.testing.assert_identical(
            saved["response_support/b0"].to_dataset(), metadata["response_support/b0"]
        )
        assert saved["design"].unit_scale_higher_prefers.values.tolist() == ["second"]


def test_saved_pick_and_mole_supports_keep_unknown_positions(tmp_path):
    import numpy as np

    pytest.importorskip("h5netcdf")
    prepared = fc.PreparedInputs.from_inputs(mixed_inputs()[0])
    metadata = prepared.result_metadata
    pick, mole = metadata["response_support/b1"], metadata["response_support/b2"]
    np.testing.assert_array_equal(pick.pattern, [[1, -1, -1], [2, -1, -1], [3, -1, -1]])
    assert pick.attrs["pattern_kind"] == "ranking_positions"
    assert pick.attrs["response_format"] == "pick"
    assert mole.pattern.shape == (12, 4)
    assert np.all(mole.pattern.values[:, 1:3] == -1)
    assert set(map(tuple, mole.pattern.values[:, [0, 3]])) == {
        (a, b) for a in range(1, 5) for b in range(1, 5) if a != b
    }
    assert mole.attrs["response_format"] == "mole"
    target = tmp_path / "partial-rank-support.nc"
    xr.DataTree.from_dict(metadata).to_netcdf(target, engine="h5netcdf")
    with xr.open_datatree(target, engine="h5netcdf") as saved:
        xr.testing.assert_identical(saved["response_support/b1"].to_dataset(), pick)
        xr.testing.assert_identical(saved["response_support/b2"].to_dataset(), mole)


@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_saved_partial_comparison_support_keeps_signature_pair_order(tmp_path):
    import numpy as np

    pytest.importorskip("h5netcdf")
    source = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a", "c"], "B": ["b", "d"]}, blocks={"partial": ["a", "b", "c", "d"]}
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p", "p", "p"],
                    "block_id": ["partial", "partial", "partial"],
                    "statement_1": ["a", "b", "a"],
                    "statement_2": ["b", "c", "c"],
                    "response": [1, 0, None],
                }
            ),
            format="comparison_long",
        ),
        model=fc.MGPPM(),
        administration=pd.DataFrame(
            {
                "person_id": ["q", "p"],
                "block_id": ["partial", "partial"],
                "status": ["not_administered", "answered"],
            }
        ),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    metadata = prepared.result_metadata
    support = metadata["response_support/b0"]
    assert support.attrs["pattern_kind"] == "comparison_signature"
    assert support.attrs["response_format"] == "rank"
    np.testing.assert_array_equal(support.pairs, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(support.pattern, [[0, 0], [0, 1], [1, 0], [1, 1]])
    assert support.pattern.shape == (4, 2)
    assert support.pattern.dims[-1] == support.pairs.dims[0]
    target = tmp_path / "comparison-support.nc"
    xr.DataTree.from_dict(metadata).to_netcdf(target, engine="h5netcdf")
    with xr.open_datatree(target, engine="h5netcdf") as saved:
        xr.testing.assert_identical(saved["response_support/b0"].to_dataset(), support)
        source_rows = saved["design"].observation_source_response_row_index.values
        np.testing.assert_array_equal(source_rows, [0, 1, 2])
        np.testing.assert_array_equal(saved["design"].observation_source_obs_index, [0, 0, 0])
        np.testing.assert_array_equal(saved["design"].obs_input_event_index, [1])


@pytest.mark.parametrize(
    "declarations",
    [
        (
            fc.TIRT(loading_anchors={"a": 1.0, "b": 1.0}),
            fc.TIRT(loading_anchors={"a": 2.0, "b": 1.0}),
        ),
        (fc.MGPPM(reference_statements=("a",)), fc.MGPPM(reference_statements=("b",))),
        (
            fc.MGPPM(priors={"ranking_intercept": pm.Normal.dist(sigma=1)}),
            fc.MGPPM(priors={"ranking_intercept": pm.Normal.dist(sigma=2)}),
        ),
        (fc.TIRT(link="probit"), fc.TIRT(link="logit")),
    ],
)
def test_inline_model_conflicts_compare_complete_declarations_in_both_row_orders(declarations):
    source = pair_input()
    frame = pd.DataFrame(
        {
            "block_id": ["pair", "pair"],
            "position": [1, 2],
            "statement_id": ["a", "b"],
            "trait": ["A", "B"],
            "keying": [1, 1],
            "model": declarations,
        }
    )
    for design in (frame, frame.iloc[::-1]):
        with pytest.raises(ValueError, match="conflicting model declarations"):
            fc.PreparedInputs.from_inputs(replace(source, design=design))


def test_equivalent_native_inline_priors_are_compared_by_law_not_rv_identity():
    frame = pd.DataFrame(
        {
            "block_id": ["pair", "pair"],
            "position": [1, 2],
            "statement_id": ["a", "b"],
            "trait": ["A", "B"],
            "keying": [1, 1],
            "model": [
                fc.MGPPM(
                    reference_statements=("a",),
                    priors={"ranking_intercept": pm.Normal.dist(sigma=2)},
                )
                for _ in range(2)
            ],
        }
    )
    first = fc.PreparedInputs.from_inputs(replace(pair_input(), design=frame))
    second = fc.PreparedInputs.from_inputs(replace(pair_input(), design=frame.iloc[::-1]))
    assert_same_preparation(first, second)


def test_old_task_keywords_are_rejected_without_a_compatibility_path():
    with pytest.raises(TypeError, match="task"):
        fc.BlockSpec("B1", task="pick")
    with pytest.raises(TypeError, match="default_task"):
        fc.FCInput("", None, default_task="pick")
    source = pair_input()
    with pytest.raises(ValueError, match="unknown override fields.*task"):
        fc.PreparedInputs.from_inputs(replace(source, blocks={"pair": {"task": "pick"}}))
    with pytest.raises(ValueError, match="legacy design columns.*task"):
        fc.PreparedInputs.from_inputs(
            replace(
                source,
                design=pd.DataFrame(
                    {
                        "block_id": ["pair"] * 2,
                        "position": [1, 2],
                        "statement_id": ["a", "b"],
                        "trait": ["A", "B"],
                        "keying": [1, 1],
                        "task": ["pick"] * 2,
                    }
                ),
            )
        )
