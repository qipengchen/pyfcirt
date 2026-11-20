"""Independent combinatorial checks of the actual ranking event support."""

import numpy as np
import pytest

from pyfcirt.common.validation import identifier
from pyfcirt.data import encoding


def test_integer_categories_with_missing_cells_keep_declared_support():
    import pandas as pd

    import pyfcirt as fc

    source = fc.ItemInput(
        {"traits": {"A": [1]}},
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1, 2], "item_id": [1, 1], "response": [1.0, np.nan]}),
            format="item_long",
        ),
    )
    state = fc.PreparedInputs.from_inputs(source)
    assert state.data["constant_data"].person.values.tolist() == [1, 2]
    assert state.data["constant_data"].sizes["obs"] == 1
    assert state.data["observed_data"].response_status.values.tolist() == ["answered", "missing"]


def test_supported_formats_and_tasks_have_no_compatibility_aliases():
    assert encoding._RESPONSE_FORMATS == {"rank", "pick", "mole", None}
    assert "comparison" not in encoding._RESPONSE_FORMATS
    assert {"block_long", "comparison_long", "item_long"} <= encoding._FORMATS


def test_calibration_identity_covers_structural_fixed_values_and_typed_people():
    import pandas as pd

    import pyfcirt as fc

    def prepared(fixed, person):
        design = pd.DataFrame(
            {
                "item_id": ["a", "b"],
                "trait": ["A", "A"],
                "fix_loading": [fixed, np.nan],
            }
        )
        responses = fc.ResponseTable(
            pd.DataFrame(
                {"person_id": [person, person], "item_id": ["a", "b"], "response": [1, 0]}
            ),
            format="item_long",
        )
        return (
            fc.Model()
            .add_items(design, responses.frame, format=responses.format)
            .prepare()
            .prepared
        )

    first = prepared(0.7, 1)
    changed_fixed = prepared(1.2, 1)
    new_person = prepared(0.7, "1")
    assert first.calibration_identity != changed_fixed.calibration_identity
    assert first.calibration_identity == new_person.calibration_identity
    assert first.person_keys != new_person.person_keys
    assert first.scoring_batch_identity != new_person.scoring_batch_identity


def test_scoring_evidence_distinguishes_changed_partial_ranking_constraints():
    import pandas as pd

    import pyfcirt as fc

    design = fc.FCDesignMapping(traits={"A": ["a", "c"], "B": ["b"]}, blocks={"B": ["a", "b", "c"]})

    def prepare(other):
        responses = fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["P"],
                    "block_id": ["B"],
                    "statement_1": ["a"],
                    "statement_2": [other],
                    "response": [1],
                }
            ),
            format="comparison_long",
        )
        return fc.PreparedInputs.from_inputs(
            fc.FCInput(design, responses, model=fc.TIRTRank(loading_anchors={"a": 1.0}))
        )

    first, changed = prepare("b"), prepare("c")
    assert first.observed_person_keys == changed.observed_person_keys
    assert first.scoring_batch_identity == prepare("b").scoring_batch_identity
    assert first.scoring_batch_identity != changed.scoring_batch_identity


@pytest.mark.parametrize("value", [7, "7"])
def test_identifier_preserves_type(value):
    assert type(identifier(value, "id")) is type(value)


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.2, "", "  "])
def test_identifier_rejects_ambiguous_values(value):
    with pytest.raises(ValueError):
        identifier(value, "id")


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("response_format", ["rank", "pick", "mole"])
def test_native_catalog_partitions_complete_orders_without_completing_middle(size, response_format):
    from itertools import permutations

    catalog = encoding.ResponseEncoding().catalog(size, response_format)
    orders = list(permutations(range(1, size + 1)))
    expected = []
    for event in catalog["labels"]:
        expected.append(
            [
                all(value is None or order[pos] == value for pos, value in enumerate(event))
                for order in orders
            ]
        )
    membership = catalog.get("membership", np.eye(len(orders), dtype=bool))
    np.testing.assert_array_equal(membership, expected)
    np.testing.assert_array_equal(np.asarray(expected).sum(axis=0), 1)
    expected_count = (
        len(orders)
        if response_format == "rank"
        else size
        if response_format == "pick"
        else size * (size - 1)
    )
    assert len(expected) == expected_count


@pytest.mark.parametrize("size", [2, 3, 4])
def test_every_recorded_pair_subset_is_an_exact_partition(size):
    from itertools import combinations, permutations

    all_pairs = list(combinations(range(size), 2))
    orders = list(permutations(range(size)))
    cache = encoding.ResponseEncoding()
    full = cache.catalog(size, "rank")
    for count in range(1, len(all_pairs) + 1):
        for pairs in combinations(all_pairs, count):
            catalog = cache.catalog(size, "rank", pairs)
            assert catalog["orders"] is full["orders"]
            assert catalog["contrasts"] is full["contrasts"]
            pick = cache.catalog(size, "pick", pairs)
            assert pick["membership"] is catalog["membership"]
            assert pick["response_format"] == "pick"
            signatures = [
                tuple(int(order.index(a) < order.index(b)) for a, b in pairs) for order in orders
            ]
            expected = [
                [signature == event for signature in signatures] for event in catalog["labels"]
            ]
            np.testing.assert_array_equal(catalog["membership"], expected)
            np.testing.assert_array_equal(np.asarray(expected).sum(axis=0), 1)


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("response_format", ["rank", "pick", "mole"])
@pytest.mark.parametrize("family", ["TIRTRank", "MGPPM", "TwoPLRank", "GGUMRank"])
def test_ranking_prepares_basis_only_for_augmented_observations(size, response_format, family):
    import pandas as pd

    import pyfcirt as fc

    names = [f"S{i}" for i in range(1, size + 1)]
    if response_format == "rank":
        response = tuple(range(1, size + 1))
    elif response_format == "pick":
        response = (1, *([None] * (size - 1)))
    else:
        response = (1, *([None] * (size - 2)), size)
    design = pd.DataFrame(
        {
            "statement_id": names,
            "block_id": "B",
            "trait": "theta",
            "position": range(1, size + 1),
            "keying": 1,
            "location_side": 1,
            "fix_loading": [1.0, *([np.nan] * (size - 1))],
        }
    )
    spec = getattr(fc, family)()
    responses = [response]
    if family == "TIRTRank":
        reversed_positions = tuple(
            None if value is None else size + 1 - value for value in response
        )
        responses += [reversed_positions, response]
    inputs = fc.FCInput(
        design,
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": range(len(responses)), "block_id": "B", "response": responses}
            ),
            format="block_long",
        ),
        model=spec,
        response_format=response_format,
    )
    prepared = fc.PreparedInputs.from_inputs(inputs)
    (batch,) = prepared.batches
    category = batch.events.labels.index(response)
    np.testing.assert_array_equal(batch.y, [batch.events.labels.index(row) for row in responses])
    np.testing.assert_array_equal(batch.events.orders[0], np.arange(size))
    np.testing.assert_array_equal(batch.events.scores[0], np.arange(size - 1, -1, -1))
    if response_format != "rank":
        assert batch.events.membership[category, 0]
    if family != "TIRTRank":
        assert batch.events.basis is None and batch.events.mask is None
        if size == 4:
            graph = fc.Model(prepared).build()
            observed = graph[batch.response_name]
            np.testing.assert_array_equal(graph.rvs_to_values[observed].eval(), batch.y)
            assert np.isfinite(graph.compile_logp()(graph.initial_point()))
        return
    basis = batch.events.basis[0]
    assert basis.shape == (size - 1, size)
    np.testing.assert_array_equal(basis @ np.ones(size), 0)
    assert np.linalg.matrix_rank(basis) == size - 1
    complete = response_format == "rank" or size == 2 or (response_format == "mole" and size == 3)
    assert (batch.events.mask is None) == complete
    np.testing.assert_array_equal(batch.events.basis[2], basis)
    assert not np.array_equal(batch.events.basis[1], basis)
    if complete:
        np.testing.assert_array_equal(basis @ np.arange(size, 0, -1), 1)
    else:
        np.testing.assert_array_equal(batch.events.mask[0], batch.events.mask[2])


def test_complete_six_statement_rank_does_not_materialize_identity_partition():
    from math import factorial

    catalog = encoding.ResponseEncoding().catalog(6, "rank")
    assert len(catalog["labels"]) == factorial(6)
    assert "membership" not in catalog
    assert catalog["orders"].shape == (factorial(6), 6)


@pytest.mark.parametrize("pairs", [[(0, -1)], [(-1, 1)], [(0, 1), (-1, -1)]])
def test_unit_events_reject_partially_specified_or_mixed_comparisons(pairs):
    import pandas as pd

    from pyfcirt.models.specification import TIRTRank

    rows = pd.DataFrame(pairs, columns=["position_1", "position_2"])
    rows["parent_index"] = np.arange(len(rows))
    rows["response"] = None
    unit = dict(
        response_format="rank",
        scale={"categories": (0, 1)},
        channel="fc",
        size=3,
    )
    with pytest.raises(ValueError, match="cannot mix block events and comparison rows"):
        encoding._unit_events(
            unit,
            rows,
            specification=TIRTRank(),
            generate=True,
            encoding=encoding.ResponseEncoding(),
        )
