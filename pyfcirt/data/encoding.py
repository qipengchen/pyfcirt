"""Normalize labelled inputs and encode actual response events once."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations, permutations, repeat
from math import factorial
from numbers import Integral, Real
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

from pyfcirt.common.validation import identifier, identifiers, require_frame
from pyfcirt.data.arrays import LayoutBuilder, merge_prepared
from pyfcirt.data.design import (
    BlockSpec,
    CategoryScale,
    ComparisonColumn,
    FCDesignMapping,
    FCInput,
    ItemDesignMapping,
    ItemInput,
    ItemOverride,
    ResponseTable,
    UnitIntervalScale,
    _fc_membership_table,
)

if TYPE_CHECKING:
    from pyfcirt.data.prepare import PreparedInputs

OBSERVED_STATUSES = frozenset({"answered", "missing", "refused", "not_administered"})
SCHEDULE_STATUSES = (OBSERVED_STATUSES - {"answered"}) | {"scheduled"}
UNIT_COLUMNS = {"fc": "block_id", "items": "item_id"}
_RESPONSE_STATES = {
    "observed": ("answered", OBSERVED_STATUSES),
    "generate": ("scheduled", SCHEDULE_STATUSES),
}
_RESPONSE_FORMATS = frozenset({"rank", "pick", "mole", None})
_RESPONSE_KEYS = ("person_id", "unit_id")
_PAIR_POSITIONS = ("position_1", "position_2")
_PAIR_FIELDS = (*_PAIR_POSITIONS, "pair_reversed")
_CHANNEL_FORMATS = {
    ("block_id", "observed"): {
        "block_long",
        "block_wide",
        "statement_rank",
        "comparison_long",
        "comparison_wide",
    },
    ("item_id", "observed"): {"item_long", "item_wide"},
    ("block_id", "generate"): {"block_schedule", "comparison_schedule"},
    ("item_id", "generate"): {"item_schedule"},
}
_FORMATS = frozenset().union(*_CHANNEL_FORMATS.values())


@dataclass(frozen=True, slots=True, kw_only=True)
class NormalizedChannel:
    """Validated channel with its local administration and comparison positions."""

    channel: str
    tables: Mapping[str, pd.DataFrame]
    input_format: str | None
    people: tuple


def _missing(value):
    if not pd.api.types.is_scalar(value):
        return False
    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _integer(value, context):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value != int(value)
    ):
        raise ValueError(f"{context} must be a finite integer; got {value!r}")
    return int(value)


def _frame(value, required, name):
    require_frame(value, required, name)
    if not value.columns.is_unique:
        raise ValueError(f"{name} has duplicate columns")
    result = value.copy(deep=True).reset_index(drop=True)
    for column in result:
        # Only object cells need inspection; frozen model declarations remain shared.
        if (name == "design" and column == "model") or not pd.api.types.is_object_dtype(
            result[column]
        ):
            continue
        kinds = {type(v) for v in result[column].to_numpy()}
        immutable = all(
            kind in {str, int, float, bool, complex, type(None)}
            or issubclass(kind, (np.integer, np.floating, np.bool_, np.str_))
            for kind in kinds
        )
        if not immutable:
            result[column] = pd.Series([deepcopy(v) for v in result[column]], dtype=object)
    return result


def _records(rows, columns=None):
    """Keep heterogeneous label types and nested events out of pandas coercion."""
    return pd.DataFrame(rows, columns=columns, dtype=object)


def _sequence(values, context):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence | np.ndarray):
        raise TypeError(f"{context} must be an ordered sequence")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{context} must be one-dimensional")
    return tuple(values)


def _categories(values, context):
    values = tuple(identifier(v, f"{context} category") for v in _sequence(values, context))
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError(f"{context} must contain at least two unique ordered categories")
    return values


def _mapping_design(value, *, fc):
    expected = FCDesignMapping if fc else ItemDesignMapping
    if not isinstance(value, (Mapping, expected)):
        raise TypeError("design must be a canonical DataFrame, design string or design mapping")
    if isinstance(value, Mapping):
        allowed = {"traits", "blocks", "keying"} if fc else {"traits"}
        if set(value) - allowed or not (allowed - {"keying"}) <= value.keys():
            raise ValueError(f"design mapping fields must be {sorted(allowed)}")
        value = expected(**value)
    if not isinstance(value.traits, Mapping) or not value.traits:
        raise TypeError("design.traits must be a nonempty mapping")
    assignments = {}
    for raw_trait, raw_members in value.traits.items():
        trait = identifier(raw_trait, "trait")
        members = _sequence(raw_members, f"Trait {trait!r} members")
        if not members:
            raise ValueError(f"Trait {trait!r} needs at least one member")
        for raw_member in members:
            member = identifier(raw_member, "statement_id" if fc else "item_id")
            if fc and member in assignments:
                raise ValueError(f"Statement {member!r} has multiple trait assignments")
            assignments.setdefault(member, []).append(trait)
    if not fc:
        return _records(
            [(item, trait) for item, values in assignments.items() for trait in values],
            ["item_id", "trait"],
        )
    if not isinstance(value.blocks, Mapping) or not value.blocks:
        raise TypeError("design.blocks must be a nonempty mapping")
    if value.keying is not None and not isinstance(value.keying, Mapping):
        raise TypeError("design.keying must be a mapping")
    keying = {} if value.keying is None else value.keying
    for statement in keying:
        identifier(statement, "keying statement_id")
    if set(keying) - set(assignments):
        raise ValueError("keying refers to unknown statements")
    blocks = {}
    for raw_block, raw_members in value.blocks.items():
        block = identifier(raw_block, "block_id")
        members = _sequence(raw_members, f"Block {block!r} members")
        if len(members) < 2:
            raise ValueError(f"Block {block!r} needs at least two distinct statements")
        blocks[block] = [identifier(member, "statement_id") for member in members]
    return _fc_membership_table(
        blocks, {statement: traits[0] for statement, traits in assignments.items()}, keying
    )


def _design_table(source, *, fc):
    value = source.design
    defaults, options = {}, {}
    if isinstance(value, pd.DataFrame):
        required = (
            ("block_id", "position", "statement_id", "trait", "keying")
            if fc
            else ("item_id", "trait")
        )
        table = _frame(value, required, "design")
    elif isinstance(value, str):
        from pyfcirt.data.design import parse_design, parse_item_design

        if fc:
            table, declarations = parse_design(value)
            defaults, options = declarations["defaults"], declarations["blocks"]
        else:
            table = parse_item_design(value)
    else:
        table = _mapping_design(value, fc=fc)
    if table.empty:
        raise ValueError("design must contain at least one unit")
    for column in ("block_id", "statement_id", "trait") if fc else ("item_id", "trait"):
        table[column] = pd.Series(identifiers(table[column], f"design.{column}"), dtype=object)
    if not fc:
        if table.duplicated(["item_id", "trait"]).any():
            raise ValueError("Duplicate item/trait design rows")
    else:
        table["position"] = [_integer(v, "position") for v in table.position]
        table["keying"] = [_integer(v, "keying") for v in table.keying]
        if not table.keying.isin((-1, 1)).all():
            raise ValueError("keying must be -1 or +1")
        if table.duplicated(["block_id", "statement_id"]).any():
            raise ValueError("A statement may appear only once in a block")
        optional = [column for column in ("location_side", "text") if column in table]
        checked = table if optional else table.loc[table.statement_id.duplicated(keep=False)]
        for statement, rows in checked.groupby("statement_id", sort=False):
            if rows.trait.nunique() != 1 or rows.keying.nunique() != 1:
                raise ValueError(f"Conflicting trait or keying for statement {statement!r}")
            for column in optional:
                if rows[column].nunique(dropna=False) != 1:
                    raise ValueError(f"Conflicting {column} for statement {statement!r}")
        codes, labels = pd.factorize(table.block_id, sort=False)
        positions = table.position.to_numpy()
        order = np.lexsort((positions, codes))
        sizes = np.bincount(codes)
        expected = np.arange(len(table)) - np.repeat(sizes.cumsum() - sizes, sizes) + 1
        bad = np.flatnonzero((sizes[codes[order]] < 2) | (positions[order] != expected))
        if len(bad):
            block = labels.tolist()[codes[order[bad[0]]]]
            raise ValueError(f"Block {block!r}: positions must be 1..K with K >= 2")
        table = table.iloc[order].reset_index(drop=True)
    if "location_side" in table:
        values = [
            None if _missing(v) else _integer(v, "location_side") for v in table.location_side
        ]
        if any(v is not None and v not in (-1, 0, 1) for v in values):
            raise ValueError("location_side must be -1, 0 or +1")
        table["location_side"] = pd.Series(values, dtype=object)
    return table, defaults, options


def _scale_record(value, *, unit, required=False):
    if value is None:
        if required:
            raise ValueError(f"Unit {unit!r} requires its complete declared scale")
        return dict(kind="none", categories=(), higher_prefers="first", labels=None)
    if not isinstance(value, CategoryScale | UnitIntervalScale):
        raise TypeError(f"Unit {unit!r}: scale must be CategoryScale or UnitIntervalScale")
    if not isinstance(value.higher_prefers, str) or value.higher_prefers not in {"first", "second"}:
        raise ValueError("scale.higher_prefers must be 'first' or 'second'")
    if isinstance(value, UnitIntervalScale):
        return dict(
            kind="unit_interval", categories=(), higher_prefers=value.higher_prefers, labels=None
        )
    categories = _categories(value.categories, f"Unit {unit!r} categories")
    labels = value.labels
    if labels is not None:
        labels = _sequence(labels, "scale.labels")
        if len(labels) != len(categories) or any(
            not isinstance(v, str) or not v.strip() for v in labels
        ):
            raise ValueError("scale.labels must contain one nonempty string per category")
    return dict(
        kind="category", categories=categories, higher_prefers=value.higher_prefers, labels=labels
    )


def _model_declaration(value, *, channel):
    """Accept only current declarations; scientific resolution belongs to layout."""
    from pyfcirt.data.design import _ModelDeclaration
    from pyfcirt.models.specification import MEASUREMENT_MODELS

    names = tuple(name for name, model in MEASUREMENT_MODELS.items() if model.channel == channel)
    if isinstance(value, str | _ModelDeclaration):
        name = value if isinstance(value, str) else value.name
        if name not in names:
            raise ValueError(f"Unknown model {name!r}; expected one of {names!r}")
        options = {} if isinstance(value, str) else value.options
        return MEASUREMENT_MODELS[name](**options)
    if not any(type(value) is MEASUREMENT_MODELS[name] for name in names):
        raise TypeError(f"Model must be one of the declarations {names!r}")
    return value


def _measurement_options(values, *, channel, context):
    """Normalize every declaration through the same constructors and response_format vocabulary."""
    from pyfcirt.data.design import _ModelDeclaration

    result = dict(values)
    if isinstance(result.get("model"), _ModelDeclaration):
        context = f"{context}, line {result['model'].line}"
    try:
        if "model" in result:
            result["model"] = _model_declaration(result["model"], channel=channel)
        if "response_format" in result and (
            result["response_format"] is not None
            and (
                not isinstance(result["response_format"], str)
                or result["response_format"] not in _RESPONSE_FORMATS
            )
        ):
            raise ValueError("response_format must be rank, pick, mole or None")
    except (TypeError, ValueError) as error:
        raise type(error)(f"{context}: {error}") from error
    return result


def _unit_override(value, *, fc, context):
    """Resolve the explicit fields of a block or item override."""
    declaration, fields = (
        (BlockSpec, ("model", "response_format", "scale"))
        if fc
        else (ItemOverride, ("model", "categories"))
    )
    if isinstance(value, declaration):
        return {
            field: option
            for field in fields
            if (option := getattr(value, field))
            is not (Ellipsis if field == "response_format" else None)
        }
    if isinstance(value, Mapping):
        if unknown := value.keys() - set(fields):
            raise ValueError(f"{context}: unknown override fields {sorted(unknown, key=repr)!r}")
        return dict(value)
    if not fc:
        raise TypeError(f"{context}: expected a model/categories mapping")
    values = _sequence(value, f"{context} override")
    if len(values) not in (1, 2):
        raise ValueError(f"{context}: expected [model] or [model, response_format]")
    return dict(zip(fields, values, strict=False))


def _design_row_options(columns, positions, *, channel, context):
    """Require declarations repeated across the same unit to agree."""
    from pyfcirt.models.priors import PriorPreparation

    result = {}
    for field, values in columns.items():
        declared = []
        for value in values[positions]:
            if value is Ellipsis:
                continue
            if _missing(value) and not (field == "response_format" and value is None):
                continue
            setting = _measurement_options({field: value}, channel=channel, context=context)
            declared.append(setting[field])
        if not declared:
            continue
        # Agreement includes identification, independently of parameter sharing.
        comparable = (
            [(type(value), PriorPreparation.key(vars(value))) for value in declared]
            if field == "model"
            else declared
        )
        if any(value != comparable[0] for value in comparable[1:]):
            raise ValueError(f"{context}: conflicting {field} declarations")
        result[field] = declared[0]
    return result


def _unit_overrides(declarations, groups, *, channel):
    """Validate override identities and normalize only explicitly supplied fields."""
    column = UNIT_COLUMNS[channel]
    fc = channel == "fc"
    overrides = {}
    if isinstance(declarations, Mapping):
        entries = declarations.items()
    else:
        declarations = _sequence(declarations, "Unit overrides")
        if any(not isinstance(value, BlockSpec if fc else ItemOverride) for value in declarations):
            raise TypeError("Unit overrides must be BlockSpec or ItemOverride declarations")
        entries = ((getattr(value, column), value) for value in declarations)
    for label, override in entries:
        unit = identifier(label, column)
        if unit not in groups:
            raise ValueError(f"Unknown {column} {unit!r} in overrides")
        if unit in overrides:
            raise ValueError(f"Duplicate override for {column} {unit!r}")
        context = f"{'Block' if fc else 'Item'} {unit!r}"
        overrides[unit] = _measurement_options(
            _unit_override(override, fc=fc, context=context), channel=channel, context=context
        )
    return overrides


def _resolved_units(design, *, channel, defaults, wildcard, inline_options, overrides):
    """Yield effective unit options in design order using the declared precedence."""
    if legacy := {"block_type", "item_type", "task"}.intersection(design.columns):
        raise ValueError(
            f"Unsupported legacy design columns {sorted(legacy)}; "
            "use 'model' and 'response_format' columns"
        )
    resolved_defaults = _measurement_options(defaults, channel=channel, context="Input defaults")
    resolved_defaults.update(
        _measurement_options(wildcard, channel=channel, context="String wildcard defaults")
    )
    column = UNIT_COLUMNS[channel]
    groups = design.groupby(column, sort=False, observed=True).indices
    overrides = _unit_overrides(overrides, groups, channel=channel)
    design_options = {
        field: design[field].to_numpy()
        for field in (("model", "response_format", "scale") if channel == "fc" else ("model",))
        if field in design
    }
    for unit in design[column].unique():
        positions = groups[unit]
        context = f"Unit {unit!r}"
        inline = _measurement_options(
            inline_options.get(unit, {}), channel=channel, context=context
        )
        inline.update(
            _design_row_options(design_options, positions, channel=channel, context=context)
        )
        yield unit, len(positions), resolved_defaults | inline | overrides.get(unit, {})


def _block_scale(unit, size, model, response_format, scale, *, beta, comparison_input):
    """Normalize the scale of a block and validate its declared response support."""
    if scale is None and response_format is None and not beta:
        # Declare binary support even when only one category was observed.
        scale = CategoryScale((0, 1))
    record = _scale_record(scale, unit=unit, required=response_format is None)
    if response_format is None:
        if size != 2 and not (model.comparison_based and comparison_input):
            raise ValueError("Raw block-level intensity responses require exactly two statements")
        if not (model.comparison_based or model.graded_events):
            raise ValueError(
                f"{type(model).__name__} requires a rank, pick or mole response_format"
            )
    if (
        response_format is not None
        and record["kind"] != "none"
        and not (
            record["kind"] == "category"
            and record["categories"] == (0, 1)
            and record["higher_prefers"] == "first"
        )
    ):
        raise ValueError(
            f"Unit {unit!r}: {response_format} requires scale omitted or "
            "CategoryScale((0, 1), higher_prefers='first'); "
            "use response_format=None for other preference scales"
        )
    return record


def _item_scale(unit, categories):
    """Normalize consecutive low-to-high item codes before building a unit record."""
    categories = _categories(categories, f"Item {unit!r}")
    if any(not isinstance(value, int) for value in categories) or any(
        upper != lower + 1 for lower, upper in zip(categories[:-1], categories[1:], strict=True)
    ):
        raise ValueError(
            f"Item {unit!r}: categories must be consecutive integers in increasing order"
        )
    return dict(kind="category", categories=categories, higher_prefers="first", labels=None)


def _unit_record(unit, model, *, channel, response_format, size, scale, beta):
    """Build one unit record from resolved and normalized declarations."""
    continuous = scale["kind"] == "unit_interval"
    if beta and not (continuous and response_format is None):
        raise ValueError(
            "Beta models require an explicit unit-interval scale and response_format=None"
        )
    if continuous and not beta:
        raise ValueError(
            f"{type(model).__name__} is categorical and cannot use a unit-interval scale"
        )
    if model.category_count is not None and len(scale["categories"]) != model.category_count:
        raise ValueError(
            f"{type(model).__name__} requires exactly {model.category_count} "
            "complete item categories"
        )
    return dict(
        channel=channel,
        unit_id=unit,
        model=model,
        response_format=response_format,
        size=size,
        scale=scale,
        likelihood_kind="rank" if model.augmented_ranking else "beta" if beta else "categorical",
    )


def _block_units(source, design, defaults, options):
    """Build FC unit records from their effective declarations."""
    input_format = source.responses.format
    comparison_input = isinstance(input_format, str) and input_format.startswith("comparison_")
    rows = []
    for unit, size, resolved in _resolved_units(
        design,
        channel="fc",
        defaults=dict(
            model=source.model, response_format=source.response_format, scale=source.scale
        ),
        wildcard=defaults,
        inline_options=options,
        overrides=source.blocks,
    ):
        model, response_format = resolved["model"], resolved["response_format"]
        beta = getattr(model, "link", "").startswith("beta_")
        scale = _block_scale(
            unit,
            size,
            model,
            response_format,
            resolved["scale"],
            beta=beta,
            comparison_input=comparison_input,
        )
        rows.append(
            _unit_record(
                unit,
                model,
                channel="fc",
                response_format=response_format,
                size=size,
                scale=scale,
                beta=beta,
            )
        )
    return _records(rows)


def _item_units(source, design, defaults, options):
    """Build single-item records from their effective declarations."""
    rows = []
    for unit, _, resolved in _resolved_units(
        design,
        channel="items",
        defaults=dict(model=source.model),
        wildcard=defaults,
        inline_options=options,
        overrides=source.items,
    ):
        scale = _item_scale(unit, resolved.get("categories", source.categories))
        rows.append(
            _unit_record(
                unit,
                resolved["model"],
                channel="items",
                response_format=None,
                size=1,
                scale=scale,
                beta=False,
            )
        )
    return _records(rows)


def _wide_rows(response_table, *, unit_column):
    frame = _frame(response_table.frame, (), "responses")
    person_column = response_table.person_id_column
    if person_column is None:
        # The original index is the declared wide person axis.
        persons = response_table.frame.index.tolist()
        columns = list(frame.columns)
    else:
        person_column = identifier(person_column, "person_id_column")
        if person_column not in frame:
            raise ValueError(f"Missing person ID column {person_column!r}")
        persons = list(frame[person_column])
        columns = [column for column in frame if column != person_column]
    persons = identifiers(persons, "person_id")
    if len(set(persons)) != len(persons):
        raise ValueError("Wide responses require unique person IDs")
    if not columns:
        raise ValueError("Wide responses require at least one response column")
    comparison = response_table.format == "comparison_wide"
    if comparison:
        columns = [identifier(column, "response column") for column in columns]
        mapping = {}
        for spec in response_table.comparison_columns:
            if not isinstance(spec, ComparisonColumn):
                raise TypeError("comparison_columns must contain ComparisonColumn declarations")
            column = identifier(spec.column, "comparison column")
            if column in mapping:
                raise ValueError(f"Duplicate comparison mapping for {column!r}")
            mapping[column] = dict(
                block_id=spec.block_id, statement_1=spec.statement_1, statement_2=spec.statement_2
            )
        if set(mapping) != set(columns):
            raise ValueError("comparison_columns must map every wide response column exactly once")
        targets = {
            field: identifiers((mapping[col][field] for col in columns), f"responses.{field}")
            for field in ("block_id", "statement_1", "statement_2")
        }
    else:
        if response_table.comparison_columns:
            raise ValueError("comparison_columns requires comparison_wide format")
        columns = identifiers(columns, f"responses.{unit_column}")
        targets = {unit_column: columns}
    return _records(
        {
            "person_id": np.repeat(np.asarray(persons, dtype=object), len(columns)),
            **{
                field: np.tile(np.asarray(labels, dtype=object), len(persons))
                for field, labels in targets.items()
            },
            "response": frame[columns].to_numpy(dtype=object).ravel(),
        }
    )


def has_statement_rows(storage_format):
    return isinstance(storage_format, str) and (
        storage_format == "statement_rank" or storage_format.startswith("comparison_")
    )


def _statement_order(units, statements, positions):
    """Key one-based presentation positions by the supplied unit and statement identities."""
    keys = pd.MultiIndex.from_arrays((units, statements))
    return pd.Series(np.asarray(positions, dtype=np.int64), index=keys)


def frozen_statement_order(constant, *, indexed=False):
    """Read presentation order from the frozen block membership array."""
    membership = constant.fc_statement.values
    block, position = np.nonzero(membership >= 0)
    statements = membership[block, position]
    if not indexed:
        block = constant.block.values[block]
        statements = constant.statement_label.values[statements]
    return _statement_order(block, statements, position + 1)


def _member_positions(order, units, statements):
    keys = pd.MultiIndex.from_arrays((units, statements))
    return order.reindex(keys, fill_value=-1).to_numpy()


def _source_rows(table, units, order, *, unit_column, purpose):
    storage_format = table.format
    if not isinstance(storage_format, str) or storage_format not in _FORMATS:
        raise ValueError(
            f"Unknown response format {storage_format!r}; expected one of {sorted(_FORMATS)}"
        )
    if storage_format not in _CHANNEL_FORMATS[unit_column, purpose]:
        raise ValueError(
            f"{storage_format!r} is incompatible with this channel and purpose={purpose!r}"
        )
    if storage_format.endswith("_wide"):
        rows = _wide_rows(table, unit_column=unit_column)
    else:
        if table.person_id_column is not None or table.comparison_columns:
            raise ValueError("person_id_column and comparison_columns apply only to wide formats")
        required = ["person_id", unit_column]
        if storage_format.startswith("comparison"):
            required.extend(("statement_1", "statement_2"))
        if storage_format == "statement_rank":
            required.extend(("statement_id", "rank"))
        elif purpose == "observed":
            required.append("response")
        rows = _frame(table.frame, required, "responses")
        for column in ("person_id", unit_column, "statement_id", "statement_1", "statement_2"):
            if column in rows:
                rows[column] = pd.Series(
                    identifiers(rows[column], f"responses.{column}"), dtype=object
                )
    if any(unit not in units for unit in rows[unit_column].unique()):
        raise ValueError("Responses contain unknown units")
    rows["unit_id"] = rows[unit_column].astype(object)
    if storage_format == "statement_rank":
        rows = _statement_ranks(rows, units, order)
    if purpose == "generate":
        if "response" in rows and any(not _missing(v) for v in rows.response):
            raise ValueError("Generation schedules cannot contain response values")
        rows["response"] = None
    return rows


def _statement_ranks(frame, units, order):
    frame["_position"] = _member_positions(order, frame.unit_id, frame.statement_id)
    rows = []
    for (person, block), selected in frame.groupby(["person_id", "unit_id"], sort=False):
        positions = selected.pop("_position")
        if (positions < 0).any() or not positions.is_unique:
            raise ValueError(f"Person {person!r}, block {block!r}: duplicate or unknown statements")
        size = units[block]["size"]
        preference = [None] * size
        for position, value in zip(positions, selected["rank"], strict=True):
            if _missing(value):
                continue
            rank = _integer(value, "rank")
            if not 1 <= rank <= size or preference[rank - 1] is not None:
                raise ValueError(f"Person {person!r}, block {block!r}: invalid or tied ranks")
            preference[rank - 1] = position
        record = dict(person_id=person, unit_id=block, response=tuple(preference))
        if "status" in selected:
            states = tuple(dict.fromkeys(selected.status))
            if len(states) != 1:
                raise ValueError("Statement-rank rows require one status per person/block")
            record["status"] = states[0]
        rows.append(record)
    fields = ["person_id", "unit_id", "response"]
    if "status" in frame:
        fields.append("status")
    return _records(rows, fields)


def _normalize_order(value, context, *, size, response_format, expected):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: ranking strings must be JSON arrays") from error
    if _missing(value):
        return None
    value = _sequence(value, f"{context}: preference order")
    if len(value) != size:
        raise ValueError(f"{context}: expected a preference-order list of length {size}")
    order = tuple(None if _missing(v) else _integer(v, context) for v in value)
    positions = {i for i, v in enumerate(order) if v is not None}
    if not positions:
        return None
    if positions != expected:
        raise ValueError(
            f"{context}: unknown ranks do not match declared {response_format} response_format"
        )
    known = [order[i] for i in positions]
    if len(set(known)) != len(known) or any(not 1 <= v <= size for v in known):
        raise ValueError(f"{context}: choices must be distinct positions in 1..{size}")
    return order


def _inside_unit_interval(value):
    return np.isfinite(value) & (0 < value) & (value < 1)


def _normalize_scalar(value, context, *, continuous, categories):
    if _missing(value):
        return None
    if continuous:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not _inside_unit_interval(value)
        ):
            raise ValueError(f"{context}: response must be finite with 0 < y < 1")
        return float(value)
    # Genuine NA cells can promote an integer category column to floating dtype.
    if isinstance(value, Real) and not isinstance(value, Integral):
        value = _integer(value, f"{context}: category")
    value = identifier(value, f"{context}: category")
    if value not in categories:
        raise ValueError(f"{context}: unknown category {value!r}; expected {categories!r}")
    return value


def normalize_administration(value, units, *, unit_column, allowed_statuses):
    """Validate a labelled table on the canonical person/unit/status columns."""
    if value is None:
        return None
    frame = _frame(value, ("person_id", unit_column, "status"), "administration")
    for name in ("person_id", unit_column):
        frame[name] = pd.Series(
            [identifier(v, f"administration.{name}") for v in frame[name]], dtype=object
        )
    frame["unit_id"] = frame[unit_column].astype(object)
    if any(unit not in units for unit in frame.unit_id.unique()):
        raise ValueError("Administration contains unknown units")
    if frame.duplicated(["person_id", "unit_id"]).any():
        raise ValueError("Duplicate person/unit administration entries")
    if any(not isinstance(v, str) or v not in allowed_statuses for v in frame.status):
        raise ValueError(f"Administration status must be one of {sorted(allowed_statuses)}")
    return frame[["person_id", "unit_id", "status"]]


def _integer_indices(values):
    return values.dtype.kind in "iu" or (
        values.dtype.kind == "O"
        and all(
            isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            for value in values
        )
    )


def _pair_row(first, second, size):
    """Position of an ordered pair in the lexicographic upper triangle."""
    return first * (2 * size - first - 1) // 2 + (second - first - 1)


def _ordered_pair(first, second):
    """Return canonical local positions and the direction of the source pair."""
    return np.minimum(first, second), np.maximum(first, second), first > second


def _schedule_positions(source, constant, person, unit):
    """Validate scheduled comparison keys and return canonical member positions."""
    positions = np.full((len(unit), 2), -1, dtype=np.int64)
    reversed_pair = np.zeros(len(unit), dtype=bool)
    names = ["schedule_statement_1", "schedule_statement_2"]
    if any(name not in source for name in names):
        raise ValueError("A generation schedule requires both statement index arrays")
    if any(source[name].dims != ("schedule",) for name in names):
        raise ValueError("Schedule statement indices must use the 'schedule' row dimension")
    first, second = (source[name].values for name in names)
    if not (_integer_indices(first) and _integer_indices(second)):
        raise ValueError("Schedule statement indices must be integers")
    pair = (first != -1) | (second != -1)
    if np.isin(unit[~pair], constant.pair_unit.values).any():
        raise ValueError("Comparison models require scheduled statement pairs")
    if pair.any():
        block = constant.unit_block.values[unit[pair]]
        if np.any(block < 0):
            raise ValueError("Item schedules use statement indices -1")
        order = frozen_statement_order(constant, indexed=True)
        a, b = (_member_positions(order, block, side[pair]) for side in (first, second))
        if np.any((a < 0) | (b < 0) | (a == b)):
            raise ValueError("Schedule comparisons must name two distinct unit members")
        left, right, reversed_pair[pair] = _ordered_pair(a - 1, b - 1)
        positions[pair] = np.column_stack((left, right))
    key = pd.MultiIndex.from_arrays((person, unit, *positions.T))
    if not key.is_unique:
        raise ValueError("The generation schedule contains duplicate events")
    return positions, reversed_pair, pair


def normalize_indexed_administration(source, constant, *, purpose):
    """Validate indexed administration and preserve its frozen entity positions."""
    generate = purpose == "generate"
    prefix = "schedule" if "schedule_person" in source else "administration"
    required = [f"{prefix}_{field}" for field in ("person", "unit", "status")]
    if any(name not in source for name in required):
        raise ValueError("administration requires person, unit and status index arrays")
    if any(source[name].dims != (prefix,) for name in required):
        raise ValueError(f"administration arrays must use the {prefix!r} row dimension")
    for name in ("unit_id", "unit_channel"):
        if name in source.coords and not source[name].equals(constant[name]):
            raise ValueError("Schedule unit coordinates must match the frozen measurement design")
    ids = source.person if "person" in source.coords else constant.person
    people = tuple(identifier(value, "person_id") for value in ids.values)
    if len(set(people)) != len(people) or generate and not people:
        raise ValueError("Administration person labels must be nonempty and unique")
    person, unit, status = (source[name].values for name in required)
    for positions, count in ((person, len(people)), (unit, constant.sizes["unit"])):
        if not _integer_indices(positions) or np.any((positions < 0) | (positions >= count)):
            raise ValueError("administration contains an invalid person or unit index")
    person, unit = person.astype(np.intp, copy=False), unit.astype(np.intp, copy=False)
    _, allowed = _RESPONSE_STATES[purpose]
    if not pd.Index(status).isin(allowed).all():
        raise ValueError(f"Administration status must be one of {sorted(allowed)}")
    columns = ["person", "unit", "status"]
    rows = pd.DataFrame(dict(person=person, unit=unit, status=status))
    if prefix == "schedule":
        positions, reversed_pair, pair = _schedule_positions(source, constant, person, unit)
        if generate:
            if np.intersect1d(unit[pair], unit[~pair]).size:
                raise ValueError("One unit cannot mix block events and scheduled comparisons")
            rows[list(_PAIR_POSITIONS)] = positions
            rows["pair_reversed"] = reversed_pair
    # Keep channel order from the frozen layout and row order within each channel.
    channels = constant.unit_channel.values
    codes = pd.Categorical(channels[unit], categories=pd.unique(channels)).codes
    rows = rows.iloc[np.argsort(codes, kind="stable")].reset_index(drop=True)
    parents = rows.groupby(columns[:2], sort=False).ngroup().to_numpy(dtype=np.int64)
    admin = rows.loc[~pd.Index(parents).duplicated(), columns].reset_index(drop=True)
    if np.any(admin.status.to_numpy()[parents] != rows.status.to_numpy()):
        raise ValueError("administration has conflicting person/unit statuses")
    scheduled = pd.DataFrame(columns=["status", "response", "administration_index", *_PAIR_FIELDS])
    if generate and prefix == "schedule":
        scheduled = rows[["status", *_PAIR_FIELDS]].assign(
            response=None, administration_index=parents
        )
    return admin, people, scheduled


_SCALAR_RESPONSE_TYPES = {
    str,
    int,
    float,
    np.str_,
    type(None),
    type(pd.NA),
    type(pd.NaT),
    *(
        np.dtype(code).type
        for code in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8")
    ),
}


def _response_values(rows, unit, *, label, reverse):
    """Normalize one unit's values and orient comparisons to its presentation order."""
    raw = rows.response.to_numpy()
    result = np.full(len(raw), None, dtype=object)

    def context(index):
        return f"Person {rows.person_id.iloc[index]!r}, unit {label!r}"

    response_format = unit["response_format"]
    if reverse is None and response_format is not None:
        size = unit["size"]
        expected = {"rank": set(range(size)), "pick": {0}, "mole": {0, size - 1}}[response_format]
        result[:] = [
            _normalize_order(
                value, context(i), size=size, response_format=response_format, expected=expected
            )
            for i, value in enumerate(raw)
        ]
        return result

    scale = unit["scale"]
    if response_format is not None:
        continuous, categories = False, (0, 1)
    else:
        continuous, categories = scale["kind"] == "unit_interval", scale["categories"]

    def normalize(value, index):
        return _normalize_scalar(
            value, context(index), continuous=continuous, categories=categories
        )

    if (
        raw.dtype.type not in _SCALAR_RESPONSE_TYPES
        and not {type(value) for value in raw} <= _SCALAR_RESPONSE_TYPES
    ):
        result[:] = [normalize(value, i) for i, value in enumerate(raw)]
    else:
        positions = np.flatnonzero(pd.notna(raw))
        values = raw[positions]
        if continuous:
            if values.dtype.kind in "OU" and any(isinstance(value, str) for value in values):
                bad = next(i for i in positions if isinstance(raw[i], str))
                normalize(raw[bad], bad)
            numeric = np.asarray(values, dtype=float)
            if (invalid := np.flatnonzero(~_inside_unit_interval(numeric))).size:
                bad = positions[invalid[0]]
                normalize(raw[bad], bad)
            result[positions] = numeric
        else:
            codes, unique = pd.factorize(values, sort=False)
            first = positions[~pd.Index(codes).duplicated()]
            converted = np.asarray(
                [normalize(value, i) for value, i in zip(unique, first, strict=True)],
                dtype=object,
            )
            result[positions] = converted[codes]

    if reverse is not None and (flipped := np.flatnonzero(reverse & pd.notna(result))).size:
        if continuous:
            reflected = 1 - np.asarray(result[flipped], dtype=float)
            if (invalid := np.flatnonzero(~_inside_unit_interval(reflected))).size:
                raise ValueError(
                    f"{context(flipped[invalid[0]])}: "
                    "reversed response cannot be represented in (0,1)"
                )
            result[flipped] = reflected
        else:
            reverse_category = dict(zip(categories, categories[::-1], strict=True))
            result[flipped] = pd.Series(result[flipped], dtype=object).map(reverse_category)
    return result


def _reject_response_rows(frame, mask, message):
    bad = np.flatnonzero(mask)
    if len(bad):
        row = frame.iloc[bad[0]]
        raise ValueError(f"Person {row.person_id!r}, unit {row.unit_id!r}: {message}")


def _canonical_response_rows(frame, order, *, comparison):
    """Validate response keys after orienting comparison rows by member position."""
    keys = list(_RESPONSE_KEYS)
    reverse = None
    if comparison:
        first = _member_positions(order, frame.unit_id, frame.statement_1)
        second = _member_positions(order, frame.unit_id, frame.statement_2)
        _reject_response_rows(
            frame,
            (first < 0) | (second < 0) | (first == second),
            "comparisons require two distinct member statements",
        )
        frame["position_1"], frame["position_2"], reverse = _ordered_pair(first - 1, second - 1)
        keys += _PAIR_POSITIONS
    _reject_response_rows(
        frame,
        frame.duplicated(keys).to_numpy(),
        "duplicate response, including reversed comparisons",
    )
    return keys, reverse


def normalize_responses(
    responses: ResponseTable | None,
    units: Mapping,
    *,
    unit_column: str,
    purpose: str,
    order: pd.Series | None = None,
    administration: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
    """Align responses to administration, including an omitted scoring channel."""
    if responses is None:
        frame = pd.DataFrame(columns=_RESPONSE_KEYS)
        comparison = False
    else:
        frame = _source_rows(responses, units, order, unit_column=unit_column, purpose=purpose)
        comparison = responses.format.startswith("comparison")
    active_status, allowed_statuses = _RESPONSE_STATES[purpose]
    generate = purpose == "generate"
    keys, reverse = _canonical_response_rows(frame, order, comparison=comparison)
    values = np.full(len(frame), None, dtype=object)
    for label, selected in frame.groupby("unit_id", sort=False).indices.items():
        unit = units[label]
        if (
            not comparison
            and unit_column == UNIT_COLUMNS["fc"]
            and unit["response_format"] is None
            and unit["size"] != 2
        ):
            raise ValueError("block-level intensity responses require two statements")
        if generate:
            continue
        values[selected] = _response_values(
            frame.iloc[selected],
            unit,
            label=label,
            reverse=None if reverse is None else reverse[selected],
        )
    states, parents = _response_statuses(
        frame,
        pd.notna(values),
        administration,
        active_status=active_status,
        allowed_statuses=allowed_statuses,
    )
    normalized = frame[keys].reset_index(drop=True).assign(response=values, status=states)
    if comparison:
        normalized["pair_reversed"] = reverse
    administration, parents = _complete_administration(
        normalized, administration, parents, active_status=active_status
    )
    normalized["administration_index"] = parents
    people = tuple(
        dict.fromkeys((*normalized.person_id.unique(), *administration.person_id.unique()))
    )
    return normalized, administration, people


def _response_statuses(frame, present, administration, *, active_status, allowed_statuses):
    """Resolve row status and link responses to explicit administration when supplied."""
    default = (
        np.where(present, "answered", "missing")
        if active_status == "answered"
        else np.full(len(frame), "scheduled", dtype=object)
    )
    parents = None
    if administration is not None:
        admin_keys = pd.MultiIndex.from_frame(administration[["person_id", "unit_id"]])
        row_keys = pd.MultiIndex.from_frame(frame[["person_id", "unit_id"]])
        parents = admin_keys.get_indexer(row_keys)
        _reject_response_rows(frame, parents < 0, "response is absent from administration")
        inherited = administration.status.to_numpy()[parents]
        default = np.where(inherited == active_status, default, inherited)
    states = frame.status.to_numpy() if "status" in frame else default
    _reject_response_rows(
        frame,
        ~pd.Series(states).isin(allowed_statuses).to_numpy(),
        f"status must be one of {sorted(allowed_statuses)}",
    )
    if active_status == "answered":
        _reject_response_rows(
            frame,
            (states == active_status) != present,
            "response status conflicts with response presence",
        )
    if administration is not None:
        _reject_response_rows(
            frame,
            (inherited != active_status) & (states != inherited),
            "response status conflicts with administration",
        )
    return states, parents


def _complete_administration(normalized, administration, parents, *, active_status):
    """Infer missing administration or check that every declared active unit has a row."""
    if administration is None:
        # Active answers dominate genuine nonresponse states. Without any active
        # answer, all states in a person/unit must agree.
        names = [active_status, *sorted(OBSERVED_STATUSES - {active_status})]
        compact = normalized[["person_id", "unit_id"]].copy()
        compact["_status"] = normalized.status.map({name: i for i, name in enumerate(names)})
        group = compact.groupby(["person_id", "unit_id"], sort=False)
        parents = group.ngroup().to_numpy()
        reduction = group["_status"].agg(["min", "max"])
        if ((reduction["min"] != 0) & (reduction["min"] != reduction["max"])).any():
            raise ValueError("Nonresponse comparison statuses conflict within person/unit")
        administration = reduction.reset_index()[["person_id", "unit_id"]]
        administration["status"] = reduction["min"].map(dict(enumerate(names))).to_numpy()
        administration = administration.astype(object)
    else:
        has_response = np.zeros(len(administration), dtype=bool)
        has_response[parents[normalized.status.to_numpy() == active_status]] = True
        missing = (administration.status.to_numpy() == active_status) & ~has_response
        if missing.any():
            row = administration.iloc[np.flatnonzero(missing)[0]]
            raise ValueError(
                f"Active administration entry {(row.person_id, row.unit_id)!r} "
                "has no response/schedule"
            )
    return administration, parents


def normalize_channel(source: FCInput | ItemInput, *, purpose: str) -> NormalizedChannel:
    """Normalize a declared input, retaining complete support and actual missingness."""
    if not isinstance(source.responses, ResponseTable):
        raise TypeError("responses must be ResponseTable")
    fc = isinstance(source, FCInput)
    design, defaults, options = _design_table(source, fc=fc)
    if fc:
        units = _block_units(source, design, defaults, options)
        channel = "fc"
    else:
        units = _item_units(source, design, defaults, options)
        channel = "items"
    unit_column = UNIT_COLUMNS[channel]
    by_unit = units.set_index("unit_id").to_dict("index")
    order = (
        _statement_order(design.block_id, design.statement_id, design.position)
        if fc and has_statement_rows(getattr(source.responses, "format", None))
        else None
    )
    responses, administration, people = normalize_responses(
        source.responses,
        by_unit,
        unit_column=unit_column,
        purpose=purpose,
        order=order,
        administration=normalize_administration(
            source.administration,
            by_unit,
            unit_column=unit_column,
            allowed_statuses=_RESPONSE_STATES[purpose][1],
        ),
    )
    tables = dict(
        design=design,
        units=units,
        responses=responses,
        administration=administration,
    )
    if fc and source.statement_features is not None:
        features = _frame(source.statement_features, ("statement_id",), "statement_features")
        features["statement_id"] = identifiers(
            features.statement_id, "statement_features.statement_id"
        )
        if features.statement_id.duplicated().any() or set(features.statement_id) - set(
            design.statement_id
        ):
            raise ValueError("statement_features contains duplicate or unknown statements")
        tables["statement_features"] = features
    return NormalizedChannel(
        channel=channel,
        tables=tables,
        input_format=source.responses.format,
        people=people,
    )


def _known_pairs(order, response_format, *, pairs=None):
    """Project a native event onto its known comparisons, leaving middle orders unknown."""
    if pairs is not None:
        yield from ((int(a), int(b), int(bit)) for (a, b), bit in zip(pairs, order, strict=True))
        return
    ranks = {position: rank for rank, position in enumerate(order) if position is not None}
    for first, second in combinations(range(1, len(order) + 1), 2):
        if response_format == "rank":
            yield first - 1, second - 1, int(ranks[first] < ranks[second])
        elif response_format == "pick":
            if first == order[0] or second == order[0]:
                yield first - 1, second - 1, int(first == order[0])
        elif response_format == "mole":
            if first == order[0] or second == order[-1]:
                yield first - 1, second - 1, 1
            elif second == order[0] or first == order[-1]:
                yield first - 1, second - 1, 0


def response_category_count(unit):
    """Count complete declared answers without allocating the ranking catalogue."""
    size = unit["size"]
    match unit["response_format"]:
        case "rank":
            return factorial(size)
        case "pick":
            return size
        case "mole":
            return size * (size - 1)
        case None:
            return len(unit["scale"]["categories"]) or None


def _ranking_geometry(size):
    orders = np.asarray(tuple(permutations(range(size))), dtype=np.int64)
    scores = (size - 1 - np.argsort(orders, axis=-1)).astype(np.float64)
    eye = np.eye(size, dtype=np.float64)
    return dict(orders=orders, scores=scores, contrasts=eye[orders[:, :-1]] - eye[orders[:, 1:]])


def _ranking_catalog(geometry, response_format, pairs):
    """Partition the existing complete orders by task or recorded comparisons."""
    orders, scores = geometry["orders"], geometry["scores"]
    size = orders.shape[-1]
    if pairs:
        first, second = np.asarray(pairs, dtype=np.int64).T
        signatures = (scores[:, first] > scores[:, second]).astype(np.int8)
        patterns, codes = np.unique(signatures, axis=0, return_inverse=True)
        labels = tuple(map(tuple, patterns.tolist()))
        membership = np.arange(len(labels))[:, None] == codes
    elif response_format == "rank":
        labels = tuple(tuple(int(v) + 1 for v in row) for row in orders)
        membership = None  # Complete orders already are the response categories.
    elif response_format in {"pick", "mole"}:
        if response_format == "pick":
            labels = tuple((first, *([None] * (size - 1))) for first in range(1, size + 1))
        else:
            labels = tuple(
                (first, *([None] * (size - 2)), last)
                for first, last in permutations(range(1, size + 1), 2)
            )
        constraints = np.asarray([[0 if v is None else v for v in event] for event in labels])
        membership = np.all(
            (constraints[:, None] == 0) | (constraints[:, None] == orders[None] + 1), axis=-1
        )
    result = dict(
        labels=labels, codes={label: index for index, label in enumerate(labels)}, **geometry
    )
    if membership is not None:
        result["membership"] = membership
    return result


class ResponseEncoding:
    """Own reusable event tables and immutable ID fragments for one encoding call."""

    def __init__(self):
        self.geometry = {}
        self.catalogs = {}
        self.signatures = {}
        self.constraints = {}
        self.fragments = {}

    def catalog(self, size, response_format, pairs=()):
        pairs = tuple(pairs)
        key = size, pairs or response_format
        if key not in self.catalogs:
            if size not in self.geometry:
                self.geometry[size] = _ranking_geometry(size)
            self.catalogs[key] = _ranking_catalog(self.geometry[size], response_format, pairs)
        return dict(self.catalogs[key], response_format=response_format, pairs=pairs)

    def native_codes(self, size, response_format):
        """Index the exact comparison signatures for category-shaped observations."""
        key = size, response_format
        if key not in self.signatures:
            codes = {}
            for event, code in self.catalog(size, response_format)["codes"].items():
                signature = [-1] * (size * (size - 1) // 2)
                for a, b, bit in _known_pairs(event, response_format):
                    signature[_pair_row(a, b, size)] = bit
                codes[tuple(signature)] = code
            self.signatures[key] = codes
        return self.signatures[key]

    def augmentation(self, distribution, size, edges):
        """Resolve distinct order regions, then gather geometry in observation order."""
        patterns = dict.fromkeys(edges)
        regions = []
        for index, pattern in enumerate(patterns):
            patterns[pattern] = index
            key = distribution, size, pattern
            if key not in self.constraints:
                self.constraints[key] = distribution.observed_region(size, pattern)
            regions.append(self.constraints[key])
        basis, mask = distribution.observed_contrasts(*map(np.stack, zip(*regions, strict=True)))
        rows = np.fromiter(map(patterns.__getitem__, edges), dtype=np.intp, count=len(edges))
        return basis[rows], None if mask is None else mask[rows]

    def observation_ids(self, channel, unit, people, pairs):
        def fragment(value):
            key = type(value), value
            if key not in self.fragments:
                self.fragments[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            return self.fragments[key]

        prefix, middle = "[" + fragment(channel) + ",", "," + fragment(unit) + ","
        pair_keys = repeat(None, len(people)) if pairs is None else map(tuple, pairs.tolist())
        return [
            prefix + fragment(person) + middle + fragment(pair) + "]"
            for person, pair in zip(people, pair_keys, strict=True)
        ]


def _graded_catalog(categories):
    scores = np.zeros((len(categories), 2), dtype=np.float64)
    half = len(categories) // 2
    scores[:half, 1] = np.arange(half, 0, -1)
    scores[-half:, 0] = np.arange(1, half + 1)
    return dict(
        response_format=None,
        labels=tuple(categories),
        scores=scores,
    )


@dataclass(frozen=True, slots=True)
class _EncodedEvents:
    # All columns follow likelihood order. A source of -1 denotes a joint
    # comparison event, whose provenance includes the administration's rows.
    parents: np.ndarray
    sources: np.ndarray
    y: np.ndarray | None
    pairs: np.ndarray | None = None
    edges: tuple | None = None


def _unit_events(unit, rows, *, specification, generate, encoding):
    """Encode aligned factor columns from validated unit-local rows."""
    response_format, scale = unit["response_format"], unit["scale"]
    pairs = rows.loc[:, list(_PAIR_POSITIONS)].to_numpy(dtype=np.int64)
    comparison_input = bool((pairs >= 0).any())
    if comparison_input and (pairs < 0).any():
        raise ValueError("One unit cannot mix block events and comparison rows")

    parents, sources = (
        rows[key].to_numpy(dtype=np.int64) for key in ("parent_index", "source_row")
    )
    values = rows.response.to_numpy()
    ranking = response_format is not None
    categories = (0, 1) if ranking else tuple(scale["categories"])
    catalog = dict(response_format=response_format, labels=categories)
    if specification.comparison_based:
        if not comparison_input and ranking:
            if generate:
                raise ValueError("Comparison models require comparison_schedule for generation")
            codes, orders = pd.factorize(values, sort=False)
            projection = np.asarray(
                [tuple(_known_pairs(order, response_format)) for order in orders], dtype=np.int64
            )
            count = projection.shape[1]
            projected = projection[codes].reshape(-1, 3)
            return [
                (
                    catalog,
                    _EncodedEvents(
                        parents.repeat(count),
                        sources.repeat(count),
                        projected[:, 2],
                        projected[:, :2],
                    ),
                )
            ]
        if not comparison_input:
            pairs = np.broadcast_to((0, 1), pairs.shape)
    elif unit["channel"] == "items":
        pairs = None
    elif specification.graded_events and scale["kind"] == "category" and not ranking:
        catalog, pairs = _graded_catalog(categories), None
    elif comparison_input:
        return _joint_comparison_events(
            unit, parents, pairs, values, specification, generate, encoding
        )
    else:
        catalog, pairs = encoding.catalog(unit["size"], response_format), None

    y, edges = None, None
    if not generate:
        if "native_response" in rows and rows.native_response.iat[0]:
            # Each generated unit uses one support. Direct item and comparison
            # outcomes are already expressed on that native support.
            y = values.astype(np.float64 if scale["kind"] == "unit_interval" else np.int64)
        elif scale["kind"] == "unit_interval":
            y = values.astype(np.float64)
            if scale["higher_prefers"] == "second":
                y = 1 - y
                if np.any((y <= 0) | (y >= 1)):
                    raise ValueError("Continuous scale reversal cannot be represented inside (0,1)")
        else:
            codes = catalog.get("codes")
            lookup = catalog["labels"].index if codes is None else codes.__getitem__
            y = np.fromiter(map(lookup, values), dtype=np.int64, count=len(values))
        if specification.augmented_ranking:
            by_pattern = {
                order: tuple(_known_pairs(order, response_format))
                for order in dict.fromkeys(values)
            }
            edges = tuple(map(by_pattern.__getitem__, values))
    return [(catalog, _EncodedEvents(parents, sources, y, pairs, edges))]


def _joint_comparison_events(unit, row_parents, pairs, values, specification, generate, encoding):
    # Missing comparisons stay -1. Compatible rankings supply support without
    # filling in unobserved comparisons.
    response_format, size = unit["response_format"], unit["size"]
    codes, parents = pd.factorize(row_parents, sort=False)
    all_pairs = tuple(combinations(range(size), 2))
    matrix = np.full((len(parents), len(all_pairs)), -1, dtype=np.int8)
    matrix[codes, _pair_row(*pairs.T, size)] = 0 if generate else values.astype(np.int8)
    patterns, inverse = np.unique(matrix, axis=0, return_inverse=True)
    if not generate:
        native = encoding.native_codes(size, response_format)
    descriptors = []
    for pattern in patterns:
        observed = np.flatnonzero(pattern >= 0)
        subset = tuple(all_pairs[index] for index in observed)
        response, edges = None, ()
        if not generate:
            response = native.get(tuple(pattern))
            if response is None:
                codes = encoding.catalog(size, response_format, subset)["codes"]
                response = codes.get(tuple(pattern[observed]))
                if response is None:
                    raise ValueError("Comparisons have no compatible complete ranking")
            else:
                subset = ()
            if specification.augmented_ranking:
                edges = tuple((*all_pairs[index], int(pattern[index])) for index in observed)
        descriptors.append((subset, response, edges))
    groups = {}
    for index, code in enumerate(inverse):
        groups.setdefault(descriptors[code][0], []).append(index)
    events = []
    for subset, selected in groups.items():
        catalog = encoding.catalog(size, response_format, subset)
        observations = [descriptors[code] for code in inverse[selected]]
        y = None if generate else np.asarray([row[1] for row in observations], dtype=np.int64)
        edges = (
            tuple(row[2] for row in observations)
            if specification.augmented_ranking and not generate
            else None
        )
        events.append(
            (
                catalog,
                _EncodedEvents(
                    parents[selected], np.full(len(selected), -1, np.int64), y, edges=edges
                ),
            )
        )
    return events


def _put_array(dataset, name, value, dims):
    array = np.asarray(value)
    for dim, size in zip(dims, array.shape, strict=True):
        if dim not in dataset.coords:
            dataset.add_coords({dim: np.arange(size, dtype=np.int64)})
    dataset[name] = dims, array
    return name


def _batch_slots(constant, arrays, record, measurement, pairs, item_traits, cut_rows, people):
    """Apply prepared gather axes to the observed rows."""
    name, obs_dim = record["name"], record["obs_dim"]
    selections = dict(position=pairs, trait_position=item_traits, cutpoint=cut_rows, person=people)
    result = []
    for slot in measurement["slots"]:
        key, axis = slot["key"], slot["axis"]
        values = constant[slot["entry_index"]].values
        if (indices := selections.get(axis)) is not None:
            values = values[indices]
        dims = (obs_dim,) if axis in (None, "person") else (obs_dim, f"{name}_{axis}")
        if values.ndim < len(dims):
            values = np.broadcast_to(values, (len(people), *values.shape))
        index = _put_array(arrays, f"{name}_{key}_entry_index", values, dims)
        result.append(dict(key=key, kind=slot["kind"], entry_index=index))
    return result


def _encode_batch(constant, unit, catalog, events, admin_people, *, index, encoding):
    """Build one batch's record and arrays for the shared observation catalogue."""
    name = f"b{index}"
    obs_dim, category_dim = f"{name}_obs", f"{name}_category"
    unit_index = unit["unit_index"]
    measurement = constant.attrs["measurements"][unit_index]
    unit_label = constant.variables["unit_id"].values[unit_index]
    channel = unit["channel"]
    if channel == "items":
        item = constant.variables["unit_item"].values[unit_index]
        full_statement = constant.variables["item_statement"].values[item : item + 1]
        item_traits = np.flatnonzero(constant.variables["item_q"].values[item])
    else:
        block = constant.variables["unit_block"].values[unit_index]
        full_statement = constant.variables["fc_statement"].values[block, : unit["size"]]
        item_traits = None
    pairs = events.pairs
    parents = events.parents
    size = len(parents)
    statements = (
        full_statement[pairs]
        if pairs is not None
        else np.broadcast_to(full_statement, (size, len(full_statement)))
    )
    people = admin_people[parents]
    person_labels = constant.variables["person"].values[people].tolist()
    obs_ids = encoding.observation_ids(channel, unit_label, person_labels, pairs)
    labels = catalog["labels"]
    arrays = LayoutBuilder(
        coords={obs_dim: obs_ids, category_dim: np.arange(len(labels), dtype=np.int64)}, attrs={}
    )
    position_dim = f"{name}_position"
    record = dict(
        name=name,
        response_name=f"{name}_response",
        channel=channel,
        unit_index=unit_index,
        obs_dim=obs_dim,
        category_dim=category_dim,
        events=dict(response_format=catalog["response_format"], labels=labels),
    )
    record["statement"] = _put_array(
        arrays, f"{name}_statement", statements, (obs_dim, position_dim)
    )
    traits = (
        constant.variables["statement_trait"].values[statements]
        if item_traits is None
        else np.broadcast_to(item_traits, (size, len(item_traits)))
    )
    record["trait"] = _put_array(
        arrays,
        f"{name}_trait",
        traits,
        (obs_dim, f"{name}_trait_position" if channel == "items" else position_dim),
    )
    # Labels/y stay in declared order. This is the sole categorical scale reversal.
    category_order = np.arange(len(labels), dtype=np.int64)
    categorical_scores = "scores" in catalog and "orders" not in catalog
    if (pairs is not None or categorical_scores) and unit["scale"]["higher_prefers"] == "second":
        category_order = category_order[::-1]
    record["category_order"] = _put_array(
        arrays, f"{name}_category_order", category_order, (category_dim,)
    )
    cut_rows, pair_index = None, np.full(size, -1, np.int64)
    if pairs is not None:
        cut_rows = _pair_row(*pairs.T, unit["size"])
        pair_index = constant[measurement["pair_indices"]].values[cut_rows]
    array_dims = {
        "orders": (f"{name}_pattern", position_dim),
        "scores": (f"{name}_pattern" if "orders" in catalog else category_dim, position_dim),
        "membership": (category_dim, f"{name}_pattern"),
        "contrasts": (f"{name}_pattern", f"{name}_difference", position_dim),
    }
    for field, dims in array_dims.items():
        if field in catalog:
            record["events"][field] = _put_array(arrays, f"{name}_{field}", catalog[field], dims)
    if catalog.get("pairs"):
        record["events"]["pairs"] = _put_array(
            arrays,
            f"{name}_pairs",
            np.asarray(catalog["pairs"], np.int64),
            (f"{name}_pair", f"{name}_pair_member"),
        )
    if events.edges is not None:
        basis, masks = encoding.augmentation(
            measurement["specification"].distribution, unit["size"], events.edges
        )
        record["events"]["basis"] = _put_array(
            arrays,
            f"{name}_basis",
            basis,
            (obs_dim, f"{name}_difference", position_dim),
        )
        if masks is not None:
            record["events"]["mask"] = _put_array(
                arrays,
                f"{name}_mask",
                masks,
                (obs_dim, position_dim, f"{name}_position_aux"),
            )
    y = events.y
    record["slots"] = _batch_slots(
        constant, arrays, record, measurement, pairs, item_traits, cut_rows, people
    )
    return record, arrays, y, dict(person=people, pair=pair_index, input_event_index=parents)


def _observation_layout(constant, observed, batches):
    """Extend the existing true-factor catalogue with prepared output provenance."""
    columns = {
        field: []
        for field in (
            "person",
            "unit",
            "pair",
            "native_position",
            "batch",
            "input_event_index",
        )
    }
    obs_ids, origins, arrays, coordinates = [], [], {}, {}
    offset = 0
    for batch_index, (record, batch, indices, sources) in enumerate(batches):
        ids = batch.variables[record["obs_dim"]].values
        size = len(ids)
        obs_ids.append(ids)
        origins.append(sources)
        for field, values in indices.items():
            columns[field].append(values)
        columns["unit"].append(np.full(size, record["unit_index"], dtype=np.int64))
        columns["native_position"].append(np.arange(size, dtype=np.int64))
        columns["batch"].append(np.full(size, batch_index, dtype=np.int64))
        record["obs_slice"] = slice(offset, offset + size)
        offset += size
    coordinates["obs"] = np.concatenate(obs_ids, dtype=str) if batches else np.array([], dtype=str)
    for field, values in columns.items():
        arrays[f"obs_{field}"] = (
            "obs",
            (np.concatenate(values, dtype=np.int64) if batches else np.array([], dtype=np.int64)),
        )
    people, units = arrays["obs_person"][1], arrays["obs_unit"][1]
    arrays["obs_person_id"] = "obs", constant.variables["person"].values[people]
    arrays["obs_population_group_index"] = "obs", constant.variables["person_group"].values[people]
    for field in ("channel", "model", "response_format", "block", "item"):
        suffix = "_index" if field in {"block", "item"} else ""
        arrays[f"obs_{field}{suffix}"] = "obs", constant.variables[f"unit_{field}"].values[units]
    origins = np.concatenate(origins) if batches else np.array([], dtype=np.int64)
    direct = origins >= 0
    obs_index, response_index = np.flatnonzero(direct), origins[direct]
    if not direct.all():
        # Joint comparison events refer to the complete recorded administration.
        grouped = pd.DataFrame(
            {
                "obs_index": np.flatnonzero(~direct),
                "administration": arrays["obs_input_event_index"][1][~direct],
            }
        ).merge(
            pd.DataFrame(
                {
                    "administration": observed.response_administration_index.values,
                    "response_index": observed.response_row.values,
                }
            ),
            on="administration",
        )
        obs_index = np.concatenate([obs_index, grouped.obs_index.to_numpy()])
        response_index = np.concatenate([response_index, grouped.response_index.to_numpy()])
        order = np.lexsort((response_index, obs_index))
        obs_index, response_index = obs_index[order], response_index[order]
    coordinates["observation_source"] = np.arange(len(obs_index), dtype=np.int64)
    arrays["observation_source_obs_index"] = "observation_source", obs_index
    arrays["observation_source_response_row_index"] = "observation_source", response_index
    return xr.Dataset(arrays, coords=coordinates)


_PAIR_COLUMNS = ("statement_1", "statement_2", "source_statement_1", "source_statement_2")


def index_response_rows(person_labels, units, channels, *, omitted=()):
    """Resolve administration once and complete omitted channels on the integer axes."""
    if any("person" not in c.tables["administration"] for c in channels):
        people = {label: index for index, label in enumerate(person_labels)}
    administration, responses = [], []
    offset = 0
    for channel in channels:
        admin, rows = (channel.tables[name] for name in ("administration", "responses"))
        channel_units = units[channel.channel]
        if "person" not in admin:
            unit_positions = {label: unit["unit_index"] for label, unit in channel_units.items()}
            admin = admin.assign(
                person=admin.person_id.map(people).to_numpy(dtype=np.int64),
                unit=admin.unit_id.map(unit_positions).to_numpy(dtype=np.int64),
            )
        admin = admin[["person", "unit", "status"]]
        if channel.channel in omitted:
            keys = ["person", "unit"]
            unit_axis = [unit["unit_index"] for unit in channel_units.values()]
            expected = pd.MultiIndex.from_product(
                (range(len(person_labels)), unit_axis), names=keys
            )
            absent = expected.difference(pd.MultiIndex.from_frame(admin[keys]), sort=False)
            if len(absent):
                admin = pd.concat(
                    [admin, absent.to_frame(index=False).assign(status="not_administered")],
                    ignore_index=True,
                )
        administration.append(admin)
        responses.append(
            pd.DataFrame(
                dict(
                    administration_index=rows.administration_index.to_numpy(dtype=np.int64)
                    + offset,
                    status=rows.status.to_numpy(),
                    response=rows.response.to_numpy(dtype=object),
                    pair_reversed=rows.get("pair_reversed", False),
                    **{
                        col: rows[col].to_numpy(dtype=np.int64)
                        if col in rows
                        else np.full(len(rows), -1, dtype=np.int64)
                        for col in _PAIR_POSITIONS
                    },
                )
            )
        )
        offset += len(admin)
    return (
        pd.concat(administration, ignore_index=True),
        pd.concat(responses, ignore_index=True),
    )


def encode_batches(
    constant: xr.Dataset, administration: pd.DataFrame, responses: pd.DataFrame, *, purpose: str
) -> tuple[xr.Dataset, xr.Dataset]:
    """Encode indexed rows and publish the true observations and their origins."""
    generate = purpose == "generate"
    encoding = ResponseEncoding()
    people, units = (administration[col].to_numpy(dtype=np.int64) for col in ("person", "unit"))
    parents = responses.administration_index.to_numpy(dtype=np.int64)
    status = responses.status.to_numpy(dtype=str)
    active_status, _ = _RESPONSE_STATES[purpose]
    active = status == active_status
    positions = responses[list(_PAIR_POSITIONS)].to_numpy(dtype=np.int64)
    response_units = units[parents]
    fields = ["response", *_PAIR_POSITIONS]
    if "native_response" in responses:
        fields.append("native_response")
    events = responses.loc[active, fields].assign(
        unit=response_units[active],
        parent_index=parents[active],
        source_row=np.flatnonzero(active),
    )
    batches, batch_coords, values = [], {}, {}
    for index, selected in events.groupby("unit", sort=True):
        unit = constant.attrs["units"][index]
        for catalog, encoded in _unit_events(
            unit,
            selected,
            specification=constant.attrs["measurements"][index]["specification"],
            generate=generate,
            encoding=encoding,
        ):
            record, arrays, y, indices = _encode_batch(
                constant,
                unit,
                catalog,
                encoded,
                people,
                index=len(batches),
                encoding=encoding,
            )
            batches.append((record, arrays, indices, encoded.sources))
            obs_dim = record["obs_dim"]
            batch_coords[obs_dim] = arrays.variables[obs_dim]
            if y is not None:
                values[record["response_name"]] = obs_dim, y
    # Global indices belong to the published provenance; event construction uses local positions.
    statements = np.full_like(positions, -1)
    compared = positions[:, 0] >= 0
    if compared.any():
        blocks = constant.unit_block.values[response_units[compared]]
        statements[compared] = constant.fc_statement.values[blocks[:, None], positions[compared]]
    source_statements = np.where(
        responses.pair_reversed.to_numpy(dtype=bool)[:, None], statements[:, ::-1], statements
    )
    observed = xr.Dataset(
        dict(
            administration_person=("administration", people),
            administration_unit=("administration", units),
            administration_status=("administration", administration.status.to_numpy(dtype=str)),
            response_person=("response_row", people[parents]),
            response_unit=("response_row", response_units),
            response_status=("response_row", status),
            response_administration_index=("response_row", parents),
            **{
                f"response_{col}": ("response_row", indices)
                for col, indices in zip(
                    _PAIR_COLUMNS, (*statements.T, *source_statements.T), strict=True
                )
            },
            **values,
        ),
        coords={
            "administration": np.arange(len(administration), dtype=np.int64),
            "response_row": np.arange(len(responses), dtype=np.int64),
            **batch_coords,
        },
    )
    pieces = [arrays for _, arrays, _, _ in batches]
    pieces.append(_observation_layout(constant, observed, batches))
    observation_variables = tuple(
        dict.fromkeys(name for piece in pieces for name in piece.variables)
    )
    result = merge_prepared([constant, *pieces])
    result.attrs.update(
        batches=[record for record, _, _, _ in batches], observation_variables=observation_variables
    )
    return result, observed


def warn_unobserved_statements(constant):
    """Report unused FC design entries, without changing the design or its anchors."""
    observed = set()
    for batch in constant.attrs["batches"]:
        if batch["channel"] != "fc":
            continue
        statements = constant.variables[batch["statement"]].values
        if "pairs" in batch["events"]:
            statements = statements[0][constant.variables[batch["events"]["pairs"]].values]
        observed.update(np.unique(statements).tolist())
    labels = constant.variables["statement_label"].values
    channels = constant.variables["statement_channel"].values
    missing = [
        label
        for index, label in enumerate(labels)
        if channels[index] == "fc" and index not in observed
    ]
    if missing:
        warn(
            f"Statements without observed comparisons: {missing!r}. "
            "Remove unused entries from the design if they are not intentional.",
            UserWarning,
            stacklevel=3,
        )


def observed_response_view(prepared, unit, representation, encoding):
    """Read a display view from the unit's prepared batches and comparison positions."""
    constant = prepared.constant_data
    index = unit["unit_index"]
    statements = (
        constant.fc_statement.values[constant.unit_block.values[index], : unit["size"]]
        if unit["channel"] == "fc"
        else np.array([], dtype=np.int64)
    )
    labels, response_format = constant.statement_label.values[statements], unit["response_format"]
    frames = []
    for batch in prepared.batches_by_unit[index]:
        if batch.y is None:
            continue
        rows = pd.DataFrame({"person": constant.person.values[batch.person], "code": batch.y})
        events, pairs = batch.events, batch.projected_pairs
        support = events.labels
        if events.orders is None or representation == "categories" and pairs is None:
            rows["category"] = rows.code.map(dict(enumerate(support)))
            rows["pair"] = None
            if (
                unit["channel"] == "fc"
                and events.orders is None
                and not (
                    representation == "categories" and response_format is None and unit["size"] == 2
                )
            ):
                if batch.specification.comparison_based:
                    positions = constant.pair_positions.values[batch.pair_index]
                    rows["pair"] = pd.MultiIndex.from_arrays(positions.T).to_flat_index()
                else:
                    # Whole-block graded events are supported only for dyads.
                    rows["pair"] = [(0, 1)] * len(rows)
        else:
            # Project each observed category once, not once per person.
            projection = []
            positions = None if pairs is None else pairs.values
            for code in rows.code.unique():
                known = _known_pairs(support[int(code)], response_format, pairs=positions)
                projection.extend((code, (first, second), bit) for first, second, bit in known)
            lookup = pd.DataFrame(projection, columns=["code", "pair", "category"])
            rows = rows.merge(lookup, on="code", sort=False)
            rows["code"] = rows["category"]  # Projected comparisons use binary codes.
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=["person", "pair", "category"])
    result = pd.concat(frames, ignore_index=True)
    if representation == "categories" and result.pair.notna().any():
        context = f"Block {constant.unit_id.values[unit['unit_index']]!r}: categories"
        if response_format is None:
            raise ValueError(f"{context} requires a recorded original ranking response_format")
        catalog = encoding.catalog(unit["size"], response_format)
        matrix = result.dropna(subset="pair").pivot(
            index="person", columns="pair", values="category"
        )
        patterns, inverse = np.unique(
            matrix.fillna(-1).to_numpy(dtype=np.int8), axis=0, return_inverse=True
        )
        first, second = np.asarray(matrix.columns.tolist()).T
        signatures = (catalog["scores"][:, first] > catalog["scores"][:, second]).astype(np.int8)
        selected = []
        for pattern in patterns:
            compatible = np.all((pattern < 0) | (signatures == pattern), axis=-1)
            possible = (
                catalog["membership"][:, compatible].any(axis=-1)
                if "membership" in catalog
                else compatible
            )
            if possible.sum() != 1:
                raise ValueError(f"{context} cannot be uniquely recovered; use pairwise")
            selected.append(int(np.flatnonzero(possible)[0]))
        recovered = pd.DataFrame({"person": matrix.index, "code": np.asarray(selected)[inverse]})
        recovered["category"] = recovered.code.map(dict(enumerate(catalog["labels"])))
        result = pd.concat(
            [result.loc[result.pair.isna()], recovered.assign(pair=None)], ignore_index=True
        )
    result = result.sort_values(["pair", "code"], kind="stable").drop(columns="code")
    if representation == "categories" and response_format is not None:
        result["category"] = result.category.map(
            lambda outcome: tuple(
                None if position is None else labels[position - 1] for position in outcome
            )
        )
    elif representation == "pairwise":
        result["pair"] = result.pair.map(lambda pair: tuple(labels[list(pair)]))
    return result


def generated_rows(prepared: PreparedInputs, values: xr.Dataset):
    """Keep direct native outcomes and expand ranking events to their recorded comparisons."""
    constant, source = prepared.constant_data, prepared.observed_data
    administration = pd.DataFrame(
        dict(
            person=source.administration_person.values,
            unit=source.administration_unit.values,
            status=np.where(
                source.administration_status.values == "scheduled",
                "answered",
                source.administration_status.values,
            ),
        )
    )
    parents, first, second = (
        source[f"response_{field}"].values
        for field in ("administration_index", "statement_1", "statement_2")
    )
    reversed_rows = source.response_source_statement_1.values != first
    status = source.response_status.values
    inactive = ~np.isin(status, ["answered", "scheduled"])
    reserved = inactive & (first >= 0)
    reserved_pairs = set(zip(parents[reserved], first[reserved], second[reserved], strict=True))
    reversed_pairs = set(
        zip(parents[reversed_rows], first[reversed_rows], second[reversed_rows], strict=True)
    )
    pair_catalog = constant.pair_positions.values
    rows = []
    for batch in prepared.batches:
        name, dimension = batch.response_name, batch.obs_dim
        if name not in values:
            raise ValueError(f"Generated values are missing {name!r}")
        value = values[name]
        if value.dims != (dimension,) or value.sizes[dimension] != constant.sizes[dimension]:
            raise ValueError(f"{name!r} must contain one replicate along {dimension!r}")
        if dimension in value.coords and not value[dimension].variable.equals(
            constant.variables[dimension]
        ):
            raise ValueError(f"{name!r} observation coordinates differ from the prepared batch")
        unit = constant.attrs["units"][batch.unit_index]
        events, scale = batch.events, unit["scale"]
        labels = events.labels
        channel = unit["channel"]
        continuous = scale["kind"] == "unit_interval"
        if channel == "fc":
            block = constant.unit_block.values[batch.unit_index]
            members = constant.fc_statement.values[block, : unit["size"]]
        pairs = batch.projected_pairs
        pair_positions = None if pairs is None else pairs.values
        by_code = {}
        for raw, parent, comparison in zip(
            value.values, batch.administration_index, batch.pair_index, strict=True
        ):
            if continuous:
                native = response = float(raw)
                if not 0 < native < 1:
                    raise ValueError(f"{name!r} contains a response outside (0,1)")
                if scale["higher_prefers"] == "second":
                    response = 1 - response
                    if not 0 < response < 1:
                        raise ValueError("Decoded continuous response is outside (0,1)")
            else:
                native = _integer(raw, f"{name!r} generated category")
                if not 0 <= native < len(labels):
                    raise ValueError(f"{name!r} contains a category outside frozen support")
            if channel == "items":
                rows.append((parent, native, -1, -1, False, "answered", True))
                continue
            direct = events.orders is None and comparison >= 0
            if events.orders is None:
                a, b = pair_catalog[comparison] if comparison >= 0 else (0, 1)
                # Block-level graded MGPPM still needs its category label.
                known = ((a, b, native if direct else labels[native]),)
            else:
                if native not in by_code:
                    by_code[native] = tuple(
                        _known_pairs(labels[native], events.response_format, pairs=pair_positions)
                    )
                known = by_code[native]
            for a, b, outcome in known:
                pair = parent, members[a], members[b]
                if pair in reserved_pairs:
                    continue
                reverse = pair in reversed_pairs
                if reverse and continuous and not 0 < 1 - response < 1:
                    raise ValueError("Decoded reversed response is outside (0,1)")
                rows.append((parent, outcome, a, b, reverse, "answered", direct))

    # Keep recorded nonresponses separate from the newly generated likelihood rows.
    positions = np.full((np.count_nonzero(inactive), 2), -1, dtype=np.int64)
    if reserved_pairs:
        blocks = constant.unit_block.values[source.response_unit.values[reserved]]
        order = frozen_statement_order(constant, indexed=True)
        positions[reserved[inactive]] = np.column_stack(
            [_member_positions(order, blocks, side[reserved]) - 1 for side in (first, second)]
        )
    rows.extend(
        zip(
            parents[inactive],
            repeat(None, len(positions)),
            *positions.T,
            reversed_rows[inactive],
            status[inactive],
            repeat(False, len(positions)),
            strict=True,
        )
    )
    return administration, _records(
        rows, ("administration_index", "response", *_PAIR_FIELDS, "status", "native_response")
    )
