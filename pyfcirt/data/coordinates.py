"""Validated arrays, native declarations and cached runtime views for model construction."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import DisjointSet

from pyfcirt.common.validation import identifier, identifiers
from pyfcirt.data.arrays import LayoutBuilder
from pyfcirt.data.grouping import fit_groups
from pyfcirt.models.priors import PriorPreparation

REGRESSION_KINDS = frozenset(("regression_coefficient", "regression_sd"))
REGRESSION_POOL_FIELDS = ("target_kind", "scale", "trait_index")
PARAMETER_INDEX_FIELDS = (
    "person_index",
    "unit_index",
    "block_index",
    "statement_index",
    "item_index",
    "trait_index",
    "trait_aux_index",
    "pair_index",
    "category_index",
    "cutpoint_position",
    "population_group_index",
    "regression_index",
    "pool_index",
    "term_index",
)

if TYPE_CHECKING:
    from pytensor.tensor.variable import TensorVariable

    from pyfcirt.data.encoding import NormalizedChannel


def select_units(design, blocks=None, items=None):
    """Resolve ordered user labels using the typed block/item-to-unit relationship."""
    if blocks is None and items is None:
        return list(range(design.sizes["unit"]))
    units = []
    for axis, labels in (("block", blocks), ("item", items)):
        if labels is None:
            continue
        if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence | np.ndarray):
            raise TypeError("blocks and items must be ordered label sequences")
        labels = tuple(identifier(label, axis) for label in labels)
        positions = design.get_index(axis).get_indexer(labels)
        if len(set(labels)) != len(labels) or np.any(positions < 0):
            raise ValueError(f"Unknown or duplicate {axis} labels: {labels!r}")
        relation = design[f"unit_{axis}"].values
        members = np.flatnonzero(relation >= 0)
        units.extend(pd.Series(members, index=relation[members]).loc[positions].tolist())
    return units


def freeze_prior_groups(constant, layout):
    """Publish each new or resized block's roles in its free-entry order."""
    if "prior_groups" not in layout:
        if (reference := layout.get("prior_role_index")) is not None:
            roles = constant.variables[reference].values
            layout["prior_groups"] = {
                str(role): tuple(np.flatnonzero(roles == role).tolist())
                for role in dict.fromkeys(roles)
            }
        else:
            size = constant.variables[layout["free_flat"]].size
            role = layout["prior_role"]
            layout["prior_groups"] = {role: range(size)} if size and role is not None else {}
    return layout["prior_groups"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterLayout:
    """One canonical final parameter and its authoritative labelled entry catalogue."""

    kind: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    fixed: np.ndarray
    free_flat: np.ndarray
    fixed_flat: np.ndarray
    entry_data: xr.Dataset
    fixed_for_scoring: bool = True
    point_parameter: bool = True

    @property
    def fixed_value(self):
        values = self.fixed.ravel()[self.fixed_flat]
        values.flags.writeable = False
        return values

    @property
    def fixed_source(self):
        values = self.entry_data.fixed_source.values[self.fixed_flat]
        values.flags.writeable = False
        return values


@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterBlock:
    """Native construction instructions mapped to canonical final entries."""

    binding_id: int
    kind: str
    key: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    entry_index: np.ndarray
    prior_role: str | None
    priors: Mapping
    component: str
    free_flat: np.ndarray
    fixed_for_scoring: bool = True
    orientation: np.ndarray | None = None
    prior_role_index: np.ndarray | None = None
    prior_groups: tuple[tuple[str, np.ndarray], ...]
    ordered: bool = False
    shared: bool = False
    bounds: tuple | None = None
    prior_shapes: Mapping[str, tuple[int, ...]] | None = None
    scale_kind: str | None = None
    scale_entry_index: np.ndarray | None = None
    scale_power: float = 1.0
    noncentered: bool = False
    centered_roles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class SlotRef:
    key: str
    kind: str
    entry_index: np.ndarray
    effects: tuple[PersonEffect, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class PersonEffect:
    """Prepared observed positions and their person and regression pool indices."""

    regression_name: str
    position: np.ndarray
    person: np.ndarray
    pool: np.ndarray
    link: str


@dataclass(frozen=True, slots=True, kw_only=True)
class EventLayout:
    """Original measurement response_format and the prepared support of actual likelihood events."""

    response_format: str | None
    labels: tuple
    orders: np.ndarray | None = None
    scores: np.ndarray | None = None
    membership: np.ndarray | None = None
    contrasts: np.ndarray | None = None
    basis: np.ndarray | None = None
    mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedBatch:
    name: str
    response_name: str
    likelihood_kind: str
    obs_slice: slice
    person: np.ndarray
    pair_index: np.ndarray
    administration_index: np.ndarray
    projected_pairs: xr.Variable | None
    unit_index: int
    specification: object
    statement: np.ndarray
    trait: np.ndarray
    category_order: np.ndarray
    events: EventLayout
    y: np.ndarray | None
    slots: tuple[SlotRef, ...]
    obs_dim: str
    category_dim: str
    statement_keying: np.ndarray

    @property
    def keying(self):
        return self.statement_keying[self.statement]

    @property
    def distribution_names(self):
        outputs = ("alpha", "beta") if self.likelihood_kind == "beta" else ("p",)
        return tuple(f"{self.name}_{output}" for output in outputs)

    @property
    def log_probability_name(self):
        return f"{self.name}_log_probability"

    @property
    def log_likelihood_name(self):
        return f"{self.name}_log_likelihood"


@dataclass(frozen=True, slots=True, kw_only=True)
class PopulationLayout:
    specification: object
    person_group: np.ndarray
    predictors: np.ndarray
    blocks: tuple[ParameterBlock, ...]
    recipe: Mapping
    centered_index: np.ndarray
    noncentered_index: np.ndarray
    theta_parameterization: str = "centered"

    @property
    def groups(self):
        return tuple(self.recipe["groups"])

    @property
    def reference_group(self):
        return self.recipe["reference_group"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RegressionTarget:
    """Frozen feature rows and coefficient pools in native prior-group order.

    Local positions identify selected scalars or whole ordered cutpoint vectors.
    Theta uses person-by-trait indices and has no local-position axis.
    """

    block: ParameterBlock | None
    prior_role: str
    row_index: np.ndarray
    pool_index: np.ndarray
    conditional: Mapping
    noncentered: bool
    local_position: np.ndarray | None
    unit_indices: tuple[int, ...] = ()

    @property
    def binding_id(self):
        return -1 if self.block is None else self.block.binding_id

    @property
    def kind(self):
        return "theta" if self.block is None else self.block.kind


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedRegression:
    """One fitted design with native templates and resolved parameter layout views."""

    name: str
    design: xr.Dataset
    blocks: tuple[ParameterBlock, ...]
    priors: Mapping[str, TensorVariable]
    coefficient_positions: np.ndarray
    coefficient_entries: np.ndarray
    targets: tuple[RegressionTarget, ...]
    standardized_coefficients: np.ndarray | None = None

    @property
    def person_effects(self) -> bool:
        return self.standardized_coefficients is not None


def _gather_entries(values, indices, inactive):
    result = np.full(indices.shape, inactive, dtype=values.dtype)
    active = indices >= 0
    result[active] = values.ravel()[indices[active]]
    result.flags.writeable = False
    return result


def block_constraint(constant, block, *, source=False):
    """Read local declarations before publication or project the canonical column."""
    arrays = constant.variables
    if (name := block.get("fixed_source" if source else "fixed")) is not None:
        return arrays[name].values
    entries = arrays[block["entry_index"]].values
    field, inactive = "fixed_value", 0.0
    if source:
        entries = entries.ravel()[arrays[block["fixed_flat"]].values]
        field, inactive = "fixed_source", "inactive"
    return _gather_entries(arrays[f"{block['kind']}_{field}"].values, entries, inactive)


def _parameter_scopes(constant):
    return (
        *constant.attrs["owners"],
        *constant.attrs["measurements"],
        constant.attrs["population"],
        *constant.attrs.get("regression_scopes", ()),
    )


def _measurement_parameter_uses(constant, bindings, rebuilt):
    """Publish final gathers for local measurement slots and collect their uses."""
    uses = []
    for measurement in constant.attrs["measurements"]:
        for slot in measurement["slots"]:
            owner, block = bindings[slot["binding_id"]]
            local = constant[slot["positions"]]
            kind = block["kind"]
            if kind in rebuilt:
                values = constant[block["entry_index"]].values.ravel()[local.values]
                variable = f"{measurement['name']}_{slot['key']}_entry_index"
                slot["entry_index"] = variable
                constant[variable] = local.dims, values
            else:
                values = constant[slot["entry_index"]].values
            entries = values.ravel()
            active = np.flatnonzero(entries >= 0)
            if not active.size:
                continue
            positions = local.values.ravel()[active]
            dims = block["dims"]
            traits = -1
            if "trait" in dims:
                traits = np.unravel_index(positions, block["shape"])[dims.index("trait")]
            elif "statement_indices" in owner and "statement" in dims[0]:
                statements = constant[owner["statement_indices"]].values[positions]
                traits = constant.statement_trait.values[statements]
            uses.append(
                np.broadcast_arrays(
                    kind,
                    entries[active],
                    measurement["unit_index"],
                    slot["key"],
                    active,
                    traits,
                )
            )
    return uses


def _regression_parameter_uses(constant, regression_plans):
    """Project each coefficient and residual pool onto its measurement units."""
    uses = []
    for scope in regression_plans:
        coefficient = next((b for b in scope["blocks"] if b.get("key") == "coefficients"), None)
        if coefficient is None:
            continue
        coefficient_map = constant[coefficient["entry_index"]].values
        residual = next((b for b in scope["blocks"] if b["key"] == "residual_sd"), None)
        grid = constant[scope["coefficient_positions"]]
        positions = grid.values
        trait_by_pool = constant[f"{grid.dims[0]}_trait_index"].values
        for target in scope["targets"]:
            if not target["unit_pools"]:
                continue
            units, pools = np.asarray(target["unit_pools"], dtype=np.int64).T
            term_positions = positions[pools]
            entries = coefficient_map.ravel()[term_positions]
            slots = term_positions
            if residual is not None:
                residual_map = constant[residual["entry_index"]].values
                entries = np.column_stack((entries, residual_map[pools]))
                slots = np.column_stack((slots, pools))
            is_residual = np.broadcast_to(
                np.arange(entries.shape[1]) == positions.shape[1], entries.shape
            )
            uses.append(
                (
                    np.where(is_residual, "regression_sd", "regression_coefficient").ravel(),
                    entries.ravel(),
                    np.repeat(units, entries.shape[1]),
                    np.where(is_residual, "residual_sd", "coefficients").ravel(),
                    slots.ravel(),
                    np.repeat(trait_by_pool[pools], entries.shape[1]),
                )
            )
    return uses


def _publish_parameter_uses(constant, prior, uses, retained, *, measurement_uses):
    """Write usage columns in measurement order, then regression order."""
    retained_uses = 0 if retained is None else len(prior.parameter_use.values[retained])
    if retained_uses or uses:
        count = retained_uses + sum(chunk[0].size for chunk in uses)
        constant.add_coords({"parameter_use": np.arange(count, dtype=np.int64)})
        fields = ("kind", "entry_index", "unit_index", "key", "slot_position", "trait_index")
        columns = zip(*uses, strict=True) if uses else [()] * len(fields)
        for field, chunks in zip(fields, columns, strict=True):
            name = f"parameter_use_{field}"
            if retained_uses:
                existing = prior.variables[name].values[retained]
                chunks = (*chunks, existing) if measurement_uses else (existing, *chunks)
            dtype = str if field in {"kind", "key"} else np.int64
            values = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            constant[name] = "parameter_use", values.astype(dtype, copy=False)


def _existing_parameter_ids(constant, blocks):
    """Read published entry IDs and the next available ID for each affected kind."""
    previous_ids, next_ids = {}, {}
    for block in blocks:
        kind = block["kind"]
        axis = f"{kind}_id"
        if "entry_index" in block and axis in constant.coords:
            entries = constant[block["entry_index"]].values.ravel()
            labels = _gather_entries(constant[axis].values, entries, -1)
            previous_ids[block["binding_id"]] = labels
            next_ids[kind] = max(next_ids.get(kind, 0), int(labels.max(initial=-1)) + 1)
    return previous_ids, next_ids


def _regression_entry_metadata(constant, regression_plans):
    """Align each regression binding to its coefficient order and pool labels."""
    regression_entries = {}
    for plan in regression_plans:
        index = plan["regression_index"]
        position_grid = constant[plan["coefficient_positions"]]
        pool_dim, term_dim = position_grid.dims
        labels = tuple(constant[f"{pool_dim}_{field}"].values for field in REGRESSION_POOL_FIELDS)
        terms = constant[term_dim].values
        order = np.argsort(position_grid.values.ravel())
        for block in plan["blocks"]:
            regression_entries[block["binding_id"]] = index, order, terms, labels
    return regression_entries


def finalize_parameters(constant: LayoutBuilder | xr.Dataset, *, changed=None) -> xr.Dataset:
    """Publish a fresh layout or replace affected kinds in an existing Dataset."""
    previous = {record["kind"]: record for record in constant.attrs.get("parameters", ())}
    update = changed is not None and bool(previous)
    scopes = _parameter_scopes(constant)
    grouped = defaultdict(list)
    for scope in scopes:
        for block in scope["blocks"]:
            if "scope" not in block:
                grouped[block["kind"]].append((scope, block))
    rebuilt = {kind: members for kind, members in grouped.items() if not update or kind in changed}
    blocks = [pair for members in rebuilt.values() for pair in members]
    bindings = {
        block["binding_id"]: (scope, block)
        for members in grouped.values()
        for scope, block in members
    }
    # Keep the previous catalogue for one block-at-a-time projection during rebuild.
    prior = constant
    previous_ids, next_ids = _existing_parameter_ids(constant, (block for _, block in blocks))
    drop_dims = {
        dim
        for kind, record in previous.items()
        if not update or kind in changed
        for dim in record["dims"]
        if dim.endswith("_id")
    }
    measurement_uses = not update or any(
        "person" in block["dims"] and "entry_index" not in block for _, block in blocks
    )
    regression_uses = not update or bool(changed & REGRESSION_KINDS)
    retained = None
    if (measurement_uses or regression_uses) and "parameter_use" in constant.sizes:
        drop_dims.add("parameter_use")
        if measurement_uses != regression_uses:
            # The two usage segments always keep their existing publication order.
            kinds = constant.variables["parameter_use_kind"].values
            split = np.count_nonzero(~np.isin(kinds, tuple(REGRESSION_KINDS)))
            retained = slice(split, None) if measurement_uses else slice(split)
    if isinstance(constant, xr.Dataset):
        # Only published datasets can have old catalogue dimensions to replace.
        arrays = LayoutBuilder(coords={}, attrs=constant.attrs)
        arrays.update(constant.drop_dims(drop_dims) if drop_dims else constant)
        constant = arrays
    records = dict(previous) if update else {}
    measurement_scopes = {
        scope["name"] for key in ("owners", "measurements") for scope in constant.attrs[key]
    }
    population_scope = constant.attrs["population"]["name"]
    defaults = dict.fromkeys(PARAMETER_INDEX_FIELDS, -1)
    defaults.update(term="", target_kind="", scale="identity")
    owners = {scope["name"]: scope for scope in constant.attrs["owners"]}
    regression_plans = constant.attrs["regression_scopes"] if regression_uses else ()
    regression_entries = _regression_entry_metadata(constant, regression_plans)
    dependent_bounds = []
    for kind, members in rebuilt.items():
        native = all(
            scope["name"] == population_scope and block["key"] != "B" for scope, block in members
        )
        dims = tuple(members[0][1]["dims"]) if native else (f"{kind}_id",)
        chunks, ids, offset, ordered_start = [], [], 0, 0
        for scope, block in members:
            binding = block["binding_id"]
            name = scope["name"]
            block["point_parameter"] = name in measurement_scopes or (
                (name == population_scope or scope.get("person_effects", False))
                and block["fixed_for_scoring"]
            )
            values = block_constraint(prior, block).ravel()
            shape, axes = tuple(block["shape"]), tuple(block["dims"])
            local_source = np.full(len(values), "", dtype="U32")
            local_source[constant[block["fixed_flat"]].values] = block_constraint(
                prior, block, source=True
            )
            active = local_source != "inactive"
            positions = np.flatnonzero(active)
            shared = bool(block.get("shared"))
            if shared and len(positions):
                declared = positions[np.isin(local_source[positions], ("design", "user"))]
                positions = declared[:1] if len(declared) else positions[:1]
            count = len(positions)
            index = dict(zip(axes, np.unravel_index(positions, shape), strict=True)) if axes else {}
            strict = block.get("orientation_strict") or {}
            side = block.get("orientation")
            role = block.get("prior_role") or ""
            roles = np.full(len(values), role, dtype=object)
            if (reference := block.get("prior_role_index")) is not None:
                roles[constant[block["free_flat"]].values] = constant[reference].values
            chunk = {
                **defaults,
                "binding_id": binding,
                "parameter_set_index": scope["parameter_set_index"],
                "local_position": positions,
                "fixed_value": values[positions],
                "fixed_source": local_source[positions],
                "orientation": 0 if side is None else constant[side].values.ravel()[positions],
                "strict_positive": bool(strict.get("positive")),
                "strict_negative": bool(strict.get("negative")),
                "shared": shared,
                "prior_role": roles[positions].astype(str),
                **{
                    name: block[name]
                    for name in (
                        "component",
                        "fixed_for_scoring",
                        "summary_default",
                        "point_parameter",
                        "domain",
                    )
                },
            }
            if "unit_index" in scope:
                unit = scope["unit_index"]
                chunk.update(unit_index=unit, block_index=constant.unit_block.values[unit])
            if "person" in index:
                chunk["person_index"] = index["person"]
            if not shared:
                owner = owners.get(scope.get("owner"), scope)
                if axes and "statement_indices" in owner:
                    statement = constant[owner["statement_indices"]]
                    if statement.dims[0] == axes[0]:
                        indices = statement.values[index[axes[0]]]
                        chunk.update(
                            statement_index=indices,
                            trait_index=constant.statement_trait.values[indices],
                        )
                if axes and "item_indices" in owner:
                    item = constant[owner["item_indices"]]
                    if item.dims[0] == axes[0]:
                        chunk["item_index"] = item.values[index[axes[0]]]
                if "pair_indices" in scope and len(shape) == 2:
                    chunk.update(
                        pair_index=constant[scope["pair_indices"]].values[index[axes[0]]],
                        cutpoint_position=index[axes[-1]],
                    )
                if "unit_index" in scope and axes and axes[-1] == scope.get("category_dim"):
                    chunk.update(
                        category_index=index[axes[-1]],
                        item_index=constant.unit_item.values[scope["unit_index"]],
                    )
                for axis in ("trait", "trait_aux", "population_group"):
                    if axis in index and (
                        axis != "population_group"
                        or constant.attrs["population"]["recipe"]["groups"]
                    ):
                        chunk[f"{axis}_index"] = index[axis]
            if block["key"] == "population_mean":
                chunk.update(term="Intercept", target_kind="theta", scale="trait")
            if regression := regression_entries.get(binding):
                reg_index, order, terms, labels = regression
                residual = kind == "regression_sd"
                pool, term = (
                    (positions, -1) if residual else np.divmod(order[positions], len(terms))
                )
                target, scale, traits = (labels_by_pool[pool] for labels_by_pool in labels)
                chunk.update(
                    regression_index=reg_index,
                    pool_index=pool,
                    term_index=term,
                    trait_index=np.where(traits >= 0, traits, chunk["trait_index"]),
                    target_kind=target,
                    scale=scale,
                    term="" if residual else terms[term].astype(str),
                )
            ordered = block["ordered"]
            width = shape[-1] if ordered else 1
            chunk.update(
                ordered=ordered,
                ordered_count=width,
                ordered_group_index=ordered_start + positions // width if ordered else -1,
                lower_kind="",
                lower_entry_index=-1,
                upper_limit=np.nan,
            )
            if ordered:
                ordered_start += len(values) // width
            if block["bounds"] is not None:
                lower, upper = block["bounds"]
                _, lower_block = bindings[lower]
                chunk.update(lower_kind=lower_block["kind"], upper_limit=upper)
                dependent_bounds.append(
                    (kind, slice(offset, offset + count), lower_block, positions)
                )
            chunks.append({key: np.broadcast_to(value, (count,)) for key, value in chunk.items()})
            if not native:
                old = previous_ids.get(binding)
                labels = old[positions] if old is not None else np.full(count, -1, dtype=np.int64)
                missing = labels < 0
                start = next_ids.get(kind, 0)
                next_ids[kind] = stop = start + int(missing.sum())
                labels[missing] = np.arange(start, stop, dtype=np.int64)
                ids.append(labels)
            local_map = np.full(len(values), -1, dtype=np.int64)
            local_map[positions] = np.arange(offset, offset + count)
            offset += count
            if shared and count:
                local_map[active] = local_map[positions[0]]
            block["entry_index"] = f"binding_{binding}_entry_index"
            constant[block["entry_index"]] = axes, local_map.reshape(shape)
            for key in ("fixed", "fixed_source"):
                if block.get(key) is not None:
                    del constant.variables[block[key]]
                    block[key] = None
        shape = tuple(members[0][1]["shape"]) if native else (offset,)
        if not native:
            constant.add_coords({dims[0]: np.concatenate(ids)})
        for field in chunks[0]:
            values = np.concatenate([chunk[field] for chunk in chunks])
            constant[f"{kind}_{field}"] = dims, values.reshape(shape)
        records[kind] = dict(
            kind=kind, dims=list(dims), shape=list(shape), columns=tuple(chunks[0])
        )
    for scope, block in blocks:
        spec = scope.get("specification")
        if spec is not None and block["key"] in spec.parameter_scales:
            scale_key, power = spec.parameter_scales[block["key"]]
            other = next(b for b in scope["blocks"] if b["key"] == scale_key)
            block.update(
                scale_kind=other["kind"], scale_entry_index=other["entry_index"], scale_power=power
            )
            block["noncentered"] = (
                block["key"] == "statement_effect"
                and spec.statement_effect_parameterization == "noncentered"
            )
    # Dependent bounds need the lower parameter's completed local-to-final map.
    for kind, entries, lower, positions in dependent_bounds:
        lower_entries = constant[lower["entry_index"]].values.ravel()
        constant[f"{kind}_lower_entry_index"].values.ravel()[entries] = lower_entries[positions]
    uses = []
    if measurement_uses:
        uses.extend(_measurement_parameter_uses(constant, bindings, rebuilt))
    uses.extend(_regression_parameter_uses(constant, regression_plans))
    _publish_parameter_uses(constant, prior, uses, retained, measurement_uses=measurement_uses)
    constant.attrs["parameters"] = [records[kind] for kind in grouped]
    return constant.build()


def _labels(values):
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result


def _owner_key(specification):
    """Parameter-sharing compatibility; never use this to compare full declarations.

    Identification choices are pooled and checked later by the constraint builder.
    """
    options = {
        f.name: getattr(specification, f.name) for f in fields(specification) if f.name != "priors"
    }
    options.pop("reference_statements", None)
    options.pop("loading_anchors", None)
    return type(specification).__name__, PriorPreparation.key(options)


def _consistent_column(frame, identifier, column):
    """Align optional design declarations, rejecting conflicting shared values."""
    values = {}
    if column not in frame:
        return values
    for label, value in zip(frame[identifier], frame[column], strict=True):
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{column} must contain real numeric values or missing cells")
        if label in values and values[label] != value:
            raise ValueError(f"{identifier} {label!r}: conflicting {column} declarations")
        values[label] = value
    return values


def _ownership(units, measurements, unit_blocks, block_statements):
    """Group compatible declarations and units connected by shared statements."""
    components = DisjointSet(range(len(units)))
    memberships, declarations = {}, {}
    for index, (unit, measurement, block) in enumerate(
        zip(units, measurements, unit_blocks, strict=True)
    ):
        spec = measurement["specification"]
        key = unit["channel"], _owner_key(spec)
        declaration = key, PriorPreparation.key(spec.priors)
        first = declarations.setdefault(declaration, index)
        if first != index:
            components.merge(index, first)
        if block >= 0:
            for statement in block_statements[block, : unit["size"]].tolist():
                first = memberships.setdefault((key, statement), index)
                if first != index:
                    components.merge(index, first)
    owners = []
    for group in sorted(components.subsets(), key=min):
        indices = sorted(group)
        name = f"o{len(owners)}"
        owners.append(
            dict(
                name=name,
                channel=units[indices[0]]["channel"],
                specification=measurements[indices[0]]["specification"],
                units=indices,
            )
        )
        for index in indices:
            measurements[index]["owner"] = name
    return owners


def _layout_statement_features(ds, channels, statement_index):
    """Align optional feature columns to the canonical statement axis."""
    for channel in channels:
        if "statement_features" not in channel.tables:
            continue
        features = channel.tables["statement_features"]
        ds.attrs["statement_features"] = []
        for column in features.columns:
            if column == "statement_id":
                continue
            if isinstance(column, (bool, np.bool_)) or not isinstance(column, (str, int)):
                raise ValueError("Statement feature column names must be strings or integers")
            values = np.full(ds.sizes["statement"], None, dtype=object)
            for label, value in zip(features.statement_id, features[column], strict=True):
                if value is None or value is pd.NA:
                    continue
                if not isinstance(
                    value, (str, bool, int, float, np.integer, np.floating, np.bool_)
                ):
                    raise ValueError(
                        "Statement features must contain scalar numbers, strings or booleans"
                    )
                if pd.isna(value):
                    continue
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ValueError("Numeric statement features must be finite or missing")
                values[statement_index[("fc", label)]] = (
                    value.item() if isinstance(value, np.generic) else value
                )
            variable = f"statement_feature_{len(ds.attrs['statement_features'])}"
            ds[variable] = "statement", values
            ds.attrs["statement_features"].append(dict(name=column, variable=variable))


def _layout_fc_membership(ds, frame, units, block_index, statement_index, trait_index):
    """Index FC membership and validate attributes of shared statements."""
    block_rows = frame.block_id.map(block_index).to_numpy(dtype=np.intp)
    sizes = np.asarray([r["size"] for r in units if r["channel"] == "fc"], dtype=np.int64)
    size = int(sizes.max())
    ds.add_coords({"position": np.arange(size, dtype=np.int64)})
    membership = np.full((len(block_index), size), -1, dtype=np.int64)
    membership[block_rows, frame.position.to_numpy(dtype=np.intp) - 1] = [
        statement_index[("fc", label)] for label in frame.statement_id
    ]
    ds["fc_statement"] = ("block", "position"), membership
    ds["fc_size"] = "block", sizes
    first = frame.drop_duplicates("statement_id")
    indices = [statement_index[("fc", label)] for label in first.statement_id]
    ds["statement_trait"].values[indices] = first.trait.map(trait_index).to_numpy()
    ds["statement_keying"].values[indices] = first.keying.to_numpy()
    if "location_side" in first:
        ds["statement_location_side"].values[indices] = first.location_side.to_numpy(dtype=float)
    for column in ("fix_loading", "fix_var"):
        for label, value in _consistent_column(frame, "statement_id", column).items():
            ds[f"statement_{column}"].values[statement_index[("fc", label)]] = value
    return membership


def _layout_item_traits(ds, rows, items, item_index, statement_index, trait_index):
    """Index the active trait and declared constraints of each item."""
    q = np.zeros((len(items), len(trait_index)), dtype=bool)
    sides, fixed = np.full(q.shape, np.nan), np.full(q.shape, np.nan)
    for row in rows:
        item, trait = item_index[row.item_id], trait_index[row.trait]
        q[item, trait] = True
        sides[item, trait] = getattr(row, "location_side", np.nan)
        value = getattr(row, "fix_loading", np.nan)
        if not pd.isna(value):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError("fix_loading must contain real numeric values or missing cells")
            fixed[item, trait] = value
    ds["item_q"] = ("item", "trait"), q
    ds["item_location_side"] = ("item", "trait"), sides
    ds["item_fix_loading"] = ("item", "trait"), fixed
    ds["item_statement"] = (
        "item",
        np.array([statement_index[("items", item)] for item in items], dtype=np.int64),
    )


def _layout_population(
    ds,
    people,
    traits,
    population,
    person_data,
    theta_parameterization,
    centered_traits,
    *,
    fc_design,
):
    """Bind fitted population groups and partition the established trait axis."""
    recipe, groups, alignment = fit_groups(person_data, people, population)
    ds.add_coords(
        {"theta_predictor": _labels(()), "population_group": _labels(recipe["groups"] or ("all",))}
    )
    ds["person_group"] = "person", groups
    ds["predictors"] = ("person", "theta_predictor"), np.empty((len(people), 0))
    if centered_traits is None:
        if theta_parameterization == "conditional":
            centered = set(fc_design.trait) if fc_design is not None else set()
        else:
            centered = set(traits) if theta_parameterization == "centered" else set()
    else:
        selected = identifiers(centered_traits, "centered_traits")
        centered = set(selected)
        if len(centered) != len(selected):
            raise ValueError("centered_traits contains duplicate trait IDs")
        if centered - set(traits):
            raise ValueError(
                f"centered_traits contains unknown trait IDs: {centered - set(traits)}"
            )
    labels = ds.trait.values
    mask = np.array([trait in centered for trait in traits], dtype=bool)
    for kind, selected in (("centered", mask), ("noncentered", ~mask)):
        dim = f"{kind}_trait"
        ds.add_coords({dim: labels[selected]})
        ds[f"{kind}_index"] = dim, np.flatnonzero(selected).astype(np.int64, copy=False)
    ds.attrs["population"] = dict(
        name="population",
        specification=population,
        recipe=recipe,
        theta_parameterization=theta_parameterization,
        blocks=[],
    )
    return alignment


def prepare_layout(
    channels: tuple[NormalizedChannel, ...],
    *,
    population,
    person_data,
    theta_parameterization: str,
    centered_traits=None,
) -> tuple[LayoutBuilder, tuple[np.ndarray, pd.Index] | None]:
    """Lay out normalized entities and return the transient person-table alignment."""
    # Dict insertion order preserves first-seen axes across response channels.
    people, traits, blocks, items, statement_keys = {}, {}, {}, {}, {}
    units, measurements, unit_ids = [], [], []
    designs = {}
    for channel in channels:
        tables = channel.tables
        name = channel.channel
        designs[name] = tables["design"]
        people.update(dict.fromkeys(channel.people))
        frame = designs[name]
        traits.update(dict.fromkeys(frame.trait))
        label_column = "statement_id" if name == "fc" else "item_id"
        statement_keys.update(dict.fromkeys((name, label) for label in frame[label_column]))
        local_ids = blocks if name == "fc" else items
        for row in tables["units"].itertuples(index=False):
            index = len(units)
            local_ids.setdefault(row.unit_id, None)
            unit_ids.append(row.unit_id)
            measurement = f"m{index}"
            labels = row.scale["labels"]
            scale = dict(
                row.scale,
                categories=list(row.scale["categories"]),
                labels=None if labels is None else list(labels),
            )
            units.append(
                dict(
                    channel=name,
                    unit_index=index,
                    response_format=row.response_format,
                    size=row.size,
                    input_format=channel.input_format,
                    scale=scale,
                    likelihood_kind=row.likelihood_kind,
                )
            )
            measurements.append(
                dict(
                    name=measurement,
                    unit_index=index,
                    specification=row.model,
                    blocks=[],
                    slots=[],
                )
            )
    people, traits, blocks, items, statement_keys = (
        list(labels) for labels in (people, traits, blocks, items, statement_keys)
    )
    identifiers(people, "person_id across channels")
    identifiers(traits, "trait across channels")
    ds = LayoutBuilder(
        coords=dict(
            person=_labels(people),
            trait=_labels(traits),
            trait_aux=_labels(traits),
            statement=np.arange(len(statement_keys), dtype=np.int64),
            block=_labels(blocks),
            item=_labels(items),
            unit=np.arange(len(units), dtype=np.int64),
            statement_label=("statement", _labels([label for _, label in statement_keys])),
            statement_channel=("statement", [channel for channel, _ in statement_keys]),
            unit_id=("unit", _labels(unit_ids)),
            unit_channel=("unit", [unit["channel"] for unit in units]),
        ),
        attrs=dict(units=units, measurements=measurements),
    )
    statement_index = {key: index for index, key in enumerate(statement_keys)}
    _layout_statement_features(ds, channels, statement_index)
    trait_index = {label: index for index, label in enumerate(traits)}
    block_index = {label: index for index, label in enumerate(blocks)}
    item_index = {label: index for index, label in enumerate(items)}
    ds["unit_model"] = (
        "unit",
        np.asarray([type(record["specification"]).__name__ for record in measurements], dtype=str),
    )
    ds["unit_response_format"] = (
        "unit",
        np.asarray([unit["response_format"] or "" for unit in units], dtype=str),
    )
    for axis, lookup, channel in (("block", block_index, "fc"), ("item", item_index, "items")):
        ds[f"unit_{axis}"] = (
            "unit",
            np.fromiter(
                (
                    lookup[u] if r["channel"] == channel else -1
                    for u, r in zip(unit_ids, units, strict=True)
                ),
                dtype=np.int64,
                count=len(units),
            ),
        )
    ds["statement_trait"] = "statement", np.full(len(statement_keys), -1, dtype=np.int64)
    ds["statement_keying"] = "statement", np.ones(len(statement_keys))
    for column, default in (
        ("location_side", np.nan),
        ("fix_loading", np.nan),
        ("fix_var", np.nan),
    ):
        ds[f"statement_{column}"] = "statement", np.full(len(statement_keys), default)
    membership = None
    if "fc" in designs:
        membership = _layout_fc_membership(
            ds, designs["fc"], units, block_index, statement_index, trait_index
        )
    item_rows = designs["items"].itertuples() if "items" in designs else ()
    _layout_item_traits(ds, item_rows, items, item_index, statement_index, trait_index)
    owners = ds.attrs["owners"] = _ownership(units, measurements, ds.unit_block.values, membership)
    for owner in owners:
        indices = owner["units"]
        if owner["channel"] == "fc":
            statements = tuple(
                dict.fromkeys(
                    int(s)
                    for index in indices
                    for s in ds.fc_statement.values[int(ds.unit_block.values[index])]
                    if s >= 0
                )
            )
        else:
            item_indices = np.array(
                [ds.unit_item.values[index] for index in indices], dtype=np.int64
            )
            statements = tuple(ds.item_statement.values[item_indices])
            dim = f"{owner['name']}_item"
            ds.add_coords({dim: _labels([items[index] for index in item_indices])})
            owner["item_indices"] = f"{owner['name']}_item_index"
            ds[owner["item_indices"]] = dim, item_indices
        dim = f"{owner['name']}_statement"
        ds.add_coords({dim: _labels([statement_keys[index][1] for index in statements])})
        owner["statement_indices"] = f"{owner['name']}_statement_index"
        ds[owner["statement_indices"]] = dim, np.asarray(statements, dtype=np.int64)
        owner["blocks"] = []
    alignment = _layout_population(
        ds,
        people,
        traits,
        population,
        person_data,
        theta_parameterization,
        centered_traits,
        fc_design=designs.get("fc"),
    )
    for index, scope in enumerate(_parameter_scopes(ds)):
        scope["parameter_set_index"] = index
    return ds, alignment


def _person_key(label):
    """Encode a normalized respondent ID for the persisted evidence history."""
    return hashlib.sha256(json.dumps(_semantic_identity_value(label)).encode()).hexdigest()


def _identity_digest(value):
    encoded = json.dumps(_semantic_identity_value(value), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _semantic_identity_value(value):
    if isinstance(value, xr.Dataset):
        return _semantic_identity_value(value.variables)
    if isinstance(value, xr.Variable):
        return _semantic_identity_value((value.dims, value.values))
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biuf":
            dtype = value.dtype.newbyteorder("<")
            flat = np.asarray(value, dtype=dtype).ravel()
            return ["array", list(value.shape), dtype.str, hashlib.sha256(flat).hexdigest()]
        if value.dtype.kind == "U":
            return ["text", list(value.shape), value.ravel().tolist()]
        return ["array", list(value.shape), [_semantic_identity_value(v) for v in value.ravel()]]
    if isinstance(value, np.generic):
        return _semantic_identity_value(value.item())
    if isinstance(value, Mapping):
        entries = [
            (_semantic_identity_value(key), _semantic_identity_value(item))
            for key, item in value.items()
        ]
        return ["mapping", sorted(entries, key=lambda entry: json.dumps(entry[0], sort_keys=True))]
    if isinstance(value, (tuple, list)):
        return ["sequence", [_semantic_identity_value(item) for item in value]]
    if value is None:
        return ["none"]
    if type(value) in {str, int, bool}:
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", value if np.isfinite(value) else str(value)]
    raise TypeError(f"Unsupported calibration semantic value: {type(value).__name__}")
