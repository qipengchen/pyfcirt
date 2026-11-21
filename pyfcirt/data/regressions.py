"""Resolve one active ordinary or conditional scheme per existing parameter role."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pymc as pm
from jax import enable_x64

from pyfcirt.common.validation import identifier
from pyfcirt.covariates.design import fit_design
from pyfcirt.covariates.specification import Regression, validate_formula
from pyfcirt.data.arrays import LayoutBuilder, merge_prepared
from pyfcirt.data.constraints import (
    _check_population_directions,
    _layout,
    check_asymptote_priors,
    freeze_prior_shapes,
)
from pyfcirt.data.coordinates import REGRESSION_KINDS, REGRESSION_POOL_FIELDS, finalize_parameters
from pyfcirt.data.grouping import theta_target_indices
from pyfcirt.models.priors import PriorPreparation, prior_shape, require_proper, subset_prior
from pyfcirt.models.specification import (
    conditional_defaults,
    regression_defaults,
)

if TYPE_CHECKING:
    from pyfcirt.data.prepare import PreparedInputs


def _active_updates(constant, declarations, *, regression=False):
    """Resolve each selector once against the candidate; the last owner wins."""
    scopes = {"fc": [], "items": [], "population": [(constant.attrs["population"], None)]}
    if any(
        isinstance(target, tuple) and target and target[0] in ("fc", "items")
        for target, _ in declarations
    ):
        owners = {owner["name"]: owner for owner in constant.attrs["owners"]}
        for owner in owners.values():
            scopes[owner["channel"]].append((owner, owner))
        for measurement in constant.attrs["measurements"]:
            owner = owners[measurement["owner"]]
            scopes[owner["channel"]].append((measurement, owner))

    resolved = {"theta": [(None, None, None, "theta")]} if regression else {}
    for target, _ in declarations:
        if target == "theta":
            if not regression:
                raise ValueError("The Gaussian theta law has no ordinary-prior selector")
            continue
        if not isinstance(target, tuple) or len(target) not in (2, 3):
            raise ValueError(f"Unsupported parameter target: {target!r}")
        if target in resolved:
            continue
        channel, *family, selector = target
        if channel not in scopes or (
            channel == "population"
            and (
                regression
                or family
                or selector not in {"population_mean", "population_sd", "trait_correlation"}
            )
        ):
            raise ValueError(f"Unsupported parameter target: {target!r}")
        entries = []
        for scope, owner in scopes[channel]:
            if family and type(scope["specification"]).__name__ != family[0]:
                continue
            for layout in scope["blocks"]:
                alias = (
                    layout["orientation"] is not None and selector in layout["prior_roles"].values()
                )
                if layout["key"] != selector and not alias:
                    continue
                if regression and not layout["regression"]:
                    raise ValueError(f"Unsupported parameter target: {target!r}")
                roles = layout["prior_groups"]
                if regression and layout.get("shared") and roles:
                    raise ValueError(
                        "1PL common loading does not support item-specific regressions"
                    )
                entries.extend(
                    (scope, owner, layout, str(role))
                    for role in roles
                    if not alias or role == selector
                )
        if not entries:
            raise ValueError(f"Parameter target {target!r} has no applicable free parameters")
        resolved[target] = entries

    claimed, active = set(), []
    for target, declaration in reversed(declarations):
        selected = []
        for entry in resolved[target]:
            layout, role = entry[2:]
            binding_id = -1 if layout is None else layout["binding_id"]
            binding_role = binding_id, role
            if binding_role not in claimed:
                selected.append(entry)
                claimed.add(binding_role)
        if selected:
            active.append((target, declaration, selected))
    active.reverse()
    return active, claimed


def _pair_keys(constant, scope):
    block = constant.unit_id.values[scope["unit_index"]]
    indices = constant[scope["pair_indices"]].values
    labels = constant.statement_label.values
    first, second = (labels[constant[f"pair_statement_{side}"].values[indices]] for side in (1, 2))
    return [(block, a, b) for a, b in zip(first, second, strict=True)]


def _canonical_pairs(spec, entities):
    columns = spec.id if isinstance(spec.id, tuple) else (spec.id,)
    if len(columns) != 3:
        raise ValueError("Cutpoint regression requires id=(block, statement_1, statement_2)")
    canonical = {}
    for key in entities:
        canonical[key] = key
        canonical[(key[0], key[2], key[1])] = key
    frame = spec.data.copy(deep=True)
    keys, undirected = [], []
    for row in frame[list(columns)].itertuples(index=False, name=None):
        key = tuple(identifier(value, "Cutpoint regression ID") for value in row)
        keys.append(canonical.get(key, key))
        undirected.append((key[0], frozenset(key[1:])))
    if len(set(undirected)) != len(undirected):
        raise ValueError("Cutpoint regression contains duplicate or reversed-pair IDs")
    for index, column in enumerate(columns):
        frame[column] = pd.Series([key[index] for key in keys], index=frame.index, dtype=object)
    return replace(spec, data=frame)


def _item_targets(constant, selected, key):
    entities, pools = {}, {}
    defaults, targets, pool_rows, labels = [], [], [], []
    for scope, owner, layout, role in selected:
        conditional = conditional_defaults(scope["specification"], role)
        kind = layout["kind"]
        scale = "log_loading" if kind == "loading" and conditional["link"] == "log" else "identity"
        shape = tuple(layout["shape"])
        free = constant[layout["free_flat"]].values
        ordered = layout["ordered"]
        if ordered and len(free) != int(np.prod(shape)):
            raise NotImplementedError("Regression requires complete free ordered cutpoint vectors")
        local_keys = (
            _pair_keys(constant, scope)
            if key == "cutpoints"
            else [(label,) for label in constant[layout["dims"][0]].values]
        )
        positions = np.take(free, layout["prior_groups"][role])
        if ordered:
            positions = np.unique(positions // shape[-1])
        rows, indices = np.empty_like(positions), np.empty_like(positions)
        dimensions = shape[:-1] if ordered else shape
        count = shape[-1] if key == "cutpoints" else None
        trait_axis = layout["dims"].index("trait") if "trait" in layout["dims"] else None
        coordinates = zip(*np.unravel_index(positions, dimensions), strict=True)
        for index, position in enumerate(coordinates):
            trait = position[trait_axis] if trait_axis is not None else -1
            signature = owner["name"], key, role, trait, count
            if signature not in pools:
                pools[signature] = len(pools)
                defaults.append((scope["specification"], role))
                pool_rows.append(set())
                labels.append((kind, scale, trait))
            entity = local_keys[position[0]]
            row = entities.setdefault(entity, len(entities))
            pool = pools[signature]
            rows[index], indices[index] = row, pool
            pool_rows[pool].add(row)
        targets.append((layout, role, positions, rows, indices, conditional))
    return list(entities), defaults, targets, pool_rows, labels


def _rank_check(matrix, label):
    """Check the final training columns, after Formulae coding and pooling."""
    with enable_x64():
        values = jnp.asarray(matrix, dtype=jnp.float64)
        rank = int(jnp.linalg.matrix_rank(values))
    if rank != values.shape[1]:
        raise ValueError(f"{label}: regression design columns are linearly dependent")


def _resolve_coefficients(spec, design, pool_defaults, qualified_priors=None):
    qualified_priors = qualified_priors or {}
    if unknown := qualified_priors.keys() - {type(model).__name__ for model, _ in pool_defaults}:
        raise ValueError(f"No applicable free parameter targets for models: {sorted(unknown)}")
    terms = design.term.values
    for family, overrides in qualified_priors.items():
        if unknown := overrides.keys() - {*terms, "common"}:
            raise ValueError(f"{family}: priors for unknown design columns: {sorted(unknown)}")
    native, roles, registered = [], [], {}

    def register(prior, *, term=None, family=None):
        # A non-scalar model override describes only that model's pools.
        scope = family if family is not None and any(prior_shape(prior)) else None
        key = term, scope, PriorPreparation.key(prior)
        if key not in registered:
            registered[key] = f"prior_{len(native)}"
            native.append(prior)
        return registered[key]

    for model, role in pool_defaults:
        intercept, _, slope_sigma = regression_defaults(model, role)
        row = []
        family = type(model).__name__
        by_model = qualified_priors.get(family, {})
        for term in terms:
            fallback = None if term == "Intercept" else "common"
            prior = by_model.get(term, by_model.get(fallback))
            model_override = prior is not None
            if prior is None:
                prior = spec.priors.get(term, spec.priors.get(fallback))
            if prior is None:
                if term == "Intercept":
                    prior = intercept
                else:
                    sigma = slope_sigma / design.attrs["prior_scale_divisors"][term]
                    if not isfinite(sigma) or sigma <= 0:
                        raise ValueError(f"Regression {term}: invalid automatic prior SD")
                    prior = PriorPreparation.default(pm.Normal.dist, mu=0, sigma=sigma)
            require_proper(prior, role=f"Regression {term}")
            if (
                spec.coefficient_noncentered
                and term != "Intercept"
                and not isinstance(prior.owner.op, pm.Normal.rv_type)
            ):
                raise NotImplementedError("Noncentered coefficients require a native Normal prior")
            row.append(register(prior, term=str(term), family=family if model_override else None))
        roles.append(row)
    return native, np.asarray(roles, dtype=object), register


def _coefficient_layout(
    constant,
    arrays,
    name,
    design,
    spec,
    pool_defaults,
    pool_labels,
    *,
    theta=False,
    qualified_priors=None,
):
    terms = list(design.term.values)
    pool_dim, term_dim = f"{name}_pool", f"{name}_term"
    coordinates = {pool_dim: np.arange(len(pool_defaults), dtype=np.int64), term_dim: terms}
    native, roles, register = _resolve_coefficients(spec, design, pool_defaults, qualified_priors)
    if theta:
        population = constant.attrs["population"]
        by_group = population["recipe"]["regression_by_group"]
        coordinates["theta_predictor"] = terms
        dims = (
            ("population_group", "theta_predictor", "trait")
            if by_group
            else ("theta_predictor", "trait")
        )
        coordinates.update(
            {dim: constant.variables[dim] for dim in dims if dim != "theta_predictor"}
        )
    arrays.add_coords(coordinates)
    for field, values in zip(REGRESSION_POOL_FIELDS, zip(*pool_labels, strict=True), strict=True):
        arrays[f"{pool_dim}_{field}"] = pool_dim, np.asarray(values)
    if theta:
        layout = _layout(
            arrays,
            "population",
            "B",
            dims,
            str(roles.flat[0]),
            kind="regression_coefficient",
            component="regression",
        )
        shape = tuple(layout["shape"])
        positions = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        positions = (
            positions.transpose(0, 2, 1).reshape(-1, len(terms)) if by_group else positions.T
        )
        flat_roles = np.empty(np.prod(shape), dtype=object)
        flat_roles[positions.ravel()] = roles.ravel()
        population["blocks"].append(layout)
        records = [dict(scope="population", key="B", binding_id=layout["binding_id"])]
    else:
        layout = _layout(
            arrays,
            name,
            "coefficients",
            (pool_dim, term_dim),
            str(roles.flat[0]),
            fixed_for_scoring=False,
            kind="regression_coefficient",
            component="regression",
        )
        positions = np.arange(roles.size, dtype=np.int64).reshape(roles.shape)
        flat_roles = roles.ravel()
        records = [layout]
    layout["noncentered"] = spec.coefficient_noncentered
    if spec.coefficient_noncentered and "Intercept" in terms:
        layout["centered_roles"] = tuple(dict.fromkeys(roles[:, terms.index("Intercept")]))
    role_variable = f"{name}_coefficient_prior_role"
    arrays[role_variable] = arrays[layout["free_flat"]].dims, flat_roles
    layout["prior_role_index"] = role_variable
    position_variable = f"{name}_coefficient_positions"
    arrays[position_variable] = (pool_dim, term_dim), positions
    if not theta:
        fixed = (
            np.full(len(pool_defaults), spec.parameter_residual_sd, dtype=float)
            if isinstance(spec.parameter_residual_sd, Real)
            else None
        )
        residual_roles = []
        for model, role in pool_defaults:
            if fixed is not None:
                residual_roles.append("residual_sd")
                continue
            prior = spec.parameter_residual_sd
            if prior is None:
                _, prior, _ = regression_defaults(model, role)
            require_proper(prior, role="Regression residual_sd")
            residual_roles.append(register(prior))
        residual = _layout(
            arrays,
            name,
            "residual_sd",
            (pool_dim,),
            residual_roles[0],
            fixed=fixed,
            sources=None if fixed is None else np.full(fixed.shape, "fixed", dtype="U16"),
            fixed_for_scoring=False,
            kind="regression_sd",
            component="regression",
            domain="positive",
        )
        if len(set(residual_roles)) > 1:
            role_variable = f"{name}_residual_sd_prior_role"
            arrays[role_variable] = (
                arrays[residual["free_flat"]].dims,
                np.asarray(residual_roles, dtype=object),
            )
            residual["prior_role_index"] = role_variable
        records.append(residual)
    priors = {f"prior_{index}": prior for index, prior in enumerate(native)}
    actual_layouts = [layout] + ([] if theta else [residual])
    records_by_role = {}
    for current in actual_layouts:
        records_by_role.update(freeze_prior_shapes(arrays, current, priors))
    # One coefficient/scale prior record mapping is the persisted source of truth.
    design.attrs["resolved_priors"] = records_by_role
    return records, position_variable


def _store_targets(constant, arrays, name, targets, spec, slots_by_binding):
    records = []
    for index, (layout, role, positions, rows, pools, conditional) in enumerate(targets):
        requires_centered = conditional["family"] == "TruncatedNormal" or layout["ordered"]
        if spec.deviation_noncentered is True and requires_centered:
            raise NotImplementedError(
                "Truncated or ordered conditionals require centered parameters"
            )
        binding = layout["binding_id"]
        noncentered = spec.deviation_noncentered
        if binding == -1:
            noncentered = False
        elif noncentered is None:
            noncentered = not requires_centered
        index_variable = f"{name}_target_{index}"
        unit_pools = []
        if positions is None:
            dims = ("person", "trait")
            arrays.add_coords({dim: constant[dim].variable for dim in dims})
        else:
            dims = (f"{index_variable}_position",)
            arrays.add_coords({dims[0]: positions})
            # For ordered thresholds, one regression target supplies the entire vector.
            stride = layout["shape"][-1] if layout["ordered"] else 1
            lookup = np.full(int(np.prod(layout["shape"])) // stride, -1, dtype=np.int64)
            lookup[positions] = pools
            for unit, local in slots_by_binding[binding]:
                indices = lookup[constant[local].values.ravel() // stride]
                unit_pools.extend(
                    (unit, pool) for pool in np.unique(indices[indices >= 0]).tolist()
                )
        arrays[index_variable] = ("regression_target_field", *dims), np.stack((rows, pools))
        conditional = dict(conditional)
        if "Intercept" in spec.priors:
            conditional["intercept_source"] = "user"
        if spec.parameter_residual_sd is not None:
            conditional["residual_sd_source"] = (
                "fixed" if isinstance(spec.parameter_residual_sd, Real) else "user"
            )
        records.append(
            dict(
                binding_id=binding,
                prior_role=role,
                indices=index_variable,
                unit_pools=tuple(unit_pools),
                conditional=conditional,
                noncentered=noncentered,
            )
        )
    return records


def _theta_design(constant, spec, feature_columns, *, aligned_rows=None):
    if spec.parameter_residual_sd is not None or spec.deviation_noncentered is not None:
        raise ValueError("Theta regression uses the existing population scale and parameterization")
    columns = (spec.id,) if isinstance(spec.id, str) else spec.id
    if len(columns) != 1:
        raise ValueError("Theta regression requires a single person ID column")
    keys = (
        pd.DataFrame({columns[0]: pd.Series(constant.person.values, dtype=object)})
        if aligned_rows is None
        else None
    )
    population = constant.attrs["population"]
    recipe = population["recipe"]
    reference = None
    if recipe["groups"]:
        reference = constant.person_group.values == recipe["groups"].index(
            recipe["reference_group"]
        )
    design = fit_design(
        spec,
        keys,
        reference_rows=reference,
        include_intercept=False,
        aligned_rows=aligned_rows,
        feature_columns=feature_columns,
    )
    if recipe["group"] in {record["name"] for record in design.attrs["feature_records"]}:
        raise ValueError("Theta formula cannot repeat the population group column")
    design.attrs["reference_group"] = recipe["reference_group"]
    x, groups = design.X.values, constant.person_group.values
    group_count = constant.sizes["population_group"]
    if recipe["regression_by_group"]:
        for group in range(group_count):
            selected = x[groups == group]
            _rank_check(np.column_stack([np.ones(len(selected)), selected]), f"Theta group {group}")
    else:
        _rank_check(np.column_stack([np.eye(group_count)[groups], x]), "Theta")
    traits = constant.sizes["trait"]
    rows, pools = theta_target_indices(groups, traits, by_group=recipe["regression_by_group"])
    pool_count = group_count * traits if recipe["regression_by_group"] else traits
    conditional = dict(
        family="Normal",
        link="identity",
        bounds=None,
        intercept_source="population_mean",
        residual_sd_source="population_covariance",
    )
    target = dict(binding_id=-1, kind="theta", dims=["person", "trait"], ordered=False)
    defaults = [(population["specification"], "population_mean")] * pool_count
    return design, defaults, [(target, "theta", None, rows, pools, conditional)]


def _parameter_update_candidate(state, declarations, *, regression=False):
    """Detach the layout, resolve replacements and retire their previous regressions."""
    result = state.to_dict()
    constant = result["/constant_data"]
    # Mutating the candidate layout must not change the source metadata.
    constant.attrs = deepcopy(constant.attrs)
    active, replaced = _active_updates(constant, declarations, regression=regression)
    constant, pruned = _prune_regressions(result, constant, replaced)
    changed_kinds = set(REGRESSION_KINDS) if pruned else set()
    return result, constant, active, replaced, changed_kinds


def _finish_parameter_update(prepared, result, constant, changed_kinds):
    """Publish the candidate layout, rebuilding only changed catalogue entries."""
    result["/constant_data"] = (
        finalize_parameters(constant, changed=changed_kinds) if changed_kinds else constant
    )
    return prepared._with_parameter_updates(result)


def _require_mutable(state):
    if any(group in state for group in ("posterior", "fixed_parameters", "estimates")):
        raise ValueError("Update priors and regressions before estimation or fixing parameters")


def _compact_regression_pools(constant, scope, priors, selection, variables):
    """Stage all retained references against one pool remap for this scope."""
    grid = constant[scope["coefficient_positions"]]
    pool_dim, term_dim = grid.dims
    targets = [(target, constant[target["indices"]].variable) for target in scope["targets"]]
    pools = np.unique(np.concatenate([array.values[1].ravel() for _, array in targets]))
    if len(pools) == grid.sizes[pool_dim]:
        return None
    priors = dict(priors)
    pool_index = np.full(grid.sizes[pool_dim], -1, dtype=np.int64)
    pool_index[pools] = np.arange(len(pools))
    selection[pool_dim] = pools
    shape = len(pools), grid.sizes[term_dim]
    variables[grid.name] = grid.dims, np.arange(np.prod(shape)).reshape(shape)
    for target, array in targets:
        rows, indices = array.values
        variables[target["indices"]] = array.dims, np.stack((rows, pool_index[indices]))
        target["unit_pools"] = tuple(
            (unit, int(pool_index[pool])) for unit, pool in target["unit_pools"]
        )
    for block in scope["blocks"]:
        width = int(np.prod(block["shape"][1:], dtype=int))
        block["shape"][0] = len(pools)
        # Both partitions reference the same flat block positions.
        flat_index = (pool_index[:, None] * width + np.arange(width)).ravel()
        for key in ("free_flat", "fixed_flat"):
            name = block[key]
            variable = constant[name]
            positions = flat_index[variable.values]
            keep = positions >= 0
            (dim,) = variable.dims
            variables[name] = dim, positions[keep]
            selection[dim] = np.flatnonzero(keep)
            if key != "free_flat":
                continue
            for role, members in block["prior_groups"].items():
                retained = np.take(keep, members)
                if retained.any() and not retained.all() and prior_shape(priors[role])[0]:
                    priors[role] = subset_prior(
                        priors[role], np.flatnonzero(retained), block["prior_shapes"][role]
                    )
    return priors


def _prune_regressions(result, constant, replaced):
    """Retain frozen encoders; remove only unreachable targets and coefficient/scale pools."""
    scopes, obsolete, closed_axes = [], set(), set()
    selection, variables, resized = {}, {}, []
    for scope in constant.attrs["regression_scopes"]:
        path = f"/{scope['path']}"
        targets = []
        for target in scope["targets"]:
            if (target["binding_id"], target["prior_role"]) not in replaced:
                targets.append(target)
                continue
            name = target["indices"]
            obsolete.add(name)
            if target["binding_id"] != -1:
                closed_axes.add(constant[name].dims[-1])
        if len(targets) == len(scope["targets"]):
            scopes.append(scope)
            continue
        if not targets:
            layouts = scope["blocks"]
            if scope["target"] == "theta":
                population = constant.attrs["population"]
                binding = layouts[0]["binding_id"]
                coefficient = next(b for b in population["blocks"] if b["binding_id"] == binding)
                population["blocks"].remove(coefficient)
                population["recipe"]["regression"] = None
                layouts = (coefficient,)
            # Block bookkeeping uses private free/fixed axes. The coefficient
            # grid owns the pool and term axes shared within this regression.
            closed_axes.update(constant[scope["coefficient_positions"]].dims)
            closed_axes.update(
                dim
                for layout in layouts
                for key in ("free_flat", "fixed_flat")
                for dim in constant[layout[key]].dims
            )
            del result[path]
            continue
        scope["targets"] = targets
        priors = _compact_regression_pools(
            constant, scope, result[path].attrs["resolved_priors"], selection, variables
        )
        if priors is not None:
            resized.append((path, scope["blocks"], priors))
        scopes.append(scope)
    if not obsolete:
        return constant, False
    constant = constant.drop_vars(obsolete).drop_dims(closed_axes)
    if resized:
        # Pool and block-local axes belong to individual scopes and can be selected together.
        constant = (
            constant.isel(selection)
            .assign_coords({dim: np.arange(len(indices)) for dim, indices in selection.items()})
            .assign(variables)
        )
        for path, blocks, priors in resized:
            resolved = {}
            for block in blocks:
                if (reference := block.get("prior_role_index")) and constant[reference].size:
                    block["prior_role"] = str(constant[reference].values[0])
                del block["prior_groups"]
                resolved.update(freeze_prior_shapes(constant, block, priors))
            result[path] = result[path].assign_attrs(resolved_priors=resolved)
    constant.attrs["regression_scopes"] = scopes
    if not scopes:
        result.pop("/covariates", None)
    return constant, True


def _remove_ordinary(selected):
    changed = set()
    for scope, _, layout, role in selected:
        if scope is not None:
            scope["specification"] = replace(
                scope["specification"],
                priors={
                    key: prior
                    for key, prior in scope["specification"].priors.items()
                    if key != role
                },
            )
            layout.get("prior_shapes", {}).pop(role, None)
            if role in {"location_positive", "location_negative"}:
                side = "positive" if role == "location_positive" else "negative"
                if not layout["orientation_strict"][side]:
                    changed.add(layout["kind"])
                    layout["orientation_strict"][side] = True
    return changed


def _measurement_regression_design(constant, target, spec, selected, feature_columns):
    """Align target entities and fit one common design before constructing prior pools."""
    key = selected[0][2]["key"]
    entities, defaults, targets, pool_rows, labels = _item_targets(constant, selected, key)
    columns = (spec.id,) if isinstance(spec.id, str) else spec.id
    if key == "cutpoints":
        spec = _canonical_pairs(spec, entities)
    elif len(columns) != 1:
        raise ValueError("Statement/item regression requires one ID column")
    frame = pd.DataFrame(
        {
            column: pd.Series([entity[j] for entity in entities], dtype=object)
            for j, column in enumerate(columns)
        }
    )
    design = fit_design(spec, frame, feature_columns=feature_columns)
    for pool, rows in enumerate(pool_rows):
        _rank_check(design.X.values[sorted(rows)], f"{target!r}, pool {pool}")
    return spec, design, defaults, targets, labels


def prepare_regressions(
    prepared: PreparedInputs,
    regressions: Sequence[tuple[str | tuple[str, str], Regression]] | None = None,
    *,
    marginal_correlation: bool | None = None,
    qualified_priors: Mapping | None = None,
    theta_alignment=None,
) -> PreparedInputs:
    """Build an atomic candidate using latest declaration order and existing layouts."""
    state = prepared.data
    if marginal_correlation is not None and not isinstance(marginal_correlation, bool):
        raise TypeError("marginal_correlation must be a bool or None")
    if regressions is None:
        declarations = []
    elif isinstance(regressions, Mapping):
        declarations = list(regressions.items())
    elif isinstance(regressions, Sequence) and not isinstance(regressions, str):
        declarations = list(regressions)
    else:
        raise TypeError("regressions must be an ordered sequence of declarations")
    current = state["constant_data"].to_dataset()
    changed = (
        marginal_correlation is not None
        and marginal_correlation != current.attrs["marginal_correlation"]
    )
    if not declarations and not changed:
        return prepared
    if declarations:
        _require_mutable(state)
    else:
        if any(group in state for group in ("posterior", "fixed_parameters", "estimates")):
            raise ValueError("marginal_correlation is fixed after estimation or fixing parameters")
        result = state.to_dict()
        result["/constant_data"] = current.assign_attrs(marginal_correlation=marginal_correlation)
        return prepared._with_parameter_updates(result)
    features_by_formula = {}
    for _, spec in declarations:
        if not isinstance(spec, Regression):
            raise TypeError("regressions values must be Regression declarations")
        if spec.formula not in features_by_formula:
            features_by_formula[spec.formula] = validate_formula(spec.formula)
    result, constant, active, replaced, changed_kinds = _parameter_update_candidate(
        state, declarations, regression=True
    )
    qualified_priors = dict(qualified_priors or {})
    slots_by_binding = defaultdict(list)
    bindings = {binding for binding, _ in replaced if binding != -1}
    if bindings:
        for measurement in constant.attrs["measurements"]:
            for slot in measurement["slots"]:
                binding = slot["binding_id"]
                if binding in bindings:
                    slots_by_binding[binding].append((measurement["unit_index"], slot["positions"]))
    if marginal_correlation is not None:
        constant.attrs["marginal_correlation"] = marginal_correlation
    next_id = constant.attrs["next_regression_id"]
    regression_offset = constant.attrs["population"]["parameter_set_index"] + 1
    changed_kinds.update(REGRESSION_KINDS)
    arrays = LayoutBuilder(coords={}, attrs={"next_binding_id": constant.attrs["next_binding_id"]})
    for target, spec, selected in active:
        name = f"r{next_id}"
        theta = target == "theta"
        feature_columns = features_by_formula[spec.formula]
        if theta:
            design, defaults, targets = _theta_design(
                constant, spec, feature_columns, aligned_rows=theta_alignment
            )
            labels = [("theta", "trait", -1)] * len(defaults)
            constant = constant.drop_dims("theta_predictor")
            constant.attrs["population"]["recipe"]["regression"] = name
        else:
            spec, design, defaults, targets, labels = _measurement_regression_design(
                constant, target, spec, selected, feature_columns
            )
            changed_kinds.update(_remove_ordinary(selected))
        layouts, positions = _coefficient_layout(
            constant,
            arrays,
            name,
            design,
            spec,
            defaults,
            labels,
            theta=theta,
            qualified_priors=qualified_priors.pop(target, None),
        )
        records = _store_targets(constant, arrays, name, targets, spec, slots_by_binding)
        constant.attrs["regression_scopes"].append(
            dict(
                name=name,
                target=target,
                regression_index=next_id,
                parameter_set_index=regression_offset + next_id,
                path=f"covariates/{name}",
                blocks=layouts,
                coefficient_positions=positions,
                targets=records,
            )
        )
        result[f"/covariates/{name}"] = design
        next_id += 1
    if qualified_priors:
        raise ValueError(
            f"Model-qualified priors have no active formula: {sorted(qualified_priors)}"
        )
    constant.attrs.update(next_regression_id=next_id, **arrays.attrs)
    constant = merge_prepared([constant, arrays])
    if (-1, "theta") in replaced:
        _check_population_directions(constant)
    return _finish_parameter_update(prepared, result, constant, changed_kinds)


def prepare_priors(prepared: PreparedInputs, priors):
    """Validate native replacements and cancel only selected conditional role targets."""
    state = prepared.data
    if not isinstance(priors, Mapping) or not priors:
        raise TypeError("priors must be a nonempty selector-to-native-prior mapping")
    _require_mutable(state)
    declarations = []
    for selector, prior in priors.items():
        target = tuple(selector.split(".")) if isinstance(selector, str) else ()
        declarations.append((target, prior))
    result, constant, active, _, changed_kinds = _parameter_update_candidate(state, declarations)
    pending = {}
    for _, prior, selected in active:
        for scope, _, layout, role in selected:
            entry = pending.setdefault(scope["name"], dict(scope=scope, blocks={}, priors={}))
            entry["priors"][role] = prior
            item = entry["blocks"].setdefault(layout["binding_id"], (layout, set()))
            item[1].add(role)
    for entry in pending.values():
        scope, native = entry["scope"], entry["priors"]
        spec = scope["specification"]
        records = {}
        for layout, roles in entry["blocks"].values():
            orientation = dict(layout.get("orientation_strict") or {})
            records.update(freeze_prior_shapes(constant, layout, native, only_roles=roles))
            if orientation != (layout.get("orientation_strict") or {}):
                changed_kinds.add(layout["kind"])
        declared = {**spec.priors, **records}
        check_asymptote_priors(scope["blocks"], declared)
        scope["specification"] = replace(spec, priors=declared)
    # Prior shapes and native bounds live in the block records. The published
    # entry catalogue changes only with regression ownership or orientation.
    return _finish_parameter_update(prepared, result, constant, changed_kinds)
