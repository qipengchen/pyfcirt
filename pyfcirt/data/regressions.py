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

from pyfcirt.common.validation import align_rows, identifier
from pyfcirt.covariates.design import fit_design, pairwise_design
from pyfcirt.covariates.pairwise import PairwiseL1
from pyfcirt.covariates.specification import Regression, _measurement_level, validate_formula
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
    person_effect_defaults,
    regression_defaults,
)

if TYPE_CHECKING:
    from pyfcirt.data.prepare import PreparedInputs


def _measurement_scopes(constant):
    owners = {owner["name"]: owner for owner in constant.attrs["owners"]}
    yield from ((owner, owner) for owner in owners.values())
    for measurement in constant.attrs["measurements"]:
        yield measurement, owners[measurement["owner"]]


def _selector_roles(layout, selector):
    roles = layout["prior_groups"]
    if layout["key"] == selector:
        return roles
    if layout["orientation"] is not None and selector in layout["prior_roles"].values():
        return (selector,) if selector in roles else ()
    return None


def _regression_target_problem(constant, layout, roles, target, *, person_effects):
    if not (layout["regression"] or (person_effects and layout.get("person_link") is not None)):
        return ValueError(f"Unsupported parameter target: {target!r}")
    if layout.get("shared") and layout["prior_groups"]:
        return ValueError("1PL common loading does not support item-specific regressions")
    if (
        roles
        and layout["ordered"]
        and constant[layout["free_flat"]].size != np.prod(layout["shape"])
    ):
        return NotImplementedError("Regression requires complete free ordered cutpoint vectors")
    return None


def available_covariate_targets(prepared, *, level, kind):
    """List the selector and predictor table for each eligible prepared target."""
    if level not in (None, "person", "statement", "pair", "item", "measurement"):
        raise ValueError("level must be person, statement, pair, item, measurement or None")
    if kind not in (None, "regression", "dif"):
        raise ValueError("kind must be regression, dif or None")
    constant = prepared.constant_data
    candidates = defaultdict(lambda: defaultdict(list))
    for scope, owner in _measurement_scopes(constant):
        family = type(scope["specification"]).__name__
        for layout in scope["blocks"]:
            selectors = (layout["key"],)
            if layout["orientation"] is not None:
                selectors += tuple(layout["prior_roles"].values())
            for selector in selectors:
                roles = _selector_roles(layout, selector)
                candidates[owner["channel"], selector][family].append((layout, roles))

    columns = ["target", "model", "level", "kind", "predictor_level"]
    rows = []
    if level in (None, "person") and kind in (None, "regression"):
        rows.append(("theta", None, "person", "regression", "person"))
    for (channel, selector), families in candidates.items():
        for family, matches in families.items():
            layout = matches[0][0]
            axis = _measurement_level(channel, layout["key"]) if layout["dims"] else "measurement"
            if level is not None and level != axis:
                continue
            target = (channel, family, selector) if len(families) > 1 else (channel, selector)
            for usage in ("regression", "dif") if kind is None else (kind,):
                if not any(roles for _, roles in matches) or any(
                    _regression_target_problem(
                        constant, layout, roles, target, person_effects=usage == "dif"
                    )
                    for layout, roles in matches
                ):
                    continue
                label = target if usage == "regression" and len(target) == 3 else ".".join(target)
                rows.append((label, family, axis, usage, "person" if usage == "dif" else axis))
    return pd.DataFrame(rows, columns=columns)


def _active_updates(constant, declarations, *, regression=False, person_effects=False):
    """Resolve each selector once against the candidate; the last owner wins."""
    scopes = {"fc": [], "items": [], "population": [(constant.attrs["population"], None)]}
    if any(
        isinstance(target, tuple) and target and target[0] in ("fc", "items")
        for target, _ in declarations
    ):
        for scope, owner in _measurement_scopes(constant):
            scopes[owner["channel"]].append((scope, owner))

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
                roles = _selector_roles(layout, selector)
                if roles is None:
                    continue
                if regression and (
                    problem := _regression_target_problem(
                        constant, layout, roles, target, person_effects=person_effects
                    )
                ):
                    raise problem
                entries.extend((scope, owner, layout, str(role)) for role in roles)
        if not entries:
            raise ValueError(f"Parameter target {target!r} has no applicable free parameters")
        resolved[target] = entries

    claimed, active = set(), []
    for target, declaration in reversed(declarations):
        selected = []
        for entry in resolved[target]:
            layout, role = entry[2:]
            binding_id = -1 if layout is None else layout["binding_id"]
            binding_role = (binding_id, role)
            if person_effects:
                binding_role += (declaration.formula.strip(),)
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


def _item_targets(constant, selected, key, *, person_effects=False, shared=False):
    entities, pools = {}, {}
    defaults, targets, pool_rows, labels = [], [], [], []
    for scope, owner, layout, role in selected:
        spec = scope["specification"]
        conditional = (
            person_effect_defaults(spec, role, link=layout.get("person_link"))
            if person_effects
            else conditional_defaults(spec, role)
        )
        kind = layout["kind"]
        link = conditional["link"]
        scale = "log_loading" if kind == "loading" and link == "log" else link
        shape = tuple(layout["shape"])
        free = constant[layout["free_flat"]].values
        ordered = layout["ordered"]
        if not person_effects:
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
        coordinates = (
            zip(*np.unravel_index(positions, dimensions), strict=True)
            if dimensions
            else [()] * len(positions)
        )
        for index, position in enumerate(coordinates):
            trait = position[trait_axis] if trait_axis is not None else -1
            if person_effects:
                signature = (
                    (kind, role, link) if shared else (layout["binding_id"], positions[index])
                )
                row = 0
            else:
                signature = owner["name"], key, role, trait, count
                row = entities.setdefault(local_keys[position[0]], len(entities))
            if signature not in pools:
                pools[signature] = len(pools)
                defaults.append((scope["specification"], role))
                pool_rows.append(set())
                labels.append((kind, scale, trait))
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


def _resolve_coefficients(
    spec, design, pool_defaults, qualified_priors=None, *, person_effects=False
):
    qualified_priors = qualified_priors or {}
    if unknown := qualified_priors.keys() - {type(model).__name__ for model, _ in pool_defaults}:
        raise ValueError(f"No applicable free parameter targets for models: {sorted(unknown)}")
    terms = design.term.values
    for family, overrides in qualified_priors.items():
        if unknown := overrides.keys() - {*terms, "common"}:
            raise ValueError(f"{family}: priors for unknown design columns: {sorted(unknown)}")
    native, roles, standardized, registered = [], [], [], {}

    def register(prior, *, term=None, family=None):
        # A non-scalar model override describes only that model's pools.
        scope = family if family is not None and any(prior_shape(prior)) else None
        key = term, scope, PriorPreparation.key(prior)
        if key not in registered:
            registered[key] = f"prior_{len(native)}"
            native.append(prior)
        return registered[key]

    for model, role in pool_defaults:
        intercept, _, slope_sigma = (
            (None, None, 1.0) if person_effects else regression_defaults(model, role)
        )
        row, automatic = [], []
        family = type(model).__name__
        by_model = qualified_priors.get(family, {})
        for term in terms:
            fallback = None if term == "Intercept" else "common"
            prior = by_model.get(term, by_model.get(fallback))
            model_override = prior is not None
            if prior is None:
                prior = spec.priors.get(term, spec.priors.get(fallback))
            automatic.append(person_effects and prior is None)
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
        standardized.append(automatic)
    return native, np.asarray(roles, dtype=object), register, np.asarray(standardized)


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
    person_effects=False,
    joint_prior=None,
):
    terms = list(design.term.values)
    pool_dim, term_dim = f"{name}_pool", f"{name}_term"
    coordinates = {pool_dim: np.arange(len(pool_defaults), dtype=np.int64), term_dim: terms}
    if joint_prior is None:
        native, roles, register, standardized = _resolve_coefficients(
            spec, design, pool_defaults, qualified_priors, person_effects=person_effects
        )
    else:
        native = [joint_prior]
        roles = np.full((len(pool_defaults), len(terms)), "prior_0", dtype=object)
        standardized = np.zeros(roles.shape, dtype=bool)
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
            fixed_for_scoring=person_effects,
            kind="regression_coefficient",
            component="regression",
        )
        positions = np.arange(roles.size, dtype=np.int64).reshape(roles.shape)
        flat_roles = roles.ravel()
        records = [layout]
    layout["noncentered"] = spec.coefficient_noncentered
    if joint_prior is not None:
        layout["joint_per_pool"] = True
    if spec.coefficient_noncentered and "Intercept" in terms:
        layout["centered_roles"] = tuple(dict.fromkeys(roles[:, terms.index("Intercept")]))
    role_variable = f"{name}_coefficient_prior_role"
    arrays[role_variable] = arrays[layout["free_flat"]].dims, flat_roles
    layout["prior_role_index"] = role_variable
    position_variable = f"{name}_coefficient_positions"
    arrays[position_variable] = (pool_dim, term_dim), positions
    if person_effects:
        arrays[f"{name}_standardized"] = (pool_dim, term_dim), standardized
    actual_layouts = [layout]
    if not theta and (not person_effects or standardized.any()):
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
                prior = (
                    PriorPreparation.default(pm.HalfNormal.dist, sigma=0.5)
                    if person_effects
                    else regression_defaults(model, role)[1]
                )
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
            fixed_for_scoring=person_effects,
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
        actual_layouts.append(residual)
    priors = {f"prior_{index}": prior for index, prior in enumerate(native)}
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


def _align_people(constant, spec):
    columns = (spec.id,) if isinstance(spec.id, str) else spec.id
    if len(columns) != 1:
        raise ValueError("Person-level regression requires a single person ID column")
    keys = pd.DataFrame({columns[0]: pd.Series(constant.person.values, dtype=object)})
    return align_rows(spec.data, keys, columns, f"Regression {spec.formula.strip()!r}")


def _person_design(spec, feature_columns, *, aligned_rows, reference_rows=None, group_columns=()):
    """Fit a person-level formula against prepared respondent rows."""
    return fit_design(
        spec,
        None,
        reference_rows=reference_rows,
        include_intercept=False,
        group_columns=group_columns,
        aligned_rows=aligned_rows,
        feature_columns=feature_columns,
    )


def _theta_design(constant, spec, feature_columns, *, aligned_rows=None):
    if spec.parameter_residual_sd is not None or spec.deviation_noncentered is not None:
        raise ValueError("Theta regression uses the existing population scale and parameterization")
    population = constant.attrs["population"]
    recipe = population["recipe"]
    reference = None
    if recipe["groups"]:
        reference = constant.person_group.values == recipe["groups"].index(
            recipe["reference_group"]
        )
    design = _person_design(
        spec,
        feature_columns,
        reference_rows=reference,
        aligned_rows=_align_people(constant, spec) if aligned_rows is None else aligned_rows,
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


def _copy_parameter_groups(state):
    """Detach the published layout metadata before changing ownership."""
    result = state.to_dict()
    constant = result["/constant_data"]
    constant.attrs = deepcopy(constant.attrs)
    return result


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
            key = (target["binding_id"], target["prior_role"])
            if scope.get("person_effects", False):
                key += (result[path].attrs["formula"],)
            if key not in replaced:
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
    person_effects: bool = False,
    dif_options: Mapping | None = None,
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
    result = _copy_parameter_groups(state)
    constant = result["/constant_data"]
    active, replaced = _resolve_regressions(constant, declarations, person_effects=person_effects)
    constant, _ = _prune_regressions(result, constant, replaced)
    if marginal_correlation is not None:
        constant.attrs["marginal_correlation"] = marginal_correlation
    constant, changed_kinds = _bind_regressions(
        constant,
        result,
        active,
        replaced,
        qualified_priors=qualified_priors,
        person_effects=person_effects,
        dif_options=dif_options,
    )
    return _finish_parameter_update(prepared, result, constant, changed_kinds)


def _resolve_regressions(constant, declarations, *, person_effects=False):
    """Check every formula's syntax, then resolve the effective target declarations."""
    features_by_formula = {}
    for target, spec in declarations:
        if person_effects and target == "theta":
            raise ValueError("Person measurement effects require a measurement target")
        if not isinstance(spec, Regression):
            raise TypeError("regressions values must be Regression declarations")
        if spec.formula not in features_by_formula:
            features_by_formula[spec.formula] = validate_formula(spec.formula)
    active, replaced = _active_updates(
        constant, declarations, regression=True, person_effects=person_effects
    )
    return [
        (target, spec, selected, features_by_formula[spec.formula])
        for target, spec, selected in active
    ], replaced


def _check_person_effect_designs(constant, groups, changed):
    """Check the combined design once for each distinct set of overlapping formulas."""
    affected = {entry[:2] for entry in changed}
    designs = defaultdict(list)
    for scope in constant.attrs["regression_scopes"]:
        if not scope.get("person_effects", False):
            continue
        path = f"/{scope['path']}"
        for target in scope["targets"]:
            key = target["binding_id"], target["prior_role"]
            if key in affected:
                designs[key].append(path)
    for paths in dict.fromkeys(tuple(paths) for paths in designs.values()):
        matrices = [groups[path].X.values for path in paths]
        labels = ", ".join(groups[path].attrs["formula"] for path in paths)
        _rank_check(np.column_stack((np.ones(len(matrices[0])), *matrices)), labels)


def _person_effect_design(constant, spec, selected, features, aligned_rows, options, overrides):
    """Prepare one respondent encoder and the coefficient law for its target pools."""
    design = _person_design(
        spec, features, aligned_rows=aligned_rows, group_columns=options.get("group_columns", ())
    )
    if options.get("categorical") and any(
        record["kind"] != "category" for record in design.attrs["feature_records"]
    ):
        raise ValueError("DIF groups require categorical columns; cast numeric group labels")
    joint_prior = None
    if (pairwise := options.get("pairwise")) is not None:
        if (
            spec.priors
            or overrides
            or spec.parameter_residual_sd is not None
            or spec.coefficient_noncentered
        ):
            raise ValueError("PairwiseL1 supplies the entire joint coefficient prior")
        design = pairwise_design(design)
        joint_prior = PairwiseL1.dist(len(design.attrs["group_basis"]), **pairwise)
    _, defaults, targets, _, labels = _item_targets(
        constant,
        selected,
        selected[0][2]["key"],
        person_effects=True,
        shared=options.get("shared", False),
    )
    return design, defaults, targets, labels, joint_prior


def _bind_regressions(
    constant,
    groups,
    active,
    replaced,
    *,
    qualified_priors=None,
    theta_alignment=None,
    person_effects=False,
    dif_options=None,
):
    """Fit encoders and bind the resolved targets on a candidate layout."""
    qualified_priors = dict(qualified_priors or {})
    slots_by_binding = defaultdict(list)
    bindings = {key[0] for key in replaced if key[0] != -1}
    if bindings:
        for measurement in constant.attrs["measurements"]:
            for slot in measurement["slots"]:
                binding = slot["binding_id"]
                if binding in bindings:
                    slots_by_binding[binding].append((measurement["unit_index"], slot["positions"]))
    next_id = constant.attrs["next_regression_id"]
    regression_offset = constant.attrs["population"]["parameter_set_index"] + 1
    changed_kinds = set(REGRESSION_KINDS)
    arrays = LayoutBuilder(coords={}, attrs={"next_binding_id": constant.attrs["next_binding_id"]})
    person_rows = {}
    for target, spec, selected, feature_columns in active:
        name = f"r{next_id}"
        theta = target == "theta"
        joint_prior = None
        overrides = qualified_priors.pop(target, None)
        if theta:
            design, defaults, targets = _theta_design(
                constant, spec, feature_columns, aligned_rows=theta_alignment
            )
            labels = [("theta", "trait", -1)] * len(defaults)
            constant = constant.drop_dims("theta_predictor")
            constant.attrs["population"]["recipe"]["regression"] = name
        elif person_effects:
            key = id(spec.data), spec.id
            if key not in person_rows:
                person_rows[key] = _align_people(constant, spec)
            options = (dif_options or {}).get((target, spec.formula.strip()), {})
            design, defaults, targets, labels, joint_prior = _person_effect_design(
                constant, spec, selected, feature_columns, person_rows[key], options, overrides
            )
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
            qualified_priors=overrides,
            person_effects=person_effects,
            joint_prior=joint_prior,
        )
        records = _store_targets(constant, arrays, name, targets, spec, slots_by_binding)
        constant.attrs["regression_scopes"].append(
            dict(
                name=name,
                target=target,
                person_effects=person_effects,
                regression_index=next_id,
                parameter_set_index=regression_offset + next_id,
                path=f"covariates/{name}",
                blocks=layouts,
                coefficient_positions=positions,
                targets=records,
            )
        )
        groups[f"/covariates/{name}"] = design
        next_id += 1
    if qualified_priors:
        raise ValueError(
            f"Model-qualified priors have no active formula: {sorted(qualified_priors)}"
        )
    constant.attrs.update(next_regression_id=next_id, **arrays.attrs)
    constant = merge_prepared([constant, arrays])
    if (-1, "theta") in replaced:
        _check_population_directions(constant)
    if person_effects:
        _check_person_effect_designs(constant, groups, replaced)
    return constant, changed_kinds


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
    result = _copy_parameter_groups(state)
    constant = result["/constant_data"]
    active, replaced = _active_updates(constant, declarations)
    constant, pruned = _prune_regressions(result, constant, replaced)
    changed_kinds = set(REGRESSION_KINDS) if pruned else set()
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
