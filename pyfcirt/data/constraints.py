"""Resolve final free/fixed parameter layouts at the data boundary only."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, repeat
from warnings import warn

import numpy as np
from scipy.cluster.hierarchy import DisjointSet

from pyfcirt.data.arrays import LayoutBuilder
from pyfcirt.data.coordinates import block_constraint, freeze_prior_groups
from pyfcirt.models.priors import (
    prior_shape,
    prior_support,
    require_proper,
    same_prior,
    validate_prior,
)
from pyfcirt.models.specification import (
    IdentificationWarning,
    prior_orientation,
    resolve_spec,
)


def _layout(
    ds,
    scope,
    key,
    dims,
    role,
    *,
    fixed=None,
    orientation=None,
    ordered=False,
    regression=False,
    fixed_for_scoring=True,
    sources=None,
    domain="real",
    kind,
    component="measurement",
    summary_default=True,
):
    binding_id = ds.attrs.get("next_binding_id", 0)
    ds.attrs["next_binding_id"] = binding_id + 1
    shape = tuple(ds.sizes[dim] for dim in dims)
    fixed = np.full(shape, np.nan) if fixed is None else np.asarray(fixed, dtype=float)
    flat = fixed.ravel()
    free, bound = np.flatnonzero(np.isnan(flat)), np.flatnonzero(~np.isnan(flat))
    prefix = f"{scope}_{key}"
    free_dim, fixed_dim = f"{prefix}_free", f"{prefix}_fixed"
    ds.add_coords(
        {
            free_dim: np.arange(len(free), dtype=np.int64),
            fixed_dim: np.arange(len(bound), dtype=np.int64),
        }
    )
    result = dict(
        key=key,
        dims=list(dims),
        shape=list(shape),
        prior_role=role,
        free_flat=f"{prefix}_free_flat",
        fixed_flat=f"{prefix}_fixed_flat",
        fixed=f"{prefix}_fixed_template",
        binding_id=binding_id,
        kind=kind,
        component=component,
        summary_default=summary_default,
        fixed_for_scoring=fixed_for_scoring,
        orientation=None,
        orientation_strict=None,
        ordered=ordered,
        regression=regression,
        domain=domain,
        bounds=None,
    )
    ds[result["free_flat"]] = free_dim, free.astype(np.int64, copy=False)
    ds[result["fixed_flat"]] = fixed_dim, bound.astype(np.int64, copy=False)
    ds[result["fixed"]] = dims, fixed
    if orientation is not None:
        result["orientation"] = f"{prefix}_orientation"
        result["orientation_strict"] = dict(positive=False, negative=False)
        result["prior_roles"] = dict(
            positive="location_positive", negative="location_negative", free="location_free"
        )
        sides = np.asarray(orientation, dtype=np.int64)
        ds[result["orientation"]] = dims, sides
        result["prior_role_index"] = reference = f"binding_{binding_id}_prior_role_index"
        side_roles = np.array(
            [result["prior_roles"][side] for side in ("negative", "free", "positive")],
            dtype=object,
        )
        ds[reference] = free_dim, side_roles[np.sign(sides.ravel()[free]) + 1]
    result["fixed_source"] = f"{prefix}_fixed_source"
    ds[result["fixed_source"]] = (
        fixed_dim,
        np.full(len(bound), "identification", dtype="U16")
        if sources is None
        else np.asarray(sources).ravel()[bound],
    )
    return result


def _declared_layouts(ds, name, declarations, axes, overrides=None, *, reference_group=None):
    """Resolve model-owned construction facts onto the actual prepared coordinates."""
    layouts = []
    for declaration in declarations:
        dims = tuple(axes.get(axis, axis) for axis in declaration.axes)
        options = dict(
            ordered=declaration.ordered and ds.sizes[dims[-1]] > 1,
            fixed_for_scoring=declaration.fixed_for_scoring,
            regression=declaration.regression,
            domain=declaration.domain,
            kind=declaration.kind or declaration.key,
            component=declaration.component,
            summary_default=declaration.summary_default,
        )
        if not np.isnan(declaration.fixed) or declaration.reference is not None:
            fixed = np.full(tuple(ds.sizes[dim] for dim in dims), declaration.fixed)
            if declaration.reference == "first":
                fixed.flat[0] = declaration.reference_value
            elif declaration.reference == "diagonal":
                diagonal = np.arange(fixed.shape[-1])
                fixed[..., diagonal, diagonal] = declaration.reference_value
            elif declaration.reference == "group":
                fixed[reference_group] = declaration.reference_value
            elif declaration.reference == "group_diagonal":
                diagonal = np.arange(fixed.shape[-1])
                fixed[reference_group, diagonal, diagonal] = declaration.reference_value
            options["fixed"] = fixed
            options["sources"] = np.full(fixed.shape, declaration.fixed_source, dtype="U16")
        options.update((overrides or {}).get(declaration.key, {}))
        role = options.pop(
            "role", None if declaration.derived else declaration.role or declaration.key
        )
        layout = _layout(ds, name, declaration.key, dims, role, **options)
        if declaration.person_link is not None:
            layout["person_link"] = declaration.person_link
        if declaration.shared:
            layout["shared"] = True
        layouts.append(layout)
    nodes = {layout["key"]: layout["binding_id"] for layout in layouts}
    for declaration, layout in zip(declarations, layouts, strict=True):
        if declaration.bounds is not None:
            lower, upper = declaration.bounds
            layout["bounds"] = (nodes[lower], upper)
    return layouts


def freeze_prior_shapes(ds, layout, priors, *, only_roles=None):
    """Validate free-block shapes and retain their native templates before construction."""
    shapes = {} if only_roles is None else dict(layout["prior_shapes"])
    records = {}
    layout_shape = tuple(layout["shape"])
    for role, positions in freeze_prior_groups(ds, layout).items():
        if only_roles is not None and role not in only_roles:
            continue
        if role not in priors:
            continue  # An active conditional binding owns this role instead.
        prior = priors[role]
        batch, event = prior_shape(prior)
        count = len(positions)
        if layout["domain"] == "correlation":
            shape = layout_shape
        elif layout.get("shared"):
            shape = ()
        elif layout["ordered"]:
            if event:
                raise ValueError(
                    f"{layout['binding_id']}: a joint ordered prior needs a user-owned pm.Model; "
                    "the iid permutation factor does not normalize a joint event"
                )
            cutpoints = layout_shape[-1]
            shape = (count // cutpoints,)
        elif layout.get("joint_per_pool"):
            width = layout_shape[-1]
            if event != (width,) or count % width:
                raise ValueError(f"{role}: a joint DIF prior must cover each full coefficient pool")
            shape = (count // width, width)
        elif event:
            if int(np.prod(event)) != count:
                raise ValueError(
                    f"{layout['binding_id']} {role}: replace the whole joint free block; "
                    "fixed entries or partial event components cannot be discarded"
                )
            shape = event
        elif batch == layout_shape and count == int(np.prod(layout_shape)):
            shape = layout_shape
        else:
            shape = (count,)
        validate_prior(prior, shape=shape, role=str(role), domain=layout["domain"])
        records[str(role)] = prior
        strict = prior_orientation(prior, role=str(role))
        if role in {"response_precision", "block_effect_scale"}:
            require_proper(prior, role=str(role))
        shapes[str(role)] = list(shape)
        if layout.get("orientation") is not None and role in {
            "location_positive",
            "location_negative",
        }:
            layout["orientation_strict"][str(role).removeprefix("location_")] = strict
    layout["prior_shapes"] = shapes
    return records


def check_asymptote_priors(layouts, priors):
    """Reject native prior pairs that can leave an empty conditional interval."""
    roles = {layout["binding_id"]: layout["prior_role"] for layout in layouts}
    for layout in layouts:
        if layout["bounds"] is None or layout["prior_role"] not in priors:
            continue
        lower, _ = layout["bounds"]
        lower_role, upper_role = roles[lower], layout["prior_role"]
        upper_prior = priors[upper_role]
        if upper_prior.owner.op.ndim_supp:
            raise ValueError(f"A joint {upper_role} law requires a user-owned pm.Model")
        if lower_role in priors:
            upper_c = prior_support(priors[lower_role])[1]
            upper_d = prior_support(upper_prior)[1]
            if np.any(upper_c > upper_d):
                raise ValueError(
                    f"{lower_role} support can exceed the {upper_role} support; "
                    "the upper conditional law would have empty support for some lower draws"
                )


def _components(rows, positions):
    """List components in statement order, with ordered members."""
    connected = DisjointSet(positions)
    for row in rows:
        first, *rest = row.tolist()
        for position in rest:
            connected.merge(first, position)
    groups = {}
    for position in connected:
        groups.setdefault(connected[position], []).append(position)
    return tuple(np.asarray(group, dtype=np.int64) for group in groups.values())


def _references(name, labels, components, specs, fixed, sources):
    explicit = tuple(
        dict.fromkeys(
            label for spec in specs for label in getattr(spec, "reference_statements", ())
        )
    )
    unknown = [label for label in explicit if label not in labels]
    if unknown:
        raise ValueError(f"Unknown reference_statements for {name}: {unknown!r}")
    for group in components:
        declared = [index for index in group if labels[index] in explicit]
        if len(declared) > 1:
            raise ValueError("A connected position group accepts only one reference statement")
        reference = int(declared[0] if declared else group[0])
        fixed[reference] = 0.0
        sources[reference] = "user_reference" if declared else "reference"


def _merge_anchors(labels, declared, specs):
    anchors = dict(zip(labels[~np.isnan(declared)], declared[~np.isnan(declared)], strict=True))
    for spec in specs:
        for label, value in getattr(spec, "loading_anchors", {}).items():
            if label not in labels:
                raise ValueError(f"Unknown loading anchor statement {label!r}")
            if label in anchors and anchors[label] != value:
                raise ValueError(f"Statement {label!r}: conflicting loading anchors")
            anchors[label] = value
    if any(not np.isfinite(value) or value < 0 for value in anchors.values()):
        raise ValueError("Loading anchors must be finite nonnegative magnitudes")
    return anchors


def _slot(layout, positions, axis=None):
    return dict(
        binding_id=layout["binding_id"],
        key=layout["key"],
        kind=layout["kind"],
        positions=positions,
        axis=axis,
    )


def _fc_owner(ds, owner, measurements):
    name = owner["name"]
    indices = ds[owner["statement_indices"]].values
    labels = ds.statement_label.values[indices]
    local = {int(index): position for position, index in enumerate(indices)}
    unit_indices = owner["units"]
    specs = [measurements[index]["specification"] for index in unit_indices]
    model = type(specs[0]).__name__
    loading_policy = specs[0].loading_identification
    declarations = {p.key: p for p in specs[0].parameter_declarations() if p.scope == "owner"}
    overrides = {}
    dim = f"{name}_statement"
    units = [ds.attrs["units"][index] for index in unit_indices]
    membership, block_indices = ds.fc_statement.values, ds.unit_block.values[unit_indices]
    rows = [
        np.fromiter(
            map(local.__getitem__, membership[block, : unit["size"]]),
            dtype=np.int64,
            count=unit["size"],
        )
        for unit, block in zip(units, block_indices, strict=True)
    ]
    components = _components(rows, range(len(indices)))
    component_index = np.empty(len(indices), dtype=np.int64)
    for index, component in enumerate(components):
        component_index[component] = index
    ds[f"{name}_component"] = dim, component_index
    owner["component"] = f"{name}_component"
    loading = ds.statement_fix_loading.values[indices].copy()
    loading_sources = np.where(np.isnan(loading), "", "design").astype("U16")
    if np.any(~np.isnan(loading) & (~np.isfinite(loading) | (loading < 0))):
        raise ValueError("fix_loading must be finite and nonnegative")
    if loading_policy in {"automatic", "required"}:
        anchors = _merge_anchors(labels, loading, specs)
        for label, value in anchors.items():
            position = int(np.flatnonzero(labels == label)[0])
            loading[position], loading_sources[position] = value, "user"
        traits = ds.statement_trait.values[indices]
        for component in components:
            if len(set(traits[component])) == 1 and np.isnan(loading[component]).all():
                if loading_policy == "required":
                    raise ValueError("A same-trait TIRTRank component requires a loading anchor")
                loading[component[0]] = declarations["loading_magnitude"].reference_value
                loading_sources[component[0]] = "automatic"
        active_traits = set(traits)
        if (
            len(active_traits) == 2
            and all(len(row) == 2 and len(set(traits[row])) == 2 for row in rows)
            and not any(np.isfinite(loading[row]).all() for row in rows)
        ):
            raise ValueError(
                f"Two-trait {model} pairs require both loadings of a reference pair. "
                "Set loading_anchors={statement_id: magnitude, ...} on the model "
                "declaration, or set design fix_loading for both reference statements."
            )
    overrides["loading_magnitude"] = dict(fixed=loading, sources=loading_sources)
    offset = "threshold" if "threshold" in declarations else "intercept"
    if offset in declarations:
        declaration = declarations[offset]
        role, graded_role = declaration.role or declaration.key, declaration.graded_role
        fixed, sources = np.full(len(indices), np.nan), np.full(len(indices), "", dtype="U16")
        ranking_rows = [
            row
            for unit, row in zip(units, rows, strict=True)
            if unit["response_format"] in {"rank", "pick", "mole"}
        ]
        if ranking_rows:
            ranking_components = (
                components
                if len(ranking_rows) == len(rows)
                else _components(ranking_rows, np.unique(np.concatenate(ranking_rows)).tolist())
            )
            _references(name, labels, ranking_components, specs, fixed, sources)
        elif graded_role is not None:
            role = graded_role
        overrides[offset] = dict(role=role, fixed=fixed, sources=sources)
        if loading_policy == "linear":
            traits = ds.statement_trait.values[indices]
            pinned = np.asarray([np.isfinite(loading[group]).any() for group in components])
            for unit, row in zip(units, rows, strict=True):
                group = component_index[row[0]]
                pinned[group] |= len(set(traits[row])) > 1 or (
                    unit["scale"]["kind"] == "category"
                    and unit["response_format"] is None
                    and len(unit["scale"]["categories"]) >= 3
                )
            if not pinned.all():
                raise ValueError(
                    "A single-trait linear ranking component requires a loading anchor"
                )
    if "residual_variance" in declarations:
        declaration = declarations["residual_variance"]
        reference_value = declaration.reference_value
        residual = ds.statement_fix_var.values[indices].copy()
        sources = np.where(np.isnan(residual), "", "design").astype("U16")
        if np.any(~np.isnan(residual) & (~np.isfinite(residual) | (residual <= 0))):
            raise ValueError("fix_var must be finite and positive")
        if np.isfinite(declaration.fixed):
            empty = np.isnan(residual)
            residual[empty], sources[empty] = declaration.fixed, declaration.fixed_source
        else:
            for row in rows:
                if len(row) == 2:
                    empty = row[np.isnan(residual[row])]
                    residual[empty], sources[empty] = reference_value, "pair_scale"
            for component in components:
                if np.isnan(residual[component]).all():
                    residual[component[0]], sources[component[0]] = (
                        reference_value,
                        "reference_scale",
                    )
        overrides["residual_variance"] = dict(fixed=residual, sources=sources)
    elif np.any(np.isfinite(ds.statement_fix_var.values[indices])):
        raise ValueError("fix_var applies only to Gaussian TIRT and TIRTRank residuals")
    if "location" in declarations:
        sides = ds.statement_location_side.values[indices]
        if not np.isin(sides, (-1, 0, 1)).all():
            raise ValueError(f"{model} requires location_side -1, 0 or +1 for every statement")
        overrides["location"] = dict(orientation=sides)
    layouts = _declared_layouts(ds, name, declarations.values(), {"statement": dim}, overrides)
    if offset in declarations and graded_role is not None and ranking_rows:
        layout = next(layout for layout in layouts if layout["key"] == offset)
        graded_rows = [
            row
            for unit, row in zip(units, rows, strict=True)
            if unit["scale"]["kind"] == "category" and unit["response_format"] is None
        ]
        if graded_rows:
            ranking = {int(index) for row in ranking_rows for index in row}
            graded = {int(index) for row in graded_rows for index in row}
            free = ds[layout["free_flat"]].values
            roles = np.array([role if index in ranking else graded_role for index in free])
            reference = f"{name}_intercept_prior_role_index"
            ds[reference] = ds[layout["free_flat"]].dims, roles
            layout["prior_role_index"] = reference
            if any(index in ranking & graded for index in free):
                owner["_overlap_roles"] = (role, graded_role)
    owner["blocks"] = layouts
    for index, row in zip(unit_indices, rows, strict=True):
        measurement = measurements[index]
        position_dim = f"{measurement['name']}_position"
        ds.add_coords({position_dim: np.arange(len(row), dtype=np.int64)})
        reference = f"{measurement['name']}_statement_positions"
        ds[reference] = position_dim, row
        measurement["slots"] = [_slot(layout, reference, "position") for layout in layouts]


def _item_owner(ds, owner, measurements):
    name = owner["name"]
    indices = ds[owner["item_indices"]].values
    dim = f"{name}_item"
    spec = owner["specification"]
    declarations = tuple(p for p in spec.parameter_declarations() if p.scope == "owner")
    q = ds.item_q.values[indices]
    loading = ds.item_fix_loading.values[indices].copy()
    sources = np.where(np.isnan(loading), "", "design").astype("U16")
    if np.any(~np.isnan(loading[q]) & (~np.isfinite(loading[q]) | (loading[q] < 0))):
        raise ValueError("fix_loading must be finite and nonnegative on active item/trait rows")
    loading[~q] = 0.0
    sources[~q] = "inactive"
    shared_loading = next(p.shared for p in declarations if p.key == "loading_magnitude")
    if shared_loading:
        declared = loading[q & np.isfinite(loading)]
        if len(declared):
            if np.any(declared <= 0):
                raise ValueError("1PL common fix_loading must be positive")
            if np.any(declared != declared[0]):
                raise ValueError("1PL requires one common fix_loading value")
            sources[q & np.isnan(loading)] = "shared"
            loading[q] = declared[0]
    overrides = {"loading_magnitude": dict(fixed=loading, sources=sources)}
    if any(p.key == "location" for p in declarations):
        sides = ds.item_location_side.values[indices]
        if not np.isin(sides[q], (-1, 0, 1)).all():
            raise ValueError("GGUM requires location_side -1, 0 or +1 on every active Q row")
        sides = np.where(q, sides, 0)
        overrides["location"] = dict(
            fixed=np.where(q, np.nan, 0.0),
            orientation=sides,
            sources=np.where(q, "", "inactive"),
        )
    layouts = _declared_layouts(ds, name, declarations, {"item": dim}, overrides)
    owner["blocks"] = layouts
    for item, index in enumerate(owner["units"]):
        measurement = measurements[index]
        prefix = measurement["name"]
        scalar = f"{prefix}_item_positions"
        matrix = f"{prefix}_item_trait_positions"
        ds[scalar] = (), np.int64(item)
        ds[matrix] = (
            "trait",
            item * ds.sizes["trait"] + np.arange(ds.sizes["trait"], dtype=np.int64),
        )
        measurement["slots"] = [
            _slot(layout, matrix, "trait_position")
            if len(layout["dims"]) == 2
            else _slot(layout, scalar)
            for layout in layouts
        ]


def _measurement(ds, measurement, pair_rows):
    name = measurement["name"]
    index = measurement["unit_index"]
    unit = ds.attrs["units"][index]
    scale = unit["scale"]
    spec = measurement["specification"]
    declarations = tuple(
        p for p in spec.parameter_declarations(block_size=unit["size"]) if p.scope == "measurement"
    )
    axes = {}
    if spec.comparison_based:
        first, second = np.triu_indices(unit["size"], k=1)
        pair_dim, cut_dim = f"{name}_pair", f"{name}_cutpoint"
        count = len(scale["categories"]) - 1 if scale["kind"] == "category" else 1
        n_pairs, offset = len(first), len(pair_rows)
        ds.add_coords(
            {
                pair_dim: np.arange(n_pairs, dtype=np.int64),
                cut_dim: np.arange(count, dtype=np.int64),
            }
        )
        block = ds.unit_block.values[index]
        members = ds.fc_statement.values[block]
        pair_rows.extend(
            zip(repeat(index, n_pairs), first, second, members[first], members[second], strict=True)
        )
        measurement["pair_indices"] = f"{name}_pair_index"
        ds[measurement["pair_indices"]] = (
            pair_dim,
            np.arange(offset, offset + n_pairs, dtype=np.int64),
        )
        axes.update(pair=pair_dim, cutpoint=cut_dim)
    if any("category" in p.axes for p in declarations):
        axes["category"] = measurement["category_dim"] = f"{name}_category"
        ds.add_coords({axes["category"]: np.asarray(scale["categories"], dtype=object)})
    layouts = _declared_layouts(ds, name, declarations, axes)
    measurement["blocks"] = layouts
    for declaration, layout in zip(declarations, layouts, strict=True):
        positions = f"{name}_{layout['key']}_positions"
        dims, shape = tuple(layout["dims"]), tuple(layout["shape"])
        ds[positions] = dims, np.arange(np.prod(shape, dtype=int), dtype=np.int64).reshape(shape)
        axis = declaration.axes[-1] if dims else None
        measurement["slots"].append(_slot(layout, positions, axis))


def _roles(ds, layouts):
    return frozenset(role for layout in layouts for role in freeze_prior_groups(ds, layout))


def _population(ds):
    record = ds.attrs["population"]
    spec, recipe = record["specification"], record["recipe"]
    reference = recipe["groups"].index(recipe["reference_group"]) if recipe["groups"] else 0
    layouts = _declared_layouts(
        ds,
        "population",
        spec.parameter_declarations(grouped=bool(recipe["groups"])),
        {},
        reference_group=reference,
    )
    record["blocks"] = layouts
    resolved = resolve_spec(spec, active_roles=_roles(ds, layouts), n=ds.sizes["trait"])
    record["specification"] = resolved
    for layout in layouts:
        freeze_prior_shapes(ds, layout, resolved.priors)


def _check_population_directions(ds):
    """Reject exact axis/pair mean directions invisible to every measurement law."""
    recipe = ds.attrs["population"]["recipe"]
    if recipe["regression"] is None and len(recipe["groups"]) <= 1:
        return
    size = ds.sizes["trait"]
    touched = np.zeros(size, dtype=bool)
    common_loadings, anchored = [], []
    for owner in ds.attrs["owners"]:
        if owner["channel"] == "items":
            q = ds.item_q.values[ds[owner["item_indices"]].values]
            layout = next(block for block in owner["blocks"] if block["key"] == "loading_magnitude")
            if layout.get("shared"):
                common_loadings.append(q.astype(np.int64))
            else:
                touched |= q.any(axis=0)
            continue
        indices = ds[owner["statement_indices"]].values
        traits = ds.statement_trait.values[indices]
        layout = next(block for block in owner["blocks"] if block["key"] == "loading_magnitude")
        if owner["specification"].loading_identification in {"automatic", "required"} and not len(
            ds[layout["free_flat"]]
        ):
            # A common shift within a connected component cancels in every block.
            groups = ds[owner["component"]].values
            _, first = np.unique(groups, return_index=True)
            weights = block_constraint(ds, layout) * ds.statement_keying.values[indices]
            anchored.append((weights, traits, first[groups]))
        else:
            touched[traits] = True
    axes = np.eye(size, dtype=np.int64)
    candidates = list(axes)
    for first, second in combinations(range(size), 2):
        candidates.extend((axes[first] - axes[second], axes[first] + axes[second]))
    for candidate in candidates:
        if np.any(candidate[touched]) or any(np.any(q @ candidate) for q in common_loadings):
            continue
        for weights, traits, reference in anchored:
            coefficients = weights * candidate[traits]
            if np.any(coefficients != coefficients[reference]):
                break
        else:
            direction = " ".join(
                f"{int(value):+d}*{trait!r}"
                for trait, value in zip(ds.trait.values, candidate, strict=True)
                if value
            )
            raise ValueError(
                f"Population mean direction {direction} is exactly likelihood-blind; "
                "ability regression or nonreference means are not identified"
            )


def _resolve_measurement_priors(ds, owners, measurements):
    """Resolve priors from active roles, then check sharing and each concrete shape."""
    scopes = (*owners, *measurements)
    active_roles = {scope["name"]: _roles(ds, scope["blocks"]) for scope in scopes}
    requests, owner_roles = {}, {}
    for owner in owners:
        overlap_roles = owner.pop("_overlap_roles", ())
        shared_roles = active_roles[owner["name"]] | frozenset(overlap_roles)
        owner_roles[owner["name"]] = overlap_roles, shared_roles
        for index in owner["units"]:
            measurement = measurements[index]
            spec = measurement["specification"]
            if id(spec) not in requests:
                requests[id(spec)] = spec, set()
            requests[id(spec)][1].update(shared_roles | active_roles[measurement["name"]])
    resolved_specs = {
        key: resolve_spec(spec, active_roles=frozenset(roles))
        for key, (spec, roles) in requests.items()
    }
    for owner in owners:
        overlap_roles, shared_roles = owner_roles[owner["name"]]
        resolved = [
            resolved_specs[id(measurements[index]["specification"])] for index in owner["units"]
        ]
        for index, spec in zip(owner["units"], resolved, strict=True):
            measurements[index]["specification"] = spec
        for role in shared_roles:
            first = resolved[0].priors[role]
            if any(not same_prior(spec.priors[role], first) for spec in resolved[1:]):
                raise ValueError(f"Shared owner {owner['name']} has incompatible {role} priors")
        if overlap_roles and not same_prior(
            resolved[0].priors[overlap_roles[0]], resolved[0].priors[overlap_roles[1]]
        ):
            raise ValueError("Shared ranking/graded statements require matching intercept priors")
        owner["specification"] = resolved[0]
        for layout in owner["blocks"]:
            if layout["orientation"] is not None:
                sides = ds[layout["orientation"]].values
                if owner["channel"] == "fc":
                    traits = ds.statement_trait.values[ds[owner["statement_indices"]].values]
                    missing = set(traits) - set(traits[sides != 0])
                else:
                    q = ds.item_q.values[ds[owner["item_indices"]].values]
                    missing = set(np.flatnonzero(q.any(axis=0))) - set(np.nonzero(sides)[1])
                if missing:
                    warn(
                        "Some unfolding traits have no signed location anchors; orientation "
                        "depends on other constraints and priors.",
                        IdentificationWarning,
                        stacklevel=3,
                    )
    for scope in scopes:
        resolved = scope["specification"]
        roles = active_roles[scope["name"]]
        if roles != resolved.priors.keys():
            scope["specification"] = replace(
                resolved,
                priors={role: prior for role, prior in resolved.priors.items() if role in roles},
            )
        for layout in scope["blocks"]:
            freeze_prior_shapes(ds, layout, resolved.priors)
        check_asymptote_priors(scope["blocks"], resolved.priors)


def freeze_layout(constant_data: LayoutBuilder) -> LayoutBuilder:
    """Select identification once and retain native declarations and index arrays."""
    ds = constant_data
    owners, measurements = ds.attrs["owners"], ds.attrs["measurements"]
    for owner in owners:
        if owner["channel"] == "fc":
            _fc_owner(ds, owner, measurements)
        else:
            _item_owner(ds, owner, measurements)
    pair_rows = []
    for measurement in measurements:
        _measurement(ds, measurement, pair_rows)
    pairs = np.asarray(pair_rows, dtype=np.int64).reshape(-1, 5)
    ds.add_coords({"pair": np.arange(len(pairs), dtype=np.int64)})
    ds["pair_positions"] = ("pair", "pair_member"), pairs[:, 1:3]
    ds["pair_unit"] = "pair", pairs[:, 0]
    ds["pair_statement_1"] = "pair", pairs[:, 3]
    ds["pair_statement_2"] = "pair", pairs[:, 4]
    _resolve_measurement_priors(ds, owners, measurements)
    _population(ds)
    for owner in owners:
        del owner["units"]
    _check_population_directions(ds)
    ds.attrs["next_regression_id"] = 0
    return ds
