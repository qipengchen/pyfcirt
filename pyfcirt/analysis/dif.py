"""Posterior contrasts from frozen respondent designs and final measurement nodes."""

from collections.abc import Mapping
from functools import cache
from itertools import combinations
from numbers import Real

import numpy as np
import pandas as pd
from arviz_stats import eti

from pyfcirt.analysis.output import posterior_samples
from pyfcirt.covariates.design import apply_design
from pyfcirt.covariates.model import effective_coefficients, offset_parameter


def _profiles(plans, at, contrasts):
    features, specs = set(), {}
    by_name = {plan.name: plan for plan in plans}
    for plan in plans:
        attrs = plan.design.attrs
        records = attrs["feature_records"]
        features.update(record["name"] for record in records)
        if columns := attrs.get("group_columns"):
            local = [(":".join(columns), columns, attrs["group_levels"], None)]
        else:
            local = [
                (record["name"], (record["name"],), tuple(record["levels"]), record["center"])
                for record in records
            ]
        for spec in local:
            specs.setdefault(spec, set()).add(plan.name)
    if not isinstance(at, Mapping) or not isinstance(contrasts, Mapping):
        raise TypeError("at and contrasts must map predictor names to values")
    if unknown := (at.keys() - features) | (contrasts.keys() - {spec[0] for spec in specs}):
        raise ValueError(f"Unknown DIF predictors: {sorted(unknown)}")

    pairs = {}
    for (key, columns, levels, center), active_plans in specs.items():
        if contrasts and key not in contrasts:
            continue
        if key in contrasts:
            custom = contrasts[key]
            if not isinstance(custom, tuple) or len(custom) != 2:
                raise ValueError(f"{key}: contrast must be a (from, to) tuple")
            endpoints = [custom]
        elif levels:
            endpoints = combinations(levels, 2)
        else:
            center = at.get(key, center)
            endpoints = [(center, center + 1)]
        for low, high in endpoints:
            if low == high:
                raise ValueError(f"{key}: contrast endpoints must differ")
            if len(columns) > 1 and any(
                not isinstance(value, tuple) or len(value) != len(columns) for value in (low, high)
            ):
                raise ValueError(f"{key}: compare tuples in the declared group column order")
            pairs.setdefault((key, columns, low, high), set()).update(active_plans)

    @cache
    def encode(name, comparison):
        attrs = by_name[name].design.attrs
        center = {
            record["name"]: (
                record["levels"][0] if record["kind"] == "category" else record["center"]
            )
            for record in attrs["feature_records"]
        }
        if group_columns := attrs.get("group_columns"):
            center.update(zip(group_columns, attrs["group_levels"][0], strict=True))
        center.update(at)
        _, columns, low, high = comparison
        rows = pd.DataFrame(
            [
                {
                    **center,
                    **dict(zip(columns, (value,) if len(columns) == 1 else value, strict=True)),
                }
                for value in (low, high)
            ]
        )
        return apply_design(by_name[name].design, rows, np.arange(2), rows.index).values

    return pairs, encode


def _draw_reader(model):
    posterior = posterior_samples(model.idata)
    fixed = model._fixed_reference
    sample_shape = (posterior.sizes["chain"], posterior.sizes["draw"])
    cache = {}

    def read(kind, entries):
        if kind not in cache:
            layout = model.prepared.parameter_layouts[kind]
            if fixed is not None and kind in fixed:
                values = fixed[kind].transpose(*layout.dims)
            elif kind in posterior:
                values = posterior[kind].transpose("chain", "draw", *layout.dims)
            elif not layout.free_flat.size:
                values = layout.fixed.ravel()
            else:
                raise ValueError(f"Missing DIF posterior parameter {kind!r}")
            if not isinstance(values, np.ndarray):
                values = values.stack(parameter_entry=layout.dims, create_index=False)
            cache[kind] = values
        values = cache[kind]
        flattened = entries.ravel()
        shape = (int(np.prod(sample_shape)), *entries.shape)
        if isinstance(values, np.ndarray):
            return np.broadcast_to(values[entries], shape)
        selected = values.isel(parameter_entry=flattened).values
        if selected.ndim == 1:
            return np.broadcast_to(selected.reshape(entries.shape), shape)
        return selected.reshape(shape)

    return read


def _coefficients(plan, read):
    draws = read("regression_coefficient", plan.coefficient_entries)
    scale = next((block for block in plan.blocks if block.key == "residual_sd"), None)
    scale_draws = None if scale is None else read(scale.kind, scale.entry_index)
    return effective_coefficients(plan, draws, scale_draws, tensor=np)


def _entry_labels(prepared, kind, positions):
    from pyfcirt.analysis.summary import _entity_labels, _measurement_pair_labels

    constant = prepared.constant_data
    layout = prepared.parameter_layouts[kind]
    labels = layout.entry_data[
        ["unit_index", "statement_index", "pair_index", "trait_index", "cutpoint_position"]
    ].isel({next(iter(layout.entry_data.dims)): positions})
    parameter = labels.to_dataframe().reset_index()
    parameter["parameter_id"] = labels.coords[next(iter(labels.dims))].values
    units = prepared.parameter_units.loc[kind].groupby(level="entry_index")["unit_index"]
    unit_labels = {
        entry: tuple(constant.unit_id.values[indices.unique()]) for entry, indices in units
    }
    parameter["units"] = [unit_labels.get(entry, ()) for entry in positions]
    parameter["statement"] = _entity_labels(
        parameter.statement_index, constant.statement_label.values
    )
    parameter["pair"] = _measurement_pair_labels(
        constant, parameter.pair_index.to_numpy(dtype=int), parameter.unit_index.to_numpy(dtype=int)
    )
    parameter["trait"] = _entity_labels(parameter.trait_index, constant.trait.values)
    parameter["position"] = parameter.cutpoint_position.where(parameter.cutpoint_position >= 0) + 1
    return parameter[["parameter_id", "units", "statement", "pair", "trait", "position"]]


def summarize_dif(model, *, contrasts=None, at=None, prob=0.95, significant_only=False):
    """Summarize the to-minus-from draws on each measurement parameter's scale."""
    if isinstance(prob, bool) or not isinstance(prob, Real) or not 0 < prob < 1:
        raise ValueError("prob must be between zero and one")
    if not isinstance(significant_only, bool):
        raise TypeError("significant_only must be boolean")
    prepared = model.prepared
    plans = prepared.person_effect_regressions
    if not plans:
        raise ValueError("The model has no respondent-level DIF terms")
    pairs, encode = _profiles(
        plans, {} if at is None else at, {} if contrasts is None else contrasts
    )
    read = _draw_reader(model)
    coefficients = {plan.name: _coefficients(plan, read) for plan in plans}
    results = []
    for kind, bindings in prepared.person_effect_pools.items():
        entries = prepared.parameter_layouts[kind].fixed.size
        for comparison, active_plans in pairs.items():
            name, _, low, high = comparison
            affected = np.zeros(entries, dtype=bool)
            for plan, pools, _ in bindings:
                if plan.name in active_plans:
                    affected |= pools >= 0
            positions = np.flatnonzero(affected)
            if not positions.size:
                continue
            base = read(kind, positions)
            values = np.broadcast_to(base, (2, *base.shape)).copy()
            for plan, pools, link in bindings:
                active_pools = pools[positions]
                active = active_pools >= 0
                if active.any():
                    matrix = encode(plan.name, comparison)
                    offsets = np.einsum(
                        "ct,net->cne", matrix, coefficients[plan.name][:, active_pools[active]]
                    )
                    values[..., active] = offset_parameter(
                        values[..., active], offsets, link, tensor=np
                    )
            difference = values[1] - values[0]
            lower, upper = eti(difference, prob=prob, axis=0).T
            detected = (lower > 0) | (upper < 0)
            results.append(
                _entry_labels(prepared, kind, positions).assign(
                    parameter=kind,
                    predictor=name,
                    reference=[low] * len(positions),
                    compared=[high] * len(positions),
                    reference_mean=values[0].mean(axis=0),
                    compared_mean=values[1].mean(axis=0),
                    mean=difference.mean(axis=0),
                    lower=lower,
                    upper=upper,
                    detected=detected,
                )
            )
    summary = pd.concat(results, ignore_index=True)
    return summary.loc[summary.detected].reset_index(drop=True) if significant_only else summary
