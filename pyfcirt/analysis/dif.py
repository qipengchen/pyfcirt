"""Posterior contrasts from frozen respondent designs and final measurement nodes."""

from collections.abc import Mapping
from functools import cache
from itertools import combinations
from numbers import Integral, Real

import numpy as np
import pandas as pd
import xarray as xr
from arviz_stats import bayes_factor as arviz_bayes_factor
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


def _contrast_offsets(matrix, coefficients):
    return np.einsum("ct,net->cne", matrix, coefficients)


def _check_bf_plan(prepared, plan, matrix):
    """Check that the contrast frees one independent coefficient under the null."""
    import pymc as pm

    from pyfcirt.covariates.pairwise import PairwiseL1RV
    from pyfcirt.covariates.spike_slab import SpikeAndSlabRV
    from pyfcirt.models.priors import describe_prior

    fitted = plan.design
    records = fitted.attrs["feature_records"]
    levels = fitted.attrs.get("group_levels")
    if levels is None and len(records) == 1 and records[0]["kind"] == "category":
        levels = records[0]["levels"]
    models = prepared.constant_data.unit_model.values
    if (
        levels is None
        or len(levels) != 2
        or matrix.shape != (2, 1)
        or matrix[0, 0] == matrix[1, 0]
        or any(models[index] != "TIRT" for target in plan.targets for index in target.unit_indices)
        or np.any(plan.standardized_coefficients)
    ):
        raise ValueError("Bayes factors require one fixed-scale, two-group TIRT coefficient")
    inclusion = set()
    for prior in plan.priors.values():
        op = prior.owner.op
        if isinstance(op, SpikeAndSlabRV):
            _, args = describe_prior(prior)
            probability, _ = (value.item() for _, value in args)
            inclusion.add(probability)
            continue
        inclusion.add(None)
        if isinstance(op, PairwiseL1RV):
            continue
        if not isinstance(op, (pm.Normal.rv_type, pm.Laplace.rv_type)):
            raise ValueError("Bayes factors require a supported continuous or spike-and-slab prior")
        _, args = describe_prior(prior)
        center, scale = (value for _, value in args)
        if center.ndim or scale.ndim or center != 0 or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Bayes factors require zero-centered, fixed-scale coefficient priors")
    if len(inclusion) != 1:
        raise ValueError("Each target requires a common prior inclusion probability")
    return inclusion.pop()


def _spike_evidence(effects, inclusion_prob):
    """Compute evidence from paired draws of an exactly zero or continuous effect."""
    draws = effects.shape[0]
    included = np.count_nonzero(effects, axis=0)
    omitted = draws - included
    status = np.full(effects.shape[1], "ok", dtype=object)
    status[np.minimum(included, omitted) < 20] = "few_indicator_draws"
    visited = (included > 0) & (omitted > 0)
    status[~visited] = "indicator_state_unobserved"
    bf10 = np.divide(
        included * (1 - inclusion_prob),
        omitted * inclusion_prob,
        out=np.full(effects.shape[1], np.nan),
        where=visited,
    )
    return {
        "BF10": bf10,
        "BF01": 1 / bf10,
        "bf_status": status,
        "inclusion_probability": included / draws,
    }


def _bf_fields(posterior, prior):
    """Compute evidence one parameter at a time so invalid KDEs stay local."""
    count = posterior.shape[1]
    evidence = np.full((count, 2), np.nan)
    status = np.full(count, None, dtype=object)
    for index, (post, before) in enumerate(zip(posterior.T, prior.T, strict=True)):
        if any(not np.isfinite(values).all() or np.ptp(values) == 0 for values in (post, before)):
            status[index] = "invalid_density"
            continue
        # The ArviZ KDE grid does not estimate the density beyond the draws.
        if any(not values.min() < 0 < values.max() for values in (post, before)):
            continue
        few_draws = any(
            min(np.count_nonzero(values < 0), np.count_nonzero(values > 0)) < 20
            for values in (post, before)
        )
        data = xr.DataTree.from_dict(
            {
                "/posterior": xr.Dataset({"effect": (("chain", "draw"), post[None])}),
                "/prior": xr.Dataset({"effect": (("chain", "draw"), before[None])}),
            }
        )
        try:
            result = arviz_bayes_factor(data, "effect")
        except (ValueError, FloatingPointError, ZeroDivisionError):
            status[index] = "invalid_density"
            continue
        if isinstance(result, dict):  # ArviZ Stats 1.0 returned a mapping.
            result = np.asarray([result["effect"][kind] for kind in ("BF10", "BF01")])
        else:
            result = result["effect"].sel(bf_type=["BF10", "BF01"]).to_numpy()
        if np.isfinite(result).all() and (result > 0).all():
            evidence[index] = result
            status[index] = "few_draws_near_zero" if few_draws else "ok"
        else:
            status[index] = "invalid_density"
    return {"BF10": evidence[:, 0], "BF01": evidence[:, 1], "bf_status": status}


def _bf_prior_reader(model, pairs, *, draws, random_seed, assume_same_prior):
    if len(pairs) != 1:
        raise ValueError("Select one categorical contrast to compute Bayes factors")
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws < 40:
        raise ValueError("prior_draws must be an integer of at least 40")
    if model._prior_conditioning() != "full prior" or model._fixed_reference is not None:
        raise ValueError("Bayes factors require the original calibration fit")
    identity = model.idata.attrs.get("dif_prior_identity")
    if identity is None:
        if not assume_same_prior:
            raise ValueError(
                "The posterior has no recorded DIF prior identity; verify the fitted priors "
                "and use assume_same_prior=True"
            )
    elif identity != model.prepared.dif_prior_identity:
        raise ValueError("The fitted DIF coefficient priors or bindings differ from the posterior")

    @cache
    def read():
        import pymc as pm

        coefficient = model.build()["regression_coefficient"]
        return pm.draw(coefficient, draws=int(draws), random_seed=random_seed).reshape(draws, -1)

    return read


def summarize_dif(
    model,
    *,
    contrasts=None,
    at=None,
    prob=0.95,
    significant_only=False,
    bayes_factor=False,
    prior_draws=4000,
    random_seed=None,
    assume_same_prior=False,
):
    """Summarize the to-minus-from draws on each measurement parameter's scale."""
    if isinstance(prob, bool) or not isinstance(prob, Real) or not 0 < prob < 1:
        raise ValueError("prob must be between zero and one")
    if not isinstance(significant_only, bool):
        raise TypeError("significant_only must be boolean")
    if not isinstance(bayes_factor, bool) or not isinstance(assume_same_prior, bool):
        raise TypeError("bayes_factor and assume_same_prior must be boolean")
    prepared = model.prepared
    plans = prepared.person_effect_regressions
    if not plans:
        raise ValueError("The model has no respondent-level DIF terms")
    pairs, encode = _profiles(
        plans, {} if at is None else at, {} if contrasts is None else contrasts
    )
    prior_reader = (
        _bf_prior_reader(
            model,
            pairs,
            draws=prior_draws,
            random_seed=random_seed,
            assume_same_prior=assume_same_prior,
        )
        if bayes_factor
        else None
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
            tested = []
            for plan, pools, link in bindings:
                active_pools = pools[positions]
                active = active_pools >= 0
                if active.any():
                    matrix = encode(plan.name, comparison)
                    selected = active_pools[active]
                    offsets = _contrast_offsets(matrix, coefficients[plan.name][:, selected])
                    if prior_reader is not None and plan.name in active_plans:
                        tested.append((plan, matrix, selected, link, offsets))
                    values[..., active] = offset_parameter(
                        values[..., active], offsets, link, tensor=np
                    )
            difference = values[1] - values[0]
            lower, upper = eti(difference, prob=prob, axis=0).T
            detected = (lower > 0) | (upper < 0)
            evidence = {}
            if prior_reader is not None:
                if len(tested) != 1 or len(tested[0][2]) != len(positions):
                    raise ValueError("Bayes factors require one free DIF coefficient per target")
                plan, matrix, selected, link, offsets = tested[0]
                inclusion_prob = _check_bf_plan(prepared, plan, matrix)
                contrast_draws = offsets[1] - offsets[0]
                if inclusion_prob is None:
                    prior = _contrast_offsets(
                        matrix, prior_reader()[:, plan.coefficient_entries[selected]]
                    )
                    evidence = _bf_fields(contrast_draws, prior[1] - prior[0])
                else:
                    evidence = _spike_evidence(contrast_draws, inclusion_prob)
                evidence["bf_scale"] = "log_loading" if link == "log" else "cutpoint"
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
                    **evidence,
                )
            )
    summary = pd.concat(results, ignore_index=True)
    return summary.loc[summary.detected].reset_index(drop=True) if significant_only else summary
