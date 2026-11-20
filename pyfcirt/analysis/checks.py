"""Response discrepancies from the existing probability graph and actual replicas."""

from __future__ import annotations

from warnings import warn

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from pyfcirt.analysis.output import posterior_samples, prepare_output
from pyfcirt.analysis.prediction import (
    _SOURCE_COORDS,
    _native_forward,
    _sample_responses,
    _seed,
    _take_states,
)


def _prediction_states(model, predictions):
    """Match saved prediction states by operation IDs and their real sample labels."""
    posterior = posterior_samples(model.idata, mcmc=True)
    replicas = predictions["posterior_predictive"].to_dataset()
    run = model.idata.attrs.get("posterior_run_id")
    if run is None:
        raise ValueError("The current posterior must declare posterior_run_id")
    if replicas.attrs.get("posterior_run_id") is None:
        raise ValueError("Predictions must declare posterior_run_id")
    if replicas.attrs["posterior_run_id"] != run:
        raise ValueError("Predictions belong to a different posterior operation")
    present = [name in replicas.coords for name in _SOURCE_COORDS]
    positions = None
    if any(present):
        if not all(present) or replicas.sizes["chain"] != 1:
            raise ValueError(
                "Selected predictions require paired source_chain/source_draw coordinates"
            )
        if any(replicas[name].dims != ("draw",) for name in _SOURCE_COORDS):
            raise ValueError("Predictive source labels must use the draw dimension")
        chain = posterior.get_index("chain").get_indexer(replicas.source_chain.values)
        draw = posterior.get_index("draw").get_indexer(replicas.source_draw.values)
        if ((chain < 0) | (draw < 0)).any():
            raise ValueError("Prediction source labels do not match the posterior")
        positions = chain * posterior.sizes["draw"] + draw
        if np.unique(positions).size != positions.size:
            raise ValueError("Predictions repeat a source posterior state")
    elif any(
        not replicas.get_index(axis).equals(posterior.get_index(axis)) for axis in ("chain", "draw")
    ):
        raise ValueError("Prediction sample labels do not match the posterior")
    probabilities = (
        predictions["response_probabilities"].to_dataset()
        if "response_probabilities" in predictions
        else None
    )
    if probabilities is not None:
        for field in ("posterior_run_id", "prediction_run_id"):
            if replicas.attrs.get(field) is None or probabilities.attrs.get(field) is None:
                raise ValueError(f"Predictions and response probabilities must declare {field}")
            if probabilities.attrs[field] != replicas.attrs[field]:
                raise ValueError(f"Response probabilities have a different {field}")
        for name in ("chain", "draw", *_SOURCE_COORDS):
            if (name in replicas.coords) != (name in probabilities.coords) or (
                name in replicas.coords and not replicas[name].equals(probabilities[name])
            ):
                raise ValueError("Response probabilities and replicas have different sample labels")
    return posterior, replicas, probabilities, positions


def _event_values(values, observed, replicated):
    """Extract the actual and replicated categories at each state in one gather."""
    values = jnp.asarray(values)
    observed = jnp.broadcast_to(jnp.asarray(observed), values.shape[:-1])
    events = jnp.stack((observed, jnp.asarray(replicated)), axis=-1)
    return jnp.take_along_axis(values, events, axis=-1)


def _pearson_terms(log_probability):
    """Compute log(1/p_y - 1) for the two selected events."""
    terms = jnp.log(-jnp.expm1(log_probability)) - log_probability
    if bool((jnp.isnan(terms) | jnp.isposinf(terms)).any()):
        raise FloatingPointError(
            "An actual event has zero or invalid model probability; Pearson terms cannot be "
            "represented even in log space. Probabilities were not floored."
        )
    return jnp.moveaxis(terms, -1, 0)


def _aggregate_pearson(log_terms, group_index, n_groups):
    """Aggregate paired event discrepancies and compare them in log space."""
    group_index = jnp.asarray(group_index)
    values = jnp.moveaxis(jnp.asarray(log_terms), -1, 0)
    maximum = jax.ops.segment_max(values, group_index, num_segments=n_groups)
    center = jnp.where(jnp.isfinite(maximum), maximum, 0)
    total = jax.ops.segment_sum(
        jnp.exp(values - center[group_index]), group_index, num_segments=n_groups
    )
    observed, replicated = jnp.moveaxis(center + jnp.log(total), 0, -1)
    count = jnp.bincount(group_index, length=n_groups)
    indicator = replicated >= observed
    high, low = jnp.maximum(replicated, observed), jnp.minimum(replicated, observed)
    difference = jnp.sign(replicated - observed) * jnp.exp(high + jnp.log(-jnp.expm1(low - high)))
    difference = jnp.where(observed == replicated, 0.0, difference)
    raw_observed, raw_replicated = jnp.exp(observed), jnp.exp(replicated)
    return observed, replicated, raw_observed, raw_replicated, difference, indicator, count


def _grouping(constant, groupby, selected):
    """Group selected real events using the existing frozen coordinate relations."""

    if groupby == "overall":
        return np.zeros(len(selected), np.int64), 1, None, {}
    axis = "unit" if groupby == "measurement" else groupby
    source = {
        "person": "obs_person",
        "unit": "obs_unit",
        "block": "obs_block_index",
        "item": "obs_item_index",
        "pair": "obs_pair",
    }
    if axis not in source:
        raise ValueError("groupby must be overall, person, block, item, measurement or pair")
    positions, groups = np.unique(constant[source[axis]].values[selected], return_inverse=True)
    if positions[0] < 0:
        if axis == "pair":
            raise ValueError("groupby='pair' requires prepared pairs for every selected event")
        channel = "fc" if axis == "block" else "items"
        raise ValueError(f"groupby={groupby!r} requires an explicit {channel} selection")
    coords = {axis: constant[axis].values[positions]}
    if axis == "unit":
        coords.update(
            unit_id=("unit", constant.unit_id.values[positions]),
            unit_channel=("unit", constant.unit_channel.values[positions]),
        )
    if axis == "pair":
        coords["unit"] = "pair", constant.pair_unit.values[positions]
        labels = constant.statement_label.values
        for side in ("statement_1", "statement_2"):
            coords[side] = "pair", labels[constant[f"pair_{side}"].values[positions]]
    return groups, len(positions), axis, coords


def chi_ppp(
    model,
    *,
    predictions=None,
    blocks=None,
    items=None,
    groupby="overall",
    draws=None,
    random_seed=None,
    save_discrepancy=False,
    store=True,
    progressbar=True,
) -> xr.Dataset:
    """Compute a Pearson posterior predictive p-value at matched MCMC states.

    Observed and replicated discrepancies use the same categorical probability
    vector. This is a Bayesian model check, not a classical chi-square test.
    """
    from pyfcirt.data.coordinates import select_units

    prepared = model.prepared
    constant = prepared.constant_data
    units = select_units(constant, blocks, items)
    selected_units = set(units)
    batches = [batch for batch in prepared.batches if batch.unit_index in selected_units]
    beta_units = {
        constant.unit_id.values[batch.unit_index]
        for batch in batches
        if batch.likelihood_kind == "beta"
    }
    if beta_units:
        raise NotImplementedError(
            f"Categorical Pearson checks do not apply to Beta units {list(beta_units)!r}; "
            "select categorical blocks/items explicitly and use continuous PPC for Beta"
        )
    if any(batch.y is None for batch in batches):
        raise ValueError("Pearson checks require actual observed responses")
    if not batches:
        raise ValueError("Pearson checks require selected actual observed events")
    selected = np.concatenate(
        [np.arange(batch.obs_slice.start, batch.obs_slice.stop) for batch in batches]
    )
    if not isinstance(save_discrepancy, bool) or not isinstance(store, bool):
        raise TypeError("save_discrepancy and store must be boolean")
    grouping, count, group_dim, group_coords = _grouping(constant, groupby, selected)
    if predictions is not None and draws is not None:
        raise ValueError("draws cannot select states again when predictions are supplied")
    if predictions is None and draws is None and "posterior_predictive" in model.idata:
        predictions = model.idata
    fresh = predictions is None
    if fresh:
        seed = _seed(random_seed)
        posterior = posterior_samples(model.idata, mcmc=True)
        replicas = _sample_responses(
            model,
            posterior,
            batches,
            [batch.log_probability_name for batch in batches],
            draws=draws,
            seed=seed,
            progressbar=progressbar,
        )
        if replicas.attrs.get("posterior_run_id") is None:
            raise ValueError("The current posterior must declare posterior_run_id")
        probabilities = None
    else:
        if not isinstance(predictions, xr.DataTree) or "posterior_predictive" not in predictions:
            raise TypeError("predictions must contain actual posterior_predictive responses")
        posterior, replicas, probabilities, positions = _prediction_states(model, predictions)
        replicas = prepare_output(prepared, replicas, "posterior_predictive")
        if probabilities is not None:
            probabilities = prepare_output(prepared, probabilities, "response_probabilities")
    terms, missing = [None] * len(batches), {}
    for index, batch in enumerate(batches):
        replicated = replicas[batch.response_name]
        if replicated.dims != ("chain", "draw", batch.obs_dim):
            raise ValueError(f"{batch.name}: replicated coordinates do not match actual events")
        codes = replicated.values
        if codes.dtype.kind not in "iu" or np.any(
            (codes < 0) | (codes >= len(batch.events.labels))
        ):
            raise ValueError(f"{batch.name}: invalid replicated category")
        dims = ("chain", "draw", batch.obs_dim, batch.category_dim)
        (name,) = batch.distribution_names
        if probabilities is not None and name in probabilities:
            p = jnp.asarray(probabilities[name].transpose(*dims).values)
            if not bool((jnp.isfinite(p) & (p >= 0) & (p <= 1)).all()):
                raise FloatingPointError("Response probabilities must be finite and within [0,1]")
            selected_p = _event_values(p, batch.y, codes)
            if bool((selected_p > 0).all()):
                terms[index] = _pearson_terms(jnp.log(selected_p))
                continue
        missing[batch.log_probability_name] = index, batch, codes
    if missing:
        evaluated = replicas
        if not fresh:
            if positions is not None:
                posterior = _take_states(
                    posterior, positions, chains=replicas.chain.values, draws=replicas.draw.values
                )
            evaluated = _native_forward(
                model._bindings, posterior, missing, progressbar=progressbar
            )
        for name, (index, batch, codes) in missing.items():
            log_p = evaluated[name].transpose("chain", "draw", batch.obs_dim, batch.category_dim)
            terms[index] = _pearson_terms(_event_values(log_p.values, batch.y, codes))
    log_terms = jnp.concatenate(terms, axis=-1)
    values = tuple(np.asarray(value) for value in _aggregate_pearson(log_terms, grouping, count))
    log_actual, log_replica, raw_actual, raw_replica, difference, indicator, n_obs = values
    coords = {"chain": replicas.chain, "draw": replicas.draw, **group_coords}
    dims = ("chain", "draw") if group_dim is None else ("chain", "draw", group_dim)
    if group_dim is None:
        log_actual, log_replica, raw_actual, raw_replica, difference, indicator = (
            value[..., 0] for value in values[:6]
        )
        n_obs = xr.DataArray(int(n_obs[0]))
    else:
        n_obs = xr.DataArray(n_obs, dims=group_dim, coords={group_dim: group_coords[group_dim]})
    result = xr.Dataset(
        {
            "log_D_obs": (dims, log_actual),
            "log_D_rep": (dims, log_replica),
            "D_obs": (dims, raw_actual),
            "D_rep": (dims, raw_replica),
            "D_diff": (dims, difference),
        },
        coords=coords,
    )
    indicator = xr.DataArray(indicator.astype(float), dims=dims, coords=coords)
    result["PPP"] = indicator.mean(("chain", "draw")).where(n_obs > 0)
    result["n_obs"] = n_obs
    for name in _SOURCE_COORDS:
        if name in replicas.coords:
            result = result.assign_coords({name: replicas[name]})
    ordered = not any(name in replicas.coords for name in _SOURCE_COORDS)
    if ordered and replicas.sizes["draw"] >= 4:
        from arviz_stats import mcse

        result["PPP_mcse"] = mcse(indicator, method="mean", sample_dims=["chain", "draw"]).where(
            (indicator.var(("chain", "draw")) > 0) & (n_obs > 0)
        )
        mcse_method = "ArviZ MCSE of ordered MCMC replication indicators"
    else:
        result["PPP_mcse"] = xr.full_like(result.PPP, np.nan)
        mcse_method = "unavailable for selected states or fewer than four ordered draws"
    overflow = not all(
        bool(jnp.isfinite(value).all()) for value in (raw_actual, raw_replica, difference)
    )
    if overflow:
        warn(
            "Raw Pearson discrepancy exceeds floating-point range; log_D and PPP remain available.",
            RuntimeWarning,
            stacklevel=2,
        )
    result.attrs = {
        "check": "chi_ppp",
        "statistic": "Pearson categorical discrepancy",
        **{
            key: replicas.attrs[key]
            for key in ("posterior_run_id", "prediction_run_id")
            if key in replicas.attrs
        },
        "groupby": groupby,
        "selected_units": [int(unit) for unit in units],
        "raw_discrepancy_overflow": int(overflow),
        "PPP_mcse_method": mcse_method,
        "conditioning": "same posterior state",
    }
    if store:
        groups = {"posterior_predictive_checks": result}
        if save_discrepancy:
            groups["discrepancy"] = xr.Dataset(
                {
                    name: (("chain", "draw", "obs"), value)
                    for name, value in zip(
                        ("log_D_obs", "log_D_rep"), np.asarray(log_terms), strict=True
                    )
                },
                coords={
                    "chain": replicas.chain,
                    "draw": replicas.draw,
                    "obs": constant.obs.values[selected],
                    **{name: replicas[name] for name in _SOURCE_COORDS if name in replicas.coords},
                },
                attrs=result.attrs,
            )
        model._publish_outputs(groups, remove=("posterior_predictive_checks", "discrepancy"))
    return result


def residuals(model, *, progressbar=True) -> xr.Dataset:
    """Return observed-minus-expected response residuals at actual posterior states."""
    posterior = posterior_samples(model.idata, mcmc=True)
    prepared = model.prepared
    if any(batch.y is None for batch in prepared.batches):
        raise ValueError("Residuals require actual observed responses")
    names = [name for batch in prepared.batches for name in batch.distribution_names]
    evaluated = _native_forward(model._bindings, posterior, names, progressbar=progressbar)
    result = {}
    for batch in prepared.batches:
        observation_coords = {batch.obs_dim: prepared.constant_data[batch.obs_dim]}
        observed = xr.DataArray(batch.y, dims=batch.obs_dim, coords=observation_coords)
        if batch.likelihood_kind == "beta":
            alpha, beta = (evaluated[name] for name in batch.distribution_names)
            value = observed - alpha / (alpha + beta)
        else:
            (name,) = batch.distribution_names
            probability = evaluated[name]
            labels = batch.events.labels
            numeric = all(
                isinstance(label, (int, float, np.integer, np.floating))
                and not isinstance(label, (bool, np.bool_))
                for label in labels
            )
            if numeric:
                scores = xr.DataArray(
                    np.asarray(labels, float),
                    dims=batch.category_dim,
                    coords={batch.category_dim: probability[batch.category_dim]},
                )
                actual = xr.DataArray(
                    np.asarray(labels)[batch.y],
                    dims=batch.obs_dim,
                    coords=observation_coords,
                )
                value = actual - (probability * scores).sum(batch.category_dim)
            else:
                value = (observed == probability[batch.category_dim]).astype(float) - probability
        result[f"{batch.name}_residual"] = value
    return xr.Dataset(
        result,
        attrs={
            "definition": "observed minus conditional expectation",
            "unordered_support": "one-hot minus category probability",
        },
    )
