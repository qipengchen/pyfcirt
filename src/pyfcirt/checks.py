"""Categorical Pearson posterior-predictive discrepancies and inclusive PPP."""

from __future__ import annotations

import warnings

import numpy as np
import pytensor.tensor as pt
import xarray as xr


def _log_pearson_terms(selected_logp):
    """log(1 / p_y - 1), equivalent to the normalized categorical Pearson sum."""
    return pt.log(-pt.expm1(selected_logp)) - selected_logp


def pearson_check(
    result,
    *,
    groupby: str = "overall",
    random_seed: int | None = None,
    progressbar: bool = True,
) -> xr.Dataset:
    """Run the tutorial's categorical Pearson check without saving all probabilities.

    Per-draw Pearson discrepancies are summed over actual person-block events;
    PPP is mean(D_rep >= D_obs). Probabilities condition on each posterior draw.
    PICK/MOLE compare their marginal response categories, not imputed full orders.
    Numerical comparisons use log discrepancies, without a probability floor.
    """
    import pymc as pm
    from pymc.model.fgraph import clone_model

    from .predictive import _idata_options, _posterior_parameters

    if groupby not in {"overall", "block"}:
        raise ValueError("groupby must be 'overall' or 'block'")
    built = result.built
    if not built.spec.observation_map:
        raise ValueError("Pearson checks require at least one effective response")
    posterior = _posterior_parameters(result)
    frozen = list(posterior.data_vars)
    checking = clone_model(built.model)
    log_observed, log_replicated, counts = {}, {}, {}
    generated = []
    with checking:
        checking.add_coord("block", np.asarray(built.inputs.block_ids, dtype=object))
        for component, group in zip(built.spec.components, built.inputs.groups, strict=True):
            categories = np.arange(len(group.events))
            probabilities = pm.logp(checking[component["observed_variable"]], categories[:, None]).T
            name = f"check_rep_{group.name}"
            replica = pm.Categorical(name, logit_p=probabilities, dims=f"{group.name}_obs")
            generated.append(name)
            rows = pt.arange(len(group.response))
            observed_terms = _log_pearson_terms(probabilities[rows, group.response])
            replicated_terms = _log_pearson_terms(probabilities[rows, replica])
            for index, block in enumerate(group.block_ids):
                selected = group.block_index == index
                counts[block] = int(selected.sum())
                log_observed[block] = (
                    pt.logsumexp(observed_terms[selected])
                    if selected.any()
                    else pt.constant(-np.inf)
                )
                log_replicated[block] = (
                    pt.logsumexp(replicated_terms[selected])
                    if selected.any()
                    else pt.constant(-np.inf)
                )
        pm.Deterministic(
            "check_log_D_obs",
            pt.stack([log_observed[b] for b in built.inputs.block_ids]),
            dims="block",
        )
        pm.Deterministic(
            "check_log_D_rep",
            pt.stack([log_replicated[b] for b in built.inputs.block_ids]),
            dims="block",
        )
        forward = pm.sample_posterior_predictive(
            posterior,
            var_names=["check_log_D_obs", "check_log_D_rep"],
            sample_vars=generated,
            freeze_vars=frozen,
            random_seed=random_seed,
            progressbar=progressbar,
            extend_inferencedata=False,
            idata_kwargs=_idata_options(checking),
        )
    replicated = forward["posterior_predictive"].to_dataset()
    observed = replicated["check_log_D_obs"].rename("log_D_obs")
    predicted = replicated["check_log_D_rep"].rename("log_D_rep")
    for values in [observed, predicted]:
        if np.isnan(values).any() or np.isposinf(values).any():
            raise FloatingPointError(
                "Categorical probability evaluation produced invalid log discrepancies"
            )
    n_obs = xr.DataArray(
        [counts[b] for b in built.inputs.block_ids],
        dims="block",
        coords={"block": np.asarray(built.inputs.block_ids, dtype=object)},
        name="n_obs",
    )
    if groupby == "overall":
        observed = observed.reduce(np.logaddexp.reduce, dim="block")
        predicted = predicted.reduce(np.logaddexp.reduce, dim="block")
        n_obs = n_obs.sum("block")
    ppp = (predicted >= observed).mean(("chain", "draw")).where(n_obs > 0)
    overflow = bool(
        (observed > np.log(np.finfo(float).max)).any()
        or (predicted > np.log(np.finfo(float).max)).any()
    )
    if overflow:
        warnings.warn(
            "Raw Pearson discrepancies exceed floating-point range; inspect log_D_obs and "
            "log_D_rep. PPP was computed on the log scale without clipping probabilities.",
            UserWarning,
            stacklevel=2,
        )
    with np.errstate(over="ignore"):
        output = xr.Dataset(
            {
                "D_obs": np.exp(observed),
                "D_rep": np.exp(predicted),
                "log_D_obs": observed,
                "log_D_rep": predicted,
                "PPP": ppp,
                "n_obs": n_obs,
            }
        )
    output.attrs = {
        "discrepancy": "categorical-Pearson",
        "comparison": "replicated >= observed",
        "groupby": groupby,
        "likelihood_target": built.spec.likelihood_kind,
        "observation_unit": built.spec.observation_unit,
        "data_fingerprint": built.spec.data_fingerprint,
        "raw_discrepancy_overflow": int(overflow),
    }
    return output
