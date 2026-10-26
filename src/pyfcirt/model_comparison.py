"""Conditional response likelihood and explicitly defined comparison statistics."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr


def _validated_log_likelihood(log_likelihood: xr.DataArray) -> xr.DataArray:
    if not isinstance(log_likelihood, xr.DataArray):
        raise TypeError("log_likelihood must be an xarray DataArray")
    if set(log_likelihood.dims) != {"chain", "draw", "obs_id"}:
        raise ValueError("log_likelihood requires exactly chain, draw and obs_id dimensions")
    result = log_likelihood.transpose("chain", "draw", "obs_id")
    if any(result.sizes[dim] == 0 for dim in result.dims):
        raise ValueError("log_likelihood dimensions must be nonempty")
    for dimension in result.dims:
        if dimension not in result.coords or not result.get_index(dimension).is_unique:
            raise ValueError(f"log_likelihood requires unique {dimension} coordinates")
    if (
        not np.issubdtype(result.dtype, np.number)
        or np.issubdtype(result.dtype, np.complexfloating)
        or not np.isfinite(result.values).all()
    ):
        raise ValueError("log_likelihood must contain finite numeric values; no draws are dropped")
    return result.astype(np.float64, copy=False)


def log_likelihood(result, *, progressbar: bool = True) -> xr.DataArray:
    """Compute conditional person-block response logp through native PyMC.

    Excludes priors and Jacobians and does not modify the fitted DataTree.
    Only actual response events are present; structural unknown ranks have
    already been marginalized by the model's categorical likelihood.
    """
    import pymc as pm
    from pymc.model.transform import remove_value_transforms

    from .predictive import _posterior_parameters

    built = result.built
    posterior = _posterior_parameters(result)
    # PyMC 6.3.2's transform-removal name helper truncates only the final
    # underscore segment of names such as cholesky_corr. Normalize value names
    # on a detached untransformed graph; never alter the fitted graph or draws.
    untransformed = remove_value_transforms(built.model)
    for variable in untransformed.free_RVs:
        untransformed.rvs_to_values[variable].name = variable.name
    computed = pm.compute_log_likelihood(
        xr.DataTree.from_dict({"posterior": posterior}),
        model=untransformed,
        var_names=built.spec.variable_roles["responses"],
        extend_inferencedata=False,
        progressbar=progressbar,
    )
    computed = computed.assign_coords({d: posterior.coords[d] for d in ("chain", "draw")})
    pieces = []
    for component in built.spec.components:
        name = component["observed_variable"]
        dimension = f"{component['id']}_obs"
        expected = [r["obs_id"] for r in built.spec.observation_map if r["variable"] == name]
        values = computed[name]
        if values.coords[dimension].values.tolist() != expected:
            raise ValueError(f"Computed observation coordinates do not match {name!r}")
        pieces.append(values.rename({dimension: "obs_id"}))
    combined = xr.concat(pieces, dim="obs_id").rename("response")
    combined.attrs = {
        "observation_unit": built.spec.observation_unit,
        "likelihood_target": built.spec.likelihood_kind,
        "data_fingerprint": built.spec.data_fingerprint,
        "model_id": built.spec.model_id,
    }
    return _validated_log_likelihood(combined)


def dic(log_likelihood: xr.DataArray) -> xr.Dataset:
    """R2jags default pV DIC: chainwise variance of total response deviance.

    Each chain needs at least two draws. pV uses ddof=1; averaging occurs
    after the within-chain calculation. No plug-in mean-parameter deviance
    or pooled between-chain variance is used.
    """
    logp = _validated_log_likelihood(log_likelihood)
    if logp.sizes["draw"] < 2:
        raise ValueError("R2jags pV DIC requires at least two draws per chain")
    deviance = -2 * logp.sum("obs_id")
    mean = deviance.mean("draw")
    penalty = deviance.var("draw", ddof=1) / 2
    result = xr.Dataset(
        {
            "Dbar_chain": mean,
            "pV_chain": penalty,
            "DIC_chain": mean + penalty,
            "Dbar": mean.mean("chain"),
            "pV": penalty.mean("chain"),
            "DIC": (mean + penalty).mean("chain"),
        }
    )
    result.attrs = {
        **logp.attrs,
        "method": "R2jags-default-chainwise-pV",
        "reference_version": "0.9-0",
        "scale": "deviance",
        "variance_ddof": 1,
    }
    return result


def waic(log_likelihood: xr.DataArray) -> xr.Dataset:
    """WAIC with ArviZ 0.22.0 population-variance and warning conventions.

    Both elpd (larger is better) and deviance (-2 * elpd, smaller is better)
    are returned with pointwise values and standard errors. Inputs retain
    their conditional/marginal target metadata; this function does not
    integrate latent variables or establish comparability of datasets.
    """
    logp = _validated_log_likelihood(log_likelihood)
    samples = logp.values.reshape((-1, logp.sizes["obs_id"]))
    lppd = np.logaddexp.reduce(samples, axis=0) - np.log(len(samples))
    penalty = samples.var(axis=0, ddof=0)
    pointwise = lppd - penalty
    unreliable = bool(np.any(penalty > 0.4))
    if unreliable:
        warnings.warn(
            "Posterior variance of log predictive densities exceeds 0.4 for one or more "
            "observations; WAIC may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
    if np.equal(pointwise.sum(), pointwise).all():
        warnings.warn(
            "Pointwise WAIC equals total WAIC; verify the pointwise likelihood definition.",
            UserWarning,
            stacklevel=2,
        )
    se = float(np.sqrt(len(pointwise) * pointwise.var(ddof=0)))
    output = xr.Dataset(
        {
            "lppd_i": ("obs_id", lppd),
            "p_waic_i": ("obs_id", penalty),
            "elpd_waic_i": ("obs_id", pointwise),
            "waic_i": ("obs_id", -2 * pointwise),
            "elpd_waic": float(pointwise.sum()),
            "p_waic": float(penalty.sum()),
            "WAIC": float(-2 * pointwise.sum()),
            "se_elpd": se,
            "se_WAIC": 2 * se,
            "warning": unreliable,
            "n_samples": len(samples),
            "n_data_points": len(pointwise),
        },
        coords={"obs_id": logp.coords["obs_id"]},
    )
    output.attrs = {
        **logp.attrs,
        "method": "WAIC",
        "reference_version": "arviz-0.22.0",
        "elpd_scale": "log",
        "WAIC_scale": "deviance",
        "variance_ddof": 0,
    }
    return output


def loo(result, *, pointwise: bool = True, reff: float | None = None, progressbar: bool = True):
    """Call current ArviZ PSIS-LOO on the same conditional person-block likelihood."""
    import arviz as az

    logp = result.log_likelihood(progressbar=progressbar)
    view = xr.DataTree.from_dict(
        {
            "posterior": result.idata["posterior"].to_dataset(),
            "log_likelihood": logp.to_dataset(),
        }
    )
    return az.loo(view, var_name="response", pointwise=pointwise, reff=reff)
