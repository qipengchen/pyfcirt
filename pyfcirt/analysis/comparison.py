"""WAIC and a stated chainwise pV DIC from existing response log likelihood."""

from __future__ import annotations

from warnings import warn

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.scipy.special import logsumexp

from pyfcirt.common.validation import has_labelled_axes


def _validated_log_likelihood(values: xr.DataArray) -> tuple[xr.DataArray, jnp.ndarray]:
    if not isinstance(values, xr.DataArray) or set(values.dims) != {"chain", "draw", "obs"}:
        raise ValueError("log_likelihood must have chain, draw and obs dimensions")
    if not has_labelled_axes(values, values.dims):
        raise ValueError("Log-likelihood axes must have nonempty unique labels")
    if values.dtype.kind not in "iuf":
        raise ValueError("Log likelihood must be finite real values; no states are dropped")
    labels = values.transpose("chain", "draw", "obs")
    logp = jnp.asarray(labels.values, dtype=float)
    if not bool(jnp.isfinite(logp).all()):
        raise ValueError("Log likelihood must be finite real values; no states are dropped")
    if values.attrs.get("inference_method", "mcmc") != "mcmc":
        raise NotImplementedError("Model comparison currently supports MCMC log likelihood")
    return labels, logp


def waic(log_likelihood: xr.DataArray) -> xr.Dataset:
    """Calculate log-scale WAIC with population-variance pointwise penalties.

    ArviZ Stats 1.3.3 has no public WAIC implementation. This formula consumes
    the existing observation factors and does not establish model comparability.
    """
    logp, values = _validated_log_likelihood(log_likelihood)
    count = logp.sizes["chain"] * logp.sizes["draw"]
    if count < 2:
        raise ValueError("WAIC requires at least two posterior states")
    lppd = logsumexp(values, axis=(0, 1)) - jnp.log(count)
    penalty = values.var(axis=(0, 1), ddof=0)
    pointwise = lppd - penalty
    unreliable = bool((penalty > 0.4).any())
    if unreliable:
        warn(
            "Pointwise log predictive variance exceeds 0.4; WAIC may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
    se = jnp.sqrt(logp.sizes["obs"] * pointwise.var(ddof=0))
    values = dict(
        lppd_i=lppd,
        p_waic_i=penalty,
        elpd_waic_i=pointwise,
        waic_i=-2 * pointwise,
        elpd_waic=pointwise.sum(),
        p_waic=penalty.sum(),
        WAIC=-2 * pointwise.sum(),
        se_elpd=se,
        se_WAIC=2 * se,
        warning=jnp.asarray(unreliable),
        n_samples=jnp.asarray(count),
        n_data_points=jnp.asarray(logp.sizes["obs"]),
    )
    # Transfer only final summaries, retaining the original observation coordinates.
    return xr.Dataset(
        {
            name: (("obs",), np.asarray(value)) if value.ndim else np.asarray(value)
            for name, value in values.items()
        },
        coords=logp.isel(chain=0, draw=0, drop=True).coords,
        attrs={
            **logp.attrs,
            "method": "WAIC",
            "variance_ddof": 0,
            "elpd_scale": "log",
            "WAIC_scale": "deviance",
        },
    )


def dic(log_likelihood: xr.DataArray) -> xr.Dataset:
    """Calculate pV = Var_draw(D)/2 within each chain, then average over chains."""
    logp, values = _validated_log_likelihood(log_likelihood)
    if logp.sizes["draw"] < 2:
        raise ValueError("Chainwise pV DIC requires at least two draws per chain")
    deviance = -2 * values.sum(axis=-1)
    mean = deviance.mean(axis=1)
    penalty = deviance.var(axis=1, ddof=1) / 2
    values = {
        "Dbar_chain": (("chain",), np.asarray(mean)),
        "pV_chain": (("chain",), np.asarray(penalty)),
        "DIC_chain": (("chain",), np.asarray(mean + penalty)),
        "Dbar": np.asarray(mean.mean()),
        "pV": np.asarray(penalty.mean()),
        "DIC": np.asarray((mean + penalty).mean()),
    }
    return xr.Dataset(
        values,
        coords={"chain": logp.chain},
        attrs={**logp.attrs, "method": "chainwise-pV", "variance_ddof": 1, "scale": "deviance"},
    )
