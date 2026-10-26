"""One sampling entry point over the native PyMC model."""

from __future__ import annotations

import importlib.metadata
import platform
import secrets
import time
from copy import deepcopy

from ._model import BuiltModel


def fit(
    built: BuiltModel,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    random_seed: int | None = None,
    nuts_sampler: str = "pymc",
    **kwargs,
):
    """Sample with PyMC and retain the executed settings and native result.

    NumPyro is an optional PyMC-dispatched sampler. Additional keyword arguments
    follow ``pymc.sample``; settings that would remove required result groups
    or substitute the configured graph are rejected.
    """
    import pymc as pm
    import xarray as xr

    from .predictive import _idata_options
    from .results import FitResult

    for name, value in {"draws": draws, "tune": tune, "chains": chains, "cores": cores}.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if name == "tune" else 1)
        ):
            raise ValueError(
                f"{name} must be {'a nonnegative' if name == 'tune' else 'a positive'} integer"
            )
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0
    ):
        raise ValueError("random_seed must be a nonnegative integer or None")
    if nuts_sampler not in {"pymc", "numpyro"}:
        raise ValueError("Supported nuts_sampler values are 'pymc' and 'numpyro'")
    reserved = {
        "model",
        "return_inferencedata",
        "var_names",
        "trace",
        "step",
        "discard_tuned_samples",
    }
    if reserved & kwargs.keys():
        raise ValueError(f"fit manages these sampling settings: {sorted(reserved & kwargs.keys())}")
    if nuts_sampler == "numpyro":
        try:
            importlib.metadata.version("numpyro")
        except importlib.metadata.PackageNotFoundError as error:
            raise ImportError("Install pyfcirt[numpyro] to use the NumPyro sampler") from error
    seed = secrets.randbits(32) if random_seed is None else random_seed
    settings = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores,
        "random_seed": seed,
        "nuts_sampler": nuts_sampler,
    }
    start = time.perf_counter()
    # PyMC may mutate nested sampler options while normalizing its arguments.
    executed_options = deepcopy(kwargs)
    with built.model:
        idata = pm.sample(
            **settings,
            **{
                **executed_options,
                "idata_kwargs": _idata_options(built.model, executed_options.get("idata_kwargs")),
            },
        )
    if (
        not isinstance(idata, xr.DataTree)
        or "posterior" not in idata
        or "sample_stats" not in idata
    ):
        raise TypeError("PyMC must return a DataTree with posterior and sample_stats groups")
    versions = {
        name: importlib.metadata.version(name)
        for name in ["pyfcirt", "pymc", "pytensor", "arviz", "numpy"]
    }
    if nuts_sampler == "numpyro":
        versions.update(
            {name: importlib.metadata.version(name) for name in ["numpyro", "jax", "jaxlib"]}
        )
    metadata = {
        **settings,
        "extra_settings": deepcopy(kwargs),
        "elapsed_seconds": time.perf_counter() - start,
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.system(),
        "actual_chains": idata["posterior"].sizes["chain"],
        "actual_draws": idata["posterior"].sizes["draw"],
    }
    if nuts_sampler == "numpyro":
        import jax

        metadata["jax_devices"] = [str(device) for device in jax.devices()]
    idata.attrs.update(
        pyfcirt_model_id=built.spec.model_id,
        pyfcirt_model_version=built.spec.model_version,
        pyfcirt_data_fingerprint=built.spec.data_fingerprint,
    )
    return FitResult(idata=idata, built=built, sampling=metadata)
