"""Native predictive sampling and explicit decoding to block-response lists."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._model import BuiltModel


def _idata_options(model, options=None):
    """Preserve heterogeneous identifiers before PyMC/ArviZ NumPy conversion."""
    output = dict(options or {})
    provided = dict(output.get("coords") or {})
    for name, labels in model.coords.items():
        if labels is None:
            continue
        labels = list(labels)
        if name in provided and list(provided[name]) != labels:
            raise ValueError(f"idata_kwargs cannot change model coordinates {name!r}")
        provided[name] = labels
    for name, labels in provided.items():
        if labels is not None:
            labels = list(labels)
            string_flags = [isinstance(value, str) for value in labels]
            provided[name] = np.asarray(
                labels, dtype=object if any(string_flags) and not all(string_flags) else None
            )
    for name, dimensions in (output.get("dims") or {}).items():
        if name in model.named_vars_to_dims and tuple(dimensions) != model.named_vars_to_dims[name]:
            raise ValueError(f"idata_kwargs cannot change model dimensions for {name!r}")
    output["coords"] = provided
    return output


def _posterior_parameters(result):
    """Select required free draws and verify their original model coordinates."""
    built = result.built
    posterior = result.idata["posterior"].to_dataset()
    names = [variable.name for variable in built.model.free_RVs]
    missing = set(names) - set(posterior.data_vars)
    if missing:
        raise ValueError(f"Posterior is missing required parameters: {sorted(missing)}")
    for name in names:
        for dimension in built.model.named_vars_to_dims.get(name, ()):
            expected = list(built.model.coords[dimension])
            if dimension not in posterior[name].coords:
                raise ValueError(f"Posterior {dimension!r} coordinates are absent")
            if posterior[name].coords[dimension].values.tolist() != expected:
                raise ValueError(f"Posterior {dimension!r} coordinates do not match the model")
    return posterior[names]


def prior_predictive(built: BuiltModel, *, draws: int = 500, random_seed: int | None = None):
    """Draw prior parameters and responses without requiring a FitResult."""
    import pymc as pm

    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    with built.model:
        result = pm.sample_prior_predictive(
            draws=draws, random_seed=random_seed, idata_kwargs=_idata_options(built.model)
        )
    result.attrs["pyfcirt_model_id"] = built.spec.model_id
    result.attrs["pyfcirt_model_version"] = built.spec.model_version
    return result


def response_frame(
    built: BuiltModel, draws, *, group: str = "prior_predictive", chain: int = 0, draw: int = 0
) -> pd.DataFrame:
    """Decode one chain/draw only, avoiding expansion of every posterior replica."""
    if group not in draws:
        raise ValueError(f"Predictive group {group!r} is absent")
    pieces = []
    for component in built.spec.components:
        name = component["observed_variable"]
        if name not in draws[group]:
            raise ValueError(f"Predictive variable {name!r} is absent")
        values = draws[group][name].isel(chain=chain, draw=draw)
        dimension = f"{component['id']}_obs"
        actual_ids = values.coords[dimension].values.tolist()
        expected_ids = [r["obs_id"] for r in built.spec.observation_map if r["variable"] == name]
        if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
            raise ValueError(f"Predictive observation coordinates do not match {name!r}")
        by_id = dict(zip(actual_ids, np.asarray(values).tolist(), strict=True))
        for record in built.spec.observation_map:
            if record["variable"] != name:
                continue
            code = by_id[record["obs_id"]]
            if isinstance(code, bool) or not isinstance(code, (int, np.integer)):
                raise ValueError(f"Invalid predictive event code for {name!r}")
            if code < 0 or code >= len(component["events"]):
                raise ValueError(f"Predictive event code out of range for {name!r}")
            event = component["events"][code]
            pieces.append(
                (
                    record["person_id"],
                    record["block_id"],
                    event if component["task"] == "graded" else list(event),
                )
            )
    return pd.DataFrame(pieces, columns=["person_id", "block_id", "response"])


def posterior_predictive(result, *, random_seed: int | None = None, progressbar: bool = True):
    """Use PyMC's forward sampler with observed variables explicitly resampled."""
    import pymc as pm

    built = result.built
    responses = built.spec.variable_roles["responses"]
    posterior = _posterior_parameters(result)
    frozen = list(posterior.data_vars)
    with built.model:
        return pm.sample_posterior_predictive(
            posterior,
            var_names=responses,
            sample_vars=responses,
            freeze_vars=frozen,
            random_seed=random_seed,
            progressbar=progressbar,
            extend_inferencedata=False,
            idata_kwargs=_idata_options(built.model),
        )
