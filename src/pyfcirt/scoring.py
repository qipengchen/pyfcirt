"""New-person scoring conditional on fixed calibration parameters."""

from __future__ import annotations

import secrets
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from ._model import BuiltModel
from .data import FCData
from .model import Model
from .models import MGPPM
from .predictive import _idata_options, _posterior_parameters
from .priors import Prior

PARAMETERS = {
    "loading_magnitude": ("statement",),
    "intercept": ("statement",),
    "trait_correlation": ("trait", "trait_aux"),
}


def _validate_calibration(parameters: xr.Dataset, built: BuiltModel) -> xr.Dataset:
    """Align an explicit point by labels and enforce the calibration constraints."""
    if not isinstance(parameters, xr.Dataset) or set(parameters.data_vars) != set(PARAMETERS):
        raise ValueError(f"calibration must be a Dataset containing exactly {list(PARAMETERS)}")
    labels = {
        "statement": list(built.inputs.statement_ids),
        "trait": list(built.inputs.trait_ids),
        "trait_aux": list(built.inputs.trait_ids),
    }
    aligned = {}
    for name, dimensions in PARAMETERS.items():
        value = parameters[name]
        if set(value.dims) != set(dimensions):
            raise ValueError(f"Calibration {name!r} requires dimensions {dimensions}")
        for dimension in dimensions:
            if dimension not in value.coords or not value.get_index(dimension).is_unique:
                raise ValueError(f"Calibration requires unique {dimension!r} coordinates")
            actual = value.coords[dimension].values.tolist()
            if any(isinstance(v, (bool, np.bool_)) for v in actual) or set(actual) != set(
                labels[dimension]
            ):
                raise ValueError(f"Calibration {dimension!r} labels do not match the model")
        if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
            value.dtype, np.complexfloating
        ):
            raise ValueError(f"Calibration {name!r} must contain real numeric values")
        value = value.sel({d: labels[d] for d in dimensions}).transpose(*dimensions).astype(float)
        if not np.isfinite(value.values).all():
            raise ValueError(f"Calibration {name!r} must be finite")
        aligned[name] = value
    output = xr.Dataset(aligned, attrs=deepcopy(parameters.attrs))
    if not (output.loading_magnitude.values > 0).all():
        raise ValueError("Calibration loading_magnitude must be strictly positive")
    correlation = output.trait_correlation.values
    if not np.allclose(correlation, correlation.T, atol=1e-10, rtol=0) or not np.allclose(
        np.diag(correlation), 1, atol=1e-10, rtol=0
    ):
        raise ValueError("Calibration trait_correlation must be symmetric with unit diagonal")
    try:
        np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError("Calibration trait_correlation must be positive definite") from error
    for group in built.inputs.groups:
        if group.task != "graded":
            if not np.allclose(
                output.intercept.values[group.statement_index].sum(axis=-1), 0, atol=1e-8, rtol=0
            ):
                raise ValueError("Ranking calibration intercepts must sum to zero in each block")
    return output


def calibration_parameters(result, *, draw: tuple | None = None) -> xr.Dataset:
    """Extract a labelled point, using mean correlation rather than mean Cholesky.

    ``draw=(chain_label, draw_label)`` selects one joint calibration draw.
    Without it, return posterior means of magnitude, intercept and correlation.
    Derived values are recomputed from free parameters through native PyMC.
    """
    import pymc as pm

    posterior = _posterior_parameters(result)
    if any(posterior.sizes[d] == 0 for d in ("chain", "draw")):
        raise ValueError("Calibration posterior must contain chains and draws")
    if draw is not None:
        if not isinstance(draw, tuple) or len(draw) != 2:
            raise ValueError("draw must be a (chain_label, draw_label) tuple")
        try:
            posterior = posterior.sel(chain=[draw[0]], draw=[draw[1]])
        except KeyError as error:
            raise ValueError(f"Unknown calibration draw {draw!r}") from error
    derived = pm.compute_deterministics(
        posterior,
        model=result.built.model,
        var_names=["intercept", "trait_correlation"],
        progressbar=False,
    )
    # PyMC's dataset mapper currently resets selected sample labels to 0..N-1.
    # Its output order follows the input; restore labels before xarray alignment.
    coordinates = _idata_options(result.built.model)["coords"]
    derived = derived.assign_coords(
        {
            **{d: coordinates[d] for d in derived.dims if d in coordinates},
            **{d: posterior.coords[d] for d in ("chain", "draw")},
        }
    )
    derived["loading_magnitude"] = posterior["loading_magnitude"]
    point = derived.mean(("chain", "draw"))
    point.attrs = {
        "source": "posterior_mean" if draw is None else "joint_posterior_draw",
        "model_data_fingerprint": result.spec.data_fingerprint,
    }
    if draw is not None:
        point.attrs.update(
            {
                "source_chain": draw[0].item() if isinstance(draw[0], np.generic) else draw[0],
                "source_draw": draw[1].item() if isinstance(draw[1], np.generic) else draw[1],
            }
        )
    return _validate_calibration(point, result.built)


def _new_data(calibrated: BuiltModel, data: FCData) -> FCData:
    if not isinstance(data, FCData):
        raise TypeError("New-person data must be FCData")
    reference = calibrated.data.design
    if set(data.design.statements["statement_id"]) - set(calibrated.inputs.statement_ids):
        raise ValueError("New data contain unknown calibration statements")
    expected = reference.to_frame().set_index(["block_id", "position"])
    given = data.design.to_frame().set_index(["block_id", "position"])
    if set(data.design.block_sizes) - set(reference.block_sizes):
        raise ValueError("New data contain unknown calibration blocks")
    columns = ["statement_id", "trait", "keying"]
    for block in data.design.block_sizes:
        if given.loc[block, columns].to_dict("records") != expected.loc[block, columns].to_dict(
            "records"
        ):
            raise ValueError(
                f"Block {block!r}: statement positions, traits or keying differ from calibration"
            )
        if data.design.response_formats[block] != reference.response_formats[block]:
            raise ValueError(f"Block {block!r}: response task differs from calibration")
    if data.design.graded_scale != reference.graded_scale:
        raise ValueError("New graded response scale differs from calibration")
    return FCData.from_frames(
        data.to_frame(),
        reference,
        administration=data.administration,
    )


def _interventions(built: BuiltModel, point: xr.Dataset) -> dict[str, np.ndarray]:
    # Cholesky(mean(correlation)) preserves unit trait variances; mean(Cholesky)
    # generally does not.
    values = {
        "loading_magnitude": point.loading_magnitude.values,
        "trait_cholesky": np.linalg.cholesky(point.trait_correlation.values),
    }
    for group, name in zip(
        built.inputs.groups, built.spec.variable_roles["intercept_components"], strict=True
    ):
        values[name] = point.intercept.values[group.statement_index]
    return {
        name: np.asarray(value, dtype=built.model[name].dtype) for name, value in values.items()
    }


def _scoring_graph(calibrated: BuiltModel, data: FCData, point: xr.Dataset) -> BuiltModel:
    import pymc as pm

    declarations = calibrated.spec.settings["priors"]
    family = MGPPM(
        priors={
            role: Prior(d["name"], **d["parameters"])
            for role, d in declarations.items()
            if role != "trait_correlation"
        }
    )
    global_prior = declarations["trait_correlation"]
    fresh = Model(
        data,
        family=family,
        priors={"trait_correlation": Prior(global_prior["name"], **global_prior["parameters"])},
    ).build()
    native = pm.do(fresh.model, _interventions(fresh, point))
    if [rv.name for rv in native.free_RVs] != ["theta"]:
        raise ValueError("Scoring must infer only new-person traits; unexpected free parameters")
    settings = {**fresh.spec.settings, "purpose": "new_person_conditional_scoring"}
    roles = {**fresh.spec.variable_roles, "item_parameters": [], "intercept_components": []}
    spec = replace(fresh.spec, settings=settings, variable_roles=roles)
    return replace(fresh, model=native, spec=spec)


@dataclass(frozen=True)
class ScoreResult:
    """Native trait draws conditional on one fixed calibration point."""

    idata: xr.DataTree
    data: FCData
    settings: dict
    sampling: dict

    @property
    def samples(self) -> xr.DataArray:
        return (
            self.idata["posterior"]["theta"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "person", "trait")
        )

    @property
    def person_status(self) -> xr.Dataset:
        return self.idata["person_status"].to_dataset()

    def summary(self, **kwargs):
        """Summarize trait draws without including calibration uncertainty."""
        import arviz as az

        if "kind" in kwargs or "sample_dims" in kwargs:
            raise ValueError("Score summaries use kind='stats'; use diagnostics separately")
        return az.summary(self.samples.to_dataset(), sample_dims=["sample"], kind="stats", **kwargs)

    def diagnostics(self, **kwargs):
        """Diagnose the single fixed-calibration scoring fit."""
        import arviz as az

        if "kind" in kwargs:
            raise ValueError("diagnostics manages kind")
        if self.sampling.get("method") == "exact_prior":
            raise ValueError("Exact prior draws have no MCMC convergence diagnostics")
        return az.summary(
            self.idata["posterior"].to_dataset(), var_names=["theta"], kind="diagnostics", **kwargs
        )

    def save(self, path):
        """Save scoring draws, fixed calibration, data and executed settings."""
        from .io import save

        return save(self, path)


def _prior_only(built: BuiltModel, *, draws: int, chains: int, random_seed: int):
    import pymc as pm

    with built.model:
        prior = pm.sample_prior_predictive(
            draws=draws * chains,
            var_names=["theta"],
            random_seed=random_seed,
            idata_kwargs=_idata_options(built.model),
        )
    posterior = {}
    for name, value in prior["prior"].data_vars.items():
        dimensions = value.dims[2:]
        posterior[name] = xr.DataArray(
            value.values.reshape((chains, draws, *value.shape[2:])),
            dims=("chain", "draw", *dimensions),
            coords={
                "chain": range(chains),
                "draw": range(draws),
                **{d: value.coords[d] for d in dimensions},
            },
        )
    return xr.DataTree.from_dict({"posterior": xr.Dataset(posterior)}), {
        "method": "exact_prior",
        "draws": draws,
        "chains": chains,
        "random_seed": random_seed,
    }


def score(
    calibrated,
    data: FCData,
    *,
    mode: str = "fixed",
    calibration: xr.Dataset | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    random_seed: int | None = None,
    nuts_sampler: str = "pymc",
    **kwargs,
) -> ScoreResult:
    """Infer new-person traits with fixed item parameters and trait correlation.

    By default, use calibration posterior means. Explicit calibration may provide
    a different fixed point. Calibration uncertainty is not propagated.
    """

    if mode != "fixed":
        raise ValueError("Only fixed-calibration scoring is supported; mode must be 'fixed'")
    if "calibration_draws" in kwargs:
        raise ValueError("calibration_draws is not supported by fixed-calibration scoring")
    built = calibrated if isinstance(calibrated, BuiltModel) else calibrated.built
    if built.spec.model_id != "mgppm":
        raise ValueError("Only validated MGPPM scoring is currently implemented")
    if nuts_sampler not in {"pymc", "numpyro"}:
        raise ValueError("Supported nuts_sampler values are 'pymc' and 'numpyro'")
    for name, value in {"draws": draws, "chains": chains, "cores": cores}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(tune, bool) or not isinstance(tune, int) or tune < 0:
        raise ValueError("tune must be a nonnegative integer")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0
    ):
        raise ValueError("random_seed must be a nonnegative integer or None")
    seed = secrets.randbits(32) if random_seed is None else random_seed
    new_data = _new_data(built, data)
    if not new_data.person_ids:
        raise ValueError("New-person scoring requires at least one person")
    if calibration is None:
        if isinstance(calibrated, BuiltModel):
            raise ValueError("A BuiltModel requires explicit calibration parameters")
        calibration = calibration_parameters(calibrated)
    point = _validate_calibration(calibration, built)
    point.attrs.setdefault("source", "user_supplied")
    graph = _scoring_graph(built, new_data, point)
    if graph.spec.observation_map:
        fitted = graph.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            nuts_sampler=nuts_sampler,
            **kwargs,
        )
        trace, metadata = fitted.idata, fitted.sampling
        trace["posterior"] = trace["posterior"].to_dataset()[["theta"]]
    else:
        trace, metadata = _prior_only(graph, draws=draws, chains=chains, random_seed=seed)
    posterior = trace["posterior"]
    if posterior.sizes["chain"] != chains or posterior.sizes["draw"] != draws:
        raise ValueError("Scoring returned different chain/draw counts than requested")
    if not np.isfinite(posterior["theta"].values).all():
        raise ValueError("Scoring returned nonfinite trait draws")
    status = new_data.to_frame(answered_only=True).groupby("person_id", sort=False).size()
    counts = np.asarray([status.get(person, 0) for person in new_data.person_ids], dtype=int)
    trace["person_status"] = xr.Dataset(
        {
            "n_observations": ("person", counts),
            "status": ("person", np.where(counts > 0, "posterior", "prior_only")),
        },
        coords={"person": np.asarray(new_data.person_ids, dtype=object)},
    )
    trace["calibration"] = point
    settings = {
        "mode": "fixed",
        "target": "fixed_calibration",
        "calibration_uncertainty": "not_propagated",
        "random_seed": seed,
        "calibration_source": deepcopy(point.attrs),
        "model_id": built.spec.model_id,
        "model_version": built.spec.model_version,
        "model_data_fingerprint": built.spec.data_fingerprint,
        "trait_prior": "zero_mean_unit_marginal_variances_with_calibration_correlation",
    }
    return ScoreResult(idata=trace, data=new_data, settings=settings, sampling=metadata)
