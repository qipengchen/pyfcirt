"""Fit runtime priors to selected PyMC free variables and paired posterior draws."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.distributions.transforms import _default_transform
from pymc.logprob.utils import replace_rvs_by_values
from pymc.model.fgraph import clone_model
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import ancestors
from scipy.special import ndtr

from pyfcirt.analysis.output import posterior_samples


def _unconstrained_draws(free, paired, transforms):
    """Evaluate PyMC's native transforms on positional arrays with sample axes."""
    if not any(transform is not None for transform in transforms.values()):
        return {name: paired[name].values for name in transforms}

    inputs = {name: free[name].type() for name in paired}
    outputs = [
        inputs[name]
        if transform is None
        else transform.forward(inputs[name], *free[name].owner.inputs)
        for name, transform in transforms.items()
    ]
    # A dependent bound uses the matching natural-scale draw of its parent RV.
    outputs = graph_replace(
        outputs, {free[name]: value for name, value in inputs.items()}, strict=False
    )
    batch = {
        value: pt.tensor(dtype=value.dtype, shape=(None, None, *value.type.shape))
        for value in inputs.values()
    }
    evaluate = pm.compile(
        list(batch.values()), vectorize_graph(outputs, batch), mode="FAST_COMPILE"
    )
    return dict(
        zip(transforms, evaluate(*(value.values for value in paired.values())), strict=True)
    )


def _options(method, components, random_seed):
    if method not in {"mvn", "gmm", "auto", "mean_field"}:
        raise ValueError("method must be 'mvn', 'gmm', 'auto' or 'mean_field'")
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise TypeError("components must be a sequence of distinct integers at least two")
    components = tuple(components)
    if not components or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 2
        for k in components
    ):
        raise ValueError("components must contain distinct integers at least two")
    if len(set(components)) != len(components):
        raise ValueError("components must not contain duplicates")
    if random_seed is not None and (
        isinstance(random_seed, (bool, np.bool_)) or not isinstance(random_seed, (int, np.integer))
    ):
        raise TypeError("random_seed must be an integer or None")
    return components, None if random_seed is None else int(random_seed)


def _marginal_pit(values, fitted):
    """Summarize empirical uniform-CDF gaps on the fitted standardized coordinates."""
    variances = fitted.covariances_
    if fitted.covariance_type == "full":
        variances = np.diagonal(variances, axis1=-2, axis2=-1)
    pit = np.zeros_like(values)
    for weight, mean, sd in zip(fitted.weights_, fitted.means_, np.sqrt(variances), strict=True):
        pit += weight * ndtr((values - mean) / sd)
    pit.sort(axis=0)
    steps = np.arange(len(values) + 1)[:, None] / len(values)
    gaps = np.maximum((steps[1:] - pit).max(axis=0), (pit - steps[:-1]).max(axis=0))
    worst = int(np.argmax(gaps))
    return {
        "median_gap": float(np.median(gaps)),
        "max_gap": float(gaps[worst]),
        "worst_coordinate": worst,
    }


def _fit(states, *, method, components, random_seed):
    """Fit one canonical kind using native Gaussian mixtures."""
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError("Fitting a posterior density requires pyfcirt[posterior]") from error

    rng = pm.util.get_random_generator(random_seed, copy=True)
    chains, draws, dimension = states.shape
    diagonal = method == "mean_field"
    fits = []

    def scale(values):
        if diagonal and len(values) < 2:
            raise ValueError("Mean-field density fitting needs at least two training states")
        if not diagonal and len(values) <= dimension:
            raise ValueError("Density fitting needs more joint training states than dimensions")
        scaler = StandardScaler()
        standardized = scaler.fit_transform(values)
        finite = np.isfinite(standardized).all()
        if diagonal:
            if not finite or not np.all(np.isfinite(scaler.var_) & (scaler.var_ > 0)):
                raise ValueError(
                    "Mean-field training coordinates must have finite positive variance"
                )
        elif not finite or np.linalg.matrix_rank(standardized) != dimension:
            raise ValueError("Joint training support coordinates must have full numerical rank")
        return scaler, standardized

    def fit(values, count, stage):
        seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        record = {"stage": stage, "components": int(count), "seed": seed}
        fits.append(record)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fitted = GaussianMixture(
                    n_components=int(count),
                    covariance_type="diag" if diagonal else "full",
                    random_state=seed,
                ).fit(values)
            if not fitted.converged_:
                raise ValueError("GaussianMixture did not converge")
            if not all(
                np.isfinite(array).all()
                for array in (fitted.weights_, fitted.means_, fitted.covariances_)
            ) or np.any(fitted.weights_ <= 0):
                raise ValueError("GaussianMixture returned invalid parameters")
            if diagonal:
                if np.any(fitted.covariances_ <= 0):
                    raise ValueError("GaussianMixture returned nonpositive variances")
            else:
                np.linalg.cholesky(fitted.covariances_)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            record.update(status="invalid", reason=str(error))
            raise ValueError(f"{stage}: invalid GaussianMixture fit: {error}") from error
        record.update(status="converged", iterations=int(fitted.n_iter_))
        return fitted

    diagnostics = {
        "requested_method": method,
        "random_seed": random_seed,
        "chains": chains,
        "draws_per_chain": draws,
        "total_states": chains * draws,
        "support_dimension": dimension,
        "fits": fits,
        "comparison_performed": False,
        "marginal_pit": {},
    }
    block_length = int(np.sqrt(draws))
    block_count = draws // block_length
    count = 1
    if method in {"mvn", "mean_field"}:
        diagnostics["selection_reason"] = f"explicit_{method}"
    elif chains < 4:
        if method == "gmm":
            raise ValueError("Explicit GMM requires at least four independently labelled chains")
        diagnostics["selection_reason"] = "insufficient_independent_chains"
    elif block_count < 2:
        if method == "gmm":
            raise ValueError("Explicit GMM requires at least two complete audit blocks")
        diagnostics["selection_reason"] = "insufficient_audit_blocks"
    elif (chains - 2) * draws <= dimension:
        if method == "gmm":
            raise ValueError("GMM audit needs more training states than support dimensions")
        diagnostics["selection_reason"] = "insufficient_audit_training_states"
    else:
        train = states[:-2].reshape((-1, dimension))
        scaler, standardized = scale(train)
        baseline = fit(standardized, 1, "training_mvn")
        audit_fit = baseline
        selection = scaler.transform(states[-2])
        audit = scaler.transform(states[-1])
        eligible = []
        diagnostics.update(
            training_states=len(train), selection_states=draws, audit_states=draws, candidates=[]
        )
        for candidate_count in components:
            candidate = {"components": int(candidate_count)}
            diagnostics["candidates"].append(candidate)
            try:
                fitted = fit(standardized, candidate_count, "training_gmm")
                scores = fitted.score_samples(selection)
                if not np.isfinite(scores).all():
                    raise ValueError("Component-selection scores are not finite")
                score = float(scores.mean())
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                candidate.update(status="invalid", reason=str(error))
                continue
            candidate.update(status="eligible", selection_score=score)
            eligible.append((score, int(candidate_count), fitted))
        if not eligible:
            if method == "gmm":
                raise ValueError("No eligible full-covariance GMM candidate")
            diagnostics["selection_reason"] = "no_eligible_gmm"
        else:
            _, candidate_count, selected = max(eligible, key=lambda item: item[0])
            difference = selected.score_samples(audit) - baseline.score_samples(audit)
            used = block_count * block_length
            if not np.isfinite(difference).all():
                raise ValueError("Audit log-density differences are not finite")
            blocks = difference[:used].reshape((block_count, block_length)).mean(axis=1)
            gain = float(blocks.mean())
            error = float(blocks.std(ddof=1) / np.sqrt(block_count))
            threshold = max(0.01, 2 * error)
            diagnostics.update(
                comparison_performed=True,
                selected_candidate_components=candidate_count,
                audit_gain=gain,
                audit_standard_error=error,
                audit_threshold=threshold,
                audit_block_length=block_length,
                audit_complete_blocks=block_count,
                audit_used_states=used,
                audit_omitted_states=draws - used,
                audit_describes="training_candidates_before_all_state_refit",
            )
            if method == "gmm" or gain > threshold:
                count = candidate_count
                audit_fit = selected
                diagnostics["selection_reason"] = (
                    "explicit_gmm" if method == "gmm" else "audit_gain_exceeds_threshold"
                )
            else:
                diagnostics["selection_reason"] = "audit_gain_does_not_exceed_threshold"
        diagnostics["marginal_pit"]["held_out_audit"] = _marginal_pit(audit, audit_fit)

    # The chosen family is refitted once, with fresh scaling and an independent seed.
    scaler, standardized = scale(states.reshape((-1, dimension)))
    final = fit(standardized, count, "all_state_refit")
    means = scaler.inverse_transform(final.means_)
    if diagonal:
        scales = scaler.scale_[None, :] * np.sqrt(final.covariances_)
        family = "mean_field"
    else:
        scales = scaler.scale_[None, :, None] * np.linalg.cholesky(final.covariances_)
        family = "mvn" if count == 1 else "gmm"
    diagnostics["marginal_pit"]["in_sample_refit"] = _marginal_pit(standardized, final)
    diagnostics.update(family=family, components=count, numerical_status="converged")
    return final.weights_, means, scales, diagnostics


@dataclass(frozen=True)
class PosteriorApproximation:
    """A fitted set of native-coordinate densities, valid only at runtime."""

    groups: Mapping
    densities: Mapping
    diagnostics: Mapping
    transforms: Mapping
    shapes: Mapping
    evidence_people: frozenset = frozenset()
    evidence_batches: frozenset = frozenset()
    design_identity: str | None = None
    coordinate_identity: str | None = None
    fixed_parameters: xr.Dataset | None = None
    _fit_id: str = field(default_factory=lambda: uuid4().hex, init=False, repr=False, compare=False)

    @property
    def var_names(self):
        return tuple(name for group in self.groups.values() for name in group)

    @classmethod
    def from_idata(
        cls,
        idata,
        *,
        model,
        var_names,
        method="mvn",
        components=(2, 3),
        random_seed=None,
        transforms=None,
    ):
        """Fit selected native RVs. A mapping groups joint coordinates by parameter kind."""
        return cls._from_native(
            posterior_samples(idata),
            model=model,
            var_names=var_names,
            method=method,
            components=components,
            random_seed=random_seed,
            transforms=transforms,
        )

    @classmethod
    def _from_native(
        cls,
        posterior,
        *,
        model,
        var_names,
        method,
        components,
        random_seed,
        transforms,
        eligible=None,
    ):
        components, random_seed = _options(method, components, random_seed)
        if isinstance(var_names, str) or not var_names:
            raise ValueError("Select at least one native free RV in var_names")
        if isinstance(var_names, Mapping):
            if any(isinstance(group, str) for group in var_names.values()):
                raise TypeError("Each parameter kind requires a sequence of native RV names")
            groups = {kind: tuple(group) for kind, group in var_names.items()}
            names = tuple(name for group in groups.values() for name in group)
        else:
            names = tuple(var_names)
            groups = {name: (name,) for name in names}
        selected = set(names)
        if not names or any(not group for group in groups.values()) or len(names) != len(selected):
            raise ValueError("Parameter kinds must select distinct native RVs")
        free = {rv.name: rv for rv in model.free_RVs}
        allowed = free.keys() if eligible is None else {rv.name for rv in eligible}
        if missing := selected - allowed:
            expected = "actual native free" if eligible is None else "shared calibration"
            raise ValueError(f"Select {expected} RVs: {sorted(missing)}")
        if any(np.dtype(free[name].dtype).kind != "f" for name in names):
            raise ValueError("Select continuous native calibration RVs")
        overrides = dict(transforms or {})
        if overrides.keys() - selected:
            raise ValueError("Transform overrides must select fitted native RVs")
        native_free = set(free.values())
        resolved, transform_inputs = {}, set()
        for name in names:
            rv = free[name]
            native = model.rvs_to_transforms[rv]
            if native is None and overrides.get(name) is None:
                native = _default_transform(rv.owner.op, rv)
            transform = resolved[name] = overrides.get(name, native)
            if transform is None and native is not None:
                raise ValueError(f"{name}: a constrained native RV cannot disable its transform")
            if transform is not None:
                transform_inputs.update(
                    ancestors(
                        [transform.forward(rv.type(), *rv.owner.inputs)], blockers=native_free
                    )
                )
        inputs_only = sorted({rv.name for rv in native_free & transform_inputs} - selected)
        shapes = model.eval_rv_shapes()
        paired = {}
        for name in (*names, *inputs_only):
            if name not in posterior:
                purpose = "native RV" if name in selected else "transform input"
                raise ValueError(f"Missing posterior draws for {purpose} {name!r}")
            values = posterior[name].transpose("chain", "draw", ...)
            axes = model.named_vars_to_dims.get(name, ())
            named = {dim for dim in axes if dim is not None}
            for dim in named:
                if dim not in values.dims or (
                    model.coords[dim] is not None
                    and tuple(values[dim].values) != tuple(model.coords[dim])
                ):
                    raise ValueError(f"{name}: posterior labels differ on {dim!r}")
            unnamed = iter(dim for dim in values.dims[2:] if dim not in named)
            dims = tuple(next(unnamed, None) if dim is None else dim for dim in axes) + tuple(
                unnamed
            )
            if None in dims or tuple(values.sizes[dim] for dim in dims) != shapes[name]:
                raise ValueError(f"{name}: posterior shape differs from the native RV")
            paired[name] = values.transpose("chain", "draw", *dims)
        coordinates = _unconstrained_draws(free, paired, resolved)
        densities, diagnostics = {}, {}
        shape = posterior.sizes["chain"], posterior.sizes["draw"]
        for kind, group in groups.items():
            states = np.concatenate(
                [coordinates[name].reshape(*shape, -1) for name in group],
                axis=-1,
            )
            if not states.size or not np.isfinite(states).all():
                raise ValueError(f"{kind}: posterior coordinates must be nonempty and finite")
            *density, diagnostics[kind] = _fit(
                states, method=method, components=components, random_seed=random_seed
            )
            for array in density:
                array.flags.writeable = False
            densities[kind] = tuple(density)
        native_shapes = {name: (shapes[name], coordinates[name].shape[2:]) for name in names}
        return cls(
            MappingProxyType(groups),
            MappingProxyType(densities),
            {"requested_method": method, "kinds": diagnostics},
            MappingProxyType(resolved),
            MappingProxyType(native_shapes),
        )

    def apply(self, model):
        """Return a native model with selected priors replaced. The input model is unchanged."""
        return self._apply(model, {})

    def _apply(self, model, fixed):
        candidate = clone_model(model)
        free = {rv.name: rv for rv in candidate.free_RVs}
        if missing := set(self.var_names) - free.keys():
            raise ValueError(
                f"Destination lacks selected native calibration RVs: {sorted(missing)}"
            )
        fixed = {candidate[name]: value for name, value in fixed.items()}
        if fixed and (overlap := set(ancestors(fixed)) & {free[name] for name in self.var_names}):
            raise ValueError(
                "Fixed parameters depend on selected native RVs: "
                f"{sorted(rv.name for rv in overlap)}"
            )
        shapes = candidate.eval_rv_shapes()
        for name in self.var_names:
            if shapes[name] != self.shapes[name][0]:
                raise ValueError(f"{name}: destination native shape differs from calibration")
        draws = {}
        with candidate:
            for index, (kind, group) in enumerate(self.groups.items()):
                weights, means, scales = self.densities[kind]
                name = f"calibrated_prior_{index}_raw"
                if len(weights) > 1:
                    draw = pm.Mixture(
                        name, w=weights, comp_dists=pm.MvNormal.dist(mu=means, chol=scales)
                    )
                elif scales.ndim == 2:
                    draw = pm.Normal(name, mu=means[0], sigma=scales[0])
                else:
                    draw = pm.MvNormal(name, mu=means[0], chol=scales[0])
                start = 0
                for variable in group:
                    rv = free[variable]
                    shape = self.shapes[variable][1]
                    stop = start + int(np.prod(shape))
                    draws[rv] = pt.cast(draw[start:stop].reshape(shape), rv.dtype)
                    start = stop
        # Apply fitted native transforms directly. These RVs are replaced in the
        # intervention, so they do not need new sampler transforms on the candidate.
        transforms = {
            rv: transform for rv in draws if (transform := self.transforms[rv.name]) is not None
        }
        if transforms:
            expressions = replace_rvs_by_values(
                list(draws), rvs_to_values={**fixed, **draws}, rvs_to_transforms=transforms
            )
            draws = {
                rv: pt.specify_shape(result, shapes[rv.name])
                for rv, result in zip(draws, expressions, strict=True)
            }
        return pm.do(candidate, fixed | draws, make_interventions_shared=False)
