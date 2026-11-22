"""Empirical reliability from explicit final ability uncertainty."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import xarray as xr

from pyfcirt.analysis.information import _FISHER_BATCH_SIZE, _point, _response_fisher
from pyfcirt.analysis.output import posterior_samples
from pyfcirt.common.validation import has_labelled_axes

if TYPE_CHECKING:
    from pyfcirt.model import Model

_SOURCE_FIELDS = (
    "inference_method",
    "source",
    "approximate",
    "uncertainty_source",
    "parameterization",
)


def empirical_reliability(
    mean: xr.DataArray, variance: xr.DataArray, *, method="ratio"
) -> xr.Dataset:
    """Use V/(V+U) or 1-U/V with score sample variance V and mean posterior variance U."""
    if method not in {"ratio", "subtraction"}:
        raise ValueError("method must be 'ratio' or 'subtraction'")
    for name, value in (("mean", mean), ("variance", variance)):
        if not isinstance(value, xr.DataArray) or set(value.dims) not in (
            {"person"},
            {"person", "trait"},
        ):
            raise ValueError(f"{name} requires person and optionally trait dimensions")
        if not has_labelled_axes(value, value.dims):
            raise ValueError(f"{name} requires nonempty unique coordinates")
        if value.dtype.kind not in "iuf" or not bool(jnp.isfinite(value.values).all()):
            raise ValueError(f"{name} must contain finite real, non-boolean values")
    if set(mean.dims) != set(variance.dims):
        raise ValueError("mean and variance must have identical dimensions and labels")
    for dim in mean.dims:
        if (
            len(mean[dim]) != len(variance[dim])
            or not mean.get_index(dim).isin(variance.get_index(dim)).all()
        ):
            raise ValueError(f"mean and variance have different {dim} labels")
    variance = variance.sel({dim: mean[dim] for dim in mean.dims}).transpose(*mean.dims)
    if mean.sizes["person"] < 2 or bool((variance < 0).any()):
        raise ValueError("Reliability requires at least two persons and nonnegative variances")
    score_variance = jnp.var(jnp.asarray(mean.values), axis=mean.get_axis_num("person"), ddof=1)
    uncertainty = jnp.mean(jnp.asarray(variance.values), axis=variance.get_axis_num("person"))
    denominator = score_variance + uncertainty if method == "ratio" else score_variance
    if not bool(jnp.isfinite(denominator).all()) or bool((denominator <= 0).any()):
        raise ValueError(f"{method} reliability has an undefined denominator")
    coefficient = (
        score_variance / denominator if method == "ratio" else 1 - uncertainty / denominator
    )
    source = {}
    for field in _SOURCE_FIELDS:
        if (
            field in mean.attrs
            and field in variance.attrs
            and mean.attrs[field] != variance.attrs[field]
        ):
            raise ValueError(f"mean and variance have conflicting {field} metadata")
        if field in variance.attrs or field in mean.attrs:
            source[field] = variance.attrs.get(field, mean.attrs.get(field))
    if "approximate" in source:
        if source["approximate"] not in (False, True):
            raise ValueError("approximate metadata must be a boolean or 0/1 flag")
        source["approximate"] = int(source["approximate"])
    dims = tuple(dim for dim in mean.dims if dim != "person")
    return xr.Dataset(
        dict(
            reliability=(dims, np.asarray(coefficient)),
            score_variance=(dims, np.asarray(score_variance)),
            mean_posterior_variance=(dims, np.asarray(uncertainty)),
            n_person=xr.DataArray(mean.sizes["person"]),
        ),
        coords={name: value for name, value in mean.coords.items() if "person" not in value.dims},
        attrs={
            **source,
            "method": method,
            "formula": "V/(V+U)" if method == "ratio" else "1-U/V",
            "ddof": 1,
            "weighting": "unweighted",
        },
    )


def reliability(model: Model, *, method="ratio", person_ids: Sequence | None = None) -> xr.Dataset:
    """Summarize genuine theta samples, including explicitly sourced approximate samples."""
    from pyfcirt.common.validation import identifier

    posterior = posterior_samples(model.idata)
    if "theta" not in posterior or set(posterior.theta.dims) != {
        "chain",
        "draw",
        "person",
        "trait",
    }:
        raise ValueError("posterior must contain final theta(chain,draw,person,trait)")
    theta = posterior.theta
    if person_ids is not None:
        if isinstance(person_ids, (str, bytes)) or not isinstance(
            person_ids, Sequence | np.ndarray
        ):
            raise TypeError("person_ids must be an ordered sequence")
        labels = tuple(identifier(value, "person_ids") for value in person_ids)
        if len(set(labels)) != len(labels) or set(labels) - set(theta.person.values):
            raise ValueError("person_ids contains duplicate or unknown persons")
        theta = theta.sel(person=np.asarray(labels, dtype=object))
    if theta.sizes["chain"] * theta.sizes["draw"] < 2:
        raise ValueError("Reliability requires at least two posterior states")
    if theta.dtype.kind not in "iuf" or not bool(jnp.isfinite(theta.values).all()):
        raise ValueError("Final theta samples must be finite real values")
    mean = theta.mean(("chain", "draw"))
    variance = theta.var(("chain", "draw"), ddof=1)
    source = {
        field: posterior.attrs.get(field, model.idata.attrs.get(field))
        for field in _SOURCE_FIELDS
        if field in posterior.attrs or field in model.idata.attrs
    }
    source.setdefault(
        "approximate", source.get("inference_method") in {"advi", "fullrank_advi", "map_laplace"}
    )
    mean.attrs = variance.attrs = source
    return empirical_reliability(mean, variance, method=method)


def information_reliability(
    model: Model,
    *,
    draw=None,
    points=4096,
    refine_points=8192,
    scrambles=2,
    random_seed=42,
    chunk_size=_FISHER_BATCH_SIZE,
) -> xr.Dataset:
    """Approximate fixed-calibration reliability from full Fisher matrices.

    This experimental coefficient is not exact posterior reliability. All
    respondents must share an answered form and a zero-mean normal population;
    individual nuisance effects and composite rank comparisons are unsupported.
    Population integration reuses the measurement kernels, not a scoring solver.
    ``chunk_size`` bounds Fisher batches and the population integration grid batches.
    """
    from jax.scipy.special import ndtri
    from scipy.stats import qmc

    posterior = posterior_samples(model.idata, mcmc=True)
    for name, value in (("points", points), ("refine_points", refine_points)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 2
            or value & (value - 1)
        ):
            raise ValueError(f"{name} must be a power of two >=2")
    if refine_points <= points:
        raise ValueError("refine_points must exceed points")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (scrambles, chunk_size)):
        raise TypeError("scrambles and chunk_size must be integers")
    if scrambles < 2 or chunk_size < 1:
        raise ValueError("Use at least two scrambles and a positive chunk_size")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    point = _point(model, None, draw, None, posterior=posterior)
    bindings = model._bindings
    constant = bindings.prepared.constant_data
    recipe = constant.attrs["population"]["recipe"]
    if recipe["regression"] is not None or len(recipe["groups"]) > 1:
        raise NotImplementedError(
            "Information reliability does not integrate covariate/group populations"
        )
    if any("person" in block.dims for block in bindings.prepared.blocks):
        raise NotImplementedError("Use posterior reliability for individual nuisance effects")
    if any(
        bindings.prepared.measurement_specs[unit["unit_index"]].comparison_based
        and unit["response_format"] in {"rank", "pick", "mole"}
        and unit["size"] > 2
        for unit in constant.attrs["units"]
    ):
        raise NotImplementedError(
            "Pairwise composite information is not joint ranking Fisher information"
        )
    theta_hat = jnp.asarray(point.theta.values)
    n_persons, n_traits = theta_hat.shape
    if n_persons < 2:
        raise ValueError("Information reliability requires at least two persons")
    covariance = jnp.asarray(point["covariance"].values).reshape(n_traits, n_traits)
    if bool(jnp.any(point["population_mean"].values != 0)):
        raise NotImplementedError(
            "Information reliability currently requires a zero-mean population"
        )
    precision = jnp.linalg.inv(covariance)
    components, empirical_information = [], jnp.zeros((n_persons, n_traits, n_traits))
    for response, parameters in bindings.iter_response_parameters(point):
        batch = response.batch
        # Each batch uses one unit. Match its statement rows within each person's form.
        order = np.lexsort((*batch.statement.T[::-1], batch.person))
        counts = np.bincount(batch.person, minlength=n_persons)
        selections = np.split(order, np.cumsum(counts[:-1]))
        first = selections[0]
        # The existing batch defines support. Compare actual prepared rows, without
        # inventing a common form or filling comparisons absent for any person.
        if not len(first) or any(
            len(indices) != len(first)
            or not np.array_equal(batch.statement[indices], batch.statement[first])
            for indices in selections[1:]
        ):
            raise ValueError("All persons must have the same actual answered form")
        ability = theta_hat[jnp.asarray(batch.person[:, None]), jnp.asarray(batch.trait)]
        matrices, _, _ = _response_fisher(
            response, ability, parameters, n_traits=n_traits, chunk_size=chunk_size
        )
        empirical_information = empirical_information.at[batch.person].add(matrices)
        selected = tuple(value[jnp.asarray(first)] if value.ndim else value for value in parameters)
        components.append((response, batch.trait[first], selected))
    if not components:
        raise ValueError("Information reliability requires an answered form")

    def variances(information):
        return jnp.diagonal(jnp.linalg.inv(information + precision), axis1=-2, axis2=-1)

    empirical_variance = variances(empirical_information).mean(axis=0)
    if not bool(jnp.isfinite(empirical_variance).all()) or bool(jnp.any(empirical_variance < 0)):
        raise ArithmeticError("The information covariance is not finite and positive")
    score_variance = theta_hat.var(axis=0, ddof=1)
    if bool(jnp.any(score_variance + empirical_variance <= 0)):
        raise ValueError("Information reliability has an undefined denominator")
    high, low = jnp.zeros((scrambles, n_traits)), jnp.zeros((scrambles, n_traits))
    factor = jnp.linalg.cholesky(covariance)
    for scramble in range(scrambles):
        uniforms = qmc.Sobol(n_traits, scramble=True, seed=random_seed + scramble).random_base2(
            refine_points.bit_length() - 1
        )
        population = ndtri(jnp.asarray(uniforms)) @ factor.T
        for start in range(0, refine_points, chunk_size):
            ability = population[start : start + chunk_size]
            count = len(ability)
            total = jnp.zeros((count, n_traits, n_traits))
            for response, traits, parameters in components:
                rows = len(traits)
                aligned = tuple(
                    jnp.broadcast_to(value, (count, *value.shape)).reshape(
                        count * rows, *value.shape[1:]
                    )
                    if value.ndim
                    else value
                    for value in parameters
                )
                indices = jnp.tile(jnp.asarray(traits), (count, 1))
                grid = ability[:, traits].reshape(count * rows, traits.shape[-1])
                matrices, _, _ = _response_fisher(
                    response,
                    grid,
                    aligned,
                    n_traits=n_traits,
                    trait_index=indices,
                    chunk_size=chunk_size,
                )
                total += matrices.reshape(count, rows, n_traits, n_traits).sum(axis=1)
            variance = variances(total)
            if not bool(jnp.isfinite(variance).all()) or bool(jnp.any(variance < 0)):
                raise ArithmeticError(
                    "Information covariance is nonfinite or has a negative variance"
                )
            high = high.at[scramble].add(variance.sum(axis=0))
            low = low.at[scramble].add(variance[: max(0, min(count, points - start))].sum(axis=0))
    high /= refine_points
    low /= points
    prior_variance = jnp.diag(covariance)
    marginal_by_scramble = 1 - high / prior_variance
    marginal = marginal_by_scramble.mean(axis=0)
    marginal_low = 1 - low.mean(axis=0) / prior_variance
    values = dict(
        empirical=score_variance / (score_variance + empirical_variance),
        marginal=marginal,
        score_variance=score_variance,
        empirical_information_variance=empirical_variance,
        marginal_information_variance=high.mean(axis=0),
        prior_variance=prior_variance,
        marginal_low=marginal_low,
        refinement_delta=marginal - marginal_low,
        scramble_sd=marginal_by_scramble.std(axis=0, ddof=1),
    )
    return xr.Dataset(
        {
            **{name: ("trait", np.asarray(value)) for name, value in values.items()},
            "marginal_by_scramble": (("scramble", "trait"), np.asarray(marginal_by_scramble)),
        },
        coords={"trait": constant.trait.values, "scramble": np.arange(scrambles)},
        attrs=dict(
            method="experimental_information_based_reliability",
            inference_method="mcmc",
            covariance_formula="inverse(full_information+inverse(population_covariance))",
            marginal_formula="1-E_Normal(0,R)[V_dd(theta)]/R_dd",
            empirical_formula="Var(theta_hat,ddof=1)/(Var(theta_hat,ddof=1)+mean(V_dd))",
            parameter_point="posterior_mean" if draw is None else list(draw),
            points_per_scramble=points,
            refine_points_per_scramble=refine_points,
            scrambles=scrambles,
            random_seed=random_seed,
            chunk_size=chunk_size,
            calibration_uncertainty="excluded",
            ddof=1,
            integration="nested scrambled Sobol; refinement is not a guaranteed error bound",
            common_form_units=sum(len(traits) for _, traits, _ in components),
        ),
    )
