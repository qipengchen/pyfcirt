"""Likelihood Fisher information from the measurement components' JAX kernels."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.scipy.special import polygamma
from rich.progress import track

from pyfcirt.analysis.output import read_parameter_point


def categorical_fisher(log_probability, theta):
    """Return probabilities, category scores and their weighted outer product."""

    def with_values(value):
        logs = log_probability(value)
        return logs, logs

    scores, logs = jax.jacrev(with_values, has_aux=True)(theta)
    probabilities = jnp.exp(logs)
    # Expectation over every possible category at this theta, rather than the
    # curvature for one recorded response or the posterior including its prior.
    return probabilities, scores, jnp.einsum("k,ki,kj->ij", probabilities, scores, scores)


def beta_fisher(shapes, theta):
    """Push native Beta shape information into the supplied ability coordinates."""

    def with_values(value):
        parameters = jnp.stack(shapes(value))
        return parameters, parameters

    derivative, parameters = jax.jacrev(with_values, has_aux=True)(theta)
    common = polygamma(1, parameters.sum())
    shape_information = jnp.diag(polygamma(1, parameters)) - common
    return parameters, derivative, derivative.T @ shape_information @ derivative


def _response_fisher(
    response, theta, parameters, *, n_traits, intermediates=False, trait_index=None
):
    """Hold actual final item/nuisance values fixed; differentiate only ability."""
    parameters = tuple(jnp.asarray(value) for value in parameters)
    n_obs = len(theta)
    if any(value.ndim and value.shape[0] != n_obs for value in parameters):
        raise ValueError(f"{response.batch.name}: component parameters lack an observation axis")
    axes = tuple(None if value.ndim == 0 else 0 for value in parameters)
    kind = response.batch.likelihood_kind
    fisher = beta_fisher if kind == "beta" else categorical_fisher

    theta = jnp.asarray(theta)
    projection = jax.nn.one_hot(
        response.batch.trait if trait_index is None else trait_index, n_traits, dtype=bool
    )

    def evaluate(**options):
        def one(ability, *values):
            return fisher(lambda point: response.kernel(point, *values, **options), ability)

        outputs, scores, local = jax.vmap(one, in_axes=(0, *axes))(theta, *parameters)
        # Sum derivatives from statements measuring the same trait.
        return (
            outputs,
            jnp.einsum("nks,nst->nkt", scores, projection),
            jnp.einsum("nsi,nst,ntj->nij", projection, local, projection),
        )

    outputs, scores, matrices = evaluate()
    diagnostics = {}
    if response.smooth is not None:
        smooth = jax.vmap(response.smooth, in_axes=(0, *axes))(theta, *parameters)
        if not bool(smooth.all()):
            raise ValueError(f"{response.batch.name}: information is undefined at a utility cusp")
    if kind == "rank":
        high_probability, high_scores, high = evaluate(n_points=16384)
        if not (
            jnp.allclose(outputs, high_probability, atol=5e-4, rtol=5e-3)
            and jnp.allclose(scores, high_scores, atol=5e-4, rtol=5e-3)
            and jnp.allclose(matrices, high, atol=5e-3, rtol=5e-3)
            and jnp.allclose(high_probability.sum(axis=-1), 1, atol=2e-3, rtol=0)
        ):
            raise ArithmeticError(f"{response.batch.name}: Gaussian information refinement failed")
        diagnostics = dict(
            probability_refinement_max=float(jnp.max(jnp.abs(high_probability - outputs))),
            score_refinement_max=float(jnp.max(jnp.abs(high_scores - scores))),
            information_refinement_max=float(jnp.max(jnp.abs(high - matrices))),
            integration_method="Genz Sobol 4096/16384; refinement is not an error bound",
        )
        outputs, scores, matrices = high_probability, high_scores, high
    if kind == "beta" and (not bool(jnp.isfinite(outputs).all()) or bool(jnp.any(outputs <= 0))):
        raise FloatingPointError(f"{response.batch.name}: Beta shapes must be finite and positive")
    if kind != "beta" and not jnp.allclose(
        outputs.sum(axis=-1),
        1,
        atol=2e-3 if kind == "rank" else 1e-6,
        rtol=0,
    ):
        raise ArithmeticError(f"{response.batch.name}: category probabilities do not sum to one")
    if not bool(jnp.isfinite(matrices).all() & jnp.isfinite(scores).all()):
        raise FloatingPointError(f"{response.batch.name}: information or derivatives are nonfinite")
    return matrices, (scores if intermediates else None), diagnostics


def _aggregate_information(matrices, person, unit, *, n_persons, n_units):
    """Use the single prepared observation relation, without regrouping responses."""
    shape = matrices.shape[-2:]
    block = jnp.zeros((n_persons, n_units, *shape), dtype=matrices.dtype)
    block = block.at[person, unit].add(matrices)
    return block, jnp.zeros((n_persons, *shape), dtype=matrices.dtype).at[person].add(matrices)


def _point(model, parameters, draw, theta, *, posterior=None):
    from pyfcirt.data.prepare import _constrain_parameter_point, _validate_parameters

    if parameters is not None and draw is not None:
        raise ValueError("parameters and draw are mutually exclusive")
    fixed = model._fixed_reference
    if parameters is None:
        point = read_parameter_point(model, posterior=posterior, draw=draw, theta=theta)
        return _constrain_parameter_point(point, model.prepared.parameter_layouts, fixed)
    return _validate_parameters(
        model.prepared,
        parameters,
        fixed_parameters=fixed,
        include_theta=True,
        theta=theta,
    )


def information(
    model,
    *,
    parameters=None,
    draw=None,
    theta=None,
    include_intermediates=False,
    store=True,
    progressbar=True,
):
    """Compute expected measurement information at one final parameter point.

    No ability-prior precision is added. Information for ordinary TIRT sums its
    fitted comparison factors; it is not a joint ranking Fisher or Godambe matrix.
    An explicit parameter point supports upstream MAP/VI/SMC analyses without
    introducing an estimator adapter or interpreting optimizer-space covariance.
    """
    if not isinstance(include_intermediates, bool) or not isinstance(store, bool):
        raise TypeError("include_intermediates and store must be boolean")
    point = _point(model, parameters, draw, theta)
    built = model._bindings
    constant = built.prepared.constant_data
    trait_count = constant.sizes["trait"]
    matrices = jnp.zeros((constant.sizes["obs"], trait_count, trait_count))
    extra, numerical_checks = {}, {}
    responses = track(
        built.iter_response_parameters(point),
        total=len(built.responses),
        description="Information",
        disable=not progressbar,
    )
    for response, aligned in responses:
        batch = response.batch
        ability = point.theta.values[batch.person[:, None], batch.trait]
        current, scores, diagnostics = _response_fisher(
            response, ability, aligned, n_traits=trait_count, intermediates=include_intermediates
        )
        matrices = matrices.at[batch.obs_slice].set(current)
        if diagnostics:
            numerical_checks[batch.name] = diagnostics
        if scores is not None:
            beta = batch.likelihood_kind == "beta"
            axis = f"{batch.name}_beta_parameter" if beta else batch.category_dim
            labels = ["alpha", "beta"] if beta else constant[axis].values
            label = "shape_jacobian" if beta else "scores"
            extra[f"{batch.name}_{label}"] = xr.DataArray(
                np.asarray(scores),
                dims=(batch.obs_dim, axis, "trait"),
                coords={
                    batch.obs_dim: constant[batch.obs_dim].values,
                    axis: labels,
                    "trait": constant.trait.values,
                },
            )
    block, test = _aggregate_information(
        matrices,
        constant.obs_person.values,
        constant.obs_unit.values,
        n_persons=constant.sizes["person"],
        n_units=constant.sizes["unit"],
    )
    result = xr.Dataset(
        extra,
        coords={
            "person": constant.person.values,
            "trait": constant.trait.values,
            "trait_aux": constant.trait_aux.values,
        },
    )
    for channel, axis in (("fc", "block"), ("items", "item")):
        selected = np.flatnonzero(constant.unit_channel.values == channel)
        if len(selected):
            result[f"{axis}_information"] = xr.DataArray(
                np.asarray(block[:, selected]),
                dims=("person", axis, "trait", "trait_aux"),
                coords={axis: constant[axis].values},
            )
    result["test_information"] = ("person", "trait", "trait_aux"), np.asarray(test)
    if include_intermediates:
        result["observation_information"] = (("obs", "trait", "trait_aux"), np.asarray(matrices))
        result = result.assign_coords(obs=constant.obs.values)
    result.attrs.update(
        information_kind="expected_likelihood_fisher",
        prior_information=0,
        parameter_source="explicit"
        if parameters is not None
        else "posterior_mean"
        if draw is None
        else "joint_draw",
        inference_method=point.attrs.get(
            "inference_method", model.idata.attrs.get("inference_method", "external")
        ),
        numerical_checks=json.dumps(numerical_checks),
        scope="sum of fitted measurement factors; TIRT uses working pairwise independence",
    )
    if store:
        model._publish_outputs({"information": result, "information/parameters": point})
    return result
