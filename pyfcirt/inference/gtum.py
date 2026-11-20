"""GTUM applicability and proposal geometry for the shared native BlackJAX runner."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _require_gtum(bindings) -> None:
    """Check the actual supported target without rebuilding a comparison model."""
    from pytensor.graph.traversal import ancestors

    from pyfcirt.models.specification import GTUM

    prepared, model = bindings.prepared, bindings.model
    if any(
        not isinstance(spec, GTUM) or spec.link != "logit" for spec in prepared.measurement_specs
    ):
        raise ValueError("Conditional MH requires original logistic GTUM measurements only")
    population = prepared.population
    if (
        population.theta_parameterization != "centered"
        or population.recipe["regression"] is not None
        or population.recipe["groups"]
    ):
        raise NotImplementedError(
            "Conditional MH requires a centered normal population without covariates/groups"
        )
    theta = bindings.parameters["theta"]
    if theta not in set(model.free_RVs) or model.rvs_to_transforms[theta] is not None:
        raise ValueError("Conditional MH requires the native untransformed theta RV")
    if model.potentials or set(model.observed_RVs) != {
        response.distribution for response in bindings.responses
    }:
        raise ValueError("Conditional MH does not support additional likelihoods or Potentials")
    nodes = list(bindings.parameters.values())
    if any(model.named_vars.get(node.name) is not node for node in nodes):
        raise ValueError(
            "Conditional MH requires the existing, unmodified final parameter bindings"
        )
    if set(model.free_RVs) - set(ancestors(nodes)):
        raise ValueError("Conditional MH does not support additional free variables")
    if any(
        response.batch.y is None or theta in set(ancestors(response.parameters))
        for response in bindings.responses
    ):
        raise ValueError(
            "Conditional MH requires observed comparisons and "
            "theta-independent measurement parameters"
        )


def _trait_slots(bindings):
    """Locate the existing final location entries, padded only for the dimension scan."""
    constant = bindings.prepared.constant_data
    layout = bindings.prepared.parameter_layouts["location"]
    traits = layout.entry_data["trait_index"].values
    count = np.bincount(traits, minlength=constant.sizes["trait"])
    width = max(1, int(count.max(initial=0)))
    slots, present = np.zeros((len(count), width), np.int64), np.zeros((len(count), width), bool)
    for trait, size in enumerate(count):
        slots[trait, :size] = np.flatnonzero(traits == trait)
        present[trait, :size] = True
    return jnp.asarray(slots), jnp.asarray(present)


def _conditional_task(bindings):
    """Get person factors and proposal bounds from the already declared native laws."""
    from pyfcirt.inference.task import TaskGraph

    model = bindings.model
    theta = bindings.parameters["theta"]
    count = bindings.prepared.constant_data.sizes["person"]
    traits = bindings.prepared.constant_data.sizes["trait"]
    person_logp = pm.logp(theta, theta)
    bound = pt.zeros(count * traits)
    for response in bindings.responses:
        batch = response.batch
        person_logp = pt.inc_subtensor(
            person_logp[batch.person], pm.logp(response.distribution, batch.y)
        )
        # Ordered logistic log scores have magnitude <=1. This is a GTUM
        # proposal bound only; the actual target remains the native log density.
        index = (batch.person[:, None] * traits + batch.trait).ravel()
        bound = pt.inc_subtensor(bound[index], pt.abs(response.parameters[0]).ravel())
    return TaskGraph.from_model(
        model,
        extra_outputs={
            "__mode_person_logp": person_logp,
            "__mode_slope_bound": pt.reshape(bound, (count, traits)),
        },
    )


def _mode_update(graph, bindings):
    """Supply verified GTUM geometry; generic proposal/MH code knows no model names."""
    from pyfcirt.inference.conditional_mode import ModeProposal, independence_step

    slots, present = _trait_slots(bindings)
    theta_name = graph.model.rvs_to_values[graph.model[bindings.parameters["theta"].name]].name
    location_name = bindings.parameters["location"].name
    mean_name, covariance_name = (
        bindings.parameters["population_mean"].name,
        bindings.parameters["covariance"].name,
    )

    def update(key, position):
        values = graph.outputs(position)
        initial_theta = position[theta_name]
        mean = values[mean_name].reshape((-1, initial_theta.shape[-1]))[0]
        covariance = values[covariance_name]
        precision = jnp.linalg.solve(
            covariance, jnp.eye(covariance.shape[-1], dtype=initial_theta.dtype)
        )
        locations = values[location_name].reshape(-1)
        bound = values["__mode_slope_bound"]

        def trait_step(theta, inputs):
            trait, key = inputs
            # Erasing the updated coordinate makes the independence condition
            # explicit. Other traits and all calibration parameters stay fixed.
            base_theta = theta.at[:, trait].set(0)
            centered = (theta - mean).at[:, trait].set(0)
            q = precision[trait, trait]
            mu = mean[trait] - centered @ precision[:, trait] / q
            radius = bound[:, trait] / q + 1
            knots = jnp.where(present[trait], locations[slots[trait]], jnp.inf)

            def target(value):
                updated = {**position, theta_name: base_theta.at[:, trait].set(value)}
                return graph.outputs(updated)["__mode_person_logp"]

            proposal = ModeProposal.from_target(
                target,
                breakpoints=knots,
                search_bounds=(mu - radius, mu + radius),
                tail_location=mu,
                tail_scale=1 / jnp.sqrt(q),
            )
            value, info = independence_step(
                key, theta[:, trait], logdensity=target, proposal=proposal
            )
            return theta.at[:, trait].set(value), info

        theta, info = jax.lax.scan(
            trait_step,
            initial_theta,
            (jnp.arange(initial_theta.shape[-1]), jax.random.split(key, initial_theta.shape[-1])),
        )
        return {**position, theta_name: theta}, dict(
            imh_acceptance=jnp.mean(info.is_accepted),
            imh_cross_interval=jnp.sum(info.crossed_region),
            imh_invalid=jnp.sum(info.invalid_proposal),
            imh_invalid_regions=jnp.sum(info.invalid_regions),
        )

    return update


def sample_gtum(
    model,
    *,
    draws=1000,
    tune=1000,
    chains=4,
    random_seed=None,
    var_names=None,
    target_accept=0.8,
    max_treedepth=10,
    initial_step_size=1.0,
    dense_mass=False,
    initvals=None,
    progressbar=False,
    compute_convergence_checks=True,
    overwrite=False,
):
    """Run NUTS and corrected regional-mode proposals through the common runner.

    This convenience interface supplies geometry for centered logistic GTUM
    without population groups/covariates. Generic native tasks can instead use
    ModeProposal and sample_task directly with their own validated geometry.
    Warmup is ordinary full-parameter NUTS; extra moves adapt nothing.
    """
    from pyfcirt.inference.runner import sample_task
    from pyfcirt.model import Model

    if not isinstance(model, Model):
        raise TypeError("sample_gtum requires a Model")
    if not isinstance(overwrite, bool) or not isinstance(compute_convergence_checks, bool):
        raise TypeError("overwrite and compute_convergence_checks must be booleans")
    if not jax.config.x64_enabled:
        raise ValueError("BlackJAX conditional MH requires JAX_ENABLE_X64")
    bindings = model._fit_bindings(overwrite)
    _require_gtum(bindings)
    selected = model._final_sampling_names(var_names)
    graph = _conditional_task(bindings)
    sampled, metadata = sample_task(
        graph,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        var_names=selected,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        initial_step_size=initial_step_size,
        dense_mass=dense_mass,
        initvals=initvals,
        progressbar=progressbar,
        extra_step=_mode_update(graph, bindings),
    )
    if compute_convergence_checks:
        from pymc.stats.convergence import log_warnings, run_convergence_checks

        log_warnings(run_convergence_checks(sampled, graph.model))
    return model._publish_samples(
        sampled,
        sampling_attrs={
            "sampling": json.dumps(
                dict(
                    engine="blackjax",
                    draws=int(draws),
                    tune=int(tune),
                    chains=int(chains),
                    target_accept=float(target_accept),
                    max_treedepth=int(max_treedepth),
                    dense_mass=dense_mass,
                    conditional_mode_experimental=True,
                    transition_order="full_parameter_NUTS_then_conditional_IMH",
                    conditional_mode_warmup="NUTS_only",
                    **metadata,
                )
            )
        },
    )
