"""Generate one actual dataset from the existing native measurement distributions."""

from __future__ import annotations

from collections.abc import Mapping
from secrets import randbits
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import xarray as xr
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from pyfcirt.model import Model
    from pyfcirt.models.specification import _MeasurementSpec


def simulate(
    model: Model | _MeasurementSpec | str | None = None,
    *,
    source="prior",
    parameters: xr.Dataset | Mapping[str, ArrayLike] | None = None,
    draw=None,
    administration: xr.Dataset | None = None,
    random_seed: int | None = None,
    n_persons: int | None = None,
    n_traits: int | None = None,
    n_blocks: int | None = None,
    block_size: int | None = None,
    response_format: Literal["rank", "pick", "mole"] | None = None,
) -> Model:
    """Generate one real response draw and return an independent, unfitted Model.

    Without a prepared model, generate a balanced forced-choice design and
    simulate from its priors. Posterior generation selects one actual joint
    sample. Fixed generation accepts a labelled parameter point or a mapping
    of canonical parameter names to arrays. Existing people retain that point's
    theta and individual effects. An explicit generation_schedule declares a
    fresh population draw: only global parameters are fixed, and the same
    population graph draws new theta and required person effects.

    Without a schedule, existing administration and recorded nonresponse rows remain.
    TIRT/GTUM produce explicit comparisons, with no joint ranking interpretation.
    Truth is stored under parameters/generation, never as a posterior or as the
    fixed conditions of the returned model. No posterior MCMC is performed.
    """
    from pyfcirt.analysis.prediction import _jax_key, _native_forward, _seed
    from pyfcirt.data.prepare import (
        _calibration_values,
        _prepare_automatic_generation,
        _prepare_generated,
        _prepare_generation,
        _validate_parameters,
    )
    from pyfcirt.model import Model
    from pyfcirt.models.assembly import ModelGraph

    automatic = not isinstance(model, Model)
    if source not in {"prior", "posterior", "fixed"}:
        raise ValueError("source must be prior, posterior or fixed")
    if (source == "fixed") != (parameters is not None):
        raise ValueError("parameters must be supplied exactly when source='fixed'")
    if source != "posterior" and draw is not None:
        raise ValueError("draw is only valid for source='posterior'")
    design_options = n_persons, n_traits, n_blocks, block_size, response_format
    if automatic:
        if source != "prior" or administration is not None:
            raise ValueError("Automatic designs require source='prior' and no administration")
    elif any(value is not None for value in design_options):
        raise ValueError(
            "Automatic design options require a model name or declaration, or no model"
        )
    seed = _seed(random_seed)
    entropy = randbits(128) if seed is None else seed
    selection, forward = jax.random.split(_jax_key(entropy))
    forward_seed = int(jax.random.bits(forward, dtype=jnp.uint32))
    if automatic:
        prepared = _prepare_automatic_generation(
            model,
            n_persons=100 if n_persons is None else n_persons,
            n_traits=3 if n_traits is None else n_traits,
            n_blocks=12 if n_blocks is None else n_blocks,
            block_size=3 if block_size is None else block_size,
            response_format=response_format,
            seed=int(jax.random.bits(selection, dtype=jnp.uint32)),
        )
        # Preparation already owns the full graph contract. Only the completed
        # generated analysis needs a Model and its published result metadata.
        graph = ModelGraph.build(prepared)
        point = selected_draw = None
    else:
        prepared = model.prepared
        point, selected_draw = _generation_point(
            model, source, parameters, draw, selection, existing_people=administration is None
        )
        if source == "prior":
            model._prior_conditioning()
        if administration is None:
            graph = model._bindings
        else:
            fixed = (
                _calibration_values(prepared, point, existing=prepared.fixed_parameters)
                if point is not None
                else None
            )
            prepared = _prepare_generation(prepared, administration, fixed_parameters=fixed)
            approximation = model._calibration_prior if source == "prior" else None
            graph = ModelGraph.build(prepared, approximation=approximation)
    names = [*prepared.point_layouts, "theta"]
    response_names = [batch.response_name for batch in prepared.batches]
    if source == "prior" or administration is not None:
        # One native call shares the same joint parameter realization with its answers.
        combined = graph.evaluation().prior_draws(draws=1, random_seed=forward_seed)
    else:
        # These singleton axes adapt a real point to native forward sampling only.
        # They are never saved or presented as a posterior sample group.
        labels = selected_draw if selected_draw is not None else (0, 0)
        trace = point.expand_dims(chain=[labels[0]], draw=[labels[1]])
        combined = _native_forward(
            graph,
            trace,
            [*names, *response_names],
            sample_names=response_names,
            random_seed=forward_seed,
            progressbar=False,
        )
    one = combined.isel(chain=0, draw=0, drop=True)
    truth = one[names]
    _validate_parameters(prepared, truth, include_theta=True)
    state = _prepare_generated(prepared, one[response_names])
    if selected_draw is not None:
        truth = truth.assign_coords(source_chain=selected_draw[0], source_draw=selected_draw[1])
    truth.attrs = {
        "source": source,
        "random_seed": "unspecified" if seed is None else seed,
        "seed_entropy": str(entropy),
        "rng": "jax.random",
        "forward_seed": forward_seed,
        "source_inference_method": (
            model.idata.attrs.get("inference_method", "external")
            if source == "posterior"
            else "none"
        ),
        "person_parameters": "fresh population draw"
        if automatic or administration is not None
        else "same people",
        "response_law": "native measurement events; TIRT/GTUM are comparisons, not joint rankings",
    }
    result = Model(state)
    result._publish_outputs({"parameters/generation": truth})
    return result


def _generation_point(model, source, parameters, draw, selection, *, existing_people):
    """Select and align the supplied point or one actual posterior draw."""
    from pyfcirt.analysis.output import posterior_samples, read_parameter_point
    from pyfcirt.data.prepare import _simulation_parameter_point, _validate_parameters

    point = selected_draw = None
    if source == "posterior":
        posterior = posterior_samples(model.idata)
        if draw is None:
            index = int(
                jax.random.randint(
                    selection, (), 0, posterior.sizes["chain"] * posterior.sizes["draw"]
                )
            )
            draw = (
                posterior.chain.values[index // posterior.sizes["draw"]],
                posterior.draw.values[index % posterior.sizes["draw"]],
            )
        point = read_parameter_point(
            model, posterior=posterior, draw=draw, include_person=existing_people
        )
        selected_draw = tuple(draw)
    elif source == "fixed":
        point = _simulation_parameter_point(model.prepared, parameters)
        if existing_people:
            point = _validate_parameters(
                model.prepared,
                point,
                fixed_parameters=model._fixed_reference,
                include_theta=True,
            )

    return point, selected_draw
