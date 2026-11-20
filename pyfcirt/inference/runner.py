"""One BlackJAX task runner and native result conversion; publication stays with callers."""

from collections.abc import Mapping
from functools import partial
from numbers import Integral, Real
from secrets import randbits
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from pyfcirt.inference.task import run_chain, warmup


def sample_task(
    graph,
    *,
    draws=1000,
    tune=1000,
    chains=4,
    random_seed=None,
    var_names=None,
    target_accept=0.8,
    initial_step_size=1.0,
    max_treedepth=10,
    dense_mass=False,
    initvals=None,
    progressbar=False,
    extra_step=None,
):
    """Sample an actual compiled task with separate warmup, execution and output timing.

    Kernels receive only state/keys/density. This host boundary validates controls,
    initializes native values, maps independent chains and converts named outputs.
    ``extra_step`` returns scalar statistics per retained iteration. FC callers
    attach their existing metadata through the common publication boundary.
    """
    from arviz_base import from_dict
    from pymc.backends.arviz import coords_and_dims_for_inferencedata, find_observations
    from pymc.sampling.jax import _postprocess_samples

    for name, value in (
        ("draws", draws),
        ("tune", tune),
        ("chains", chains),
        ("max_treedepth", max_treedepth),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(dense_mass, bool) or not isinstance(progressbar, bool):
        raise TypeError("dense_mass and progressbar must be booleans")
    if progressbar:
        raise ValueError("Specialized BlackJAX sampling requires progressbar=False")
    if (
        isinstance(target_accept, (bool, np.bool_))
        or not isinstance(target_accept, Real)
        or not 0 < target_accept < 1
    ):
        raise ValueError("target_accept must be strictly between zero and one")
    if (
        isinstance(initial_step_size, (bool, np.bool_))
        or not isinstance(initial_step_size, Real)
        or not np.isfinite(initial_step_size)
        or initial_step_size <= 0
    ):
        raise ValueError("initial_step_size must be finite and positive")
    if random_seed is not None and (
        isinstance(random_seed, (bool, np.bool_))
        or not isinstance(random_seed, Integral)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be None or a nonnegative integer")
    if isinstance(var_names, str):
        raise TypeError("var_names must be a sequence of native output names")
    available = set(graph.output_names).intersection(graph.model.named_vars)
    selected = (
        [name for name in graph.output_names if name in available]
        if var_names is None
        else list(var_names)
    )
    if not selected or len(set(selected)) != len(selected) or set(selected) - available:
        raise ValueError("Storage selection must contain distinct available native output names")
    seeds = np.random.SeedSequence(randbits(128) if random_seed is None else int(random_seed))
    key = jax.random.wrap_key_data(jnp.asarray(seeds.generate_state(2, dtype=np.uint32)))
    warm_key, draw_key = jax.random.split(key)
    if initvals is None or isinstance(initvals, Mapping):
        starts = [initvals] * chains
    else:
        starts = list(initvals)
        if len(starts) != chains:
            raise ValueError("initvals must contain one native initial-value mapping per chain")
    start = perf_counter()
    initial = [
        graph.initial_state(
            random_seed=int(seed.generate_state(1, dtype=np.uint32)[0]),
            initvals=values,
        )
        for seed, values in zip(seeds.spawn(chains), starts, strict=True)
    ]
    states = jax.tree.map(lambda *values: jnp.stack(values), *initial)
    initialization = perf_counter() - start
    parallel = jax.default_backend() == "cpu" and jax.local_device_count() >= chains
    mapper = jax.pmap if parallel else jax.vmap
    adapt = jax.jit(
        mapper(
            partial(
                warmup,
                logdensity=graph.logdensity,
                tune=int(tune),
                target_accept=float(target_accept),
                initial_step_size=float(initial_step_size),
                dense_mass=dense_mass,
                max_num_doublings=int(max_treedepth),
            )
        )
    )
    warm_keys, draw_keys = jax.random.split(warm_key, chains), jax.random.split(draw_key, chains)
    start = perf_counter()
    adapted = adapt.lower(warm_keys, states).compile()
    compile_seconds = perf_counter() - start
    start = perf_counter()
    states, tuning = adapted(warm_keys, states)
    jax.block_until_ready((states, tuning))
    warmup_seconds = perf_counter() - start

    def retained(key, state, step_size, inverse_mass_matrix):
        return run_chain(
            key,
            state,
            logdensity=graph.logdensity,
            draws=int(draws),
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=int(max_treedepth),
            extra_step=extra_step,
        )

    start = perf_counter()
    retain = (
        jax.jit(mapper(retained))
        .lower(draw_keys, states, tuning["step_size"], tuning["inverse_mass_matrix"])
        .compile()
    )
    compile_seconds += perf_counter() - start
    start = perf_counter()
    _, (samples, info) = retain(
        draw_keys, states, tuning["step_size"], tuning["inverse_mass_matrix"]
    )
    jax.block_until_ready((samples, info))
    sampling_seconds = perf_counter() - start
    start = perf_counter()

    def transform(*values):
        logp, *outputs = graph.evaluate(*values)
        named = dict(zip(graph.output_names, outputs, strict=True))
        return [logp, *(named[name] for name in selected)]

    values = _postprocess_samples(
        transform,
        [samples.position[name] for name in graph.value_names],
        postprocessing_vectorize="scan",
        donate_samples=False,
    )
    statistics = dict(
        lp=values[0],
        diverging=info.nuts.is_divergent,
        nuts_energy=info.nuts.energy,
        tree_depth=info.nuts.num_trajectory_expansions,
        n_steps=info.nuts.num_integration_steps,
        acceptance_rate=info.nuts.acceptance_rate,
    )
    if extra_step is not None:
        if set(statistics).intersection(info.extra):
            raise ValueError("Extra transition statistics conflict with native sampler names")
        statistics.update(info.extra)
    _, dims = coords_and_dims_for_inferencedata(graph.model)
    # Native tuples can contain e.g. integer 1 and string "1". A native Index
    # preserves those identities where NumPy's default tuple conversion does not.
    coords = {
        name: pd.Index(labels) for name, labels in graph.model.coords.items() if labels is not None
    }
    result = from_dict(
        data=dict(
            posterior={
                name: np.asarray(value) for name, value in zip(selected, values[1:], strict=True)
            },
            sample_stats={name: np.asarray(value) for name, value in statistics.items()},
            observed_data=find_observations(graph.model),
        ),
        coords=coords,
        dims=dims,
        sample_dims=["chain", "draw"],
    )
    result["sample_stats"].attrs["energy_scope"] = "NUTS substep only; not full-chain BFMI"
    return result, dict(
        initialization_seconds=initialization,
        compile_seconds=compile_seconds,
        warmup_seconds=warmup_seconds,
        sampling_seconds=sampling_seconds,
        postprocessing_seconds=perf_counter() - start,
        random_seed=int(seeds.entropy),
        tuned_step_sizes=np.asarray(tuning["step_size"]).tolist(),
        chain_method="parallel" if parallel else "vectorized",
        actual_platform=jax.default_backend(),
    )
