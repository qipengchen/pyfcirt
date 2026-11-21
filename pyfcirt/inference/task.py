"""Continuous native BlackJAX tasks for validated conditional-mode proposals."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import numpy as np


class TaskInfo(NamedTuple):
    """Native NUTS diagnostics and optional invariant-move diagnostics."""

    nuts: object
    extra: object


def warmup(
    key,
    position,
    *,
    logdensity,
    tune,
    target_accept=0.8,
    initial_step_size=1.0,
    dense_mass=False,
    max_num_doublings=10,
):
    """Reuse native window adaptation without retaining unused warmup histories."""
    adaptation = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        is_mass_matrix_diagonal=not dense_mass,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_accept,
        adaptation_info_fn=lambda state, info, adaptation: None,
        max_num_doublings=max_num_doublings,
    )
    result, _ = adaptation.run(key, position, num_steps=tune)
    return result.state, result.parameters


def run_chain(
    key,
    state,
    *,
    logdensity,
    draws,
    step_size,
    inverse_mass_matrix,
    max_num_doublings=10,
    extra_step=None,
):
    """Compose native NUTS with an optional independently valid extra transition."""
    kernel = blackjax.nuts(
        logdensity, step_size, inverse_mass_matrix, max_num_doublings=max_num_doublings
    ).step

    def retained(state, key):
        nuts_key, extra_key = jax.random.split(key)
        state, info = kernel(nuts_key, state)
        diagnostics = None
        if extra_step is not None:
            position, diagnostics = extra_step(extra_key, state.position)
            # An extra move changes both the cached density and gradient.
            state = blackjax.nuts.init(position, logdensity)
        return state, (state, TaskInfo(info, diagnostics))

    return jax.lax.scan(retained, state, jax.random.split(key, draws))


@dataclass(frozen=True)
class TaskGraph:
    """One native continuous task target and output compilation.

    The task contains only the required new evidence and conditional priors, or
    the caller selects factors explicitly. No FC layout or model names enter
    this bridge. Runtime extra outputs use original RV coordinates, not old
    transformed value variables.
    """

    model: object
    value_names: tuple[str, ...]
    output_names: tuple[str, ...]
    evaluate: Callable

    @classmethod
    def from_model(
        cls,
        model,
        *,
        terms=None,
        output_names=None,
        extra_outputs=None,
    ):
        """Clone/bind a task once and compile its target and requested expressions."""
        from pymc.model.fgraph import clone_model
        from pymc.pytensorf import expand_inner_graph
        from pymc.sampling.jax import get_jaxified_graph
        from pytensor.compile.mode import JAX
        from pytensor.compile.sharedvalue import SharedVariable
        from pytensor.graph.fg import FunctionGraph
        from pytensor.graph.replace import graph_replace
        from pytensor.graph.traversal import ancestors, graph_inputs, walk
        from pytensor.raise_op import CheckAndRaise
        from pytensor.tensor.random.type import RandomType

        extra = {} if extra_outputs is None else dict(extra_outputs)
        registered = {*model.data_vars, *model.dim_lengths.values()}
        if any(
            isinstance(value, SharedVariable)
            and not isinstance(value.type, RandomType)
            and value not in registered
            for value in graph_inputs(
                [
                    *model.basic_RVs,
                    *model.deterministics,
                    *model.potentials,
                    *model.rvs_to_values.values(),
                    *extra.values(),
                ]
            )
        ):
            raise ValueError("Register shared task inputs with pm.Data before graph construction")
        native = clone_model(model)
        logp = native.logp(
            None if terms is None else [native[name] for name in terms], jacobian=True
        )
        candidates = (
            [native[name] for name in output_names]
            if output_names is not None
            else native.free_RVs + native.deterministics
        )
        names = [node.name for node in candidates]
        if len(names) != len(set(names)) or set(names).intersection(extra):
            raise ValueError("Task output names must be distinct from runtime and row outputs")
        relocated = (
            graph_replace(
                list(extra.values()),
                replace={node: native[name] for name, node in model.named_vars.items()},
                strict=False,
            )
            if extra
            else []
        )
        expressions = native.replace_rvs_by_values([*candidates, *relocated])
        names.extend(extra)
        graph = FunctionGraph(outputs=[logp, *expressions], clone=False)
        # Native JAX mode propagates bounds preservation into nested graph bodies.
        JAX.including("local_check_parameter_to_ninf_switch").optimizer.rewrite(graph)
        logp, *expressions = graph.outputs
        if any(
            node.owner and isinstance(node.owner.op, CheckAndRaise)
            for node in walk(graph.outputs, expand_inner_graph)
        ):
            raise ValueError("The task contains an assertion without a JAX support rewrite")
        active = set(ancestors([logp]))
        continuous = [v for v in native.value_vars if v in active]
        if not continuous or any(not value.dtype.startswith("float") for value in continuous):
            raise ValueError("The task must have continuous variables and no other discrete RVs")
        unused = set(native.value_vars) - set(continuous)
        selected, outputs = [], []
        for name, expression in zip(names, expressions, strict=True):
            if unused.intersection(ancestors([expression])):
                if output_names is not None or name in extra:
                    raise ValueError(f"{name}: output depends on variables outside the target")
                continue
            selected.append(name)
            outputs.append(expression)
        evaluate = get_jaxified_graph(continuous, [logp, *outputs])
        return cls(native, tuple(v.name for v in continuous), tuple(selected), evaluate)

    def logdensity(self, position):
        """Evaluate the original transformed task density, including Potentials."""
        return self.evaluate(*(position[name] for name in self.value_names))[0]

    def outputs(self, position):
        """Evaluate final nodes and runtime expressions from the same graph."""
        values = self.evaluate(*(position[name] for name in self.value_names))[1:]
        return dict(zip(self.output_names, values, strict=True))

    def initial_positions(self, seeds, *, initvals=None):
        """Use PyMC's per-chain initializers with one compilation for shared settings."""
        from pymc.initial_point import make_initial_point_fns_per_chain

        if isinstance(initvals, Mapping):
            overrides = dict(initvals)
        elif initvals is None:
            overrides = None
        else:
            overrides = list(initvals)
            if len(overrides) != len(seeds):
                raise ValueError("initvals must contain one native initial-value mapping per chain")
            overrides = [dict(values or {}) for values in overrides]
        initialize = make_initial_point_fns_per_chain(
            model=self.model, overrides=overrides, chains=len(seeds)
        )
        positions = []
        for function, seed in zip(initialize, seeds, strict=True):
            initial = function(seed)
            position = {name: jnp.asarray(initial[name]) for name in self.value_names}
            if not np.isfinite(self.logdensity(position)):
                raise ValueError("The native initial point has nonfinite task density")
            positions.append(position)
        return positions
