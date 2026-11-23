"""Construct each frozen regression once, before its final parameter targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytensor.tensor as pt
from pytensor import wrap_jax
from pytensor.tensor.variable import TensorVariable

if TYPE_CHECKING:
    from pyfcirt.data.coordinates import PreparedRegression


@dataclass(frozen=True, slots=True, kw_only=True)
class RegressionBinding:
    """Native conditional inputs in the prepared target's active order."""

    mean: object
    scale: object | None
    conditional: Mapping
    noncentered: bool


def _measurement_mean(matrix, coefficients):
    return jnp.matmul(matrix, coefficients.T)


def indexed_mean(matrix, coefficients, rows, pools):
    """Gather the active design and coefficient rows before taking their products."""
    return jnp.einsum("...t,...t->...", matrix[rows], coefficients[pools])


def effective_coefficients(plan, coefficients, scale, *, tensor):
    """Convert standardized draws to the fitted design's coefficient units."""
    if scale is None:
        return coefficients
    return coefficients * tensor.where(plan.standardized_coefficients, scale[..., :, None], 1.0)


def offset_parameter(baseline, offset, link, *, tensor):
    """Apply a respondent offset on the prepared parameter's link scale."""
    return baseline * tensor.exp(offset) if link == "log" else baseline + offset


def build_regressions(
    plans: tuple[PreparedRegression, ...],
    parameters: Mapping[str, TensorVariable],
) -> tuple[
    Mapping[tuple[int, str], RegressionBinding],
    Mapping[str, tuple[TensorVariable, TensorVariable]],
]:
    """Bind the coefficient and design nodes once for both regression layers."""
    target_bindings, person_effects = {}, {}
    for plan in plans:
        coefficients = pt.reshape(parameters["regression_coefficient"], (-1,))[
            plan.coefficient_entries
        ]
        matrix = pt.as_tensor_variable(plan.design.X.values, dtype=coefficients.dtype)
        if plan.targets[0].kind == "theta":
            (target,) = plan.targets
            mean = wrap_jax(indexed_mean, allow_eval=False)(
                matrix, coefficients, target.row_index, target.pool_index
            )
            target_bindings[(target.binding_id, target.prior_role)] = RegressionBinding(
                mean=mean, scale=None, conditional=target.conditional, noncentered=False
            )
            continue
        scale_layout = next((block for block in plan.blocks if block.key == "residual_sd"), None)
        scale = (
            None
            if scale_layout is None
            else parameters[scale_layout.kind].ravel()[scale_layout.entry_index]
        )
        if plan.person_effects:
            person_effects[plan.name] = (
                matrix,
                effective_coefficients(plan, coefficients, scale, tensor=pt),
            )
            continue
        means = wrap_jax(_measurement_mean, allow_eval=False)(matrix, coefficients)
        for target in plan.targets:
            pools = target.pool_index
            target_bindings[target.binding_id, target.prior_role] = RegressionBinding(
                mean=means[target.row_index, pools],
                scale=scale[pools],
                conditional=target.conditional,
                noncentered=target.noncentered,
            )
    return target_bindings, person_effects


def apply_person_effects(values, effects, bindings):
    """Apply each bound effect only to matched observed parameter positions."""
    for effect in effects:
        matrix, coefficients = bindings[effect.regression_name]
        offset = wrap_jax(indexed_mean, allow_eval=False)(
            matrix, coefficients, effect.person, effect.pool
        )
        baseline = values[effect.position]
        actual = offset_parameter(baseline, offset, effect.link, tensor=pt)
        values = pt.set_subtensor(values[effect.position], actual)
    return values
