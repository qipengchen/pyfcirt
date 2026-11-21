"""Construct each frozen regression once, before its final parameter targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytensor.tensor as pt
from pytensor import wrap_jax
from pytensor.tensor.variable import TensorVariable

from pyfcirt.models.graph import RegressionBinding

if TYPE_CHECKING:
    from pyfcirt.data.coordinates import PreparedRegression


def _measurement_mean(matrix, coefficients):
    return jnp.matmul(matrix, coefficients.T)


def _theta_mean(matrix, coefficients, rows, pools):
    """Select each person's group before multiplying to avoid a group grid."""
    return jnp.einsum("...t,...t->...", matrix[rows], coefficients[pools])


def build_regressions(
    plans: tuple[PreparedRegression, ...],
    parameters: Mapping[str, TensorVariable],
) -> Mapping[tuple[int, str], RegressionBinding]:
    """Create beta/tau and one design product from already resolved plans."""
    target_bindings = {}
    for plan in plans:
        coefficients = pt.reshape(parameters["regression_coefficient"], (-1,))[
            plan.coefficient_entries
        ]
        matrix = pt.as_tensor_variable(plan.design.X.values, dtype=coefficients.dtype)
        if plan.targets[0].kind == "theta":
            (target,) = plan.targets
            mean = wrap_jax(_theta_mean, allow_eval=False)(
                matrix, coefficients, target.row_index, target.pool_index
            )
            target_bindings[(target.binding_id, target.prior_role)] = RegressionBinding(
                mean=mean, scale=None, conditional=target.conditional, noncentered=False
            )
            continue
        scale_layout = next(block for block in plan.blocks if block.key == "residual_sd")
        scale = pt.reshape(parameters[scale_layout.kind], (-1,))[scale_layout.entry_index]
        means = wrap_jax(_measurement_mean, allow_eval=False)(matrix, coefficients)
        for target in plan.targets:
            pools = target.pool_index
            target_bindings[target.binding_id, target.prior_role] = RegressionBinding(
                mean=means[target.row_index, pools],
                scale=scale[pools],
                conditional=target.conditional,
                noncentered=target.noncentered,
            )
    return target_bindings
