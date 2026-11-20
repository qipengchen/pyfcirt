"""Stable categorical density and aggregation of complete response patterns."""

import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt
from jax import lax
from jax.scipy.special import logsumexp
from pymc.distributions.shape_utils import convert_shape, convert_size, find_size
from pytensor import wrap_jax
from pytensor.tensor.basic import infer_static_shape


def response_size(*, shape=None, size=None):
    """Use PyMC's shape conversion for scalar categorical/Beta response batches."""
    if shape is not None and size is not None:
        raise ValueError("Specify only one of shape or size")
    return find_size(shape=convert_shape(shape), size=convert_size(size), ndim_supp=0)


def _tensor(value, *, name):
    """Give the native JAX bridge static shapes without evaluating model inputs."""
    value = pt.as_tensor_variable(value)
    shape = value.type.shape
    if None not in shape:
        return value
    _, shape = infer_static_shape(value.shape)
    if None in shape:
        raise ValueError(
            f"{name} requires a fully determined shape; use pt.specify_shape(value, shape)"
        )
    return pt.specify_shape(value, tuple(shape))


def ranking_orders(orders):
    """Validate the declared complete ranking catalogue at the PyMC boundary."""
    orders = _tensor(orders, name="orders")
    if orders.ndim != 2 or min(orders.type.shape) < 2:
        raise ValueError("orders must have shape (pattern, statement), with both counts >= 2")
    if not orders.dtype.startswith(("int", "uint")):
        raise TypeError("orders must contain integer statement positions")
    return orders


def event_membership(membership, patterns, *, name):
    """Validate the shared Boolean event-by-pattern axes without evaluating data."""
    if membership is None:
        return None
    membership = _tensor(membership, name="membership")
    if membership.dtype != "bool" or membership.ndim < 2:
        raise ValueError("membership must be a Boolean (..., event, pattern) tensor")
    if membership.type.shape[-1] != patterns.type.shape[0]:
        raise ValueError(f"membership and {name} must have the same pattern count")
    return membership


def event_log_probs(pattern_logp, membership=None):
    """Sum mutually exclusive patterns into the supplied nonempty event partition.

    ``membership`` is Boolean (..., event, pattern). Its leading axes broadcast
    with those of ``pattern_logp``; None preserves the complete-pattern support.
    The partition is supplied by the caller, without clipping or normalization.
    """
    pattern_logp = jnp.asarray(pattern_logp)
    if membership is None:
        return pattern_logp
    membership = jnp.asarray(membership)
    selected = jnp.where(membership, pattern_logp[..., None, :], -jnp.inf)
    nonempty = jnp.any(membership, axis=-1)
    # Empty complements have exactly zero mass. Keep log(0) out of the
    # differentiated branch so a full event retains its zero adjoint.
    mass = logsumexp(jnp.where(nonempty[..., None], selected, 0.0), axis=-1)
    return jnp.where(nonempty, mass, -jnp.inf)


def selected_logp(value, logp):
    """Select categories on the log scale after broadcasting their batch axes."""
    value, logp = jnp.asarray(value), jnp.asarray(logp)
    # The gather broadcasts leading axes without tiling the category table.
    rank = max(value.ndim + 1, logp.ndim)
    value = lax.broadcast_to_rank(value, rank - 1)
    logp = lax.broadcast_to_rank(logp, rank)
    valid = (
        jnp.isfinite(value) & (value >= 0) & (value < logp.shape[-1]) & (value == jnp.floor(value))
    )
    # Invalid values use a safe index only; they still receive zero probability.
    index = jnp.where(valid, value, 0).astype(jnp.int64)
    selected = jnp.take_along_axis(logp, index[..., None], axis=-1)[..., 0]
    return jnp.where(valid, selected, -jnp.inf)


class StableCategorical:
    """One stable category density paired with native PyMC generation."""

    @classmethod
    def dist(cls, logp, *, size=None):
        return pm.CustomDist.dist(
            logp, dist=cls.random_dist, logp=cls.logp, signature="(k)->()", size=size
        )

    @staticmethod
    def logp(value, logp):
        """Select log probabilities without exponentiating the observed tail."""
        value, logp = _tensor(value, name="value"), _tensor(logp, name="logp")
        return wrap_jax(selected_logp, allow_eval=False)(value, logp)

    @staticmethod
    def random_dist(logp, size):
        return pm.Categorical.dist(logit_p=logp, size=size)
