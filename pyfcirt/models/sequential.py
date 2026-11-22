"""Two distribution classes sharing one successive-choice ranking calculation."""

import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt
from jax import lax

from pyfcirt.models import categorical
from pyfcirt.models.graph import response_rv


def sequential_log_probs(utility, orders, membership=None):
    """Sum successive choices, then aggregate complete orders into observed events."""
    ordered = jnp.take(jnp.asarray(utility), jnp.asarray(orders), axis=-1)
    remaining = lax.associative_scan(jnp.logaddexp, ordered, axis=ordered.ndim - 1, reverse=True)
    # The final remaining statement is selected with probability one.
    pattern_logp = jnp.sum(ordered[..., :-1] - remaining[..., :-1], axis=-1)
    return categorical.event_log_probs(pattern_logp, membership)


class _SequentialDistribution(pm.Distribution):
    """Share construction/binding; each concrete model supplies its endorsement law."""

    @classmethod
    def bind(cls, spec, batch, *, ability, values):
        loading = values["loading_magnitude"]
        if cls.keyed:
            loading = loading * pt.cast(batch.keying, loading.dtype)
        arguments = (
            categorical._tensor(ability, name="theta"),
            loading,
            *(values[key] for key in cls.offset_keys),
        )
        return response_rv(
            cls.kernel,
            arguments,
            batch=batch,
            orders=batch.events.orders,
            membership=batch.events.membership,
        )

    @classmethod
    def _create(cls, *parameters, orders, membership=None, size=None):
        parameters = tuple(
            categorical._tensor(value, name=f"parameter {i}") for i, value in enumerate(parameters)
        )
        orders = categorical.ranking_orders(orders)
        membership = categorical.event_membership(membership, orders, name="orders")
        return response_rv(cls.kernel, parameters, orders=orders, membership=membership, size=size)

    @classmethod
    def kernel(cls, theta, *parameters, orders, membership=None):
        return sequential_log_probs(cls._utility(theta, *parameters), orders, membership)


class TwoPLRankDistribution(_SequentialDistribution):
    """Linear endorsement with successive-choice ranking events."""

    keyed = True
    offset_keys = ("intercept",)

    @classmethod
    def dist(cls, theta, loading, intercept, *, orders, membership=None, size=None, shape=None):
        return cls._create(
            theta,
            loading,
            intercept,
            orders=orders,
            membership=membership,
            size=categorical.response_size(shape=shape, size=size),
        )

    @staticmethod
    def _utility(theta, loading, intercept):
        return jnp.asarray(loading) * jnp.asarray(theta) - jnp.asarray(intercept)


class GGUMRankDistribution(_SequentialDistribution):
    """GGUM endorsement; tau retains its declared sign, without ordering or repair."""

    keyed = False
    offset_keys = ("location", "threshold")

    @classmethod
    def dist(
        cls, theta, loading, location, threshold, *, orders, membership=None, size=None, shape=None
    ):
        return cls._create(
            theta,
            loading,
            location,
            threshold,
            orders=orders,
            membership=membership,
            size=categorical.response_size(shape=shape, size=size),
        )

    @staticmethod
    def _utility(theta, loading, location, threshold):
        loading = jnp.asarray(loading)
        x = loading * (jnp.asarray(theta) - jnp.asarray(location))
        return jnp.logaddexp(x, 2 * x) - loading * threshold - jnp.logaddexp(0, 3 * x)
