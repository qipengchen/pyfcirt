"""Native single-item distributions with one shared graph and binding workflow."""

import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt
from jax.nn import log_softmax, softplus

from pyfcirt.models import categorical
from pyfcirt.models.graph import response_rv


class _ItemDistribution(pm.Distribution):
    """Share node construction and coordinates, without branching on model names."""

    @classmethod
    def bind(cls, spec, batch, *, ability, values):
        arguments = (ability, *(values[key] for key in cls.parameter_keys))
        return response_rv(cls.kernel, arguments, batch=batch)

    @classmethod
    def _dist(cls, *parameters, size=None):
        parameters = tuple(
            categorical._tensor(value, name=f"parameter {i}") for i, value in enumerate(parameters)
        )
        conditions, message = cls._constraints(*parameters)
        return response_rv(cls.kernel, parameters, size=size, checks=conditions, message=message)


class DichotomousIRTDistribution(_ItemDistribution):
    """One 1--4PL law; the prepared layout fixes unused loading/asymptote entries."""

    parameter_keys = ("loading_magnitude", "intercept", "lower_asymptote", "upper_asymptote")

    @classmethod
    def dist(cls, theta, loading, intercept, *, lower=0.0, upper=1.0, size=None, shape=None):
        """Intercept is subtracted from the trait dot product; it is not difficulty."""
        return cls._dist(
            theta,
            loading,
            intercept,
            lower,
            upper,
            size=categorical.response_size(shape=shape, size=size),
        )

    @staticmethod
    def kernel(theta, loading, intercept, lower, upper):
        eta = jnp.sum(jnp.asarray(theta) * jnp.asarray(loading), axis=-1) - intercept
        lower, upper = jnp.asarray(lower), jnp.asarray(upper)
        log_span = jnp.log(upper - lower)
        no = jnp.logaddexp(jnp.log1p(-upper), log_span - softplus(eta))
        yes = jnp.logaddexp(jnp.log(lower), log_span - softplus(-eta))
        return jnp.stack((no, yes), axis=-1)

    @staticmethod
    def _constraints(theta, loading, intercept, lower, upper):
        return (lower >= 0, lower < upper, upper <= 1), "0 <= lower < upper <= 1"


class GPCMDistribution(_ItemDistribution):
    """Unordered category intercepts with an exact zero baseline."""

    parameter_keys = ("loading_magnitude", "category_intercept")

    @classmethod
    def dist(cls, theta, loading, category_intercept, *, size=None, shape=None):
        return cls._dist(
            theta,
            loading,
            category_intercept,
            size=categorical.response_size(shape=shape, size=size),
        )

    @staticmethod
    def kernel(theta, loading, category_intercept):
        eta = jnp.sum(jnp.asarray(theta) * jnp.asarray(loading), axis=-1)
        category_intercept = jnp.asarray(category_intercept)
        logits = eta[..., None] * jnp.arange(category_intercept.shape[-1]) + category_intercept
        return log_softmax(logits, axis=-1)

    @staticmethod
    def _constraints(theta, loading, intercept):
        if intercept.ndim < 1 or intercept.type.shape[-1] < 2:
            raise ValueError("category_intercept requires a final category axis of length >= 2")
        return (pt.eq(intercept[..., 0], 0),), "category_intercept must start at zero"


class GGUMDistribution(_ItemDistribution):
    """Distance GGUM with signed locations and freely estimated step parameters."""

    parameter_keys = ("loading_magnitude", "location", "threshold")

    @classmethod
    def dist(cls, theta, loading, location, threshold, *, size=None, shape=None):
        return cls._dist(
            theta,
            loading,
            location,
            threshold,
            size=categorical.response_size(shape=shape, size=size),
        )

    @classmethod
    def kernel(cls, theta, loading, location, threshold):
        theta, loading = jnp.asarray(theta), jnp.asarray(loading)
        location, threshold = jnp.asarray(location), jnp.asarray(threshold)
        distance_squared = jnp.sum((loading * (theta - location)) ** 2, axis=-1)
        count = threshold.shape[-1]
        coefficients = (2 * count - 1) / 2 - jnp.arange(count)
        cumulative = jnp.sum(loading, axis=-1)[..., None] * jnp.cumsum(threshold, axis=-1)
        logits = cls._logcosh_sqrt(distance_squared[..., None] * coefficients**2) - cumulative
        return log_softmax(logits, axis=-1)

    @staticmethod
    def _constraints(theta, loading, location, threshold):
        if threshold.ndim < 1 or threshold.type.shape[-1] < 2:
            raise ValueError("threshold requires a final category axis of length >= 2")
        return (pt.eq(threshold[..., 0], 0),), "threshold must start at zero"

    @staticmethod
    def _logcosh_sqrt(squared):
        """Use the analytic series near zero; keep the unused sqrt branch finite."""
        squared = jnp.asarray(squared)
        series = squared * (0.5 + squared * (-1 / 12 + squared * (1 / 45 - 17 * squared / 2520)))
        root = jnp.sqrt(jnp.maximum(squared, 1e-4))
        exact = root + softplus(-2 * root) - jnp.log(2.0)
        return jnp.where(squared < 1e-4, series, exact)
