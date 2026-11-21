"""Shared comparison response laws and fixed-node Gaussian region integration."""

from functools import partial
from numbers import Integral

import jax.numpy as jnp
import numpy as np
import pytensor.tensor as pt
from jax import lax, vmap
from jax.nn import log_sigmoid
from jax.scipy.special import log_ndtr, ndtri_exp
from scipy.stats import qmc

from pyfcirt.models import categorical
from pyfcirt.models.graph import response_rv


def comparison_rv(
    kernel,
    arguments,
    direction,
    *,
    size,
    beta,
    precision,
    checks=(),
    restriction="",
    batch=None,
    smooth=None,
    **options,
):
    """Prepare one argument set for the native likelihood and Fisher information."""
    if beta != (precision is not None):
        raise ValueError(
            "Beta comparisons require precision"
            if beta
            else "Categorical comparisons do not accept precision"
        )
    arguments = {name: categorical._tensor(value, name=name) for name, value in arguments.items()}
    orientation = categorical._tensor(direction, name="direction")
    valid = pt.eq(pt.abs(orientation), 1)
    if beta:
        precision = categorical._tensor(precision, name="precision")
        arguments["precision"] = precision
        support = precision > 0
        message = "precision > 0"
    else:
        cutpoints = arguments["cutpoints"]
        if cutpoints.ndim < 1 or cutpoints.type.shape[-1] < 1:
            raise ValueError("cutpoints require a final ordered-boundary axis of length >= 1")
        support = cutpoints[..., 1:] > cutpoints[..., :-1]
        message = "cutpoints must be strictly increasing"
    return response_rv(
        kernel,
        tuple(arguments.values()),
        size=size,
        batch=batch,
        beta=beta,
        smooth=smooth,
        direction=orientation if batch is None else direction,
        checks=(valid, support, *checks),
        message="direction must be +/-1; " + message + restriction,
        **options,
    )


def utility_difference(utility):
    """Read the signed contrast on the prepared two-statement axis."""
    if utility.ndim < 1 or utility.shape[-1] != 2:
        raise ValueError("Comparison utilities require a final two-statement axis")
    return utility[..., 0] - utility[..., 1]


def interval_log_probs(boundaries, *, link="probit"):
    """Return ordered-interval log probabilities without subtracting rounded CDFs."""
    boundaries = jnp.asarray(boundaries)
    if link == "probit":
        logcdf = log_ndtr
    elif link == "logit":
        logcdf = log_sigmoid
    else:
        raise ValueError("Interval link must be 'probit' or 'logit'")
    first = logcdf(boundaries[..., :1])
    lower, upper = boundaries[..., :-1], boundaries[..., 1:]
    # Reflect right-tail intervals before evaluating either CDF branch.
    left = jnp.where(upper <= 0, lower, -upper)
    right = jnp.where(upper <= 0, upper, -lower)
    log_right = logcdf(right)
    middle = log_right + jnp.log(-jnp.expm1(logcdf(left) - log_right))
    return jnp.concatenate((first, middle, logcdf(-boundaries[..., -1:])), axis=-1)


def beta_shapes(eta, precision, *, link):
    """Compute complementary Beta shapes, exposing underflow rather than clipping it."""
    eta, precision = jnp.asarray(eta), jnp.asarray(precision)
    if link == "probit":
        logcdf = log_ndtr
    elif link == "logit":
        logcdf = log_sigmoid
    else:
        raise ValueError("Beta comparison link must be 'probit' or unscaled 'logit'")
    return precision * jnp.exp(logcdf(eta)), precision * jnp.exp(logcdf(-eta))


def sobol_rule(dimension, *, n_points, seed):
    """Validate a fixed-node rule before its first marginal likelihood is needed."""
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if (
        isinstance(n_points, bool)
        or not isinstance(n_points, Integral)
        or n_points < 128
        or n_points & (n_points - 1)
    ):
        raise ValueError("n_points must be a power of two, at least 128")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return partial(_sobol_nodes, int(dimension), int(n_points), int(seed))


def sobol_nodes(dimension, *, n_points=4096, seed=914):
    """Prepare fixed scrambled Sobol nodes for a dimension-D Gaussian integral."""
    return sobol_rule(dimension, n_points=n_points, seed=seed)()


def _sobol_nodes(dimension, n_points, seed):
    if dimension == 1:
        return np.empty((n_points, 0), dtype=np.float64)
    nodes = qmc.Sobol(dimension - 1, scramble=True, seed=seed).random_base2(
        n_points.bit_length() - 1
    )
    if not np.all((nodes > 0) & (nodes < 1)):
        raise ValueError("Sobol nodes must be strictly inside (0, 1); choose a different seed")
    return nodes


def region_logcdf(upper, mean, covariance, *, nodes, chunk_size=128):
    """Approximate P(X <= upper) by chunked Genz conditional integration.

    Batch axes broadcast; the final axis is Gaussian dimension D. ``nodes`` has
    shape (n_points, D-1). Independent regions are vmapped, while dimensions and
    node chunks use scans so no observation-by-node-by-dimension history is kept.
    No mass floor, normalization or alternate integration backend is applied.
    """
    upper, mean, covariance = jnp.asarray(upper), jnp.asarray(mean), jnp.asarray(covariance)
    dtype = jnp.result_type(upper, mean, covariance, 0.0)
    upper, mean, covariance = upper.astype(dtype), mean.astype(dtype), covariance.astype(dtype)
    nodes = jnp.asarray(nodes, dtype=dtype)
    dimension = mean.shape[-1]
    if covariance.shape[-2:] != (dimension, dimension) or upper.shape[-1] != dimension:
        raise ValueError("upper, mean and covariance must have aligned Gaussian dimensions")
    if nodes.ndim != 2 or nodes.shape[1] != dimension - 1 or nodes.shape[0] < 1:
        raise ValueError("nodes must have shape (n_points, dimension - 1)")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    limits = upper - mean
    if dimension == 1:
        return log_ndtr(limits[..., 0] / jnp.sqrt(covariance[..., 0, 0]))
    batch_shape = jnp.broadcast_shapes(limits.shape[:-1], covariance.shape[:-2])
    limits = jnp.broadcast_to(limits, (*batch_shape, dimension)).reshape(-1, dimension)
    chol = jnp.linalg.cholesky(covariance)
    chol = jnp.broadcast_to(chol, (*batch_shape, dimension, dimension)).reshape(
        -1, dimension, dimension
    )
    count = nodes.shape[0]
    padding = (-count) % chunk_size
    chunks = jnp.pad(nodes, ((0, padding), (0, 0)), constant_values=0.5).reshape(
        -1, chunk_size, dimension - 1
    )
    valid = (jnp.arange(count + padding) < count).reshape(-1, chunk_size)

    def integrate(limit, factor):
        def chunk_step(state, chunk):
            previous_max, total = state
            uniforms, keep = chunk
            log_uniforms = jnp.log(uniforms)

            def dimension_step(state, index):
                previous, log_weight = state
                shift = jnp.sum(previous * factor[index], axis=-1)
                mass = log_ndtr((limit[index] - shift) / factor[index, index])
                previous = lax.cond(
                    index < dimension - 1,
                    lambda values: values.at[:, index].set(
                        ndtri_exp(log_uniforms[:, index] + mass)
                    ),
                    lambda values: values,
                    previous,
                )
                return (previous, log_weight + mass), None

            initial = (
                jnp.zeros((chunk_size, dimension), dtype=limit.dtype),
                jnp.zeros((chunk_size,), dtype=limit.dtype),
            )
            (_, log_weight), _ = lax.scan(dimension_step, initial, jnp.arange(dimension))
            # Padding belongs only to the final numerical chunk, never to response support.
            weights = jnp.where(keep, log_weight, -jnp.inf)
            maximum = lax.stop_gradient(jnp.maximum(previous_max, jnp.max(weights)))
            shift = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
            total = total * jnp.exp(previous_max - shift) + jnp.exp(weights - shift).sum()
            return (maximum, total), None

        (maximum, total), _ = lax.scan(
            chunk_step,
            (jnp.asarray(-jnp.inf, dtype=limit.dtype), jnp.asarray(0.0, dtype=limit.dtype)),
            (chunks, valid),
        )
        # A shifted positive-weight average stays <= 1 even near certain events;
        # subtracting two separately rounded log normalizers can exceed it.
        return maximum + jnp.log(total / count)

    return vmap(integrate)(limits, chol).reshape(batch_shape)
