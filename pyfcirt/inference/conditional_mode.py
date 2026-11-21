"""Regional normal proposals and native independence-MH for explicit scalar blocks.

Arrays represent conditionally independent scalar blocks, not a general joint
Hessian. Callers establish separability, regional concavity and tail coverage.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from blackjax.mcmc.proposal import safe_energy_diff, static_binomial_sampling
from blackjax.mcmc.random_walk import RWState
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm


class ModeProposal(NamedTuple):
    """A normalized, untruncated normal mixture in the actual updated coordinates."""

    centers: jax.Array
    scales: jax.Array
    log_weights: jax.Array
    breakpoints: jax.Array
    invalid_regions: jax.Array

    @classmethod
    def from_target(
        cls,
        logdensity,
        *,
        breakpoints,
        search_bounds,
        tail_location,
        tail_scale,
        iterations: int = 31,
    ):
        """Reuse regional mode/curvature search and explicit normal tail defense.

        The target returns one scalar log density per independent block. Its
        current coordinate is not an input to this construction. Runtime arrays
        must obey the caller's validated search and defense contracts.
        """
        dtype = jnp.result_type(tail_location, tail_scale, *search_bounds, 1.0)
        mean, sigma, lower, upper = jnp.broadcast_arrays(
            *(
                jnp.asarray(value, dtype=dtype)
                for value in (tail_location, tail_scale, *search_bounds)
            )
        )
        knots = jnp.sort(jnp.broadcast_to(breakpoints, (*mean.shape, jnp.shape(breakpoints)[-1])))
        low = jnp.concatenate([jnp.full_like(mean[..., None], -jnp.inf), knots], axis=-1)
        high = jnp.concatenate([knots, jnp.full_like(mean[..., None], jnp.inf)], axis=-1)
        left, right = jnp.maximum(low, lower[..., None]), jnp.minimum(high, upper[..., None])
        nonempty = (left < right) & jnp.isfinite(left) & jnp.isfinite(right)
        left, right = (
            jnp.where(nonempty, left, mean[..., None]),
            jnp.where(nonempty, right, mean[..., None]),
        )
        # Probe inside each region: derivatives at an absolute-value knot may
        # use the other side's convention. Empty regions never contribute mass.
        left, right = jnp.nextafter(left, right), jnp.nextafter(right, left)
        gradient = jax.grad(lambda value: jnp.sum(logdensity(value)))
        derivative = jax.vmap(gradient, in_axes=-1, out_axes=-1)

        def curvature(value):
            # For separable blocks Hessian-times-ones is the diagonal. This
            # avoids allocating a block-count by block-count Hessian.
            return -jax.jvp(gradient, (value,), (jnp.ones_like(value),))[1]

        def bisect(bounds, unused):
            lo, hi = bounds
            middle = (lo + hi) / 2
            positive = derivative(middle) > 0
            return (jnp.where(positive, middle, lo), jnp.where(positive, hi, middle)), None

        (lo, hi), _ = jax.lax.scan(bisect, (left, right), xs=None, length=iterations)
        center = jnp.where(
            derivative(left) <= 0,
            left,
            jnp.where(derivative(right) >= 0, right, (lo + hi) / 2),
        )
        value = jax.vmap(logdensity, in_axes=-1, out_axes=-1)(center)
        precision = jax.vmap(curvature, in_axes=-1, out_axes=-1)(center)
        valid = nonempty & jnp.isfinite(value) & jnp.isfinite(precision) & (precision > 0)
        scales = 1 / jnp.sqrt(jnp.where(valid, precision, 1))
        raw_weights = jnp.where(valid, value + jnp.log(scales), -jnp.inf)
        has_regions = jnp.any(valid, axis=-1)
        normalization = jnp.where(has_regions, logsumexp(raw_weights, axis=-1), 0)
        weights = raw_weights - normalization[..., None] + jnp.log(0.9)
        return cls(
            jnp.concatenate([mean[..., None], center], axis=-1),
            jnp.concatenate([sigma[..., None], scales], axis=-1),
            jnp.concatenate([jnp.where(has_regions, jnp.log(0.1), 0)[..., None], weights], axis=-1),
            knots,
            nonempty & ~valid,
        )

    def sample(self, key):
        """Draw one value per explicitly independent scalar block."""
        selection, noise = jax.random.split(key)
        component = jax.random.categorical(selection, self.log_weights, axis=-1)[..., None]
        center = jnp.take_along_axis(self.centers, component, axis=-1)[..., 0]
        scale = jnp.take_along_axis(self.scales, component, axis=-1)[..., 0]
        return center + scale * jax.random.normal(noise, center.shape, dtype=center.dtype)

    def logdensity(self, value):
        """Evaluate all mixture weights and scales, without truncation or floors."""
        return logsumexp(
            self.log_weights
            + norm.logpdf(jnp.asarray(value)[..., None], self.centers, self.scales),
            axis=-1,
        )


class ModeInfo(NamedTuple):
    """Per-block correction and proposal-geometry diagnostics."""

    is_accepted: jax.Array
    acceptance_rate: jax.Array
    crossed_region: jax.Array
    invalid_proposal: jax.Array
    invalid_regions: jax.Array


def independence_step(key, value, *, logdensity, proposal: ModeProposal):
    """Correct a fixed proposal against the actual scalar/separable graph target.

    Reuse the acceptance calculation inside BlackJAX's independent MH. Evaluate
    separable targets once as vectors, then vmap only the native accept/reject
    primitive; do not replicate the whole observation graph for every block.
    """
    sample_key, accept_key = jax.random.split(key)
    value = jnp.asarray(value, dtype=proposal.centers.dtype)
    candidate = proposal.sample(sample_key)
    previous = RWState(value, logdensity(value))
    proposed = RWState(candidate, logdensity(candidate))

    previous_proposal = proposal.logdensity(value)
    candidate_proposal = proposal.logdensity(candidate)
    ratio = safe_energy_diff(
        -previous.logdensity - candidate_proposal,
        -proposed.logdensity - previous_proposal,
    )
    updated, (accepted, probability, _) = jax.vmap(static_binomial_sampling)(
        jax.random.split(accept_key, value.size),
        ratio.reshape(-1),
        jax.tree.map(lambda x: x.reshape(-1), previous),
        jax.tree.map(lambda x: x.reshape(-1), proposed),
    )
    accepted, probability = accepted.reshape(value.shape), probability.reshape(value.shape)
    crossed = jnp.sum(value[..., None] >= proposal.breakpoints, axis=-1) != jnp.sum(
        candidate[..., None] >= proposal.breakpoints, axis=-1
    )
    invalid = (
        ~jnp.isfinite(candidate)
        | ~jnp.isfinite(candidate_proposal)
        | jnp.isnan(proposed.logdensity)
        | jnp.isposinf(proposed.logdensity)
    )
    return updated.position.reshape(value.shape), ModeInfo(
        accepted,
        probability,
        accepted & crossed,
        invalid,
        jnp.sum(proposal.invalid_regions, axis=-1),
    )
