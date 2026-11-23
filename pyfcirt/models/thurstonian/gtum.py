"""Negative-distance Thurstonian comparisons with an unscaled logistic link."""

from numbers import Real

import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt

from pyfcirt.models import categorical
from pyfcirt.models.thurstonian.normal import (
    beta_shapes,
    comparison_rv,
    interval_log_probs,
    utility_difference,
)


class GTUMDistribution(pm.Distribution):
    """The model-owned parameter, probability and native response graph."""

    @classmethod
    def dist(
        cls,
        theta,
        loading,
        location,
        cutpoints,
        *,
        link="logit",
        person_block_effect=0.0,
        precision=None,
        direction=1,
        size=None,
        shape=None,
    ):
        """Return an unnamed GTUM category or Beta RV, with actual signed locations.

        Parameters end in two aligned statements. Categorical cutpoints end in K-1
        increasing raw boundaries; the aligned person/block effect enters before
        direction. Beta cutpoints are one unordered offset per row, with no category
        axis or block effect. No residual scale or 1.702 multiplier is introduced.
        The absolute-distance kink at theta=location is preserved.
        """
        return cls._create(
            theta,
            loading,
            location,
            cutpoints,
            link=link,
            person_block_effect=person_block_effect,
            precision=precision,
            direction=direction,
            size=categorical.response_size(shape=shape, size=size),
        )

    @classmethod
    def bind(cls, spec, batch, *, ability, values):
        """Attach the already prepared comparisons without reinterpreting the design."""

        loading, location = values["loading_magnitude"], values["location"]
        cuts = values["cutpoints"]
        return cls._create(
            ability,
            loading,
            location,
            cuts[..., 0] if spec.link == "beta_logit" else cuts,
            link=spec.link,
            person_block_effect=values.get("person_block_effect", 0.0),
            precision=values.get("response_precision"),
            batch=batch,
        )

    @classmethod
    def _create(
        cls,
        theta,
        loading,
        location,
        cutpoints,
        *,
        link="logit",
        person_block_effect=0.0,
        precision=None,
        direction=1,
        batch=None,
        size=None,
    ):
        """Build the native RV from the same negative-distance numerical outputs."""
        if link not in {"logit", "beta_logit"}:
            raise ValueError("GTUM link must be logit or beta_logit")
        beta = link == "beta_logit"
        arguments = dict(theta=theta, loading=loading, location=location, cutpoints=cutpoints)
        checks = ()
        if beta:
            if isinstance(person_block_effect, Real) and person_block_effect != 0:
                raise ValueError("GTUM Beta does not accept a nonzero person_block_effect")
            effect = categorical._tensor(person_block_effect, name="person_block_effect")
            checks = (pt.eq(effect, 0),)
        else:
            arguments["person_block_effect"] = person_block_effect
        return comparison_rv(
            cls._beta_parameters if beta else cls.kernel,
            arguments,
            direction,
            beta=beta,
            precision=precision,
            size=size,
            batch=batch,
            smooth=cls.smooth,
            checks=checks,
            restriction="; person_block_effect must be zero" if checks else "",
        )

    @staticmethod
    def kernel(theta, loading, location, cutpoints, person_block_effect, direction):
        """Use the aligned block effect before orienting the raw utility contrast."""
        contrast = GTUMDistribution._contrast(theta, loading, location) + jnp.asarray(
            person_block_effect
        )
        z = jnp.asarray(cutpoints) - jnp.asarray(direction)[..., None] * contrast[..., None]
        return interval_log_probs(z, link="logit")

    @staticmethod
    def _beta_parameters(theta, loading, location, intercept, precision, direction):
        eta = jnp.asarray(direction) * (
            GTUMDistribution._contrast(theta, loading, location) - jnp.asarray(intercept)
        )
        return beta_shapes(eta, precision, link="logit")

    @staticmethod
    def _contrast(theta, loading, location):
        utility = -jnp.asarray(loading) * jnp.abs(jnp.asarray(theta) - jnp.asarray(location))
        return utility_difference(utility)

    @staticmethod
    def smooth(theta, loading, location, *unused):
        return jnp.all(jnp.asarray(theta) != jnp.asarray(location), axis=-1)
