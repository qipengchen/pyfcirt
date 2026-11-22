"""MGPPM statement parameters and one native response distribution."""

import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt
from jax.nn import log_softmax

from pyfcirt.models import categorical
from pyfcirt.models.graph import response_rv


class MGPPMDistribution(pm.Distribution):
    """Score-weighted complete patterns and their observed event partition."""

    @classmethod
    def dist(cls, theta, loading, intercept, *, scores, membership=None, size=None, shape=None):
        """Return an unnamed MGPPM category RV without creating parameter priors.

        theta and loading end in aligned statement axes; other leading axes
        broadcast. Membership partitions complete patterns into RANK/PICK/MOLE or
        other explicitly declared events; None makes each complete pattern an event.
        """
        size = categorical.response_size(shape=shape, size=size)
        theta = categorical._tensor(theta, name="theta")
        loading = categorical._tensor(loading, name="loading")
        intercept = categorical._tensor(intercept, name="intercept")
        scores = categorical._tensor(scores, name="scores")
        if scores.ndim != 2 or scores.type.shape[0] < 2:
            raise ValueError(
                "scores must have shape (pattern, statement), with at least two patterns"
            )
        membership = categorical.event_membership(membership, scores, name="scores")
        return response_rv(
            cls.kernel, (theta, loading, intercept), scores=scores, membership=membership, size=size
        )

    @classmethod
    def bind(cls, spec, batch, *, ability, values):
        loading = values["loading_magnitude"] * pt.cast(
            batch.keying, values["loading_magnitude"].dtype
        )
        return response_rv(
            cls.kernel,
            (categorical._tensor(ability, name="theta"), loading, values["intercept"]),
            batch=batch,
            scores=batch.events.scores,
            membership=batch.events.membership,
        )

    @staticmethod
    def kernel(theta, loading, intercept, *, scores, membership=None):
        """Apply MGPPM's score-weighted law, then aggregate the actual response events.

        The final parameter axis is statement. ``scores[pattern, statement]`` is
        supplied explicitly: graded categories retain their own scores rather than
        being interpreted as a complete-ranking catalogue.
        """
        utility = jnp.asarray(loading) * jnp.asarray(theta) - jnp.asarray(intercept)
        logits = utility @ jnp.asarray(scores).T
        return categorical.event_log_probs(log_softmax(logits, axis=-1), membership)
