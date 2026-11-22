"""Linear Thurstonian categorical and continuous-Beta comparison laws."""

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

TIRT_LOGIT_MULTIPLIER = 1.702


class TIRTDistribution(pm.Distribution):
    """The model-owned parameter, probability and native response graph."""

    @classmethod
    def dist(
        cls,
        theta,
        loading,
        cutpoints,
        *,
        link="probit",
        statement_effect=None,
        residual_variance=None,
        precision=None,
        direction=1,
        size=None,
        shape=None,
    ):
        """Return an unnamed TIRT category or Beta RV without creating parameter priors.

        Comparison parameters end in two aligned statements. Categorical cutpoints
        end in K-1 ordered boundaries; effects and residual variances are explicit.
        Beta cutpoints instead contain one unordered offset per response row, with
        no category axis, effects or residual variances. Only categorical logit uses
        the 1.702 scale factor; Beta links retain their unscaled mean.
        """
        return cls._create(
            theta,
            loading,
            cutpoints,
            link=link,
            statement_effect=statement_effect,
            residual_variance=residual_variance,
            precision=precision,
            direction=direction,
            size=categorical.response_size(shape=shape, size=size),
        )

    @classmethod
    def bind(cls, spec, batch, *, ability, values):
        """Use the prepared pairs and the same numerical outputs for every analysis."""

        loading = values["loading_magnitude"] * pt.cast(
            batch.keying, values["loading_magnitude"].dtype
        )
        cuts = values["cutpoints"]
        return cls._create(
            ability,
            loading,
            cuts[..., 0] if spec.link.startswith("beta_") else cuts,
            link=spec.link,
            statement_effect=values.get("statement_effect"),
            residual_variance=values.get("residual_variance"),
            precision=values.get("response_precision"),
            batch=batch,
        )

    @classmethod
    def _create(
        cls,
        theta,
        loading,
        cutpoints,
        *,
        link="probit",
        statement_effect=None,
        residual_variance=None,
        precision=None,
        direction=1,
        batch=None,
        size=None,
    ):
        """Build one native response RV and retain its existing numerical outputs."""
        if link not in {"probit", "logit", "beta_probit", "beta_logit"}:
            raise ValueError("TIRT link must be probit, logit, beta_probit or beta_logit")
        beta = link.startswith("beta_")
        arguments = dict(theta=theta, loading=loading, cutpoints=cutpoints)
        checks = ()
        if beta:
            if statement_effect is not None or residual_variance is not None:
                raise ValueError(
                    "TIRT Beta links do not accept statement_effect or residual_variance"
                )
        else:
            if statement_effect is None or residual_variance is None:
                raise ValueError("Categorical TIRT requires statement_effect and residual_variance")
            residual_variance = categorical._tensor(residual_variance, name="residual_variance")
            if residual_variance.ndim < 1 or residual_variance.type.shape[-1] != 2:
                raise ValueError("residual_variance requires a final two-statement axis")
            arguments.update(statement_effect=statement_effect, residual_variance=residual_variance)
            checks = (residual_variance > 0,)
        return comparison_rv(
            cls._beta_parameters if beta else cls.kernel,
            arguments,
            direction,
            beta=beta,
            precision=precision,
            size=size,
            link=link.removeprefix("beta_"),
            batch=batch,
            checks=checks,
            restriction="; residual_variance > 0" if checks else "",
        )

    @staticmethod
    def kernel(
        theta,
        loading,
        cutpoints,
        statement_effect,
        residual_variance,
        direction,
        *,
        link,
    ):
        """Use raw utility cutpoints with the categorical residual comparison scale."""
        contrast = TIRTDistribution._contrast(theta, loading, statement_effect)
        scale = jnp.sqrt(jnp.sum(jnp.asarray(residual_variance), axis=-1))
        z = jnp.asarray(cutpoints) - jnp.asarray(direction)[..., None] * contrast[..., None]
        z = z / scale[..., None]
        return interval_log_probs(
            (TIRT_LOGIT_MULTIPLIER if link == "logit" else 1.0) * z, link=link
        )

    @staticmethod
    def _beta_parameters(theta, loading, intercept, precision, direction, *, link):
        """Apply the unscaled link to the oriented contrast minus its one offset."""
        eta = jnp.asarray(direction) * (
            TIRTDistribution._contrast(theta, loading) - jnp.asarray(intercept)
        )
        return beta_shapes(eta, precision, link=link)

    @staticmethod
    def _contrast(theta, loading, statement_effect=0.0):
        utility = jnp.asarray(loading) * jnp.asarray(theta) + jnp.asarray(statement_effect)
        return utility_difference(utility)
