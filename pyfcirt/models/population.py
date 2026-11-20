"""One population factory: native ability nodes and their shared JAX law."""

import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import wrap_jax

MARGINAL_CORRELATION = "theta_correlation_marginal"


def population_output_attrs(person_count):
    return {
        MARGINAL_CORRELATION: {
            "reference_population": "current analysis persons",
            "N": person_count,
            "weighting": "empirical_person_proportions",
            "source_person_coordinate": "person",
        }
    }


class PopulationFactory:
    """Consume a frozen population layout; do not refit covariates or identify a design."""

    @classmethod
    def build(cls, layout, *, parameter_factory, regression=None):
        parameters = parameter_factory.bindings
        layouts = parameter_factory.layouts
        mean = parameters["population_mean"]
        covariance_layout = layouts["covariance"]
        correlation = parameters["trait_correlation"]
        sd = parameters["population_sd"]
        scale = sd if len(covariance_layout.shape) == 3 else sd[0]
        covariance = parameter_factory.bind(
            "covariance", wrap_jax(cls._covariance, allow_eval=False)(correlation, scale)
        )
        person_mean, person_covariance = wrap_jax(cls.moments, allow_eval=False)(
            layout.person_group,
            mean,
            covariance,
            regression_mean=None if regression is None else regression.mean,
        )
        person_count, trait_count = len(layout.person_group), covariance_layout.shape[-1]
        if layout.theta_parameterization == "centered":
            theta = pm.MvNormal(
                "theta",
                mu=person_mean,
                cov=person_covariance,
                shape=(person_count, trait_count),
                dims=("person", "trait"),
            )
        elif layout.theta_parameterization == "noncentered":
            z = pm.Normal(
                "theta_raw", 0, 1, shape=(person_count, trait_count), dims=("person", "trait")
            )
            # Whiten every trait; saved final theta remains the downstream input.
            # Native batched linear algebra also lowers directly to JAX.
            theta = pm.Deterministic(
                "theta",
                person_mean
                + pt.matmul(pt.linalg.cholesky(person_covariance), z[..., None])[..., 0],
                dims=("person", "trait"),
            )
        else:
            centered_count, remaining_count = (
                len(layout.centered_index),
                len(layout.noncentered_index),
            )
            if centered_count:
                centered_mean, centered_covariance = wrap_jax(
                    cls._centered_moments, allow_eval=False
                )(person_mean, person_covariance, layout.centered_index)
                theta_centered = pm.MvNormal(
                    "theta_centered_raw",
                    mu=centered_mean,
                    cov=centered_covariance,
                    shape=(person_count, centered_count),
                    dims=("person", "centered_trait"),
                )
            else:
                theta_centered = pt.zeros((person_count, 0))
            z = (
                pm.Normal(
                    "theta_noncentered_raw",
                    0,
                    1,
                    shape=(person_count, remaining_count),
                    dims=("person", "noncentered_trait"),
                )
                if remaining_count
                else pt.zeros((person_count, 0))
            )
            theta = pm.Deterministic(
                "theta",
                wrap_jax(cls.conditional_theta, allow_eval=False)(
                    theta_centered,
                    z,
                    person_mean,
                    person_covariance,
                    layout.centered_index,
                    layout.noncentered_index,
                ),
                dims=("person", "trait"),
            )
        parameters["theta"] = theta
        return person_mean, person_covariance

    @classmethod
    def fixed_covariance_values(cls, covariance, *, shared_correlation, sd_shape):
        """Resolve fixed population values using the declared correlation and scale axes."""
        correlation, sd = map(np.asarray, cls._from_covariance(covariance))
        if shared_correlation and correlation.ndim > 2:
            tolerance = 8 * np.finfo(correlation.dtype).eps
            if not np.allclose(correlation, correlation[:1], rtol=tolerance, atol=tolerance):
                raise ValueError("Fixed covariance groups violate their declared shared R")
            correlation = correlation[0]
        return dict(
            trait_correlation=correlation,
            population_sd=sd.reshape(sd_shape),
        )

    @staticmethod
    def moments(group_index, group_mean, covariance, *, regression_mean=None):
        mean = jnp.asarray(group_mean)[group_index]
        if regression_mean is not None:
            mean = mean + jnp.asarray(regression_mean)
        covariance = jnp.asarray(covariance)
        person_covariance = (
            covariance[group_index]
            if covariance.ndim == 3
            else jnp.broadcast_to(covariance, (len(group_index), *covariance.shape))
        )
        return mean, person_covariance

    @classmethod
    def marginal_moments(cls, person_mean, person_covariance):
        """Mean, SD and correlation under the current empirical person distribution."""
        mean = jnp.asarray(person_mean)
        marginal_mean = mean.mean(axis=0)
        centered = mean - marginal_mean
        # Total covariance averages the conditional population covariance and
        # adds variation in covariate/group-dependent means across these people.
        covariance = jnp.asarray(person_covariance).mean(axis=0) + centered.T @ centered / len(mean)
        correlation, sd = cls._from_covariance(covariance)
        diagonal = jnp.arange(len(sd))
        return marginal_mean, sd, correlation.at[diagonal, diagonal].set(1.0)

    @staticmethod
    def conditional_theta(
        theta_centered, z_noncentered, mean, covariance, centered_index, noncentered_index
    ):
        """Use the Gaussian conditional law to retain correlations across both partitions."""
        mean, covariance = jnp.asarray(mean), jnp.asarray(covariance)
        centered_index, noncentered_index = (
            jnp.asarray(centered_index),
            jnp.asarray(noncentered_index),
        )
        result = jnp.zeros_like(mean).at[:, centered_index].set(theta_centered)
        if noncentered_index.shape[0] == 0:
            return result
        conditional_mean = mean[:, noncentered_index]
        conditional_covariance = covariance[:, noncentered_index[:, None], noncentered_index]
        if centered_index.shape[0]:
            centered_covariance = covariance[:, centered_index[:, None], centered_index]
            cross = covariance[:, noncentered_index[:, None], centered_index]
            residual = jnp.asarray(theta_centered) - mean[:, centered_index]
            # Use the same factorization for the conditional mean and Schur complement.
            rhs = jnp.concatenate((residual[..., None], jnp.swapaxes(cross, -1, -2)), axis=-1)
            adjustment = cross @ jnp.linalg.solve(centered_covariance, rhs)
            conditional_mean += adjustment[..., 0]
            conditional_covariance -= adjustment[..., 1:]
        remaining = (
            conditional_mean
            + (jnp.linalg.cholesky(conditional_covariance) @ jnp.asarray(z_noncentered)[..., None])[
                ..., 0
            ]
        )
        return result.at[:, noncentered_index].set(remaining)

    @staticmethod
    def _covariance(correlation, sd):
        return (
            jnp.asarray(correlation) * jnp.asarray(sd)[..., :, None] * jnp.asarray(sd)[..., None, :]
        )

    @staticmethod
    def _from_covariance(covariance):
        covariance = jnp.asarray(covariance)
        sd = jnp.sqrt(jnp.diagonal(covariance, axis1=-2, axis2=-1))
        return covariance / (sd[..., :, None] * sd[..., None, :]), sd

    @staticmethod
    def _centered_moments(mean, covariance, index):
        return mean[:, index], covariance[:, index[:, None], index]
