"""Gaussian ranking events, with CDF-free generation and lazy marginal density."""

from functools import cached_property, partial
from graphlib import CycleError, TopologicalSorter
from numbers import Integral

import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from jax import jit, lax
from jax import random as jr
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.transforms import LogTransform, Transform
from pytensor import wrap_jax

from pyfcirt.models import categorical
from pyfcirt.models.graph import ResponseBinding, observe_augmented, slot_values
from pyfcirt.models.thurstonian.normal import region_logcdf, sobol_nodes, sobol_rule


class TIRTRankDistribution(pm.Distribution):
    """Gaussian ranking law with model-owned lazy density and CDF-free generation.

    PyMC callbacks share numerical integration settings and consume their actual
    graph inputs, including after cloning or conditioning.
    """

    @classmethod
    def dist(
        cls,
        theta,
        loading,
        intercept,
        residual_variance,
        *,
        orders,
        contrasts,
        membership=None,
        n_points=4096,
        seed=914,
        chunk_size=128,
        size=None,
        shape=None,
    ):
        """Return an unnamed marginal Gaussian ranking-event RV.

        theta/loading end in (statement, trait), intercept/variance in statement.
        Each contrast row encodes positive adjacent utility differences for its
        complete order. Membership partitions orders into actual response events.
        Generation draws Gaussian utilities; logp uses a fixed-node QMC
        approximation with the shared stable inverse log-CDF kernel.
        No ranking probabilities are clipped or renormalized.
        """
        theta = categorical._tensor(theta, name="theta")
        loading = categorical._tensor(loading, name="loading")
        intercept = categorical._tensor(intercept, name="intercept")
        residual_variance = categorical._tensor(residual_variance, name="residual_variance")
        if residual_variance.ndim < 1:
            raise ValueError("residual_variance must end in the complete statement axis")
        residual_variance = check_parameters(
            residual_variance,
            residual_variance > 0,
            msg="residual_variance must be positive",
            can_be_replaced_by_ninf=False,
        )
        mean, covariance = wrap_jax(cls._utility_moments, allow_eval=False)(
            theta, loading, intercept, residual_variance
        )
        return cls._create(
            mean,
            covariance,
            orders=orders,
            contrasts=contrasts,
            membership=membership,
            n_points=n_points,
            seed=seed,
            chunk_size=chunk_size,
            size=categorical.response_size(shape=shape, size=size),
        )

    @classmethod
    def bind(cls, spec, batch, *, ability, parameters):
        values = slot_values(batch, parameters)
        loading = values["loading_magnitude"] * pt.cast(
            batch.keying, values["loading_magnitude"].dtype
        )
        intercept, residual_variance = values["intercept"], values["residual_variance"]
        mean, covariance = wrap_jax(cls._utility_moments, allow_eval=False)(
            ability[..., None], loading[..., None], intercept, residual_variance
        )
        distribution = cls._create(
            mean,
            covariance,
            orders=batch.events.orders,
            contrasts=batch.events.contrasts,
            membership=batch.events.membership,
        )
        augmentation = (
            ()
            if batch.y is None
            else observe_augmented(batch, *cls.augmentation(batch, mean, covariance))
        )
        return ResponseBinding(
            batch=batch,
            parameters=(loading, intercept, residual_variance),
            kernel=partial(
                cls.kernel, contrasts=batch.events.contrasts, membership=batch.events.membership
            ),
            distribution=distribution,
            augmentation=augmentation,
        )

    @classmethod
    def _create(
        cls,
        mean,
        covariance,
        *,
        orders,
        contrasts,
        membership=None,
        n_points=4096,
        seed=914,
        chunk_size=128,
        size=None,
    ):
        mean = categorical._tensor(mean, name="mean")
        covariance = categorical._tensor(covariance, name="covariance")
        orders = categorical.ranking_orders(orders)
        contrasts = categorical._tensor(contrasts, name="contrasts")
        pattern_count, statement_count = orders.type.shape
        if contrasts.type.shape != (pattern_count, statement_count - 1, statement_count):
            raise ValueError("contrasts must have shape (pattern, statement - 1, statement)")
        if mean.ndim < 1 or mean.type.shape[-1] != statement_count:
            raise ValueError("mean must end in the complete statement axis")
        if covariance.ndim < 2 or covariance.type.shape[-2:] != (statement_count, statement_count):
            raise ValueError("covariance must end in the complete statement-by-statement axes")
        membership = categorical.event_membership(membership, orders, name="orders")
        event = _GaussianEvent(
            statement_count - 1, n_points=n_points, seed=seed, chunk_size=chunk_size
        )
        members = () if membership is None else (membership,)
        signature = "(s),(s,s),(h,s),(h,d,s)" + (",(k,h)" if members else "") + "->()"
        return pm.CustomDist.dist(
            mean,
            covariance,
            orders,
            contrasts,
            *members,
            logp=event._logp,
            random=event._native_random,
            signature=signature,
            dtype="int64",
            size=size,
        )

    @classmethod
    def kernel(
        cls,
        theta,
        loading,
        intercept,
        residual_variance,
        *,
        contrasts,
        membership=None,
        n_points=4096,
        chunk_size=128,
    ):
        mean, covariance = cls._utility_moments(
            theta[..., None], loading[..., None], intercept, residual_variance
        )
        return cls._log_probs(
            mean,
            covariance,
            contrasts,
            membership,
            nodes=sobol_nodes(loading.shape[-1] - 1, n_points=n_points),
            chunk_size=chunk_size,
        )

    @staticmethod
    def observed_region(size, edges):
        """Order the region from worst to best while retaining only the known comparisons."""
        predecessors = {node: set() for node in range(size)}
        for first, second, bit in edges:
            winner, loser = (first, second) if bit else (second, first)
            predecessors[winner].add(loser)
        sorter = TopologicalSorter(predecessors)
        try:
            sorter.prepare()
        except CycleError as exc:
            raise ValueError("Cyclic comparisons cannot be represented as a ranking event") from exc
        order = []
        while sorter.is_active():
            ready = sorted(sorter.get_ready())
            order.extend(ready)
            sorter.done(*ready)
        mask = np.asarray([[other in predecessors[node] for other in order] for node in order])
        return mask, np.asarray(order, dtype=np.int64)

    @staticmethod
    def observed_contrasts(masks, orders):
        """Use adjacent differences for total orders and a common origin for partial orders."""
        size = orders.shape[-1]
        positions = np.arange(size - 1)
        eye = np.eye(size)[orders]
        if masks[:, positions + 1, positions].all():
            return eye[:, 1:] - eye[:, :-1], None
        return eye[:, 1:] - eye[:, :1], masks

    @classmethod
    def augmentation(cls, batch, mean, covariance):
        """Define native Gaussian contrasts and a membership distribution per event."""
        mask = batch.events.mask
        transform = _PositiveDifferences() if mask is None else _PartialOrderDifferences(mask)
        rows, statements = batch.statement.shape
        shape = (rows, statements - 1)
        mean, covariance = wrap_jax(cls._contrast_moments, allow_eval=False)(
            mean, covariance, batch.events.basis
        )
        differences = pm.CustomDist.dist(
            mean,
            covariance,
            dist=lambda mu, cov, size: pm.MvNormal.dist(mu=mu, cov=cov, size=size),
            support_point=lambda rv, size, mu, cov: transform.backward(
                pt.zeros(shape, dtype=rv.dtype)
            ),
            signature="(d),(d,d)->(d)",
            shape=shape,
        )

        def membership(contrast):
            return pm.DiracDelta.dist(c=pt.cast(transform.in_region(contrast), "int64"))

        return differences, transform, membership

    @staticmethod
    def _contrast_moments(mean, covariance, basis):
        basis = jnp.asarray(basis)
        return (basis @ jnp.asarray(mean)[..., None])[..., 0], (
            basis @ jnp.asarray(covariance) @ jnp.swapaxes(basis, -1, -2)
        )

    @staticmethod
    def _utility_moments(theta, loading, intercept, residual_variance):
        """Supply the same Gaussian utility law to augmentation, density and generation."""
        mean = jnp.sum(jnp.asarray(theta) * jnp.asarray(loading), axis=-1) + jnp.asarray(intercept)
        variance = jnp.asarray(residual_variance)
        covariance = jnp.eye(variance.shape[-1], dtype=jnp.result_type(variance, 0.0))
        return mean, covariance * variance[..., None, :]

    @classmethod
    def _log_probs(cls, mean, covariance, contrasts, membership=None, *, nodes, chunk_size=128):
        """Integrate complete orders, evaluating unions through their smaller tail."""
        differences, difference_covariance = cls._contrast_moments(
            jnp.asarray(mean)[..., None, :], jnp.asarray(covariance)[..., None, :, :], contrasts
        )
        # For continuous differences, P(D > 0) = P(-D <= 0).
        logs = region_logcdf(
            jnp.zeros((differences.shape[-1],), dtype=differences.dtype),
            -differences,
            difference_covariance,
            nodes=nodes,
            chunk_size=chunk_size,
        )
        if membership is None:
            return logs
        events = categorical.event_log_probs(logs, membership)
        complement = categorical.event_log_probs(logs, ~jnp.asarray(membership))
        use_complement = complement < events
        # Integrate the true excluded orders, not an estimated-total residual.
        # Sanitize the unused subtraction before autodiff, without changing its output.
        tail = jnp.where(use_complement, complement, -jnp.inf)
        return jnp.where(use_complement, jnp.log1p(-jnp.exp(tail)), events)

    @staticmethod
    @partial(jit, static_argnames=("sample_shape",))
    def _random(key, mean, covariance, orders, membership=None, *, sample_shape=()):
        """Draw utilities, sort them and encode their event without evaluating any CDF."""
        mean, covariance, orders = jnp.asarray(mean), jnp.asarray(covariance), jnp.asarray(orders)
        member_batch = () if membership is None else jnp.shape(membership)[:-2]
        batch_shape = jnp.broadcast_shapes(mean.shape[:-1], covariance.shape[:-2], member_batch)
        utility = jr.multivariate_normal(
            key, mean, covariance, shape=tuple(sample_shape) + batch_shape, method="cholesky"
        )
        order = jnp.argsort(-utility, axis=-1)
        pattern = jnp.all(order[..., None, :] == orders, axis=-1)
        if membership is not None:
            pattern = jnp.any(jnp.asarray(membership) & pattern[..., None, :], axis=-1)
        return jnp.argmax(pattern, axis=-1)

    @staticmethod
    def _project_differences(raw, mask):
        """Map K-1 relative utilities into the recorded partial-order region.

        The first reverse-topological utility is the zero reference. Other sinks
        remain unrestricted; PICK/MOLE and incomplete comparisons never acquire
        an invented middle order. The Gaussian difference law integrates out the
        omitted reference level, rather than conditioning its realized value.
        """
        raw, mask = jnp.asarray(raw), jnp.asarray(mask)

        def step(utilities, index):
            selected = mask[..., index, :]
            constrained = jnp.any(selected, axis=-1)
            bound = _predecessor_bound(utilities, selected, constrained)
            coordinate = raw[..., index - 1]
            slack = jnp.exp(jnp.where(constrained, coordinate, 0.0))
            value = jnp.where(constrained, bound + slack, coordinate)
            return utilities.at[..., index].set(value), None

        utilities, _ = lax.scan(
            step,
            jnp.zeros((*raw.shape[:-1], raw.shape[-1] + 1), dtype=raw.dtype),
            jnp.arange(1, raw.shape[-1] + 1),
        )
        return utilities[..., 1:]


def _predecessor_bound(utilities, mask, constrained):
    bound = jnp.max(jnp.where(mask, utilities, -jnp.inf), axis=-1)
    return jnp.where(constrained, bound, 0.0)


def _partial_order_slack(differences, mask):
    mask = mask[..., 1:, :]
    utilities = jnp.concatenate((jnp.zeros_like(differences[..., :1]), differences), axis=-1)
    bound = _predecessor_bound(utilities[..., None, :], mask, jnp.any(mask, axis=-1))
    return differences - bound


class _PositiveDifferences(LogTransform):
    """Apply the native log transform to each component of a vector event."""

    ndim_supp = 1

    def log_jac_det(self, value, *inputs):
        return super().log_jac_det(value, *inputs).sum(axis=-1)

    def in_region(self, value):
        return pt.all(value > 0, axis=-1)


class _PartialOrderDifferences(Transform):
    """Use positive slacks for constrained coordinates and preserve all other coordinates."""

    name = "partial_order"
    ndim_supp = 1

    def __init__(self, mask):
        self.mask = pt.as_tensor_variable(mask)
        self.shape = (*mask.shape[:-2], mask.shape[-1] - 1)
        self.constrained = pt.any(self.mask[..., 1:, :], axis=-1)
        self._project = wrap_jax(TIRTRankDistribution._project_differences, allow_eval=False)
        self._slack = wrap_jax(_partial_order_slack, allow_eval=False)

    def backward(self, value, *inputs):
        return self._project(pt.specify_shape(value, self.shape), self.mask)

    def forward(self, value, *inputs):
        slack = self._slack(pt.specify_shape(value, self.shape), self.mask)
        return pt.where(self.constrained, pt.log(pt.where(self.constrained, slack, 1)), value)

    def log_jac_det(self, value, *inputs):
        value = pt.specify_shape(value, self.shape)
        return pt.where(self.constrained, value, 0).sum(axis=-1)

    def in_region(self, value):
        slack = self._slack(pt.specify_shape(value, self.shape), self.mask)
        return pt.all(~self.constrained | (slack > 0), axis=-1)


class _GaussianEvent:
    """Share fixed integration nodes across uses of the same native distribution."""

    def __init__(self, dimension, *, n_points, seed, chunk_size):
        self._sample_nodes = sobol_rule(dimension, n_points=n_points, seed=seed)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self._chunk_size = chunk_size

    @cached_property
    def _nodes(self):
        return categorical._tensor(self._sample_nodes(), name="nodes")

    def log_probabilities(self, mean, covariance, orders, contrasts, *members):
        membership = members[0] if members else None
        return wrap_jax(TIRTRankDistribution._log_probs, allow_eval=False)(
            categorical._tensor(mean, name="mean"),
            categorical._tensor(covariance, name="covariance"),
            categorical._tensor(contrasts, name="contrasts"),
            None if membership is None else categorical._tensor(membership, name="membership"),
            nodes=self._nodes,
            chunk_size=self._chunk_size,
        )

    def _logp(self, value, mean, covariance, orders, contrasts, *members):
        return categorical.StableCategorical.logp(
            value, self.log_probabilities(mean, covariance, orders, contrasts, *members)
        )

    def _native_random(self, mean, covariance, orders, contrasts, *members, rng=None, size=None):
        mean, covariance, orders = jnp.asarray(mean), jnp.asarray(covariance), jnp.asarray(orders)
        membership = jnp.asarray(members[0]) if members else None
        count = orders.shape[-1]
        if size is not None:
            shape = tuple(int(value) for value in size)
            mean = jnp.broadcast_to(mean, (*shape, count))
            covariance = jnp.broadcast_to(covariance, (*shape, count, count))
            if membership is not None:
                membership = jnp.broadcast_to(membership, (*shape, *membership.shape[-2:]))
        # This is only the native PyMC RNG/host-array boundary, not a second generator.
        rng = np.random.default_rng() if rng is None else rng
        words = rng.integers(0, 2**32, size=2, dtype=np.uint32)
        key = jr.wrap_key_data(jnp.asarray(words), dtype="threefry2x32")
        return np.asarray(
            TIRTRankDistribution._random(key, mean, covariance, orders, membership), dtype=np.int64
        )
