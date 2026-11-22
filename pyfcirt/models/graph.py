"""Final parameter construction and lightweight references to actual model nodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial, wraps

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import implicit_size_from_params, rv_size_is_none
from pymc.distributions.transforms import _default_transform
from pymc.logprob.transforms import IntervalTransform, LogTransform
from pytensor import config, wrap_jax

from pyfcirt.models.categorical import StableCategorical
from pyfcirt.models.priors import (
    bind_prior,
    prior_support,
    prior_with_parameters,
    scalar_prior_bounds,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RegressionBinding:
    """Native conditional inputs in the prepared target's active order."""

    mean: object
    scale: object | None
    conditional: Mapping
    noncentered: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseBinding:
    """Bind the response law and the observation registered in this model view.

    The distribution is the native marginal response RV. Augmented fitting
    retains its contrast and membership RVs separately in `augmentation`.
    """

    batch: object
    parameters: tuple
    kernel: object
    distribution: object
    augmentation: tuple = ()
    smooth: object | None = None


class ParameterFactory:
    """Construct final parameter nodes from the sole frozen layout.

    All fixed/free, location, ordered and scale paths consume the prepared
    contract; they do not choose identification or reinterpret input tables.
    """

    def __init__(self, prepared):
        self.layouts = prepared.parameter_layouts
        self.blocks = prepared.blocks_by_kind
        self.bindings = {}

    def build(self, kinds, *, regressions=None):
        """Construct each final kind once, using its prepared native blocks."""
        for kind in kinds:
            if kind in self.bindings:
                continue
            layout = self.layouts[kind]
            value = pt.constant(
                np.where(np.isnan(layout.fixed), 0.0, layout.fixed).ravel(), dtype=config.floatX
            )
            for block in self.blocks[kind]:
                if block.prior_role is None or not len(block.free_flat):
                    continue
                dependencies = []
                if block.scale_kind is not None:
                    dependencies.append(block.scale_kind)
                if block.bounds is not None:
                    dependencies.append(block.bounds[0])
                if dependencies:
                    self.build(dependencies, regressions=regressions)
                for entries, draw in self._entry_updates(block, regressions):
                    value = pt.set_subtensor(value[entries], draw.ravel())
            self.bind(kind, value.reshape(layout.shape))
        return self.bindings

    def bind(self, kind, value):
        """Register a canonical node and retain its prepared layout."""
        layout = self.layouts[kind]
        node = pm.Deterministic(kind, value, dims=layout.dims)
        self.bindings[kind] = node
        return node

    def _entry_updates(self, block, regressions):
        """Yield native draws and the prepared final positions they supply."""
        name = f"_parameter_{block.binding_id}_raw"
        entries = block.entry_index.ravel()
        if block.key == "trait_correlation":
            # The joint LKJ draw supplies the whole correlation matrix.
            draw = bind_prior(block.priors[block.prior_role], name=name, shape=block.shape)
            yield entries, draw @ pt.swapaxes(draw, -1, -2)
        elif block.bounds is not None:
            yield entries[block.free_flat], self._bounded_draw(block)
        elif block.scale_kind is not None:
            scale_source = self.bindings[block.scale_kind].ravel()
            scale = scale_source[block.scale_entry_index] ** block.scale_power
            shape = block.prior_shapes[block.prior_role]
            flat_scale = pt.broadcast_to(scale, block.shape).ravel()
            free_scale = flat_scale[block.free_flat].reshape(shape)
            draw = self.scaled(
                name,
                block.priors[block.prior_role],
                free_scale,
                shape=shape,
                noncentered=block.noncentered,
            )
            yield entries[block.free_flat], draw
        else:
            entries = entries[block.free_flat]
            for selected, draw in self._free_draws(block, regressions):
                # A common discrimination has one canonical entry and one draw.
                yield entries[:1] if block.shared else entries[selected], draw

    def _bounded_draw(self, block):
        """Bind the native conditional law with the effective truncation interval."""
        lower_kind, lower_entries, upper_limit = block.bounds
        lower_binding = self.bindings[lower_kind]
        prior = block.priors[block.prior_role]
        shape = block.prior_shapes[block.prior_role]
        floor, ceiling = scalar_prior_bounds(prior)
        dtype = np.result_type(prior.dtype, lower_binding.dtype)
        lower = pt.reshape(pt.reshape(lower_binding, (-1,))[lower_entries], shape)
        return pm.Truncated(
            f"_parameter_{block.binding_id}_raw",
            prior,
            lower=pt.maximum(lower, pt.cast(floor, dtype)),
            upper=pt.minimum(pt.cast(ceiling, dtype), pt.cast(upper_limit, dtype)),
            shape=shape,
        )

    @classmethod
    def _free_draws(cls, layout, regressions):
        """Yield native free draws in each prepared prior group."""
        multiple = layout.prior_role_index is not None
        for role, positions in layout.prior_groups:
            binding = None if regressions is None else regressions.get((layout.binding_id, role))
            if binding is not None:
                yield (
                    positions,
                    cls.regression(layout, role=role, size=len(positions), binding=binding),
                )
                continue
            name = f"_parameter_{layout.binding_id}{'_' + role if multiple else ''}"
            declaration = layout.priors[role]
            shape = layout.prior_shapes[role]
            if layout.ordered:
                draw = cls.ordered(
                    f"{name}_raw", declaration, rows=shape[0], count=layout.shape[-1]
                )
            else:
                noncentered = layout.noncentered and role not in layout.centered_roles
                if layout.noncentered:
                    mode = "noncentered" if noncentered else "centered"
                    name = f"{name}_{mode}"
                name = f"{name}_raw"
                if noncentered:
                    node = declaration.owner
                    mean, scale = node.op.dist_params(node)
                    draw = cls.scaled(
                        name,
                        pm.Normal.dist(mu=0, sigma=1),
                        scale,
                        offset=mean,
                        shape=shape,
                        noncentered=True,
                    )
                else:
                    draw = bind_prior(declaration, name=name, shape=shape)
            yield positions, draw

    @classmethod
    def regression(cls, layout, *, role, size, binding):
        """Bind the prepared conditional inputs to their native distribution."""
        name = f"_parameter_{layout.binding_id}_{role}_raw"
        mean, scale = binding.mean, binding.scale
        if layout.ordered:
            count = layout.shape[-1]
            return cls.ordered(
                name, pm.Normal.dist(mu=mean, sigma=scale), rows=size // count, count=count
            )
        family = binding.conditional["family"]
        if binding.noncentered:
            draw = cls.scaled(
                name,
                pm.Normal.dist(mu=0, sigma=1),
                scale,
                offset=mean,
                shape=(size,),
                noncentered=True,
            )
            return pt.exp(draw) if family == "LogNormal" else draw
        distribution = getattr(pm, family)
        bounds = binding.conditional["bounds"]
        options = {} if bounds is None else dict(zip(("lower", "upper"), bounds, strict=True))
        return distribution(name, mu=mean, sigma=scale, shape=(size,), **options)

    @classmethod
    def ordered(cls, name, prior, *, rows, count):
        """Sort iid native draws and retain their normalized order-statistic density."""
        op = prior.owner.op
        parameters = op.dist_params(prior.owner)
        inputs, _ = op.signature.split("->")
        if count == 1:
            return pm.modelcontext(None).register_rv(
                prior_with_parameters(
                    prior, *cls._ordered_parameters(prior, parameters), size=(rows, count)
                ),
                name,
            )

        return pm.CustomDist(
            name,
            pt.zeros(count),
            *parameters,
            dist=partial(cls._ordered_dist, prior=prior, count=count),
            logp=partial(cls._ordered_logp, prior=prior, count=count),
            signature=f"(n){',' + inputs if inputs else ''}->(n)",
            shape=(rows, count),
            default_transform=cls._ordered_transform(prior),
            # Keep seeded prior initialization without model-level initval overrides,
            # which native model cloning cannot transport.
            support_point=cls._ordered_support_point,
        )

    @classmethod
    def scaled(cls, name, prior, scale, *, shape, noncentered, offset=0.0):
        """Use a native affine density or a direct expression of a standardized RV."""
        if noncentered:
            raw = bind_prior(prior, name=f"{name}_raw", shape=shape)
            return offset + scale * raw
        op = prior.owner.op
        parameters = op.dist_params(prior.owner)
        transform = None
        if _default_transform(op, prior) is not None:
            lower, upper = scalar_prior_bounds(prior)

            def bounds(size, scale, offset, *inputs):
                return (
                    None if lower is None else offset + scale * lower,
                    None if upper is None else offset + scale * upper,
                )

            transform = IntervalTransform(bounds)
        inputs, output = op.signature.split("->")
        return pm.CustomDist(
            name,
            scale,
            offset,
            *parameters,
            dist=partial(cls._scaled_dist, prior=prior),
            signature=f"{output},{output}{',' + inputs if inputs else ''}->{output}",
            default_transform=transform,
            shape=shape,
        )

    @classmethod
    def _ordered_dist(cls, template, *parameters_and_size, prior, count):
        *parameters, size = parameters_and_size
        if rv_size_is_none(size):
            size = (
                implicit_size_from_params(*parameters, ndims_params=prior.owner.op.ndims_params)
                if parameters
                else pt.constant([], dtype="int64")
            )
        value = prior_with_parameters(
            prior, *cls._ordered_parameters(prior, parameters), size=pt.concatenate([size, [count]])
        )
        # Native sorting accepts symbolic draw sizes and lowers to JAX when compiled.
        return pt.sort(value, axis=-1)

    @staticmethod
    def _ordered_support_point(rv, size, template, *parameters):
        return rv

    @classmethod
    def _ordered_logp(cls, value, template, *parameters, prior, count):
        value = pt.specify_shape(value, (*value.type.shape[:-1], count))
        distribution = prior_with_parameters(prior, *cls._ordered_parameters(prior, parameters))
        density = pm.logp(distribution, value)
        return pt.where(
            pt.all(pt.diff(value, axis=-1) > 0, axis=-1),
            pt.sum(density, axis=-1)
            + pt.gammaln(pt.as_tensor_variable(count + 1, dtype=density.dtype)),
            -np.inf,
        )

    @classmethod
    def _ordered_parameters(cls, prior, parameters):
        return tuple(
            pt.expand_dims(parameter, axis=-(ndim + 1))
            for parameter, ndim in zip(parameters, prior.owner.op.ndims_params, strict=True)
        )

    @staticmethod
    def _scaled_dist(scale, offset, *parameters_and_size, prior):
        *parameters, size = parameters_and_size
        return offset + scale * prior_with_parameters(prior, *parameters, size=size)

    @staticmethod
    def _ordered_transform(prior):
        transform = _default_transform(prior.owner.op, prior)
        if transform is None:
            return pm.distributions.transforms.Ordered()
        if isinstance(transform, LogTransform):
            return pm.distributions.transforms.Ordered(positive=True)
        ascending = True
        if isinstance(transform, IntervalTransform):
            lower, upper = prior_support(prior)
            upper_only = np.isneginf(lower) & np.isfinite(upper)
            if np.any(upper_only) and not np.all(upper_only):
                raise ValueError("Ordered interval priors require one common monotone direction")
            ascending = not np.all(upper_only)
            bounds = tuple(
                None
                if np.ndim(bound) == 0 and bound == unbounded
                else pt.as_tensor_variable(bound)[..., None]
                for bound, unbounded in ((lower, -np.inf), (upper, np.inf))
            )
            transform = IntervalTransform(lambda *inputs: bounds)
        return pm.distributions.transforms.Chain(
            [transform, pm.distributions.transforms.Ordered(ascending=ascending)]
        )


def slot_values(batch, parameters):
    """Gather final nodes using their sole, already prepared parameter references."""
    result = {}
    for slot in batch.slots:
        # Keep a one-dimensional gather and restore the prepared kernel shape.
        values = pt.take(parameters[slot.kind].ravel(), slot.entry_index.ravel(), axis=0)
        shape = slot.entry_index.shape
        result[slot.key] = pt.specify_shape(pt.cast(values, config.floatX).reshape(shape), shape)
    return result


def register_response(batch, distribution, *, forward=False):
    """Register a marginal RV for observed responses or forward generation."""
    if batch.y is None and not forward:
        return distribution
    return pm.modelcontext(None).register_rv(
        distribution, batch.response_name, observed=batch.y, dims=(batch.obs_dim,)
    )


def response_rv(
    kernel,
    arguments,
    *,
    batch=None,
    size=None,
    beta=False,
    checks=(),
    message="",
    smooth=None,
    **options,
):
    """Build a native response and bind the same kernel inputs for information."""
    if batch is not None and not beta:
        order = batch.category_order

        @wraps(kernel)
        def in_observed_order(*values, **settings):
            return kernel(*values, **settings)[..., order]

        response_kernel = in_observed_order
    else:
        response_kernel = kernel
    output = wrap_jax(response_kernel, allow_eval=False)(*arguments, **options)
    if beta:
        alpha, beta_shape = output
        alpha = check_parameters(alpha, *checks, msg=message)
        distribution = pm.Beta.dist(alpha=alpha, beta=beta_shape, size=size)
    else:
        logs = check_parameters(output, *checks, msg=message) if checks else output
        distribution = StableCategorical.dist(logs, size=size)
    if batch is None:
        return distribution
    return ResponseBinding(
        batch=batch,
        parameters=arguments[1:],
        kernel=partial(response_kernel, **options),
        distribution=register_response(batch, distribution),
        smooth=smooth,
    )


def observe_augmented(batch, contrasts, transform, membership):
    """Register the contrast RV and its conditioned membership in the active graph."""
    model = pm.modelcontext(None)
    differences = model.register_rv(
        contrasts,
        f"{batch.name}_contrast_raw",
        transform=transform,
        dims=(batch.obs_dim, f"{batch.name}_difference"),
    )
    observed = model.register_rv(
        membership(differences),
        f"{batch.name}_event_in_region",
        observed=np.ones(batch.statement.shape[0], dtype=np.int64),
        dims=(batch.obs_dim,),
    )
    return differences, observed
