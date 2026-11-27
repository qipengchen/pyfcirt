"""One boundary for constant native PyMC prior templates and graph ownership.

Descriptions expose constant inputs for equality and display. Construction uses the template.
Dependent/custom laws belong in a user-owned ``pm.Model`` with the measurement
distributions. Native parameter order and event semantics are retained throughout.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from inspect import Parameter, signature

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.distribution import SymbolicRandomVariable
from pymc.distributions.shape_utils import change_dist_size, rv_size_is_none
from pymc.distributions.transforms import _default_transform
from pymc.exceptions import NotConstantValueError
from pymc.logprob.transforms import IntervalTransform, LogOddsTransform, LogTransform
from pymc.printing import str_for_dist
from pymc.pytensorf import constant_fold, resolve_shapes, rvs_in_graph
from pytensor import config
from pytensor.graph.basic import Constant
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable

from pyfcirt.covariates.pairwise import PairwiseL1RV
from pyfcirt.covariates.spike_slab import SpikeAndSlabRV

_preparation = ContextVar("prior_preparation", default=None)


class PriorPreparation:
    """Inspect immutable prior templates once within a single preparation call.

    Cached records are copied before publication. Templates are only inputs to
    runtime binding: each model parameter still creates its own independent RV.
    Context-local lifetime prevents reuse after a caller changes input values.
    """

    def __init__(self):
        self.templates = {}
        self.labels = {}
        self.domains = set()
        self.defaults = {}

    def __enter__(self):
        self.token = _preparation.set(self)
        return self

    def __exit__(self, *exc):
        _preparation.reset(self.token)

    @staticmethod
    def key(value):
        """Hash runtime metadata directly, including native templates and NumPy keys."""
        if isinstance(value, TensorVariable):
            return _inspect(value).key
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    ((PriorPreparation.key(k), PriorPreparation.key(v)) for k, v in value.items()),
                    key=repr,
                )
            )
        if isinstance(value, np.ndarray):
            return value.dtype.str, value.shape, value.tobytes()
        if isinstance(value, (tuple, list)):
            return tuple(PriorPreparation.key(v) for v in value)
        if isinstance(value, np.generic):
            value = value.item()
        return type(value), value

    @staticmethod
    def default(factory, **arguments):
        """Reuse one native default per law and its actual arguments."""
        context = _preparation.get()
        if context is None:
            return factory(**arguments)
        # Native parameter tensors can distinguish an integer from an equal float.
        key = factory, config.floatX, PriorPreparation.key(arguments)
        if key not in context.defaults:
            context.defaults[key] = factory(**arguments)
        return context.defaults[key]


def _unsupported(role, law, reason):
    return ValueError(
        f"{role}: native prior {law} {reason}. Use a user-owned pm.Model with the "
        "measurement distributions and retain its builder for this law"
    )


def _native(prior):
    if not isinstance(prior, TensorVariable) or prior.owner is None:
        raise TypeError("A prior must be an unnamed native pm.*.dist random variable")
    if prior.name is not None:
        raise ValueError("A prior template must be unnamed and unobserved; use pm.*.dist")
    op = prior.owner.op
    if not (
        type(op).__module__.startswith(("pytensor.", "pymc."))
        or isinstance(op, (PairwiseL1RV, SpikeAndSlabRV))
    ) or not (
        isinstance(op, RandomVariable)
        or isinstance(op, SymbolicRandomVariable)
        and op.extended_signature is not None
    ):
        raise _unsupported("prior", type(op).__qualname__, "is not a native RV template")
    if np.dtype(prior.dtype).kind != "f":
        raise ValueError(f"{str_for_dist(prior, include_params=False)}: a prior must be continuous")
    parameters = tuple(op.dist_params(prior.owner))
    if rvs_in_graph(parameters):
        raise _unsupported(
            "prior", str_for_dist(prior, include_params=False), "has unrelated random ancestors"
        )
    return op, parameters, op.size_param(prior.owner)


def _constants(prior, values):
    if all(isinstance(value, Constant) for value in values):
        return tuple(value.data for value in values)
    try:
        return constant_fold(values)
    except NotConstantValueError as error:
        law = str_for_dist(prior, include_params=False)
        raise _unsupported("prior", law, "requires constant native parameters") from error


def _labels(op, count):
    context = _preparation.get()
    key = type(op), count
    if context is not None and key in context.labels:
        return list(context.labels[key])
    methods = (
        (op.rng_fn, op.make_node, op.__call__)
        if isinstance(op, RandomVariable)
        else (type(op).rv_op,)
    )
    for method in methods:
        arguments = [
            name
            for name, value in signature(method).parameters.items()
            if value.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
            and name not in {"rng", "size"}
        ]
        if len(arguments) == count:
            if context is not None:
                context.labels[key] = arguments
            return arguments
    return [f"arg:{i}" for i in range(count)]


@dataclass(frozen=True, slots=True)
class _Template:
    batch: tuple
    event: tuple
    arguments: tuple
    key: tuple


def _inspect(prior):
    """Fold the native inputs once while preparing a candidate."""
    context = _preparation.get()
    if context is not None and prior in context.templates:
        return context.templates[prior]
    op, parameters, size = _native(prior)
    size = () if rv_size_is_none(size) else (size,)
    shape = prior.type.shape
    shape_expr = resolve_shapes([prior.shape]) if None in shape else ()
    inputs = _constants(prior, (*shape_expr, *parameters, *size))
    if shape_expr:
        shape, *inputs = inputs
    batch_size, inputs = (tuple(inputs[-1]), inputs[:-1]) if size else (None, inputs)
    values = tuple(np.asarray(value).view() for value in inputs)
    for value in values:
        value.flags.writeable = False
    if any(value.dtype.kind not in "biuf" for value in values):
        raise ValueError("Native prior parameters must be real numeric arrays")
    shape = tuple(map(int, shape))
    batch_ndim = len(shape) - op.ndim_supp
    key = op, prior.dtype, batch_size, PriorPreparation.key(values)
    template = _Template(shape[:batch_ndim], shape[batch_ndim:], values, key)
    if context is not None:
        context.templates[prior] = template
    return template


def prior_shape(prior):
    template = _inspect(prior)
    return template.batch, template.event


def describe_prior(prior):
    """Return the family name and named constant arguments for presentation."""
    template = _inspect(prior)
    names = _labels(prior.owner.op, len(template.arguments))
    return str_for_dist(prior, include_params=False), zip(names, template.arguments, strict=True)


def scalar_prior_bounds(prior):
    """Read interval endpoints from the native scalar value transform."""
    node = prior.owner
    transform = _default_transform(node.op, prior)
    if transform is None:
        return None, None
    if isinstance(transform, LogTransform):
        return 0.0, None
    if isinstance(transform, LogOddsTransform):
        return 0.0, 1.0
    if isinstance(transform, IntervalTransform):
        return transform.args_fn(*node.inputs)
    raise ValueError("This native transform cannot express scalar bounds")


def prior_support(prior: TensorVariable) -> tuple[object, object]:
    """Read constant scalar bounds supplied by the native prior."""
    try:
        bounds = scalar_prior_bounds(prior)
    except ValueError:
        # Joint/circular transforms have no scalar interval to validate.
        return -np.inf, np.inf
    endpoints = tuple(
        default if bound is None else bound
        for bound, default in zip(bounds, (-np.inf, np.inf), strict=True)
    )
    if any(isinstance(bound, TensorVariable) for bound in endpoints):
        return _constants(prior, tuple(map(pt.as_tensor_variable, endpoints)))
    return endpoints


def prior_coordinates(prior, *, noncentered):
    """Describe the native value map, including constants used by that map."""
    op = prior.owner.op
    if noncentered:
        return "affine", _inspect(prior).arguments
    transform = _default_transform(op, prior)
    if isinstance(transform, IntervalTransform):
        return "interval", prior_support(prior)
    return type(transform).__qualname__, op.ndim_supp


def same_prior(left: TensorVariable, right: TensorVariable) -> bool:
    """Compare canonical plain native configurations rather than symbolic equality."""
    return _inspect(left).key == _inspect(right).key


def require_proper(prior: TensorVariable, *, role: str) -> None:
    """Reject native improper laws where the role requires a proper distribution."""
    _inspect(prior)
    if isinstance(prior.owner.op, (pm.Flat.rv_type, pm.HalfFlat.rv_type)):
        raise ValueError(f"{role} requires a proper prior")


def subset_prior(prior, positions, original_shape) -> TensorVariable:
    """Retain selected independent batch entries without changing their native law."""
    template = _inspect(prior)
    positions = np.asarray(positions, dtype=np.int64)
    if template.event:
        if np.array_equal(positions, np.arange(np.prod(original_shape))):
            return prior
        raise ValueError("An inseparable native joint prior must be replaced as a whole block")
    selected = []
    for value, core_ndim in zip(template.arguments, prior.owner.op.ndims_params, strict=True):
        core_shape = value.shape[-core_ndim:] if core_ndim else ()
        if value.ndim == core_ndim:
            selected.append(value)
        else:
            values = np.broadcast_to(value, (*original_shape, *core_shape))
            selected.append(values.reshape((-1, *core_shape))[positions])
    return prior_with_parameters(prior, *selected, size=(len(positions),))


def validate_prior(
    prior: TensorVariable, *, shape: tuple[int, ...], role: str = "prior", domain: str = "real"
) -> None:
    """Check the prepared batch/event axes and domain-specific model invariants."""
    try:
        template = _inspect(prior)
    except ValueError as error:
        raise ValueError(f"{role}: {error}") from error
    shape = tuple(shape)
    context = _preparation.get()
    key = template.key, domain, shape
    if context is not None and key in context.domains:
        return
    batch, event = template.batch, template.event
    if event and shape[-len(event) :] != event:
        raise ValueError(f"Native joint event {event} does not match requested free shape {shape}")
    target_batch = shape[: -len(event)] if event else shape
    if np.broadcast_shapes(batch, target_batch) != target_batch:
        raise ValueError(f"Native batch shape {batch} cannot broadcast to {target_batch}")
    if domain == "unit":
        lower, upper = prior_support(prior)
        if not (np.all(np.asarray(lower) >= 0) and np.all(np.asarray(upper) <= 1)):
            raise _unsupported(
                role,
                str_for_dist(prior, include_params=False),
                "does not establish support inside [0, 1]",
            )
    elif domain == "correlation" and not isinstance(prior.owner.op, pm.LKJCorr.rv_type):
        raise _unsupported(
            role,
            str_for_dist(prior, include_params=False),
            "does not provide an LKJ correlation matrix",
        )
    if context is not None:
        context.domains.add(key)


def prior_with_parameters(prior, *parameters, size=None):
    """Bind inputs with the original PyMC or PyTensor distribution operator."""
    op = prior.owner.op
    with config.change_flags(floatX=prior.dtype):
        if isinstance(op, SymbolicRandomVariable):
            return op.rebuild_rv(*parameters, size=size)
        _, rebuilt = op(
            *parameters, size=size, rng=pt.random.shared_rng(seed=None), return_next_rng=True
        )
        return rebuilt


def bind_prior(
    prior: TensorVariable, *, name: str, shape: tuple[int, ...], dims=None
) -> TensorVariable:
    """Give one independent owner a native draw, preserving event axes/transforms."""
    op = prior.owner.op
    if isinstance(op, SpikeAndSlabRV):
        return op.bind(prior, name=name, shape=shape, dims=dims)
    event_ndim = op.ndim_supp
    batch_shape = shape[:-event_ndim] if event_ndim else shape
    # Native symbolic resizing rebuilds the law, so preserve its dtype here too.
    with config.change_flags(floatX=prior.dtype):
        owned = change_dist_size(prior, new_size=batch_shape, expand=False)
    return pm.modelcontext(None).register_rv(owned, name, dims=dims)
