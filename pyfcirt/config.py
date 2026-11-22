"""Process-wide device and floating-point settings through native library APIs."""

from typing import Literal

import jax
import pymc.sampling.jax  # noqa: F401 -- initialize PyMC's native CPU-chain setup before arrays
import pytensor
from jax._src.xla_bridge import backends_are_initialized
from jax.extend.backend import clear_backends


def set_platform(
    platform: str | None = None,
    *,
    precision: Literal["float32", "float64"] = "float64",
) -> None:
    """Choose device and precision before preparing data or building graphs.

    None preserves JAX device selection without initializing backends. Explicit
    selections validate the requested backend and keep CPU available. Existing
    arrays and initialized devices follow native JAX behavior; use a fresh
    session when changing platforms. Failed updates restore the prior settings.
    Precision applies to JAX and PyTensor.
    Importing PyFCIRT applies the float64 default once.
    """
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")
    updates = {}
    if platform == "gpu":
        # An explicit GPU list requires every vendor backend; native preference
        # resolves the alias to an available GPU and also controls parallel sampling.
        updates.update(jax_platforms=None, jax_platform_name="gpu")
    elif platform is not None:
        updates.update(
            jax_platform_name="", jax_platforms="cpu" if platform == "cpu" else f"{platform},cpu"
        )
    current = jax.config.values
    previous_precision = pytensor.config.floatX
    may_initialize = platform is not None and not backends_are_initialized()
    try:
        for name, value in updates.items():
            jax.config.update(name, value)
        if platform is not None:
            jax.devices(platform)
        # Set PyTensor first. JAX's native precision update writes the global value.
        pytensor.config.floatX = precision
        jax.config.update("jax_enable_x64", precision == "float64")
    except BaseException:
        if pytensor.config.floatX != previous_precision:
            pytensor.config.floatX = previous_precision
        for name in updates:
            jax.config.update(name, current[name])
        # Clear only clients created during this call.
        if may_initialize and backends_are_initialized():
            clear_backends()
        raise


set_platform()
