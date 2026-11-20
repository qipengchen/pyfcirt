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
    selections validate the requested backend. Accelerators retain CPU for covariate
    encoding. Existing arrays and initialized devices follow native JAX behavior;
    use a fresh session when changing platforms. Failed backend selection
    restores the prior settings. Precision applies to JAX and PyTensor.
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
    if platform is not None:
        current = jax.config.values
        initialized = backends_are_initialized()
        try:
            for name, value in updates.items():
                jax.config.update(name, value)
            jax.devices(platform)
        except BaseException:
            for name in updates:
                jax.config.update(name, current[name])
            # A failed probe can initialize a different default backend. Clear
            # only clients created during the probe.
            if not initialized and backends_are_initialized():
                clear_backends()
            raise
    jax.config.update("jax_enable_x64", precision == "float64")
    pytensor.config.floatX = precision


set_platform()
