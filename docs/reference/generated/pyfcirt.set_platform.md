# set_platform

```{eval-rst}
.. autofunction:: pyfcirt.set_platform
```

Choose a computation device and floating-point precision.

## Parameters

`platform` : str or None
: `"cpu"`, `"gpu"`, or a native JAX backend name such as `"cuda"`.
  Explicit selections check device availability. `None` preserves JAX's selection.

`precision` : {"float32", "float64"}
: Numeric precision for JAX and PyTensor. Default: `"float64"`.

## Returns

None
: Configures the process. A failed update restores the previous settings.

## Notes

Call before preparing data or building graphs. Use a fresh Python session
when changing platforms. The CPU backend remains available when using an accelerator.

## Examples

```python
import jax
import pyfcirt as fc

fc.set_platform("cpu", precision="float64")
print(jax.default_backend(), jax.config.x64_enabled)
```

```text
cpu True
```

## See also

[GPU setup](../../examples/gpu.md).
