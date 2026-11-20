# set_platform

```{eval-rst}
.. autofunction:: pyfcirt.set_platform
```

Choose a computation device and floating-point precision.

## Parameters

`platform` : str or None
: `"cpu"`, `"gpu"` or `None` to preserve JAX's device selection.
  Explicit selections check that the device is available.

`precision` : {"float32", "float64"}
: Numeric precision for JAX and PyTensor. Default: `"float64"`.

## Returns

None
: Configures the process. Failed device selection restores the previous settings.

## Notes

Call before preparing data or building graphs. Use a fresh Python session
when changing platforms. The CPU remains available for covariate encoding.

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
