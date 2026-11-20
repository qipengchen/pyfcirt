"""Verify the installed package or the GPU device in a release candidate."""

import sys

import pyfcirt as fc

assert fc.__version__ == sys.argv[1], fc.__version__
if "--gpu" in sys.argv:
    import jax
    import jax.numpy as jnp

    assert jax.default_backend() == "gpu", jax.devices()
    result = jax.jit(lambda x: x * x)(jnp.arange(4, dtype=jnp.float32))
    assert result.device.platform == "gpu" and result.tolist() == [0.0, 1.0, 4.0, 9.0]
    print(f"GPU: {result.device}; squared values: {result.tolist()}")
else:
    example = fc.load_example("tirt_study", model="TIRT", n_persons=3)
    model = fc.Model().prepare(example.data)
    assert model.build().observed_RVs
    assert not model.data_summary().empty
    print(f"PyFCIRT {fc.__version__}: prepared and built a bundled TIRT example")
