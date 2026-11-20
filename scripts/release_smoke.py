"""Verify the installed release through its public API."""

import sys

import jax
import numpy as np

import pyfcirt as fc

assert fc.__version__ == sys.argv[1], fc.__version__
example = fc.load_example("tirt_study", model="TIRT", n_persons=3)
model = fc.Model().prepare(example.data)
assert model.build().observed_RVs
assert not model.data_summary().empty
print(f"PyFCIRT {fc.__version__}: prepared and built a bundled TIRT example")

if "--gpu" in sys.argv:
    assert jax.default_backend() == "gpu", jax.devices()
    print(f"GPU devices: {jax.devices()}")
    generated = fc.simulate(
        "MGPPM", n_persons=10, n_traits=2, n_blocks=3, block_size=2, random_seed=42
    )
    result = generated.fit(
        nuts_sampler="blackjax", chains=1, draws=20, tune=20, progressbar=False, random_seed=42
    )
    theta = result.posterior.theta
    assert theta.sizes["chain"] == 1 and theta.sizes["draw"] == 20
    assert np.isfinite(theta.values).all()
    print("BlackJAX GPU smoke test completed (1 chain, 20 draws)")
