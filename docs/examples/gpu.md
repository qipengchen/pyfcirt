# Use JAX on a GPU

Install PyFCIRT following [installation](../installation.md), then choose a
compatible build from the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
For CUDA 12 on a supported Linux NVIDIA system:

```console
python -m pip install -U "jax[cuda12]"
```

Check the JAX guide for the driver requirements for your machine. Select the GPU
before preparing data; float64 and BlackJAX are already package defaults:

```python
import pyfcirt as fc

fc.set_platform("gpu")
example = fc.load_example("tirt_study", model="TIRT")
model = fc.Model()
model.prepare(example.data)
model.fit(chains=4, random_seed=42)
```

BlackJAX and NumPyro use vectorized chains on GPU by default. Vectorized BlackJAX
also defaults to `progressbar=False`, because its native progress callback cannot
run inside that vectorization. Explicit settings pass through to PyMC.
On CPU, the default chain method is parallel, with CPU concurrency detected by
PyMC. `cores` does not create additional GPU devices.

`set_platform("gpu")` validates GPU availability and retains CPU for covariate
encoding. Native Windows JAX uses CPU; use a supported Linux environment for
NVIDIA GPU analysis. [The estimator example](estimation_backends.md) separates
sampler selection from graph compilation. Inspect convergence and simulation
precision as well as elapsed time.
