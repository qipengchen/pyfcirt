# Use JAX on a GPU

Start with the [installation commands](../guide/getting_started.md#installation).
JAX supports NVIDIA GPU acceleration on Linux. GPU support under WSL 2 is
experimental. CUDA 13 requires an NVIDIA GPU with compute capability 7.5 or
newer and, on Linux, driver version 580 or newer. Check the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)
for platform-specific requirements.

For Docker, configure the host for NVIDIA GPU access and start the CUDA 13
image with `--gpus all`. To use CUDA 12 through pip, install:

```console
python -m pip install --pre "pyfcirt[cuda12]"
```

If you encounter compatibility issues, a different CUDA version may help.

Verify that JAX lists a CUDA device. For Docker, pass the command after the image name.

```console
python -c "import jax; print(jax.devices())"
```

Select the GPU before preparing data. Float64 and BlackJAX are package defaults:

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
