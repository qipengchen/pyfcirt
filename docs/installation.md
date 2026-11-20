# Installation

Use **Python 3.12 or newer**. Install PyFCIRT from GitHub with one command:

```console
python -m pip install "pyfcirt[plots,io,covariates] @ https://github.com/qipengchen/pyfcirt/archive/refs/heads/main.zip"
```

This includes plotting, NetCDF saving and covariate formulas. It installs the
current `main` source without a separate Git checkout; CPU analysis needs no CUDA setup.

Check the installation:

```console
python -c "import pyfcirt; print(pyfcirt.__version__)"
```

Continue with the [Quick start](guide/getting_started.md).

## Optional NVIDIA GPU support

On a compatible Linux machine, follow the [GPU setup](examples/gpu.md).
For installation or import errors, see [Troubleshooting](howto/troubleshooting.md).
Sampler and precision settings belong in the [fitting guide](reference/generated/pyfcirt.Model.fit.md).
