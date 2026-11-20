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

## Beta packages

Once version `0.1.0b1` is published, install the same beta from PyPI or
the Anaconda.org beta channel.

```console
python -m pip install "pyfcirt==0.1.0b1"
```

```console
conda create -n pyfcirt-beta --override-channels -c qipengchen/label/beta -c conda-forge python=3.12 pyfcirt=0.1.0b1
conda activate pyfcirt-beta
python -c "import pyfcirt as fc; assert fc.__version__ == '0.1.0b1'; print(fc.__version__)"
```

The beta container tags are `0.1.0b1-cpu` and `0.1.0b1-cuda13`.
Use an NVIDIA Docker host with a CUDA 13-compatible driver for the GPU image.
Confirm that JAX selects the GPU inside the container.

```console
docker run --rm ghcr.io/qipengchen/pyfcirt:0.1.0b1-cpu python -c "import pyfcirt; print(pyfcirt.__version__)"
docker run --rm --gpus all ghcr.io/qipengchen/pyfcirt:0.1.0b1-cuda13 python -c "import pyfcirt, jax; assert jax.default_backend() == 'gpu'; print(jax.devices())"
```
