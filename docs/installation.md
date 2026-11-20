# Installation

For pip or conda, use Python 3.12 or later.
Until the beta is published, [install from GitHub](#from-github).

## pip

```console
python -m pip install --pre "pyfcirt[plots,io,covariates]"
```

This command includes plotting, NetCDF saving, and covariate formulas. For
NVIDIA GPU sampling with JAX, add the `cuda12` or `cuda13` extra. For driver
requirements, see [GPU setup](examples/gpu.md).

### From GitHub

```console
python -m pip install "pyfcirt[plots,io,covariates] @ https://github.com/qipengchen/pyfcirt/archive/refs/heads/main.zip"
```

## conda

Install into your current conda environment.

```console
conda install -c qipengchen/label/beta -c conda-forge pyfcirt
```

## Docker

`docker run` pulls the image from GitHub Container Registry and opens Python.

**CPU**

```console
docker run --rm -it ghcr.io/qipengchen/pyfcirt:0.1.0b1-cpu
```

**NVIDIA GPU (CUDA 12)**

Use a Docker host configured for NVIDIA GPU access.

```console
docker run --rm -it --gpus all ghcr.io/qipengchen/pyfcirt:0.1.0b1-cuda12
```

## Quick check

Open Python and import PyFCIRT.

```python
import pyfcirt
```

Continue with the [Quick start](guide/getting_started.md).
