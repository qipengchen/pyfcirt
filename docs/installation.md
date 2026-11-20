# Installation

Choose one installation method. pip and conda require **Python 3.12 or newer**.
The registry commands below become available when the beta is published.

## pip

```console
python -m pip install --pre "pyfcirt[plots,io,covariates]"
```

The extras add plotting, NetCDF saving, and covariate formulas. For GPU
sampling with JAX, add `cuda12` or `cuda13` to the extras. See the
[GPU setup](examples/gpu.md) for driver requirements.

To install from GitHub before the beta is available:

```console
python -m pip install "pyfcirt[plots,io,covariates] @ https://github.com/qipengchen/pyfcirt/archive/refs/heads/main.zip"
```

## conda

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

```console
docker run --rm -it --gpus all ghcr.io/qipengchen/pyfcirt:0.1.0b1-cuda12
```

## Quick check

Run `import pyfcirt` in Python, then continue to the
[Quick start](guide/getting_started.md).
