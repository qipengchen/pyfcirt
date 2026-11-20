# Installation

For pip or conda, use Python 3.12 or later.
Until the beta is published, [install from GitHub](#from-github).

## pip

```console
python -m pip install --pre pyfcirt
```

For NVIDIA GPU sampling on Linux or WSL 2 with CUDA 13, use the CUDA extra.

```console
python -m pip install --pre "pyfcirt[cuda13]"
```

Use `pyfcirt[cuda12]` if your system requires CUDA 12. See the
[GPU setup](examples/gpu.md) for driver requirements.

Add the `plots`, `io`, or `covariates` extras when you need plotting,
NetCDF saving, or covariate formulas.

### From GitHub

```console
python -m pip install "pyfcirt @ https://github.com/qipengchen/pyfcirt/archive/refs/heads/main.zip"
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

**CUDA 12 image**

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
