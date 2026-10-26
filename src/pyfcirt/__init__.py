"""Bayesian forced-choice IRT modeling with PyMC and ArviZ."""

from importlib.metadata import version

from .data import FCData, FCDesign
from .model import Model
from .priors import Prior

__version__ = version("pyfcirt")


def load(path):
    """Load portable native draws and metadata; graph reconstruction is explicit."""
    from .io import load as load_result

    return load_result(path)


__all__ = ["FCData", "FCDesign", "Model", "Prior", "load", "__version__"]
