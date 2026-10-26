"""Serializable prior declarations using PyMC distribution names and parameters."""

from __future__ import annotations

import json
from copy import deepcopy
from inspect import Parameter, signature


class Prior:
    """Describe a distribution without creating a random variable.

    Parameters follow PyMC's naming conventions. The model validates their
    compatibility with each role; dimensions and identification are model-owned.
    Custom executable factories and hierarchical declarations are not supported
    by this initial serializable interface.
    """

    def __init__(self, name: str, **parameters: float) -> None:
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError("Prior name must be a public PyMC distribution name")
        try:
            parameters = json.loads(json.dumps(parameters, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Prior parameters must be finite JSON-compatible values") from error
        self._name = name
        self._parameters = parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    def to_dict(self) -> dict:
        return {"name": self.name, "parameters": self.parameters}

    def __repr__(self) -> str:
        arguments = ", ".join(f"{k}={v!r}" for k, v in self._parameters.items())
        return f"Prior({self._name!r}{', ' if arguments else ''}{arguments})"


def validate_prior(prior: Prior, role: str) -> None:
    """Check a role's support and scalar hyperparameters before graph construction."""
    import pymc as pm
    from pymc.distributions.continuous import PositiveContinuous, UnitContinuous

    if not isinstance(prior, Prior):
        raise TypeError(f"{role}: expected a Prior declaration")
    distribution = getattr(pm, prior.name, None)
    if not isinstance(distribution, type) or not issubclass(distribution, pm.Distribution):
        raise ValueError(f"{role}: {prior.name!r} is not a PyMC distribution")
    parameters = prior.parameters
    reserved = {
        "shape",
        "size",
        "dims",
        "observed",
        "transform",
        "initval",
        "n",
        "n_zerosum_axes",
        "support_shape",
    }
    if set(parameters) & reserved:
        raise ValueError(f"{role}: dimensions and identification parameters are managed by Model")
    for name, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{role}.{name}: this version requires scalar numeric hyperparameters")
        positive_parameters = {"sigma", "alpha", "beta", "nu", "lam", "eta", "tau", "b", "scale"}
        if name in positive_parameters and value <= 0:
            raise ValueError(f"{role}.{name} must be strictly positive")
    if "lower" in parameters and "upper" in parameters:
        if parameters["upper"] <= parameters["lower"]:
            raise ValueError(f"{role}: upper must exceed lower")
    if role == "loading_magnitude":
        positive = issubclass(distribution, (PositiveContinuous, UnitContinuous))
        bounded_positive = (
            prior.name in {"Uniform", "TruncatedNormal"}
            and parameters.get("lower", 0 if prior.name == "Uniform" else -float("inf")) >= 0
        )
        if not positive and not bounded_positive:
            raise ValueError("loading_magnitude requires a distribution with positive support")
    elif role == "ranking_intercept":
        if prior.name != "ZeroSumNormal" or set(parameters) - {"sigma"}:
            raise ValueError("ranking_intercept requires ZeroSumNormal with a scalar sigma")
    elif role == "trait_correlation":
        if prior.name != "LKJCorr" or set(parameters) - {"eta"}:
            raise ValueError("trait_correlation requires LKJCorr with a scalar eta")
    elif role == "graded_intercept":
        if not issubclass(distribution, pm.Continuous):
            raise ValueError("graded_intercept requires an independent scalar distribution")
    if role in {"loading_magnitude", "graded_intercept"}:
        declared = {
            name
            for name, argument in signature(distribution.dist).parameters.items()
            if argument.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        }
        unknown = set(parameters) - declared
        if unknown:
            raise ValueError(f"{role}: invalid parameters for {prior.name}: {sorted(unknown)}")
        try:
            random_variable = distribution.dist(**parameters)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{role}: invalid parameters for {prior.name}: {error}") from error
        if random_variable.owner.op.ndim_supp != 0:
            raise ValueError(f"{role}: expected a scalar-support distribution")
