"""Finite model declarations; resolve native defaults once during preparation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import ClassVar, Literal
from warnings import warn

import numpy as np
import pymc as pm
from pytensor.tensor.variable import TensorVariable

from pyfcirt.models.likert import DichotomousIRTDistribution, GGUMDistribution, GPCMDistribution
from pyfcirt.models.mgppm import MGPPMDistribution
from pyfcirt.models.priors import (
    PriorPreparation,
    prior_support,
)
from pyfcirt.models.sequential import GGUMRankDistribution, TwoPLRankDistribution
from pyfcirt.models.thurstonian.gtum import GTUMDistribution
from pyfcirt.models.thurstonian.rank import TIRTRankDistribution
from pyfcirt.models.thurstonian.tirt import TIRT_LOGIT_MULTIPLIER, TIRTDistribution


class IdentificationWarning(UserWarning):
    """A declared prior relaxes a default orientation convention."""


@dataclass(frozen=True)
class _RoleDefault:
    """Ordinary factory and an independent, explicit built-in conditional scheme."""

    ordinary: Callable
    family: str | None = None
    center: float = 0.0
    bounds: tuple | None = None
    link: str = "identity"
    coefficient_sd: float = 1.0
    residual_sd: float = 1.0


def _normal_role(mu=0.0, sigma=1.0, *, bounds=None):
    return _RoleDefault(
        ordinary=(
            (lambda: pm.Normal.dist(mu=mu, sigma=sigma))
            if bounds is None
            else (
                lambda: pm.TruncatedNormal.dist(
                    mu=mu, sigma=sigma, lower=bounds[0], upper=bounds[1]
                )
            )
        ),
        family="Normal" if bounds is None else "TruncatedNormal",
        center=mu,
        bounds=bounds,
    )


def _loading_role(mu=0.0, sigma=0.5):
    return _RoleDefault(
        ordinary=lambda: pm.LogNormal.dist(mu=mu, sigma=sigma),
        family="LogNormal",
        center=mu,
        link="log",
        coefficient_sd=0.25,
        residual_sd=0.2,
    )


def _copy_priors(priors, *, allowed: frozenset[str]) -> Mapping:
    """Check declaration keys/types; prepared layouts own native domain validation."""
    values = {} if priors is None else dict(priors)
    if unknown := values.keys() - allowed:
        raise ValueError(f"Unknown prior roles: {sorted(unknown)}")
    for role, prior in values.items():
        if not isinstance(prior, TensorVariable):
            raise TypeError(f"{role}: a prior must be a native pm.*.dist random variable")
    return MappingProxyType(values)


@dataclass(frozen=True, kw_only=True)
class _Parameter:
    """Static construction facts; preparation resolves axes and design constraints."""

    key: str
    kind: str | None = None
    axes: tuple[str, ...]
    role: str | None = None
    scope: str = "owner"
    component: str = "measurement"
    summary_default: bool = True
    fixed: float = np.nan
    fixed_source: str = "model"
    reference_value: float = np.nan
    graded_role: str | None = None
    reference: Literal["first", "diagonal", "group", "group_diagonal"] | None = None
    derived: bool = False
    domain: Literal["real", "positive", "unit"] = "real"
    bounds: tuple[str, float] | None = None
    ordered: bool = False
    shared: bool = False
    regression: bool = False
    person_link: Literal["identity", "log"] | None = None
    fixed_for_scoring: bool = True


_STATEMENT_LOADING = _Parameter(
    key="loading_magnitude", kind="loading", axes=("statement",), regression=True
)
_ITEM_LOADING = replace(_STATEMENT_LOADING, axes=("item", "trait"))
_STATEMENT_LOCATION = _Parameter(key="location", axes=("statement",), regression=True)
_RANKING_INTERCEPT = _Parameter(
    key="intercept",
    kind="statement_intercept",
    axes=("statement",),
    role="ranking_intercept",
    regression=True,
)
_RESIDUAL_VARIANCE = _Parameter(
    key="residual_variance",
    axes=("statement",),
    reference_value=0.5,
    domain="positive",
    person_link="log",
)
_STATEMENT_LOCATION_PRIORS = {
    "location_positive": _normal_role(2, bounds=(0, None)),
    "location_negative": _normal_role(-2, bounds=(None, 0)),
    "location_free": _normal_role(),
}
_RESIDUAL_VARIANCE_PRIOR = _RoleDefault(lambda: pm.HalfNormal.dist(sigma=1))
_ITEM_LOADING_PRIOR = _loading_role(mu=0.25)
_CATEGORY_INTERCEPT = _Parameter(
    key="category_intercept",
    axes=("category",),
    scope="measurement",
    reference="first",
    reference_value=0.0,
    fixed_source="reference",
)
_CUTPOINTS = _Parameter(
    key="cutpoints",
    kind="cutpoint",
    axes=("pair", "cutpoint"),
    scope="measurement",
    ordered=True,
    regression=True,
)
_BETA_COMPARISON = (
    replace(_CUTPOINTS, kind="comparison_offset"),
    _Parameter(
        key="response_precision", axes=(), scope="measurement", domain="positive", person_link="log"
    ),
)


@dataclass(frozen=True, kw_only=True)
class _MeasurementSpec:
    """Share immutable declarations, without a model construction protocol."""

    priors: Mapping[str, TensorVariable] | None = None
    distribution: ClassVar[type[pm.Distribution]]
    channel: ClassVar = "fc"
    defaults: ClassVar[Mapping] = {}
    extra_roles: ClassVar[frozenset[str]] = frozenset()
    parameter_scales: ClassVar[Mapping] = {}
    response_link: ClassVar[str | None] = None
    # Separate likelihood factors, independent of the incoming response table format.
    comparison_based: ClassVar[bool] = False
    augmented_ranking: ClassVar[bool] = False
    graded_events: ClassVar[bool] = False
    category_count: ClassVar[int | None] = None
    loading_identification: ClassVar[str | None] = None
    link_multipliers: ClassVar[Mapping] = {}
    _parameters: ClassVar[tuple[_Parameter, ...]] = ()

    def parameter_declarations(self, *, block_size=None):
        """Return static final-parameter facts, independent of arrays and graph state."""
        return self._parameters

    def __deepcopy__(self, memo):
        return self  # Frozen declarations retain native prior templates by identity.

    def __post_init__(self):
        object.__setattr__(
            self,
            "priors",
            _copy_priors(self.priors, allowed=self.defaults.keys() | self.extra_roles),
        )


@dataclass(frozen=True, kw_only=True)
class MGPPM(_MeasurementSpec):
    """Dominance rankings normalized over complete ranking patterns."""

    distribution: ClassVar = MGPPMDistribution

    reference_statements: tuple = ()
    loading_identification: ClassVar = "linear"
    graded_events: ClassVar = True
    _parameters: ClassVar = (
        _STATEMENT_LOADING,
        replace(_RANKING_INTERCEPT, graded_role="graded_intercept"),
    )
    defaults: ClassVar = {
        "loading_magnitude": _loading_role(),
        "ranking_intercept": _normal_role(sigma=2),
        "graded_intercept": _normal_role(sigma=2),
    }


@dataclass(frozen=True, kw_only=True)
class TwoPLRank(_MeasurementSpec):
    """Dominance statements with a successive-choice ranking law."""

    distribution: ClassVar = TwoPLRankDistribution

    reference_statements: tuple = ()
    loading_identification: ClassVar = "linear"
    _parameters: ClassVar = (_STATEMENT_LOADING, _RANKING_INTERCEPT)
    defaults: ClassVar = {
        "loading_magnitude": _loading_role(),
        "ranking_intercept": _normal_role(sigma=2),
    }


@dataclass(frozen=True, kw_only=True)
class GGUMRank(_MeasurementSpec):
    """Ideal-point endorsement with a successive-choice ranking law."""

    distribution: ClassVar = GGUMRankDistribution

    reference_statements: tuple = ()
    _parameters: ClassVar = (
        _STATEMENT_LOADING,
        _Parameter(
            key="threshold", kind="statement_threshold", axes=("statement",), regression=True
        ),
        _STATEMENT_LOCATION,
    )
    defaults: ClassVar = {
        "loading_magnitude": _loading_role(sigma=0.25),
        **_STATEMENT_LOCATION_PRIORS,
        "threshold": _normal_role(sigma=2),
    }


@dataclass(frozen=True, kw_only=True)
class TIRT(_MeasurementSpec):
    """Linear Thurstonian utilities for binary, graded or Beta pair comparisons."""

    distribution: ClassVar = TIRTDistribution

    link: Literal["probit", "logit", "beta_probit", "beta_logit"] = "probit"
    residuals: Literal["estimated", "fixed"] = "estimated"
    statement_effect_parameterization: Literal["centered", "noncentered"] = "noncentered"
    loading_anchors: Mapping | None = None
    defaults: ClassVar = {
        "loading_magnitude": _loading_role(),
        "cutpoints": _normal_role(),
        "residual_variance": _RESIDUAL_VARIANCE_PRIOR,
        "statement_effect": _normal_role(),
    }
    extra_roles: ClassVar = frozenset({"response_precision"})
    comparison_based: ClassVar = True
    loading_identification: ClassVar = "automatic"
    parameter_scales: ClassVar = {"statement_effect": ("residual_variance", 0.5)}
    link_multipliers: ClassVar = {"logit": TIRT_LOGIT_MULTIPLIER}
    _parameters: ClassVar = (
        replace(_STATEMENT_LOADING, reference_value=1.0),
        _Parameter(key="statement_effect", axes=("statement",), summary_default=False),
        _RESIDUAL_VARIANCE,
    )

    def parameter_declarations(self, *, block_size=None):
        loading, effect, residual = self._parameters
        if self.link.startswith("beta_"):
            return (loading, *_BETA_COMPARISON)
        if self.residuals == "fixed":
            residual = replace(residual, fixed=residual.reference_value, fixed_source="fixed_scale")
        return loading, effect, residual, _CUTPOINTS

    def __post_init__(self):
        super().__post_init__()
        if self.link not in {"probit", "logit", "beta_probit", "beta_logit"}:
            raise ValueError("TIRT link must be probit, logit, beta_probit or beta_logit")
        if self.residuals not in {"fixed", "estimated"}:
            raise ValueError("TIRT residuals must be fixed or estimated")
        if self.statement_effect_parameterization not in {"centered", "noncentered"}:
            raise ValueError("statement_effect_parameterization must be centered or noncentered")
        object.__setattr__(
            self, "loading_anchors", MappingProxyType(dict(self.loading_anchors or {}))
        )


@dataclass(frozen=True, kw_only=True)
class GTUM(_MeasurementSpec):
    """Ideal-point distance utilities for logistic or Beta pair comparisons."""

    distribution: ClassVar = GTUMDistribution

    link: Literal["logit", "beta_logit"] = "logit"
    block_effect: bool = False
    defaults: ClassVar = {
        "loading_magnitude": _loading_role(),
        **_STATEMENT_LOCATION_PRIORS,
        "cutpoints": _normal_role(),
        "block_effect_scale": _RoleDefault(lambda: pm.HalfNormal.dist(sigma=1)),
        "person_block_effect": _normal_role(),
    }
    extra_roles: ClassVar = frozenset({"response_precision"})
    comparison_based: ClassVar = True
    parameter_scales: ClassVar = {"person_block_effect": ("block_effect_scale", 1.0)}
    _parameters: ClassVar = (_STATEMENT_LOADING, _STATEMENT_LOCATION)

    def parameter_declarations(self, *, block_size=None):
        comparison = _BETA_COMPARISON if self.link == "beta_logit" else (_CUTPOINTS,)
        values = self._parameters + comparison
        if self.block_effect and block_size is not None and block_size > 2:
            values += (
                _Parameter(
                    key="block_effect_scale", axes=(), scope="measurement", domain="positive"
                ),
                _Parameter(
                    key="person_block_effect",
                    axes=("person",),
                    component="observation",
                    scope="measurement",
                    fixed_for_scoring=False,
                ),
            )
        return values

    def __post_init__(self):
        super().__post_init__()
        if self.link not in {"logit", "beta_logit"}:
            raise ValueError("GTUM link must be logit or beta_logit")
        if not isinstance(self.block_effect, bool):
            raise TypeError("GTUM block_effect must be bool")
        if self.link == "beta_logit" and self.block_effect:
            raise ValueError("GTUM Beta responses do not include a block effect")


@dataclass(frozen=True, kw_only=True)
class TIRTRank(_MeasurementSpec):
    """Joint Gaussian ranking events with an augmented fitting likelihood."""

    distribution: ClassVar = TIRTRankDistribution

    loading_anchors: Mapping | None = None
    reference_statements: tuple = ()
    loading_identification: ClassVar = "required"
    augmented_ranking: ClassVar[bool] = True
    _parameters: ClassVar = (_STATEMENT_LOADING, _RANKING_INTERCEPT, _RESIDUAL_VARIANCE)
    defaults: ClassVar = {
        "loading_magnitude": _loading_role(),
        "ranking_intercept": _normal_role(),
        "residual_variance": _RESIDUAL_VARIANCE_PRIOR,
    }

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self, "loading_anchors", MappingProxyType(dict(self.loading_anchors or {}))
        )


@dataclass(frozen=True, kw_only=True)
class DichotomousIRT(_MeasurementSpec):
    """Binary 1PL--4PL response laws with a shared estimated loading in 1PL."""

    distribution: ClassVar = DichotomousIRTDistribution
    channel: ClassVar = "items"

    kind: Literal["1PL", "2PL", "3PL", "4PL"] = "2PL"
    category_count: ClassVar = 2
    response_link: ClassVar = "logit"
    _parameters: ClassVar = (
        _ITEM_LOADING,
        _Parameter(key="intercept", kind="item_intercept", axes=("item",), regression=True),
        _Parameter(key="lower_asymptote", axes=("item",), domain="unit"),
        _Parameter(
            key="upper_asymptote", axes=("item",), domain="unit", bounds=("lower_asymptote", 1.0)
        ),
    )

    def parameter_declarations(self, *, block_size=None):
        fixed = {
            "lower_asymptote": 0.0 if self.kind in {"1PL", "2PL"} else np.nan,
            "upper_asymptote": 1.0 if self.kind != "4PL" else np.nan,
        }
        return tuple(
            replace(
                value,
                shared=value.key == "loading_magnitude" and self.kind == "1PL",
                fixed=fixed.get(value.key, value.fixed),
            )
            for value in self._parameters
        )

    defaults: ClassVar = {
        "loading_magnitude": _ITEM_LOADING_PRIOR,
        "intercept": _normal_role(),
        "lower_asymptote": _RoleDefault(lambda: pm.Beta.dist(alpha=1, beta=4)),
        "upper_asymptote": _RoleDefault(lambda: pm.Beta.dist(alpha=4, beta=1)),
    }

    def __post_init__(self):
        super().__post_init__()
        if self.kind not in {"1PL", "2PL", "3PL", "4PL"}:
            raise ValueError("DichotomousIRT kind must be 1PL, 2PL, 3PL or 4PL")


@dataclass(frozen=True, kw_only=True)
class GPCM(_MeasurementSpec):
    """Generalized partial credit with unordered category intercepts."""

    distribution: ClassVar = GPCMDistribution
    channel: ClassVar = "items"

    _parameters: ClassVar = (_ITEM_LOADING, _CATEGORY_INTERCEPT)

    defaults: ClassVar = {
        "loading_magnitude": _ITEM_LOADING_PRIOR,
        "category_intercept": _normal_role(),
    }


@dataclass(frozen=True, kw_only=True)
class GGUM(_MeasurementSpec):
    """Distance unfolding with signed locations and freely estimated category steps."""

    distribution: ClassVar = GGUMDistribution
    channel: ClassVar = "items"

    _parameters: ClassVar = (
        _ITEM_LOADING,
        _Parameter(key="location", axes=("item", "trait"), regression=True),
        replace(_CATEGORY_INTERCEPT, key="threshold", kind="category_threshold"),
    )

    defaults: ClassVar = {
        "loading_magnitude": _loading_role(),
        "location_positive": _normal_role(1, bounds=(0, None)),
        "location_negative": _normal_role(-1, bounds=(None, 0)),
        "location_free": _normal_role(),
        "threshold": _normal_role(),
    }


def _correlation_default(*, n=None):
    if n is None or n < 2:
        raise ValueError("A free trait-correlation prior requires at least two traits")
    return pm.LKJCorr.dist(n=n, eta=1)


MEASUREMENT_MODELS = {
    model.__name__: model
    for model in (MGPPM, TwoPLRank, GGUMRank, TIRT, GTUM, TIRTRank, DichotomousIRT, GPCM, GGUM)
}


@dataclass(frozen=True, kw_only=True)
class PopulationSpec(_MeasurementSpec):
    """Trait population priors, grouping and residual-scale identification."""

    group: str | None = None
    reference_group: str | int | None = None
    regression_by_group: bool = False
    theta_residual_scale: Literal["unit", "group"] | None = None
    theta_correlation: Literal["group", "shared"] = "group"
    defaults: ClassVar = {
        "trait_correlation": _RoleDefault(_correlation_default),
        "population_mean": _normal_role(),
        "population_sd": _RoleDefault(lambda: pm.HalfNormal.dist(sigma=1)),
    }

    def parameter_declarations(self, *, grouped=False):
        """Declare population laws; preparation supplies actual group coordinates."""
        matrix_axes = ("trait", "trait_aux")
        covariance_axes = ("population_group", *matrix_axes) if grouped else matrix_axes
        correlation_axes = covariance_axes if self.theta_correlation == "group" else matrix_axes
        free_scale = grouped and self.theta_residual_scale == "group"
        return (
            _Parameter(
                key="trait_correlation",
                domain="correlation",
                component="population",
                axes=correlation_axes,
                reference="diagonal",
                reference_value=1.0,
                fixed_source="identification",
                fixed_for_scoring=False,
            ),
            _Parameter(
                key="population_mean",
                component="regression",
                axes=("population_group", "trait"),
                fixed=np.nan if grouped else 0.0,
                reference="group" if grouped else None,
                reference_value=0.0,
                fixed_source="identification",
            ),
            _Parameter(
                key="population_sd",
                component="population",
                axes=("population_group", "trait"),
                fixed=np.nan if free_scale else 1.0,
                reference="group" if free_scale else None,
                reference_value=1.0,
                fixed_source="identification",
                fixed_for_scoring=False,
            ),
            _Parameter(
                key="covariance",
                component="population",
                axes=covariance_axes,
                derived=True,
                reference="group_diagonal" if free_scale else "diagonal",
                reference_value=1.0,
                fixed_source="identification",
            ),
        )

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.regression_by_group, bool):
            raise TypeError("regression_by_group must be bool")
        if self.theta_correlation not in ("group", "shared"):
            raise ValueError("theta_correlation must be group or shared")
        if self.theta_residual_scale is None:
            object.__setattr__(self, "theta_residual_scale", "group" if self.group else "unit")
        if self.theta_residual_scale not in ("unit", "group"):
            raise ValueError("theta_residual_scale must be unit or group")
        if self.group is not None:
            if not isinstance(self.group, str) or not self.group or self.group == "person_id":
                raise ValueError("group must name a person_data column other than person_id")
        elif self.regression_by_group:
            raise ValueError("regression_by_group requires group")
        elif self.reference_group is not None:
            raise ValueError("reference_group requires group")
        elif self.theta_residual_scale == "group":
            raise ValueError("theta_residual_scale='group' requires group")


_GTUM_BETA_LOADING = _loading_role(sigma=0.25)


def _role_default(spec, role):
    declaration = type(spec)
    link = getattr(spec, "link", None)
    if issubclass(declaration, GTUM) and link == "beta_logit" and role == "loading_magnitude":
        return _GTUM_BETA_LOADING
    if role not in declaration.defaults:
        raise ValueError(f"{declaration.__name__} has no built-in default for {role}")
    return declaration.defaults[role]


def default_prior(spec, role, *, n=None) -> TensorVariable:
    """Resolve an ordinary template; model bindings own independent random draws."""
    factory = _role_default(spec, role).ordinary
    if role == "trait_correlation":
        return PriorPreparation.default(factory, n=n)
    return PriorPreparation.default(factory)


def conditional_defaults(spec, role) -> dict:
    """Describe the built-in regression law independently of the ordinary prior."""
    default = _role_default(spec, role)
    if default.family is None:
        raise ValueError(f"{role} has no built-in conditional regression scheme")
    return {
        "family": default.family,
        "link": default.link,
        "bounds": None if default.bounds is None else list(default.bounds),
        "intercept_source": "built_in_role",
        "residual_sd_source": "automatic",
    }


def person_effect_defaults(spec, role, *, link=None):
    """Describe the scale of the respondent deviation from a measurement baseline."""
    if link is None:
        default = conditional_defaults(spec, role)
        link = "log" if default["bounds"] is not None else default["link"]
    return dict(
        family="Normal",
        link=link,
        bounds=None,
        intercept_source="measurement_baseline",
        residual_sd_source="automatic",
    )


def regression_defaults(spec, role) -> tuple[TensorVariable, TensorVariable, float]:
    """Use common regression priors while retaining ordinary laws and signed centers."""
    default = _role_default(spec, role)
    if default.family is None:
        raise ValueError(f"{role} has no built-in conditional regression scheme")
    center = default.center if role in {"location_positive", "location_negative"} else 0.0
    return (
        PriorPreparation.default(
            pm.Normal.dist, mu=center, sigma=0.5 if role == "loading_magnitude" else 1.0
        ),
        PriorPreparation.default(pm.HalfNormal.dist, sigma=default.residual_sd),
        default.coefficient_sd,
    )


def resolve_spec(spec, *, active_roles: frozenset[str], n=None):
    """Resolve only ordinary roles that remain free in the prepared design."""
    if inactive := spec.priors.keys() - active_roles:
        raise ValueError(f"Priors supplied for inactive roles: {sorted(inactive)}")
    if missing := active_roles - (spec.defaults.keys() | spec.priors.keys()):
        raise ValueError(f"Explicit priors required for: {sorted(missing)}")
    priors = {
        role: spec.priors[role] if role in spec.priors else default_prior(spec, role, n=n)
        for role in sorted(active_roles)
    }
    return replace(spec, priors=priors)


def prior_orientation(prior: TensorVariable, *, role: str) -> bool | None:
    """Read whether the prior preserves the declared orientation and warn if relaxed."""
    if role not in {"loading_magnitude", "location_positive", "location_negative"}:
        return None
    lower, upper = prior_support(prior)
    strict = (
        np.all(np.asarray(upper) <= 0)
        if role == "location_negative"
        else np.all(np.asarray(lower) >= 0)
    )
    if not strict:
        warn(
            f"{role}: {type(prior.owner.op).__qualname__} relaxes the default orientation; "
            "values are not repaired",
            IdentificationWarning,
            stacklevel=2,
        )
    return bool(strict)
