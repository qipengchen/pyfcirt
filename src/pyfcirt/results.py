"""Labelled posterior results using native ArviZ statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._model import BuiltModel

if TYPE_CHECKING:
    import xarray as xr


@dataclass(frozen=True)
class FitResult:
    """Keep the native DataTree, statistical specification and executed settings.

    Sampling warnings remain PyMC warnings. A successful fit is not a claim of
    convergence; inspect ArviZ diagnostics before using posterior estimates.
    """

    idata: xr.DataTree
    built: BuiltModel
    sampling: dict

    @property
    def spec(self):
        return self.built.spec

    def parameter_draws(self, role: str = "item_parameters"):
        """Return a labelled Dataset for a declared parameter role."""
        roles = self.spec.variable_roles
        if role not in roles or role in {"responses", "intercept_components"}:
            raise ValueError(f"Unknown public parameter role {role!r}")
        return self.idata["posterior"].to_dataset()[roles[role]]

    def summary(self, role: str = "item_parameters", **kwargs):
        """Delegate summary and interval calculations to ArviZ."""
        import arviz as az

        return az.summary(self.parameter_draws(role), **kwargs)

    def diagnostics(self, role: str = "item_parameters", **kwargs):
        """Return ArviZ's R-hat, ESS and MCSE without new estimators."""
        if "kind" in kwargs:
            raise ValueError("diagnostics manages kind; use summary for other summaries")
        return self.summary(role, kind="diagnostics", **kwargs)

    def posterior_predictive(self, *, random_seed: int | None = None, progressbar: bool = True):
        """Generate new responses while retaining posterior parameter draws."""
        from .predictive import posterior_predictive

        return posterior_predictive(self, random_seed=random_seed, progressbar=progressbar)

    def response_frame(self, draws, *, chain: int = 0, draw: int = 0):
        """Decode one posterior replicated dataset to original response labels."""
        return self.built.response_frame(
            draws, group="posterior_predictive", chain=chain, draw=draw
        )

    def log_likelihood(self, *, progressbar: bool = True):
        """Return labelled conditional response logp, one value per person-block."""
        from .model_comparison import log_likelihood

        return log_likelihood(self, progressbar=progressbar)

    def dic(self, *, progressbar: bool = True):
        """Compute R2jags default chainwise pV DIC from response logp."""
        from .model_comparison import dic

        return dic(self.log_likelihood(progressbar=progressbar))

    def waic(self, *, progressbar: bool = True):
        """Compute legacy-ArviZ WAIC from response logp."""
        from .model_comparison import waic

        return waic(self.log_likelihood(progressbar=progressbar))

    def loo(self, *, pointwise: bool = True, reff: float | None = None, progressbar: bool = True):
        """Delegate PSIS-LOO to current ArviZ."""
        from .model_comparison import loo

        return loo(self, pointwise=pointwise, reff=reff, progressbar=progressbar)

    def pearson_check(
        self,
        *,
        groupby: str = "overall",
        random_seed: int | None = None,
        progressbar: bool = True,
    ):
        """Return the tutorial's categorical Pearson discrepancies and PPP."""
        from .checks import pearson_check

        return pearson_check(
            self, groupby=groupby, random_seed=random_seed, progressbar=progressbar
        )

    def calibration(self, *, draw: tuple | None = None):
        """Extract a posterior-mean point or one joint labelled calibration draw."""
        from .scoring import calibration_parameters

        return calibration_parameters(self, draw=draw)

    def score(self, data, **kwargs):
        """Infer new-person traits without updating item calibration."""
        from .scoring import score

        return score(self, data, **kwargs)

    def save(self, path, *, include_log_likelihood: bool = False):
        """Save a portable analysis directory without serializing executable code."""
        from .io import save

        return save(self, path, include_log_likelihood=include_log_likelihood)
