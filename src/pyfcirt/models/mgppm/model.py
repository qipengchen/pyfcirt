"""MGPPM family settings and explicit PyMC graph construction."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ...priors import Prior, validate_prior
from ._data import MGPPMInputs

DEFAULT_PRIORS = {
    "loading_magnitude": Prior("LogNormal", mu=0.0, sigma=2.0),
    "ranking_intercept": Prior("ZeroSumNormal", sigma=1.0),
    "graded_intercept": Prior("Normal", mu=0.0, sigma=1.0),
}


class MGPPM:
    """Declare MGPPM's item priors; data tasks determine the response mechanism."""

    def __init__(self, *, priors: dict[str, Prior] | None = None) -> None:
        priors = {} if priors is None else dict(priors)
        unknown = set(priors) - set(DEFAULT_PRIORS)
        if unknown:
            raise ValueError(f"Unknown MGPPM prior roles: {sorted(unknown)}")
        if any(not isinstance(prior, Prior) for prior in priors.values()):
            raise TypeError("MGPPM priors must be Prior declarations")
        self._priors = deepcopy(priors)

    @property
    def priors(self) -> dict[str, Prior]:
        return deepcopy(self._priors)

    def resolved_priors(self) -> dict[str, Prior]:
        return {**deepcopy(DEFAULT_PRIORS), **self.priors}


def build_graph(inputs: MGPPMInputs, priors: dict[str, Prior]):
    """Construct one native graph with shared person and statement parameters."""
    import pymc as pm
    import pytensor.tensor as pt

    from ._likelihood import event_log_probs, statement_utilities

    coords = {
        "person": np.asarray(inputs.person_ids, dtype=object),
        "statement": np.asarray(inputs.statement_ids, dtype=object),
        "trait": np.asarray(inputs.trait_ids, dtype=object),
        "trait_aux": np.asarray(inputs.trait_ids, dtype=object),
    }
    for group in inputs.groups:
        coords[f"{group.name}_block"] = np.asarray(group.block_ids, dtype=object)
        coords[f"{group.name}_position"] = np.arange(1, group.scores.shape[1] + 1)
        coords[f"{group.name}_obs"] = list(group.obs_ids)
    observed_names = {}
    intercept_names = []
    with pm.Model(coords=coords) as model:
        n_traits = len(inputs.trait_ids)
        if n_traits > 1:
            factor = pm.LKJCorr(
                "trait_cholesky",
                n=n_traits,
                dims=("trait", "trait_aux"),
                **priors["trait_correlation"].parameters,
            )
        else:
            factor = pt.ones((1, 1))
        pm.Deterministic("trait_correlation", factor @ factor.T, dims=("trait", "trait_aux"))
        theta = pm.MvNormal("theta", mu=np.zeros(n_traits), chol=factor, dims=("person", "trait"))
        declaration = priors["loading_magnitude"]
        magnitude = getattr(pm, declaration.name)(
            "loading_magnitude", dims="statement", **declaration.parameters
        )
        pm.Deterministic("loading", magnitude * inputs.keying, dims="statement")
        intercept_parts, positions = [], []
        for group in inputs.groups:
            role = "graded_intercept" if group.task == "graded" else "ranking_intercept"
            declaration = priors[role]
            kwargs = declaration.parameters
            if role == "ranking_intercept":
                kwargs["n_zerosum_axes"] = 1
            name = f"{group.name}_intercept"
            value = getattr(pm, declaration.name)(
                name, **kwargs, dims=(f"{group.name}_block", f"{group.name}_position")
            )
            intercept_parts.append(value.reshape((-1,)))
            positions.append(group.statement_index.ravel())
            intercept_names.append(name)
        order = np.argsort(np.concatenate(positions))
        beta = pm.Deterministic(
            "intercept", pt.concatenate(intercept_parts)[order], dims="statement"
        )
        for group in inputs.groups:
            utility = statement_utilities(theta, magnitude, beta, inputs, group)
            log_probs = event_log_probs(utility, group.scores, group.membership)
            name = f"response_{group.name}"
            pm.Categorical(
                name, logit_p=log_probs, observed=group.response, dims=f"{group.name}_obs"
            )
            observed_names[group.name] = name
    return model, observed_names, intercept_names


def resolve_priors(family: MGPPM, global_priors: dict[str, Prior], inputs: MGPPMInputs):
    """Resolve defaults without learning hyperparameters from response values."""
    priors = family.resolved_priors()
    priors["trait_correlation"] = global_priors.get("trait_correlation", Prior("LKJCorr", eta=1.0))
    active = {"loading_magnitude", "trait_correlation"}
    active.update(
        "graded_intercept" if g.task == "graded" else "ranking_intercept" for g in inputs.groups
    )
    for role in sorted(active):
        validate_prior(priors[role], role)
    return {role: prior for role, prior in priors.items() if role in active}
