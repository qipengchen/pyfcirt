"""Public statistical specification and explicit build entry point."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pandas as pd

from ._model import BuiltModel, ModelSpec
from .data import FCData
from .models.mgppm._data import prepare_data
from .models.mgppm.model import MGPPM, build_graph, resolve_priors
from .priors import Prior


class Model:
    """Configure a forced-choice model without implicitly fitting it.

    The first implemented family is MGPPM. Global priors configure trait
    correlation; family priors configure item roles. Block family assignments
    can explicitly select the same MGPPM configuration. Mixed families, local
    prior overrides and cross-task ranking/graded joint fits are not yet supported.
    """

    def __init__(
        self,
        data: FCData,
        *,
        family: MGPPM | None = None,
        priors: dict[str, Prior] | None = None,
        block_families: dict | None = None,
    ) -> None:
        if not isinstance(data, FCData):
            raise TypeError("data must be an FCData")
        family = MGPPM() if family is None else family
        if not isinstance(family, MGPPM):
            raise TypeError("Only the implemented MGPPM family is currently supported")
        priors = {} if priors is None else dict(priors)
        if set(priors) - {"trait_correlation"}:
            raise ValueError("Global priors support trait_correlation; item priors belong to MGPPM")
        if any(not isinstance(p, Prior) for p in priors.values()):
            raise TypeError("Global priors must be Prior declarations")
        assignments = {} if block_families is None else dict(block_families)
        if set(assignments) - set(data.design.block_sizes):
            raise ValueError("block_families contains unknown block IDs")
        expected = {r: p.to_dict() for r, p in family.resolved_priors().items()}
        for block, selected in assignments.items():
            if not isinstance(selected, MGPPM):
                raise TypeError(f"Block {block!r}: only MGPPM is currently implemented")
            if {r: p.to_dict() for r, p in selected.resolved_priors().items()} != expected:
                raise NotImplementedError(
                    "Block-specific prior configurations are not yet supported"
                )
        self._data = data
        self._family = deepcopy(family)
        self._priors = deepcopy(priors)
        self._built: BuiltModel | None = None

    @property
    def data(self) -> FCData:
        return self._data

    def prior_summary(self) -> pd.DataFrame:
        """Show every active resolved prior and its explicit/default provenance."""
        inputs = prepare_data(self._data)
        priors = resolve_priors(self._family, self._priors, inputs)
        return pd.DataFrame(
            [
                {
                    "role": role,
                    "distribution": prior.name,
                    "parameters": prior.parameters,
                    "labels": list(inputs.trait_ids)
                    if role == "trait_correlation"
                    else list(inputs.statement_ids)
                    if role == "loading_magnitude"
                    else [
                        block
                        for group in inputs.groups
                        if (group.task == "graded") == (role == "graded_intercept")
                        for block in group.block_ids
                    ],
                    "source": "user"
                    if role in self._family.priors or role in self._priors
                    else "default",
                    "constraint": "block sum zero"
                    if role == "ranking_intercept"
                    else "unit trait variances"
                    if role == "trait_correlation"
                    else "positive magnitude"
                    if role == "loading_magnitude"
                    else "independent",
                }
                for role, prior in priors.items()
            ]
        )

    def build(self) -> BuiltModel:
        """Build once for this configuration; expose the native graph in .model."""
        if self._built is not None:
            return self._built
        inputs = prepare_data(self._data)
        if not inputs.person_ids:
            raise ValueError(
                "At least one person is required; use administration for prior-only data"
            )
        if self._data.design.summary()["reused_statement_ids"]:
            raise NotImplementedError("Shared statements require validated cross-block constraints")
        tasks = {g.task for g in inputs.groups}
        if "graded" in tasks and len(tasks) > 1:
            raise NotImplementedError(
                "Joint graded/ranking fits need separate identification validation"
            )
        for group in inputs.groups:
            for block, statement in zip(group.block_ids, group.statement_index, strict=True):
                if len(set(inputs.statement_trait[statement])) < 2:
                    raise ValueError(
                        f"Block {block!r}: a single-trait block cannot identify all free slopes"
                    )
        priors = resolve_priors(self._family, self._priors, inputs)
        native, observed_names, intercept_names = build_graph(inputs, priors)
        components, observation_map = [], []
        observed = self._data.to_frame(answered_only=True).set_index(["person_id", "block_id"])
        for group in inputs.groups:
            components.append(
                {
                    "id": group.name,
                    "family": "MGPPM",
                    "task": group.task,
                    "block_ids": list(group.block_ids),
                    "observed_variable": observed_names[group.name],
                    "events": [list(e) if isinstance(e, tuple) else e for e in group.events],
                    "scores": group.scores.tolist(),
                    "encoding_version": "1",
                }
            )
            for local, (obs_id, person, block) in enumerate(
                zip(group.obs_ids, group.person_index, group.block_index, strict=True)
            ):
                person_id, block_id = inputs.person_ids[person], group.block_ids[block]
                observation_map.append(
                    {
                        "obs_id": obs_id,
                        "person_id": person_id,
                        "block_id": block_id,
                        "variable": observed_names[group.name],
                        "local_index": local,
                        "task": group.task,
                        "response": deepcopy(observed.loc[(person_id, block_id), "response"]),
                    }
                )
        core = {
            "design": self._data.design.to_frame()[
                ["block_id", "position", "statement_id", "trait", "keying"]
            ].to_dict("records"),
            "responses": self._data.to_frame()[["person_id", "block_id", "response"]].to_dict(
                "records"
            ),
            "graded_scale": self._data.design.graded_scale,
            "administration": None
            if self._data.administration is None
            else self._data.administration[["person_id", "block_id", "status"]].to_dict("records"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(core, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest()
        roles = {
            "traits": ["theta"],
            "trait_correlation": ["trait_correlation"],
            "item_parameters": ["loading_magnitude", "loading", "intercept"],
            "intercept_components": intercept_names,
            "responses": list(observed_names.values()),
        }
        spec = ModelSpec(
            model_id="mgppm",
            model_version="2",
            settings={
                "priors": {r: p.to_dict() for r, p in priors.items()},
                "parameterization": "centered",
                "prior_sources": self.prior_summary().to_dict("records"),
            },
            components=components,
            observation_map=observation_map,
            variable_roles=roles,
            observation_unit="person-block",
            likelihood_kind="conditional_response",
            data_fingerprint=fingerprint,
        )
        self._built = BuiltModel(model=native, spec=spec, data=self._data, inputs=inputs)
        return self._built

    def fit(self, **kwargs):
        """Build if needed and sample through the common PyMC entry point."""
        return self.build().fit(**kwargs)

    def prior_predictive(self, *, draws: int = 500, random_seed: int | None = None):
        """Build if needed and generate before any posterior fit."""
        return self.build().prior_predictive(draws=draws, random_seed=random_seed)
