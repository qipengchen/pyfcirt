"""Align population groups without fitting or applying covariate encodings."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from pyfcirt.common.validation import align_rows, identifier, identifiers, require_frame


def theta_target_indices(
    person_groups: np.ndarray, trait_count: int, *, by_group: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Bind person×trait targets to rows and frozen group-then-trait pools."""
    shape = (len(person_groups), trait_count)
    rows = np.broadcast_to(np.arange(shape[0], dtype=np.int64)[:, None], shape).copy()
    pools = np.broadcast_to(np.arange(trait_count, dtype=np.int64), shape).copy()
    if by_group:
        pools += person_groups[:, None] * trait_count
    return rows, pools


def _group_labels(person_data, group, positions, people):
    require_frame(person_data, (group,), "person_data")
    labels = person_data[group].iloc[positions]
    if (missing := labels.isna().to_numpy()).any():
        absent = people[missing][:10].tolist()
        raise ValueError(f"Population group {group!r} is missing for person IDs: {absent!r}")
    return pd.Index(identifiers(labels, group), dtype=object)


def fit_groups(person_data, person_ids, population) -> tuple[dict, np.ndarray]:
    """Freeze group order and reference using the aligned training people.

    Extra rows and unused columns are ignored; duplicate person IDs always fail.
    The presence of a theta regression is checked when regressions are prepared.
    """
    group = population.group
    reference = population.reference_group
    if person_data is not None:
        keys = pd.DataFrame({"person_id": pd.Series(person_ids, dtype=object)})
        positions, people = align_rows(person_data, keys, ("person_id",), "person_data")
    elif group is not None:
        raise ValueError("person_data is required for the population group column")
    groups = []
    if group is None:
        indices = np.zeros(len(person_ids), dtype=np.int64)
    else:
        labels = _group_labels(person_data, group, positions, people)
        indices, levels = pd.factorize(labels, sort=False)
        groups = levels.tolist()
        reference = groups[0] if reference is None else identifier(reference, group)
        if reference not in groups:
            raise ValueError(f"Unknown reference_group {reference!r}")
    recipe = dict(
        group=group,
        groups=groups,
        reference_group=reference,
        regression_by_group=population.regression_by_group,
        theta_residual_scale=population.theta_residual_scale,
        regression=None,
    )
    return recipe, indices


def apply_groups(
    recipe: Mapping, person_data: pd.DataFrame, positions: np.ndarray, people: pd.Index
) -> np.ndarray:
    """Encode the group labels of already matched people in the fitted order."""
    labels = _group_labels(person_data, recipe["group"], positions, people)
    indices = pd.Index(recipe["groups"], dtype=object).get_indexer(labels)
    if (unknown := indices < 0).any():
        raise ValueError(f"Unknown population groups: {labels[unknown].unique()[:10].tolist()!r}")
    return indices
