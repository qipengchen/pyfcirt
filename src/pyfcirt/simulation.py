"""Fixed-parameter MGPPM response generation with explicit labelled inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FCData, FCDesign
from .models.mgppm._data import prepare_data
from .models.mgppm._likelihood import numpy_event_log_probs


def _statement_values(values: pd.Series, labels: tuple, name: str) -> np.ndarray:
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise TypeError(f"{name} must be a Series with unique statement IDs as its index")
    if set(values.index) != set(labels):
        raise ValueError(f"{name} must contain exactly the design's statement IDs")
    result = values.loc[list(labels)].to_numpy(dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def simulate_mgppm(
    design: FCDesign,
    theta: pd.DataFrame,
    loading_magnitude: pd.Series,
    intercept: pd.Series,
    *,
    random_seed: int | None = None,
) -> FCData:
    """Generate one response for each supplied person and design block.

    Parameters
    ----------
    design : FCDesign
        Tasks and reference statement positions. Ranking supports sizes 2--4;
        graded supports the tutorial's five-category pairs.
    theta : pandas.DataFrame
        Fixed trait values with person IDs as the index and trait IDs as columns.
    loading_magnitude : pandas.Series
        Positive fixed magnitudes indexed by statement ID. Design keying supplies
        their signs; the input is not a signed loading vector.
    intercept : pandas.Series
        Fixed beta values indexed by statement ID. Ranking blocks must sum to zero;
        graded intercepts are not centered. Larger beta lowers utility.
    random_seed : int, optional
        Seed for a local NumPy generator; the global RNG is not modified.

    Returns
    -------
    FCData
        Preference-order lists for rank/PICK/MOLE and original category labels
        for graded pairs. The function does not draw priors or estimate parameters.
    """
    if not isinstance(design, FCDesign):
        raise TypeError("design must be an FCDesign")
    if not isinstance(theta, pd.DataFrame) or not theta.index.is_unique:
        raise TypeError("theta must be a DataFrame with unique person IDs as its index")
    if theta.empty or not theta.columns.is_unique:
        raise ValueError("theta requires at least one person and unique trait columns")
    traits = set(design.statements["trait"])
    if set(theta.columns) != traits:
        raise ValueError("theta columns must contain exactly the design's trait IDs")
    formats = design.response_formats
    sizes = design.block_sizes
    scale = design.graded_scale
    first_event = {}
    for block, task in formats.items():
        size = sizes[block]
        if task == "graded":
            first_event[block] = scale["categories"][0]
        elif task == "rank":
            first_event[block] = list(range(1, size + 1))
        elif task == "pick":
            first_event[block] = [1] + [None] * (size - 1)
        else:
            first_event[block] = [1] + [None] * (size - 2) + [size]
    template = pd.DataFrame(
        [(person, block, first_event[block]) for person in theta.index for block in sizes],
        columns=["person_id", "block_id", "response"],
    )
    inputs = prepare_data(FCData.from_frames(template, design))
    trait_values = theta.loc[list(inputs.person_ids), list(inputs.trait_ids)].to_numpy(dtype=float)
    if not np.isfinite(trait_values).all():
        raise ValueError("theta must contain finite trait values")
    loading = _statement_values(loading_magnitude, inputs.statement_ids, "loading_magnitude")
    if np.any(loading <= 0):
        raise ValueError("loading_magnitude must be strictly positive")
    beta = _statement_values(intercept, inputs.statement_ids, "intercept")
    rng = np.random.default_rng(random_seed)
    generated = {}
    for group in inputs.groups:
        if group.task != "graded" and not np.allclose(
            beta[group.statement_index].sum(axis=1), 0, atol=1e-10, rtol=0
        ):
            raise ValueError(f"{group.name}: ranking intercepts must sum to zero within each block")
        statement = group.statement_index[group.block_index]
        utility = (loading * inputs.keying)[statement] * trait_values[
            group.person_index[:, None], inputs.statement_trait[statement]
        ] - beta[statement]
        probability = np.exp(numpy_event_log_probs(utility, group.scores, group.membership))
        cumulative = probability.cumsum(axis=-1)
        cumulative[:, -1] = 1.0  # Close floating-point rounding error in the CDF only.
        sampled = (rng.random((len(probability), 1)) >= cumulative).sum(axis=-1)
        for person, block, event in zip(
            group.person_index, group.block_index, sampled, strict=True
        ):
            value = group.events[event]
            generated[(inputs.person_ids[person], group.block_ids[block])] = (
                value if group.task == "graded" else list(value)
            )
    template["response"] = pd.Series(
        [generated[(p, b)] for p, b in zip(template.person_id, template.block_id, strict=True)],
        dtype=object,
    )
    return FCData.from_frames(template, design)
