"""Prespecified, labelled populations for bounded MGPPM recovery checks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from pyfcirt import FCData, FCDesign, Model, Prior
from pyfcirt.models import MGPPM
from pyfcirt.simulation import simulate_mgppm


@dataclass
class RecoveryCase:
    model: Model
    new_data: FCData
    truth: xr.Dataset
    new_theta: xr.DataArray
    utility_contrasts: np.ndarray


def recovery_case(task, seed):
    rng = np.random.default_rng(seed)
    traits, blocks, keying, formats, intercepts = {}, {}, {}, {}, {}
    contrasts = []
    loading = {}
    trait_ids = ["T0", "T1", "T2"]
    for block in range(18):
        size = 2 if task == "graded" else [2, 3, 4][block % 3]
        block_id = f"B{block:02d}"
        statements = [f"{block_id}-S{position}" for position in range(size)]
        blocks[block_id] = statements
        formats[block_id] = (
            "graded" if task == "graded" else ["rank", "pick", "mole"][(block // 3) % 3]
        )
        beta = np.linspace(-0.25, 0.25, size) + rng.normal(0, 0.1, size)
        if task != "graded":
            beta -= beta.mean()
        rows = np.zeros((size, 3))
        for position, statement in enumerate(statements):
            trait = (block + position) % 3
            traits.setdefault(trait_ids[trait], []).append(statement)
            keying[statement] = -1 if (block + position // 2) % 2 else 1
            loading[statement] = rng.uniform(0.8, 1.2)
            intercepts[statement] = beta[position]
            rows[position, trait] = loading[statement] * keying[statement]
        contrasts.append(rows if task == "graded" else rows[1:] - rows[:1])
    options = (
        {"graded_scale": {"categories": [0, 1, 2, 3, 4], "higher_prefers": "position_1"}}
        if task == "graded"
        else {}
    )
    design = FCDesign.from_mapping(
        traits=traits,
        blocks=blocks,
        keying=keying,
        allow_variable_block_size=True,
        block_response_formats=formats,
        **options,
    )
    correlation = np.array([[1, 0.25, -0.15], [0.25, 1, 0.2], [-0.15, 0.2, 1]])
    theta = pd.DataFrame(
        rng.multivariate_normal(np.zeros(3), correlation, size=180),
        index=[f"P{i:03d}" for i in range(180)],
        columns=trait_ids,
    )
    new_theta = pd.DataFrame(
        rng.multivariate_normal(np.zeros(3), correlation, size=30),
        index=[f"new-P{i:02d}" for i in range(30)],
        columns=trait_ids,
    )
    magnitudes, beta = pd.Series(loading), pd.Series(intercepts)
    data = simulate_mgppm(design, theta, magnitudes, beta, random_seed=seed + 1000)
    new_data = simulate_mgppm(design, new_theta, magnitudes, beta, random_seed=seed + 2000)
    role = "graded_intercept" if task == "graded" else "ranking_intercept"
    model = Model(
        data,
        family=MGPPM(
            priors={
                "loading_magnitude": Prior("LogNormal", mu=0, sigma=0.4),
                role: Prior("Normal", mu=0, sigma=0.5)
                if task == "graded"
                else Prior("ZeroSumNormal", sigma=0.5),
            }
        ),
        priors={"trait_correlation": Prior("LKJCorr", eta=2)},
    )
    truth = xr.Dataset(
        {
            "loading_magnitude": ("statement", magnitudes.to_numpy()),
            "intercept": ("statement", beta.to_numpy()),
            "theta": (("person", "trait"), theta.to_numpy()),
            "trait_correlation": (("trait", "trait_aux"), correlation),
        },
        coords={
            "statement": magnitudes.index.tolist(),
            "person": theta.index.tolist(),
            "trait": trait_ids,
            "trait_aux": trait_ids,
        },
    )
    return RecoveryCase(
        model,
        new_data,
        truth,
        xr.DataArray(
            new_theta.to_numpy(),
            dims=("person", "trait"),
            coords={"person": new_theta.index.tolist(), "trait": trait_ids},
            name="theta",
        ),
        np.concatenate(contrasts),
    )
