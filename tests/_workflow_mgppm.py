"""Small calibration specifications; construction only, no posterior sampling."""

import pandas as pd

from pyfcirt import FCData, FCDesign, Model, Prior
from pyfcirt.models import MGPPM


def build_workflow_case(task):
    """Share the exact calibration specification with saved-fit reference tests."""
    size = 3 if task == "rank" else 2
    options = {}
    if task == "graded":
        options["graded_scale"] = {"categories": [0, 1, 2, 3, 4], "higher_prefers": "position_1"}
    design = FCDesign.from_mapping(
        traits={f"T{i}": [f"S{i}", f"S{i + size}"] for i in range(size)},
        blocks={"B1": [f"S{i}" for i in range(size)], "B2": [f"S{i + size}" for i in range(size)]},
        response_format=task,
        keying={"S1": -1},
        **options,
    )
    frame = pd.DataFrame(
        {
            "person_id": ["P", "P", "Q", "Q"],
            "block_id": ["B1", "B2", "B1", "B2"],
            "response": [[2, 3, 1], [1, 3, 2], [3, 1, 2], [2, 1, 3]]
            if task == "rank"
            else [1, 3, 4, 2],
        }
    )
    role = "ranking_intercept" if task == "rank" else "graded_intercept"
    item_priors = {
        "loading_magnitude": Prior("LogNormal", mu=0, sigma=0.3),
        role: Prior("ZeroSumNormal", sigma=0.4)
        if task == "rank"
        else Prior("Normal", mu=0, sigma=0.5),
    }
    return Model(
        FCData.from_frames(frame, design),
        family=MGPPM(priors=item_priors),
        priors={"trait_correlation": Prior("LKJCorr", eta=2)},
    ).build()
