"""Optional one-off MGPPM sampling check, outside the formal test suite.

Run through scripts/dev.py to keep results and caches in external storage.
This tiny example checks execution only; it is not parameter recovery.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["rank", "graded"], default="rank")
    parser.add_argument("--sampler", choices=["pymc", "numpyro"], default="numpyro")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    built = build_workflow_case(args.task)
    options = {}
    if args.sampler == "numpyro":
        options["nuts"] = {"chain_method": "vectorized", "postprocessing_backend": "cpu"}
    fitted = built.fit(
        draws=20,
        tune=30,
        chains=2,
        cores=1,
        random_seed=61,
        nuts_sampler=args.sampler,
        progressbar=False,
        compute_convergence_checks=False,
        **options,
    )
    posterior = fitted.idata["posterior"]
    for name in ["theta", "loading_magnitude", "intercept", "trait_correlation"]:
        if not np.isfinite(posterior[name]).all():
            raise RuntimeError(f"Nonfinite {name}")
    fitted.save(args.output / "fit")
    predicted = fitted.posterior_predictive(random_seed=62, progressbar=False)
    if len(fitted.response_frame(predicted)) != 4:
        raise RuntimeError("Prediction observation count changed")
    report = {
        "purpose": "manual_execution_check_not_convergence_or_recovery_acceptance",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_spec": built.spec.to_dict(),
        "sampling": fitted.sampling,
        "divergences": int(fitted.idata["sample_stats"].diverging.sum()),
        "finite_posterior": True,
        "prediction_completed": True,
    }
    (args.output / "check.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
