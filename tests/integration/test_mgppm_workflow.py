"""Reuse explicitly supplied calibration draws without running posterior MCMC."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyfcirt.results import FitResult
from tests._workflow_mgppm import build_workflow_case


@pytest.mark.slow
@pytest.mark.parametrize("task", ["rank", "graded"])
@pytest.mark.parametrize("sampler", ["pymc", "numpyro"])
def test_pearson_check_reuses_saved_real_fit(task, sampler, tmp_path):
    root = os.environ.get("PYFCIRT_REFERENCE_FITS")
    if not root:
        pytest.skip("Set PYFCIRT_REFERENCE_FITS to reuse saved real workflow fits")
    built = build_workflow_case(task)
    manifest = json.loads((Path(root) / f"{task}-{sampler}.json").read_text(encoding="utf8"))
    assert manifest["model_spec"] == built.spec.to_dict()
    with xr.open_datatree(Path(root) / f"{task}-{sampler}.nc", engine="h5netcdf") as stored:
        fitted = FitResult(idata=stored.load(), built=built, sampling=manifest["sampling"])
    check = fitted.pearson_check(groupby="block", random_seed=74, progressbar=False)
    assert int(check.n_obs.sum()) == 4
    assert check.PPP.dims == ("block",)
    assert ((check.PPP >= 0) & (check.PPP <= 1)).all()
    assert np.isfinite(check.D_obs).all()
    directory = Path(os.environ.get("PYFCIRT_RESULTS_DIR", str(tmp_path))) / "pearson-smoke"
    directory.mkdir(parents=True, exist_ok=True)
    check.to_netcdf(directory / f"{task}-{sampler}.nc", engine="h5netcdf")
