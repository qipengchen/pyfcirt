"""Direct external reference calls; no reference packages are imported in this process."""

import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyfcirt.model_comparison import dic, waic
from pyfcirt.results import FitResult
from tests._workflow_mgppm import build_workflow_case
from tests.test_model_comparison import labelled_logp

pytestmark = pytest.mark.reference
HERE = Path(__file__).parent


def reference_inputs(case):
    if case == "moderate":
        return labelled_logp()
    if case == "chain_difference":
        deviance = np.array([[2, 4, 6, 8], [20, 24, 28, 32]], dtype=float)
        return labelled_logp(-deviance[..., None] * np.array([0.1, 0.2, 0.2]))
    if case == "constant":
        return labelled_logp(np.full((2, 4, 3), -0.5))
    if case == "extreme":
        return labelled_logp(np.full((2, 4, 3), -1000000.0))
    if case == "single_observation":
        return labelled_logp(np.full((2, 4, 1), -0.2))
    return labelled_logp(np.zeros((2, 4, 3)))


def check_legacy_waic(logp, directory):
    executable = os.environ.get("PYFCIRT_LEGACY_ARVIZ_PYTHON")
    if not executable:
        pytest.skip("Set PYFCIRT_LEGACY_ARVIZ_PYTHON to the isolated ArviZ 0.22.0 Python")
    source = directory / "waic_input.json"
    target = directory / "waic_reference.json"
    source.write_text(
        json.dumps({"values": logp.values.tolist(), "obs_id": logp.obs_id.values.tolist()}),
        encoding="utf8",
    )
    arguments = ["-I", str(HERE / "legacy_waic.py"), str(source), str(target)]
    command = [executable, *arguments]
    root = os.environ.get("PYFCIRT_RESULTS_DIR")
    if root:
        command = [
            executable,
            str(HERE.parents[1] / "scripts" / "dev.py"),
            "--storage-root",
            str(Path(root).parent),
            "--",
            "python",
            *arguments,
        ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    reference = json.loads(target.read_text(encoding="utf8"))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        actual = waic(logp)
    for scale, total, pointwise, se in [
        ("log", "elpd_waic", "elpd_waic_i", "se_elpd"),
        ("deviance", "WAIC", "waic_i", "se_WAIC"),
    ]:
        expected = reference[scale]
        assert expected["scale"] == scale
        assert_allclose(actual[total], expected["elpd_waic"], rtol=1e-10, atol=1e-10)
        assert_allclose(actual[pointwise], expected["pointwise"], rtol=1e-10, atol=1e-10)
        assert_allclose(actual[se], expected["se"], rtol=1e-10, atol=1e-10)
        assert_allclose(actual.p_waic, expected["p_waic"], rtol=1e-10, atol=1e-10)
        assert bool(actual.warning) == expected["warning"]
        assert int(actual.n_samples) == expected["n_samples"]
        assert int(actual.n_data_points) == expected["n_data_points"]
        assert len(captured) == len(expected["warnings"])
    return reference


def check_r2jags_dic(logp, directory):
    executable = os.environ.get("PYFCIRT_RSCRIPT")
    source_root = os.environ.get("PYFCIRT_R2JAGS_SOURCE")
    if not executable or not source_root:
        pytest.skip("Set PYFCIRT_RSCRIPT and PYFCIRT_R2JAGS_SOURCE for the pinned source reference")
    pin = json.loads((HERE / "r2jags_source.json").read_text(encoding="utf8"))
    for relative, digest in pin["sha256_lf"].items():
        raw = (Path(source_root) / relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == digest, f"Reference source changed: {relative}"
    source, target = directory / "deviance.csv", directory / "dic_reference.csv"
    deviance = -2 * logp.sum("obs_id")
    pd.DataFrame(deviance.values.T, columns=[str(v) for v in logp.chain.values]).to_csv(
        source, index=False, float_format="%.17g"
    )
    environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    completed = subprocess.run(
        [
            executable,
            "--vanilla",
            str(HERE / "r2jags_dic.R"),
            str(source),
            str(target),
            source_root,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    expected = pd.read_csv(target).set_index("chain")
    actual = dic(logp)
    for label, suffix in [(v, "_chain") for v in logp.chain.values] + [("all", "")]:
        for field in ["Dbar", "pV", "DIC"]:
            value = actual[field + suffix].sel(chain=label) if suffix else actual[field]
            assert_allclose(value, expected.loc[str(label), field], rtol=1e-10, atol=1e-10)
    return {
        "pin": pin,
        "metadata": pd.read_csv(str(target) + ".metadata.csv").to_dict("records"),
        "stderr": completed.stderr,
        "values": expected.reset_index().to_dict("records"),
    }


def save_evidence(name, record):
    root = os.environ.get("PYFCIRT_RESULTS_DIR")
    if root:
        directory = Path(root) / "reference-metrics"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf8")


@pytest.mark.parametrize(
    "case", ["moderate", "chain_difference", "constant", "extreme", "single_observation", "zero"]
)
def test_waic_matches_actual_arviz_022(case, tmp_path):
    logp = reference_inputs(case)
    reference = check_legacy_waic(logp, tmp_path)
    save_evidence(f"waic-{case}", {"input": logp.values.tolist(), "reference": reference})


@pytest.mark.parametrize(
    "case", ["moderate", "chain_difference", "constant", "extreme", "single_observation", "zero"]
)
def test_dic_matches_original_r2jags_postprocessing(case, tmp_path):
    logp = reference_inputs(case)
    reference = check_r2jags_dic(logp, tmp_path)
    save_evidence(f"dic-{case}", {"input": logp.values.tolist(), "reference": reference})


@pytest.mark.parametrize("task", ["rank", "graded"])
@pytest.mark.parametrize("sampler", ["pymc", "numpyro"])
def test_saved_real_fit_metrics_match_both_references(task, sampler, tmp_path):
    root = os.environ.get("PYFCIRT_REFERENCE_FITS")
    if not root:
        pytest.skip("Set PYFCIRT_REFERENCE_FITS to the saved small workflow fits")
    built = build_workflow_case(task)
    manifest = json.loads((Path(root) / f"{task}-{sampler}.json").read_text(encoding="utf8"))
    assert manifest["model_spec"] == built.spec.to_dict(), "Saved fit specification does not match"
    with xr.open_datatree(Path(root) / f"{task}-{sampler}.nc", engine="h5netcdf") as original:
        result = FitResult(idata=original.load(), built=built, sampling=manifest["sampling"])
    logp = result.log_likelihood(progressbar=False)
    expected_waic = check_legacy_waic(logp, tmp_path)
    expected_dic = check_r2jags_dic(logp, tmp_path)
    # NetCDF roundtrip retains observations, chain membership and metric values.
    stored = tmp_path / "log_likelihood.nc"
    logp.to_netcdf(stored, engine="h5netcdf")
    with xr.open_dataarray(stored, engine="h5netcdf") as reloaded:
        xr.testing.assert_identical(logp, reloaded)
        xr.testing.assert_identical(dic(logp), dic(reloaded))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            xr.testing.assert_identical(waic(logp), waic(reloaded))
    # Reuse the existing real fit; never start another sampler in this test.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        native_loo = result.loo(progressbar=False)
    assert int(native_loo.n_data_points) == len(built.spec.observation_map)
    assert np.isfinite(native_loo.elpd)
    save_evidence(
        f"real-{task}-{sampler}",
        {
            "input": logp.to_dict(),
            "sampling": result.sampling,
            "waic_reference": expected_waic,
            "dic_reference": expected_dic,
            "loo": {
                "elpd": float(native_loo.elpd),
                "warning": bool(native_loo.warning),
                "warnings": [str(w.message) for w in captured],
            },
            "purpose": "numerical_reference_not_convergence_acceptance",
        },
    )
