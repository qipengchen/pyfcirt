"""Portable labelled data/draws and explicit reconstruction without pickle."""

import json
import os
import subprocess
import sys
import warnings
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyfcirt import FCData, FCDesign, Model, load
from pyfcirt.io import SavedFit
from pyfcirt.model_comparison import dic, waic
from pyfcirt.results import FitResult
from tests.test_model_comparison import comparison_case as comparison_case


def test_fit_roundtrip_rebuild_and_native_prediction(comparison_case, tmp_path, monkeypatch):
    original = comparison_case
    before = original.idata.copy(deep=True)
    directory = original.save(tmp_path / "fit", include_log_likelihood=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading must not execute model construction or sampling")

    with monkeypatch.context() as patch:
        patch.setattr(Model, "build", forbidden)
        saved = load(directory)
    assert isinstance(saved, SavedFit)
    assert saved.spec.to_dict() == original.spec.to_dict()
    pd.testing.assert_frame_equal(saved.data.to_frame(), original.built.data.to_frame())
    assert saved.data.design.response_formats == original.built.data.design.response_formats
    assert saved.data.design.graded_scale == original.built.data.design.graded_scale
    xr.testing.assert_identical(saved.idata["posterior"], original.idata["posterior"])
    xr.testing.assert_identical(original.idata, before)
    rebuilt = saved.rebuild()
    assert rebuilt.spec.to_dict() == original.spec.to_dict()
    logp = original.log_likelihood(progressbar=False)
    restored = rebuilt.log_likelihood(progressbar=False)
    xr.testing.assert_identical(logp, restored)
    xr.testing.assert_identical(dic(logp), dic(restored))
    with pytest.warns(UserWarning) if np.any(logp.values.var(axis=(0, 1)) > 0.4) else nullcontext():
        first = waic(logp)
    with pytest.warns(UserWarning) if bool(first.warning) else nullcontext():
        second = waic(restored)
    xr.testing.assert_identical(first, second)
    with warnings.catch_warnings(record=True) as first_warnings:
        warnings.simplefilter("always")
        before_loo = original.loo(reff=1, progressbar=False)
    with warnings.catch_warnings(record=True) as second_warnings:
        warnings.simplefilter("always")
        after_loo = rebuilt.loo(reff=1, progressbar=False)
    np.testing.assert_allclose(before_loo.elpd, after_loo.elpd, rtol=1e-12, atol=1e-12)
    xr.testing.assert_equal(before_loo.elpd_i, after_loo.elpd_i)
    xr.testing.assert_equal(before_loo.pareto_k, after_loo.pareto_k)
    assert [str(w.message) for w in first_warnings] == [str(w.message) for w in second_warnings]
    predicted = original.posterior_predictive(random_seed=93, progressbar=False)
    repeated = rebuilt.posterior_predictive(random_seed=93, progressbar=False)
    xr.testing.assert_equal(predicted["posterior_predictive"], repeated["posterior_predictive"])


def test_rebuilt_fit_can_score_without_refitting_calibration(comparison_case, tmp_path):
    original = comparison_case
    saved = load(original.save(tmp_path / "fit-for-scoring"))
    empty = pd.DataFrame(columns=["person_id", "block_id", "response"])
    administration = pd.DataFrame(
        {
            "person_id": ["new-P"],
            "block_id": [original.built.inputs.block_ids[0]],
            "status": ["missing"],
        }
    )
    data = FCData.from_frames(empty, original.built.data.design, administration=administration)
    options = {"draws": 4, "chains": 1, "random_seed": 94}
    actual = saved.rebuild().score(data, **options)
    expected = original.score(data, **options)
    xr.testing.assert_identical(actual.idata, expected.idata)
    assert actual.settings == expected.settings


def test_mixed_integer_string_ids_preserve_identity(tmp_path):
    design = FCDesign.from_mapping(
        traits={1: [10, 30], "1": ["10", "30"]},
        blocks={1: [10, "10"], "1": [30, "30"]},
        keying={30: -1},
    )
    data = FCData.from_frames(
        pd.DataFrame(
            {
                "person_id": [1, "1", 1, "1"],
                "block_id": [1, 1, "1", "1"],
                "response": [[2, 1], [1, 2], [1, 2], [2, 1]],
            }
        ),
        design,
    )
    built = Model(data).build()
    forward = built.prior_predictive(draws=4, random_seed=95)
    assert forward["prior"].coords["trait"].values.tolist() == [1, "1"]
    assert forward["prior"].coords["statement"].values.tolist() == [10, "10", 30, "30"]
    original = FitResult(
        idata=xr.DataTree.from_dict({"posterior": forward["prior"].to_dataset()}),
        built=built,
        sampling={},
    )
    saved = load(original.save(tmp_path / "mixed-labels"))
    assert saved.data.person_ids == (1, "1")
    assert list(saved.data.design.block_sizes) == [1, "1"]
    for dimension in ["person", "statement", "trait", "trait_aux"]:
        assert (
            saved.idata["posterior"].coords[dimension].values.tolist()
            == original.idata["posterior"].coords[dimension].values.tolist()
        )
    xr.testing.assert_identical(saved.idata, original.idata)
    xr.testing.assert_allclose(
        saved.rebuild().log_likelihood(progressbar=False),
        original.log_likelihood(progressbar=False),
    )


def test_statement_rank_encoding_and_structural_na_survive(tmp_path):
    design = FCDesign.from_mapping(
        traits={"T1": ["A"], "T2": ["B"]}, blocks={"pair": ["A", "B"]}, response_format="pick"
    )
    selected = FCData.from_frames(
        pd.DataFrame({"person_id": ["P"], "block_id": ["pair"], "response": [[1, None]]}), design
    )
    data = FCData.from_frames(selected.to_rank_frame(), design, encoding="statement_rank")
    built = Model(data).build()
    forward = built.prior_predictive(draws=4, random_seed=96)
    original = FitResult(
        idata=xr.DataTree.from_dict({"posterior": forward["prior"].to_dataset()}),
        built=built,
        sampling={},
    )
    saved = load(original.save(tmp_path / "long-na"))
    assert saved.data.input_encoding == "statement_rank"
    assert saved.data.to_frame().iloc[0].response == [1, None]
    pd.testing.assert_frame_equal(saved.data.original_responses, data.original_responses)
    assert saved.rebuild().spec.to_dict() == original.spec.to_dict()


def test_score_result_roundtrip_keeps_fixed_calibration(comparison_case, tmp_path):
    fitted = comparison_case
    data = fitted.built.data
    empty = pd.DataFrame(columns=["person_id", "block_id", "response"])
    administration = pd.DataFrame(
        {
            "person_id": ["new-P"],
            "block_id": [fitted.built.inputs.block_ids[0]],
            "status": ["missing"],
        }
    )
    unanswered = FCData.from_frames(empty, data.design, administration=administration)
    score = fitted.score(
        unanswered,
        draws=4,
        chains=2,
        random_seed=97,
    )
    restored = load(score.save(tmp_path / "scores"))
    xr.testing.assert_identical(restored.idata, score.idata)
    assert restored.settings == score.settings
    assert restored.sampling == score.sampling
    pd.testing.assert_frame_equal(restored.summary(), score.summary())
    assert restored.person_status["status"].item() == "prior_only"


def test_save_refuses_overwrite_and_load_detects_corruption(comparison_case, tmp_path):
    directory = comparison_case.save(tmp_path / "existing")
    with pytest.raises(FileExistsError):
        comparison_case.save(directory)
    with (directory / "metadata.json").open("a", encoding="utf8") as target:
        target.write(" ")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load(directory)


def test_custom_callbacks_are_not_serialized_or_executed(comparison_case, tmp_path):
    def callback():
        raise AssertionError("Executable values must never be called by serialization")

    changed = replace(comparison_case, sampling={"extra_settings": {"callback": callback}})
    with pytest.raises(TypeError, match="Unsupported portable value"):
        changed.save(tmp_path / "callback")
    assert not (tmp_path / "callback").exists()


@pytest.mark.parametrize(
    "edit", ["potential", "prior_parameter", "deterministic", "dimension", "observed_values"]
)
def test_custom_native_graph_requires_own_construction_code(comparison_case, tmp_path, edit):
    import pymc as pm
    import pytensor.tensor as pt

    from pyfcirt.io import _rebuild

    graph = _rebuild(comparison_case.spec, comparison_case.built.data).model
    if edit == "potential":
        with graph:
            pm.Potential("outside_saved_spec", graph["theta"].sum() * 0.01)
    elif edit == "prior_parameter":
        sigma = graph["loading_magnitude"].owner.inputs[-1]
        graph["loading_magnitude"].owner.inputs[-1] = pt.constant(0.01, dtype=sigma.dtype)
    elif edit == "deterministic":
        # Changing an expression keeps every public variable name and operator type.
        loading = graph["loading"]
        loading.owner.inputs[0] = -loading.owner.inputs[0]
    elif edit == "dimension":
        size = graph.dim_lengths["person"]
        size.set_value(size.get_value() + 1)
    else:
        observed = graph.observed_RVs[0]
        value = graph.rvs_to_values[observed]
        n_events = len(comparison_case.spec.components[0]["events"])
        graph.rvs_to_values[observed] = pt.constant((value.data + 1) % n_events, dtype=value.dtype)
    changed = replace(comparison_case, built=replace(comparison_case.built, model=graph))
    with pytest.raises(ValueError, match="Modified native graphs"):
        changed.save(tmp_path / "custom")


def test_new_process_rebuilds_predicts_and_scores(comparison_case, tmp_path):
    original = comparison_case
    directory = original.save(tmp_path / "separate-process")
    output = tmp_path / "new-process-output"
    script = Path(__file__).parent / "_resume_analysis.py"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(script), str(directory), str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_allclose(
        np.load(output / "logp.npy"),
        original.log_likelihood(progressbar=False),
        rtol=1e-12,
        atol=1e-12,
    )
    predictive = original.posterior_predictive(random_seed=101, progressbar=False)
    expected = original.response_frame(predictive).to_dict("records")
    assert json.loads((output / "predicted.json").read_text(encoding="utf8")) == expected
    empty = pd.DataFrame(columns=["person_id", "block_id", "response"])
    admin = pd.DataFrame(
        {
            "person_id": ["fresh-new"],
            "block_id": [original.built.inputs.block_ids[0]],
            "status": ["missing"],
        }
    )
    data = FCData.from_frames(empty, original.built.data.design, administration=admin)
    np.testing.assert_allclose(
        np.load(output / "scores.npy"),
        original.score(data, draws=4, chains=1, random_seed=102).samples,
        rtol=1e-12,
        atol=1e-12,
    )


def test_features_nullable_columns_and_text_survive(tmp_path):
    features = pd.DataFrame(
        {
            "statement_id": ["A", "B"],
            "likert_mean": [2.2, np.nan],
            "rating_n": pd.array([3, None], dtype="Int64"),
        }
    )
    frame = pd.DataFrame(
        {
            "block_id": ["pair", "pair"],
            "position": [1, 2],
            "statement_id": ["A", "B"],
            "trait": ["T1", "T2"],
            "keying": [1, -1],
            "text": ["Statement A", "Statement B"],
        }
    )
    design = FCDesign.from_frame(frame, statement_features=features)
    data = FCData.from_frames(
        pd.DataFrame({"person_id": ["P"], "block_id": ["pair"], "response": [[1, 2]]}), design
    )
    built = Model(data).build()
    prior = built.prior_predictive(draws=4, random_seed=103)
    fitted = FitResult(
        idata=xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()}),
        built=built,
        sampling={},
    )
    restored = load(fitted.save(tmp_path / "features"))
    pd.testing.assert_frame_equal(restored.data.design.statement_features, features)
    pd.testing.assert_frame_equal(restored.data.design.to_frame(), design.to_frame())


@pytest.mark.slow
@pytest.mark.parametrize("task", ["rank", "graded"])
@pytest.mark.parametrize("sampler", ["pymc", "numpyro"])
def test_saved_actual_fit_bundle_roundtrip(task, sampler, tmp_path):
    root = os.environ.get("PYFCIRT_REFERENCE_FITS")
    if not root:
        pytest.skip("Set PYFCIRT_REFERENCE_FITS to reuse actual calibration fits")
    from tests._workflow_mgppm import build_workflow_case

    built = build_workflow_case(task)
    metadata = json.loads((Path(root) / f"{task}-{sampler}.json").read_text(encoding="utf8"))
    assert metadata["model_spec"] == built.spec.to_dict()
    with xr.open_datatree(Path(root) / f"{task}-{sampler}.nc", engine="h5netcdf") as source:
        original = FitResult(idata=source.load(), built=built, sampling=metadata["sampling"])
    restored = load(original.save(tmp_path / "real-trace"))
    xr.testing.assert_identical(restored.idata, original.idata)
    assert restored.sampling == original.sampling
    xr.testing.assert_identical(
        restored.rebuild().log_likelihood(progressbar=False),
        original.log_likelihood(progressbar=False),
    )


def test_legacy_score_layout_is_rejected_before_loading_draws(tmp_path):
    import hashlib

    (tmp_path / "metadata.json").write_text("{}", encoding="utf8")
    (tmp_path / "posterior.nc").write_bytes(b"legacy draws need not be opened")
    manifest = {
        "format": "pyfcirt",
        "format_version": 1,
        "kind": "score",
        "sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ["metadata.json", "posterior.nc"]
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="Unsupported scoring bundle"):
        load(tmp_path)
