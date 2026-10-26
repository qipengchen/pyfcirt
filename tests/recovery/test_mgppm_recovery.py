"""Deterministic population-design and labelled summary checks; no MCMC."""

import numpy as np
import pytest
import xarray as xr

from tests._recovery_mgppm import recovery_case


@pytest.mark.parametrize("task", ["rank", "graded"])
@pytest.mark.parametrize("seed", [11, 29, 47])
def test_recovery_design_has_observable_trait_directions(task, seed):
    case = recovery_case(task, seed)
    assert case.model.data.summary()["n_observations"] == 180 * 18
    assert case.new_data.summary()["n_observations"] == 30 * 18
    assert np.linalg.svd(case.utility_contrasts, compute_uv=False).min() > 1
    assert np.linalg.eigvalsh(case.truth.trait_correlation).min() > 0
    assert not set(case.truth.person.values) & set(case.new_theta.person.values)
    sizes = set(case.model.data.design.block_sizes.values())
    assert sizes == ({2} if task == "graded" else {2, 3, 4})
    assert set(case.model.data.design.response_formats.values()) == (
        {"graded"} if task == "graded" else {"rank", "pick", "mole"}
    )
    if seed == 11:
        built = case.model.build()
        assert np.isfinite(built.model.compile_logp()(built.model.initial_point()))


def _accuracy(draws, truth, sample_dims):
    draws = draws.sel({dimension: truth.coords[dimension] for dimension in truth.dims})
    mean = draws.mean(sample_dims)
    bounds = draws.quantile([0.05, 0.95], dim=sample_dims)
    return {
        "rmse": float(np.sqrt(((mean - truth) ** 2).mean())),
        "coverage_90": float(
            ((truth >= bounds.isel(quantile=0)) & (truth <= bounds.isel(quantile=1))).mean()
        ),
        "mean_interval_width": float((bounds.isel(quantile=1) - bounds.isel(quantile=0)).mean()),
    }


def test_accuracy_aligns_labels_and_handles_both_sampling_layouts():
    truth = xr.DataArray([2.0, 1.0], dims="statement", coords={"statement": ["B", "A"]})
    samples = xr.DataArray(
        [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]],
        dims=("sample", "statement"),
        coords={"statement": ["A", "B"]},
    )
    expected = {"rmse": np.sqrt(0.5), "coverage_90": 1.0, "mean_interval_width": 2.7}
    score = _accuracy(samples, truth, "sample")
    calibration = _accuracy(
        samples.rename(sample="draw").expand_dims(chain=[0]), truth, ("chain", "draw")
    )
    for name, value in expected.items():
        np.testing.assert_allclose(score[name], value)
        np.testing.assert_allclose(calibration[name], value)
