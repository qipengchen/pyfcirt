"""Independent Pearson arithmetic, block aggregation and native forward checks."""

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyfcirt import FCData, Model
from pyfcirt.checks import _log_pearson_terms
from pyfcirt.results import FitResult
from tests.test_model_comparison import comparison_case as comparison_case


def test_log_discrepancy_matches_one_hot_pearson_without_clipping():
    values = pt.dvector("selected_logp")
    compute = pytensor.function([values], _log_pearson_terms(values), mode="NUMBA")
    probabilities = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    response = np.array([2, 1])
    one_hot = np.eye(3)[response]
    expected = ((one_hot - probabilities) ** 2 / probabilities).sum(axis=-1)
    actual = compute(np.log(probabilities[np.arange(2), response]))
    assert_allclose(np.exp(actual), expected, rtol=1e-12, atol=1e-12)
    extreme = compute(np.array([-1000.0, -1000000.0, 0.0]))
    assert_allclose(extreme[:2], [1000.0, 1000000.0])
    assert extreme[2] == -np.inf


def test_check_preserves_graph_and_matches_observed_logp(comparison_case):
    result = comparison_case
    original = result.idata.copy(deep=True)
    variables = list(result.built.model.named_vars)
    check = result.pearson_check(random_seed=67, progressbar=False)
    repeated = result.pearson_check(random_seed=67, progressbar=False)
    xr.testing.assert_identical(result.idata, original)
    assert list(result.built.model.named_vars) == variables
    xr.testing.assert_identical(check, repeated)
    logp = result.log_likelihood(progressbar=False)
    expected = np.expm1(-logp.values).sum(axis=-1)
    assert_allclose(check.D_obs, expected, rtol=1e-10, atol=1e-10)
    expected_ppp = (check.D_rep >= check.D_obs).mean(("chain", "draw"))
    xr.testing.assert_equal(check.PPP, expected_ppp)
    assert int(check.n_obs) == len(result.spec.observation_map)
    assert "check_log_D_obs" not in result.built.model.named_vars


def test_block_check_aggregation_and_structural_missingness(comparison_case):
    result = comparison_case
    by_block = result.pearson_check(groupby="block", random_seed=68, progressbar=False)
    overall = result.pearson_check(random_seed=68, progressbar=False)
    assert by_block.block.values.tolist() == list(result.built.inputs.block_ids)
    assert int(by_block.n_obs.sum()) == len(result.spec.observation_map)
    assert_allclose(by_block.D_obs.sum("block"), overall.D_obs, rtol=1e-12, atol=1e-12)
    assert_allclose(by_block.D_rep.sum("block"), overall.D_rep, rtol=1e-12, atol=1e-12)
    assert np.all((by_block.PPP >= 0) & (by_block.PPP <= 1))


def test_log_scale_ppp_handles_actual_raw_discrepancy_overflow(comparison_case):
    result = comparison_case
    altered = result.idata.copy(deep=True)
    altered["posterior"]["theta"].values[...] = 0
    for group in result.built.inputs.groups:
        size = group.scores.shape[1]
        altered["posterior"][f"{group.name}_intercept"].values[...] = np.linspace(1000, -1000, size)
    fixed = FitResult(idata=altered, built=result.built, sampling={})
    with pytest.warns(UserWarning, match="exceed floating-point range"):
        check = fixed.pearson_check(random_seed=69, progressbar=False)
    assert bool(check.attrs["raw_discrepancy_overflow"])
    assert np.isfinite(check.log_D_obs).all()
    assert np.isposinf(check.D_obs).all()
    assert float(check.PPP) == 0


def test_invalid_check_grouping_is_explicit(comparison_case):
    with pytest.raises(ValueError, match="groupby"):
        comparison_case.pearson_check(groupby="person")


def test_ppp_counts_ties_and_only_resamples_responses(comparison_case, monkeypatch):
    import pymc as pm

    result = comparison_case
    blocks = list(result.built.inputs.block_ids)
    values = xr.Dataset(
        {
            "check_log_D_obs": (
                ("chain", "draw", "block"),
                np.full((1, 3, len(blocks)), np.log(2.0)),
            ),
            "check_log_D_rep": (
                ("chain", "draw", "block"),
                np.broadcast_to(np.log([1.0, 2.0, 3.0])[None, :, None], (1, 3, len(blocks))),
            ),
        },
        coords={"chain": [0], "draw": [0, 1, 2], "block": blocks},
    )

    def forward(trace, **kwargs):
        assert set(trace.data_vars) == {rv.name for rv in result.built.model.free_RVs}
        assert set(kwargs["freeze_vars"]) == set(trace.data_vars)
        assert set(kwargs["sample_vars"]) == {
            f"check_rep_{group.name}" for group in result.built.inputs.groups
        }
        return xr.DataTree.from_dict({"posterior_predictive": values})

    monkeypatch.setattr(pm, "sample_posterior_predictive", forward)
    check = result.pearson_check(progressbar=False)
    assert_allclose(check.PPP, 2 / 3, atol=1e-15)


def test_unanswered_blocks_have_no_fabricated_ppp(comparison_case):
    result = comparison_case
    first = result.built.inputs.block_ids[0]
    responses = result.built.data.to_frame(answered_only=True)
    responses = responses.loc[responses.block_id != first]
    answered = set(zip(responses.person_id, responses.block_id, strict=True))
    administration = pd.DataFrame(
        [
            {
                "person_id": person,
                "block_id": block,
                "status": "answered" if (person, block) in answered else "missing",
            }
            for person in result.built.inputs.person_ids
            for block in result.built.inputs.block_ids
        ]
    )
    data = FCData.from_frames(responses, result.built.data.design, administration=administration)
    built = Model(data).build()
    fixed = FitResult(idata=result.idata, built=built, sampling={})
    if responses.empty:
        with pytest.raises(ValueError, match="at least one effective response"):
            fixed.pearson_check(progressbar=False)
        return
    check = fixed.pearson_check(groupby="block", random_seed=70, progressbar=False)
    assert int(check.n_obs.sel(block=first)) == 0
    assert np.isnan(check.PPP.sel(block=first))
    assert (check.D_obs.sel(block=first) == 0).all()
