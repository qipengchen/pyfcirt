"""Independent scalar WAIC and chainwise DIC definitions."""

from math import exp, log
from statistics import pvariance

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyfcirt import dic, waic


def labelled_logp(values=None):
    if values is None:
        values = -np.arange(24).reshape(2, 4, 3) / 100 - 0.1
    return xr.DataArray(
        values,
        dims=("chain", "draw", "obs"),
        coords={"chain": ["C0", "C1"], "draw": [0, 1, 2, 3], "obs": range(values.shape[-1])},
        attrs={"inference_method": "mcmc"},
    )


def test_dic_is_chainwise_total_deviance_variance():
    deviance = np.array([[2, 4, 6, 8], [20, 24, 28, 32]], dtype=float)
    logp = labelled_logp(-deviance[..., None] * np.array([0.1, 0.2, 0.2]))
    result = dic(logp)
    assert_allclose(result.Dbar_chain, [5, 26], atol=1e-12)
    assert_allclose(result.pV_chain, [10 / 3, 40 / 3], atol=1e-12)
    assert_allclose(result.DIC, 15.5 + 25 / 3, atol=1e-12)
    assert not np.isclose(result.pV, deviance.var(ddof=1) / 2)
    assert not np.isclose(result.pV, ((-2 * logp).var("draw", ddof=1).sum("obs") / 2).mean())


def test_waic_matches_scalar_probability_definition():
    logp = labelled_logp()
    expected, penalties = [], []
    for observation in range(3):
        samples = logp.values[..., observation].ravel().tolist()
        penalties.append(pvariance(samples))
        expected.append(log(sum(exp(v) for v in samples) / len(samples)) - penalties[-1])
    result = waic(logp)
    assert_allclose(result.elpd_waic_i, expected, atol=1e-12)
    assert_allclose(result.p_waic_i, penalties, atol=1e-12)
    assert_allclose(result.WAIC, -2 * sum(expected), atol=1e-12)
    assert_allclose(result.se_elpd, (3 * pvariance(expected)) ** 0.5, atol=1e-12)
    assert_allclose(result.se_WAIC, 2 * result.se_elpd)


def test_reordering_preserves_metrics_but_chain_reassignment_changes_dic():
    original = labelled_logp()
    permuted = original.isel(chain=[1, 0], draw=[3, 1, 0, 2], obs=[2, 0, 1])
    assert_allclose(dic(original).DIC, dic(permuted).DIC)
    assert_allclose(waic(original).WAIC, waic(permuted).WAIC)
    regrouped = original.copy(data=original.values.swapaxes(0, 1).reshape(original.shape))
    assert not np.isclose(dic(original).DIC, dic(regrouped).DIC)
    assert_allclose(waic(original).WAIC, waic(regrouped).WAIC)


@pytest.mark.parametrize("function", [dic, waic])
@pytest.mark.parametrize(
    "invalid",
    ["nan", "infinity", "duplicate", "missing_coordinate", "wrong_dim", "empty", "complex", "advi"],
)
def test_metrics_reject_invalid_arrays(function, invalid):
    values = labelled_logp()
    if invalid == "nan":
        values.values[0, 0, 0] = np.nan
    elif invalid == "infinity":
        values.values[0, 0, 0] = -np.inf
    elif invalid == "duplicate":
        values = values.assign_coords(obs=[0, 0, 1])
    elif invalid == "missing_coordinate":
        values = values.drop_vars("obs")
    elif invalid == "wrong_dim":
        values = values.rename(obs="person")
    elif invalid == "empty":
        values = values.isel(draw=slice(0))
    elif invalid == "complex":
        values = values.astype(complex)
    else:
        values.attrs["inference_method"] = "advi"
    with pytest.raises(NotImplementedError if invalid == "advi" else ValueError):
        function(values)


def test_extreme_finite_values_and_waic_variance_warning():
    values = labelled_logp(np.full((2, 4, 3), -1e6))
    assert_allclose(waic(values).WAIC, 6e6, atol=1e-8)
    assert_allclose(dic(values).pV, 0)
    integers = labelled_logp(np.full((2, 4, 3), -2_000_000_000_000_000_000, dtype=np.int64))
    assert_allclose(dic(integers).Dbar, 1.2e19, rtol=1e-15)
    with pytest.warns(UserWarning, match="exceeds 0.4"):
        assert bool(waic(labelled_logp(-np.arange(24).reshape(2, 4, 3) * 0.3)).warning)
    with pytest.raises(ValueError, match="two draws"):
        dic(values.isel(draw=[0]))
    with pytest.raises(ValueError, match="two posterior states"):
        waic(values.isel(chain=[0], draw=[0]))
