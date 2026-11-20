"""Plots consume existing draws and discrepancy arrays without fitting."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pyfcirt as fc

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def check():
    return xr.Dataset(
        {
            "D_diff": (("chain", "draw", "unit"), [[[-1.0, 0.0], [1.0, 0.0]]]),
            "PPP": ("unit", [0.5, np.nan]),
            "n_obs": ("unit", [2, 0]),
        },
        coords={"chain": [0], "draw": [0, 1], "unit": ["observed", "empty"]},
        attrs={"groupby": "block"},
    )


def test_ppp_uses_saved_discrepancies_and_empty_group_annotation(check):
    before = check.copy(deep=True)
    figure = fc.plot_ppp(check)
    assert [ax.get_title() for ax in figure.axes] == ["observed", "empty"]
    assert "No observed responses" in [t.get_text() for t in figure.axes[1].texts]
    xr.testing.assert_identical(check, before)
    selected = fc.plot_ppp(check, coords={"unit": "observed"})
    assert len(selected.axes) == 1


@pytest.mark.parametrize("quantile_range", [None, (0.01, 0.99)])
@pytest.mark.parametrize("invalid", [np.inf, -np.inf, np.nan])
def test_ppp_rejects_overflow_and_wrong_group_selection(check, quantile_range, invalid):
    with pytest.raises(ValueError, match="existing grouping"):
        fc.plot_ppp(check, coords={"person": [0]})
    check.D_diff.values[0, 0, 0] = invalid
    with pytest.raises(FloatingPointError, match="plotting range"):
        fc.plot_ppp(check, quantile_range=quantile_range)


@pytest.fixture
def tail_check():
    differences = np.r_[-1e9, np.linspace(1, 2, 100), 1e9]
    return xr.Dataset(
        {
            "D_diff": (("chain", "draw"), differences[None, :]),
            "PPP": (differences >= 0).mean(),
            "n_obs": 2,
        },
        attrs={"groupby": "overall"},
    )


@pytest.mark.parametrize("bins", ["auto", 8, np.linspace(1, 2, 6), np.linspace(1.4, 1.6, 3)])
def test_ppp_quantile_range_limits_display_without_changing_saved_check(tail_check, bins):
    before = tail_check.copy(deep=True)
    differences = tail_check.D_diff.values.ravel()
    lower, upper = np.quantile(differences, (0.05, 0.95))
    displayed = differences[(differences >= lower) & (differences <= upper)]
    expected_counts, expected_edges = np.histogram(displayed, bins=bins)

    axis = fc.plot_ppp(tail_check, bins=bins, quantile_range=(0.05, 0.95)).axes[0]

    np.testing.assert_allclose([bar.get_height() for bar in axis.patches], expected_counts)
    np.testing.assert_allclose([bar.get_x() for bar in axis.patches], expected_edges[:-1])
    np.testing.assert_allclose([bar.get_width() for bar in axis.patches], np.diff(expected_edges))
    np.testing.assert_allclose(axis.get_xlim(), (0, expected_edges[-1]))
    assert axis.lines[0].get_xdata() == [0, 0]
    text = axis.texts[0].get_text()
    assert f"PPP = {float(before.PPP):.3f}" in text
    assert "outside display" not in text
    xr.testing.assert_identical(tail_check, before)


def test_ppp_default_and_full_quantile_range_keep_all_draws(tail_check):
    differences = tail_check.D_diff.values.ravel()
    expected_counts, expected_edges = np.histogram(differences, bins=6)
    for quantile_range in (None, (0, 1)):
        axis = fc.plot_ppp(tail_check, bins=6, quantile_range=quantile_range).axes[0]
        np.testing.assert_allclose([bar.get_height() for bar in axis.patches], expected_counts)
        np.testing.assert_allclose([bar.get_x() for bar in axis.patches], expected_edges[:-1])
        assert "outside display" not in axis.texts[0].get_text()


@pytest.mark.parametrize(
    "quantile_range",
    [
        (-0.1, 0.9),
        (0.5, 0.5),
        (0.8, 0.2),
        (0, 1.1),
        (np.nan, 1),
        (0, np.inf),
        (0,),
        (0, 0.5, 1),
        0.5,
        "bad",
    ],
)
def test_ppp_rejects_invalid_quantile_range(check, quantile_range):
    with pytest.raises(ValueError, match="quantile_range|convert"):
        fc.plot_ppp(check, quantile_range=quantile_range)


@pytest.mark.parametrize("constant", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("bins", ["auto", 3])
def test_ppp_quantile_range_handles_grouped_constant_and_empty_samples(check, constant, bins):
    check.D_diff.values[:, :, 0] = constant
    figure = fc.plot_ppp(check, bins=bins, quantile_range=(0.01, 0.99))
    axis, empty = figure.axes
    assert sum(bar.get_height() for bar in axis.patches) == 2
    assert "outside display" not in axis.texts[0].get_text()
    lower, upper = axis.get_xlim()
    assert lower <= min(0, constant) <= max(0, constant) <= upper
    assert lower < upper
    assert min(bar.get_x() for bar in axis.patches) >= lower
    assert max(bar.get_x() + bar.get_width() for bar in axis.patches) <= upper
    assert "No observed responses" in empty.texts[0].get_text()


def test_ppp_quantile_range_is_local_to_each_group(check):
    check.D_diff.values[:, :, 1] = [10.0, 20.0]
    check.n_obs.values[1] = 2
    figure = fc.plot_ppp(check, quantile_range=(0, 0.5))
    np.testing.assert_allclose(figure.axes[0].get_xlim(), (-1.5, 0))
    np.testing.assert_allclose(figure.axes[1].get_xlim(), (0, 10.5))
    for axis in figure.axes:
        assert sum(bar.get_height() for bar in axis.patches) == 1
        assert "outside display" not in axis.texts[0].get_text()


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("bins", ["auto", 8])
def test_ppp_quantile_range_avoids_interpolated_sparse_tail_gaps(tail_check, sign, bins):
    check = tail_check.isel(draw=slice(0, 100)).copy(deep=True)
    differences = sign * np.r_[np.arange(99), 1e100]
    check.D_diff.values[0] = differences
    check["PPP"] = (differences >= 0).mean()
    before = check.copy(deep=True)
    assert np.max(np.abs(np.quantile(differences, (0.01, 0.99)))) > 1e90
    expected_counts, expected_edges = np.histogram(sign * np.arange(1, 99), bins=bins)

    axis = fc.plot_ppp(check, quantile_range=(0.01, 0.99), bins=bins).axes[0]

    np.testing.assert_allclose([bar.get_height() for bar in axis.patches], expected_counts)
    np.testing.assert_allclose([bar.get_x() for bar in axis.patches], expected_edges[:-1])
    np.testing.assert_allclose(axis.get_xlim(), (min(sign * 98, 0), max(sign * 98, 0)))
    assert "outside display" not in axis.texts[0].get_text()
    assert f"PPP = {float(before.PPP):.3f}" in axis.texts[0].get_text()
    xr.testing.assert_identical(check, before)


@pytest.mark.parametrize("quantile_range", [(0.25, 0.75), (0.49999999999999994, 0.5)])
@pytest.mark.parametrize("bins", ["auto", 3, [-2, 0, 2]])
def test_ppp_empty_quantile_selection_uses_native_empty_bins(check, quantile_range, bins):
    check = check.isel(unit=0, drop=True)
    check.D_diff.values[0] = [1e100, 1e100 + 1e90]
    before = check.copy(deep=True)
    expected_counts, expected_edges = np.histogram([], bins=bins)

    axis = fc.plot_ppp(check, quantile_range=quantile_range, bins=bins).axes[0]

    np.testing.assert_allclose([bar.get_height() for bar in axis.patches], expected_counts)
    np.testing.assert_allclose(
        [bar.get_x() for bar in axis.patches], expected_edges[:-1], atol=1e-15
    )
    np.testing.assert_allclose(axis.get_xlim(), (min(expected_edges[0], 0), expected_edges[-1]))
    assert "outside display" not in axis.texts[0].get_text()
    xr.testing.assert_identical(check, before)


def test_native_ridge_accepts_labelled_dataset_and_tree():
    pytest.importorskip("arviz_plots")
    from scipy.special import ndtri

    values = ndtri((np.arange(40) + 0.5) / 40).reshape(1, 40, 1, 1)
    posterior = xr.Dataset(
        {"theta": (("chain", "draw", "person", "trait"), values)},
        coords={"chain": [0], "draw": range(40), "person": ["p"], "trait": ["T"]},
    )
    for data in (posterior, xr.DataTree.from_dict({"posterior": posterior})):
        assert fc.plot_ridge(data, colors="C0") is not None


@pytest.mark.parametrize(
    "options", [{"max_rows": 0}, {"col_wrap": False}, {"colors": []}, {"coords": {"draw": [0]}}]
)
def test_ridge_rejects_invalid_options(options):
    pytest.importorskip("arviz_plots")
    posterior = xr.Dataset(
        {"theta": (("chain", "draw", "person"), np.ones((1, 2, 1)))},
        coords={"chain": [0], "draw": [0, 1], "person": ["p"]},
    )
    with pytest.raises(ValueError):
        fc.plot_ridge(posterior, **options)


def test_ppp_scalar_truncation_empty_selection_and_panel_shape_contracts(check):
    with pytest.raises(TypeError, match="Dataset"):
        fc.plot_ppp(xr.Dataset())
    with pytest.raises(ValueError, match="max_panels"):
        fc.plot_ppp(check, max_panels=False)
    with pytest.raises(ValueError, match="at most one"):
        fc.plot_ppp(check.expand_dims(other=[0]))
    with pytest.raises(ValueError, match="at least one"):
        fc.plot_ppp(check, coords={"unit": []})
    truncated = fc.plot_ppp(check, max_panels=1)
    assert "Showing 1 of 2" in truncated._suptitle.get_text()
    scalar = check.isel(unit=0, drop=True)
    scalar.attrs["groupby"] = "overall"
    figure = fc.plot_ppp(scalar, max_panels=None)
    assert figure.axes[0].get_title() == "Overall"
    many = xr.concat([scalar] * 4, dim=xr.IndexVariable("unit", list("abcd")))
    assert len(fc.plot_ppp(many).axes) == 4


@pytest.mark.parametrize("problem", ["missing", "one_state", "nan", "boolean"])
def test_ridge_rejects_missing_or_invalid_actual_samples(problem):
    pytest.importorskip("arviz_plots")
    posterior = xr.Dataset(
        {"theta": (("chain", "draw"), [[0.0, 1.0]])}, coords={"chain": [0], "draw": [0, 1]}
    )
    if problem == "missing":
        posterior = xr.Dataset()
    elif problem == "one_state":
        posterior = posterior.isel(draw=[0])
    elif problem == "nan":
        posterior.theta.values[0, 0] = np.nan
    else:
        posterior["theta"] = posterior.theta.astype(bool)
    with pytest.raises(ValueError):
        fc.plot_ridge(posterior)


def test_ridge_scalar_constant_entries_typed_ids_and_row_truncation():
    pytest.importorskip("arviz_plots")
    posterior = xr.Dataset(
        {
            "theta": (("chain", "draw", "person"), np.tile([0.0, 1.0, 2.0], (1, 4, 1))),
            "scalar": (("chain", "draw"), [[3.0, 3.0, 3.0, 3.0]]),
        },
        coords={"chain": [0], "draw": range(4), "person": np.array([7, "7", "p"], object)},
    )
    assert fc.plot_ridge(posterior, var_name="scalar", ci_probs=()) is not None
    assert fc.plot_ridge(posterior, max_rows=2, ci_probs=(), colors=["red"]) is not None
    assert fc.plot_ridge(posterior, coords={"person": "p"}, max_rows=None) is not None


def test_ridge_checks_finite_values_only_in_bounded_rows():
    pytest.importorskip("arviz_plots")
    posterior = xr.Dataset(
        {"theta": (("chain", "draw", "person"), np.tile([0.0, 1.0, np.nan], (1, 4, 1)))},
        coords={"chain": [0], "draw": range(4), "person": ["a", "b", "c"]},
    )
    assert fc.plot_ridge(posterior, max_rows=2, ci_probs=()) is not None
    with pytest.raises(ValueError, match="finite real values"):
        fc.plot_ridge(posterior, max_rows=3, ci_probs=())
    with pytest.raises(ValueError, match="finite real values"):
        fc.plot_ridge(posterior, coords={"person": ["c"]}, max_rows=1, ci_probs=())


def test_ridge_bounds_lazy_reads_before_materializing_samples():
    pytest.importorskip("arviz_plots")
    from xarray.backends import BackendArray
    from xarray.core import indexing

    samples = np.tile([0.0, 1.0, np.nan], (1, 4, 1))
    reads = []

    class BoundedSamples(BackendArray):
        shape = samples.shape
        dtype = samples.dtype

        def __getitem__(self, key):
            return indexing.explicit_indexing_adapter(
                key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
            )

        def _getitem(self, key):
            rows = np.arange(self.shape[-1])[key[-1]]
            assert np.all(rows < 2), "An excluded posterior row was materialized"
            reads.append(np.asarray(rows))
            return samples[key]

    posterior = xr.Dataset(
        {
            "theta": xr.Variable(
                ("chain", "draw", "person"), indexing.LazilyIndexedArray(BoundedSamples())
            )
        },
        coords={"chain": [0], "draw": range(4), "person": ["a", "b", "c"]},
    )
    assert not reads
    assert fc.plot_ridge(posterior, max_rows=2, ci_probs=()) is not None
    assert reads
    assert set(np.concatenate([value.ravel() for value in reads])) == {0, 1}


@pytest.mark.parametrize("statistic", ["count", "proportion"])
def test_response_plot_preserves_distinct_typed_categories(statistic):
    sns = pytest.importorskip("seaborn")
    source = fc.FCInput(
        fc.FCDesignMapping(traits={"A": ["left", "right"]}, blocks={"block": ["left", "right"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": range(6),
                    "block_id": "block",
                    "response": [1, "1", "1", "'1'", "'1'", "'1'"],
                }
            ),
            format="block_long",
        ),
        model="MGPPM",
        response_format=None,
        scale=fc.CategoryScale((1, "1", "'1'")),
    )
    long = fc.response_data(fc.PreparedInputs.from_inputs(source))
    before = long.copy(deep=True)
    grid = fc.plot_response_distribution(long, statistic=statistic)
    assert isinstance(grid, sns.FacetGrid)
    axis = grid.axes.flat[0]
    assert len({tick.get_text() for tick in axis.get_xticklabels()}) == 3
    heights = [bar.get_height() for bar in axis.patches]
    np.testing.assert_allclose(
        heights, np.array([1, 2, 3]) / (6 if statistic == "proportion" else 1)
    )
    pd.testing.assert_frame_equal(long, before)
