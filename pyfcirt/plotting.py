"""Small plots of existing checks and native ArviZ posterior distributions."""

from itertools import product

import numpy as np
import pandas as pd
import xarray as xr


def plot_response_distribution(summary: pd.DataFrame, *, statistic="proportion", **kwargs):
    """Plot reusable response labels; return Seaborn's native FacetGrid."""
    import seaborn as sns

    from pyfcirt.analysis.summary import response_counts

    if "n_valid" not in summary:
        summary = response_counts(summary, groupby=kwargs.get("hue"))
    if summary.empty:
        raise ValueError("No observed responses to plot")
    if statistic not in {"proportion", "count"}:
        raise ValueError("statistic must be proportion or count")
    facets = summary.columns[: summary.columns.get_loc("category")].tolist()
    hue = kwargs.get("hue")
    if hue is not None:
        if hue not in facets or hue in {"block", "item", "pair"}:
            raise ValueError("hue must name a grouping column")
        facets.remove(hue)
    if kwargs.pop("errorbar", None) is not None or kwargs.pop("sharex", False):
        raise ValueError("Use errorbar=None and local category axes for precomputed proportions")
    panels = summary.groupby(facets, sort=False, observed=True, dropna=False).ngroup()
    first_rows = ~panels.duplicated()
    titles = summary.loc[first_rows, [f"{name}_label" for name in facets]].set_axis(
        facets, axis="columns"
    )
    labels = titles.apply(
        lambda row: " / ".join(f"{name}={value}" for name, value in row.dropna().items()),
        axis=1,
    )
    # IDs containing the title separator can produce the same text for distinct panels.
    if not labels.is_unique:
        labels = titles.apply(lambda row: repr(tuple(row.dropna().items())), axis=1)
    plot_data = summary.assign(
        _panel=panels.map(dict(zip(panels[first_rows], labels, strict=True)))
    )
    kwargs.setdefault("col_wrap", 3)
    return (
        sns.catplot(
            data=plot_data,
            kind="bar",
            x="category_label",
            y=statistic,
            col="_panel",
            errorbar=None,
            sharex=False,
            **kwargs,
        )
        .set_titles("{col_name}")
        .set_axis_labels(
            "Response category", "Observed proportion" if statistic == "proportion" else "Count"
        )
    )


def plot_ppp(
    check: xr.Dataset, *, coords=None, bins="auto", quantile_range=None, max_panels=12, figsize=None
):
    """Plot saved D_rep-D_obs aggregates without changing check statistics.

    ``quantile_range=(low, high)`` limits each histogram to its empirical quantile
    interval without changing the saved check. Probabilities must satisfy
    ``0 <= low < high <= 1``; the default displays every draw. Automatic/integer
    bins use retained draws, and axes follow native bin edges; explicit bin edges
    are retained. The zero reference remains visible. Changing the grouping
    requires a new check.
    """
    import matplotlib.pyplot as plt

    if (
        not isinstance(check, xr.Dataset)
        or not {"D_diff", "PPP", "n_obs"} <= check.data_vars.keys()
    ):
        raise TypeError("check must be the Dataset returned by chi_ppp")
    group = check.attrs["groupby"]
    if check.PPP.ndim > 1:
        raise ValueError("A saved check must have at most one aggregation axis")
    dimension = next(iter(check.PPP.dims), None)
    if max_panels is not None and (
        isinstance(max_panels, bool) or not isinstance(max_panels, int) or max_panels < 1
    ):
        raise ValueError("max_panels must be a positive integer or None")
    if quantile_range is not None:
        quantile_range = np.asarray(quantile_range, dtype=float)
        if quantile_range.shape != (2,) or not 0 <= quantile_range[0] < quantile_range[1] <= 1:
            raise ValueError("quantile_range must satisfy 0 <= low < high <= 1")
    if coords:
        if dimension is None or set(coords) != {dimension}:
            raise ValueError("coords must select labels on the check's existing grouping axis")
        check = check.sel(
            {key: [value] if np.isscalar(value) else value for key, value in coords.items()}
        )
    labels = ["Overall"] if dimension is None else check[dimension].values.tolist()
    count = len(labels) if max_panels is None else min(len(labels), max_panels)
    if not count:
        raise ValueError("Select at least one aggregate")
    columns = min(count, 3)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=figsize or (4 * columns, 3 * rows)
    )
    for index, (axis, label) in enumerate(zip(axes.flat, labels[:count], strict=False)):
        values = check if dimension is None else check.isel({dimension: index})
        if int(values.n_obs) == 0:
            axis.text(0.5, 0.5, "No observed responses", ha="center", transform=axis.transAxes)
        else:
            differences = values.D_diff.values.ravel()
            if not np.isfinite(differences).all():
                plt.close(figure)
                raise FloatingPointError(
                    "Raw differences exceed plotting range; inspect log_D_obs/log_D_rep"
                )
            annotation = f"PPP = {float(values.PPP):.3f}"
            if quantile_range is not None:
                lower, upper = np.quantile(differences, quantile_range)
                differences = differences[(differences >= lower) & (differences <= upper)]
            _, edges, _ = axis.hist(differences, bins=bins, color="C0", alpha=0.7)
            if quantile_range is not None:
                axis.set_xlim(min(edges[0], 0), max(edges[-1], 0))
            axis.text(
                0.98,
                0.96,
                annotation,
                ha="right",
                va="top",
                transform=axis.transAxes,
            )
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=str(label), xlabel=r"$D_{rep}-D_{obs}$", ylabel="Posterior predictive draws")
    for axis in list(axes.flat)[count:]:
        figure.delaxes(axis)
    if count < len(labels):
        figure.suptitle(f"Showing {count} of {len(labels)} {group} groups")
    figure.tight_layout()
    return figure


def plot_ridge(
    data: xr.Dataset | xr.DataTree,
    var_name="theta",
    *,
    coords=None,
    max_rows=12,
    col_wrap=4,
    ci_probs=(0.5, 0.95),
    figsize=None,
    colors=None,
):
    """Compose native ArviZ ridges and HDIs; return its editable PlotCollection."""
    import arviz_plots as azp
    from matplotlib.collections import PathCollection
    from matplotlib.colors import is_color_like

    posterior = data["posterior"].to_dataset() if isinstance(data, xr.DataTree) else data
    if not isinstance(posterior, xr.Dataset) or var_name not in posterior:
        raise ValueError("data must contain the requested posterior variable")
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer or None")
    if isinstance(col_wrap, bool) or not isinstance(col_wrap, int) or col_wrap < 1:
        raise ValueError("col_wrap must be a positive integer")
    palette = ["C0"] if colors is None else [colors] if isinstance(colors, str) else list(colors)
    if not palette or any(not is_color_like(color) for color in palette):
        raise ValueError("colors must contain Matplotlib colors")
    values = posterior[var_name]
    if coords:
        if set(coords) - set(values.dims) or {"chain", "draw"}.intersection(coords):
            raise ValueError("coords selects parameter dimensions, not sample axes")
        values = values.sel(
            {key: [value] if np.isscalar(value) else value for key, value in coords.items()}
        )
    if (
        not {"chain", "draw"}.issubset(values.dims)
        or values.sizes["chain"] * values.sizes["draw"] < 2
    ):
        raise ValueError("Ridges require at least two actual posterior states")
    dimensions = [dim for dim in values.dims if dim not in {"chain", "draw"}]
    row = next(
        (dim for dim in ("person", "statement", "item") if dim in dimensions),
        next(iter(dimensions), None),
    )
    total = values.sizes[row] if row else 1
    if row:
        values = values.isel({row: slice(None, max_rows)}).isel({row: slice(None, None, -1)})
    values = values.compute()
    if values.dtype.kind not in "iuf" or not values.size or not np.isfinite(values.data).all():
        raise ValueError("Selected samples must be nonempty finite real values")
    selected = values.sizes[row] if row else 1
    facets = [dim for dim in dimensions if dim != row]
    # String and integer IDs may have identical print forms; preserve distinct labels.
    display = {}
    for dim in dimensions:
        labels = values[dim].values.tolist()
        text = [str(label) for label in labels]
        display[dim] = [repr(label) for label in labels] if len(set(text)) != len(text) else text
    values = values.assign_coords(display)
    dataset = values.to_dataset(name=var_name)
    panels = int(np.prod([values.sizes[dim] for dim in facets]))
    columns = min(col_wrap, panels)
    pc = azp.PlotCollection.wrap(
        dataset,
        cols=facets or ["__variable__"],
        col_wrap=col_wrap,
        aes={"y": [row] if row else ["__variable__"]},
        backend="matplotlib",
        figure_kwargs={
            "figsize": figsize
            or (3.2 * columns, ((panels + columns - 1) // columns) * (1.4 + 0.35 * selected)),
            "sharex": True,
        },
    )
    minima = values.min(("chain", "draw"))
    maxima = values.max(("chain", "draw"))
    constant = bool((minima == maxima).any())
    stats = {}
    if constant:
        lower, upper = float(minima.min()), float(maxima.max())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        shape = tuple(values.sizes[dim] for dim in dimensions) + (51,)
        stats = {"dist": {"bins": np.broadcast_to(np.linspace(lower, upper, 51), shape)}}
    # Histograms also display fixed entries, whose KDE has no positive bandwidth.
    azp.plot_ridge(
        dataset,
        plot_collection=pc,
        combined=True,
        kind="hist" if constant else "kde",
        ridge_height=0.75,
        labels=[],
        visuals={"ticklabels": False, "remove_axis": False},
        stats=stats,
    )
    if ci_probs:
        azp.plot_forest(
            dataset,
            plot_collection=pc,
            combined=True,
            point_estimate="mean",
            ci_kind="hdi",
            ci_probs=list(ci_probs),
            labels=[],
            visuals={"ticklabels": False, "remove_axis": False},
        )
    for index, labels in enumerate(product(*(values[dim].values.tolist() for dim in facets))):
        selection = dict(zip(facets, labels, strict=True))
        axis = pc.get_target(var_name, selection)
        color = palette[index % len(palette)]
        for line in axis.lines:
            line.set_color(color)
        for artist in axis.collections:
            artist.set_edgecolor(color)
            artist.set_facecolor("white" if isinstance(artist, PathCollection) else color)
        axis.set_yticks(np.arange(selected), values[row].values.tolist() if row else [var_name])
        axis.set(
            title=", ".join(str(label) for label in labels) or var_name,
            xlabel=var_name,
            ylim=(-0.5, selected - 0.05),
        )
    if selected < total:
        pc.add_title(f"{var_name}: first {selected} of {total} {row}")
    return pc
