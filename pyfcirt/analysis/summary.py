"""Read frozen specifications and summarize existing final samples without a graph."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from pyfcirt.analysis.output import posterior_samples
from pyfcirt.data.coordinates import select_units
from pyfcirt.models.priors import PriorPreparation

if TYPE_CHECKING:
    from pyfcirt.data.prepare import PreparedInputs
    from pyfcirt.model import Model


def _prepared_inputs(data):
    from pyfcirt.data.prepare import PreparedInputs
    from pyfcirt.model import Model

    prepared = data.prepared if isinstance(data, Model) else data
    if not isinstance(prepared, PreparedInputs):
        raise TypeError("Supply a Model or PreparedInputs; use native ArviZ for bare traces")
    return prepared


def _model_response_format(model, response_format):
    return f"{model} / {response_format}" if response_format else model


_MEASUREMENT_GROUP = ("channel", "model_response_format", "link", "size", "n_categories")


def _measurement_table(prepared):
    arrays = prepared.constant_data.variables
    table = pd.DataFrame(
        prepared.measurement_overview,
        columns=["size", "link", "link_multiplier", "n_categories", "n_response_categories"],
    )
    models, formats = (arrays[name].values for name in ("unit_model", "unit_response_format"))
    return table.rename_axis("unit_index").assign(
        channel=arrays["unit_channel"].values,
        unit=arrays["unit_id"].values,
        model_response_format=list(map(_model_response_format, models, formats)),
    )


def data_summary(
    data: Model | PreparedInputs,
    *,
    by: Literal["model", "block", "item"] = "model",
    details: bool = False,
) -> pd.DataFrame:
    """Summarize recorded administrations, deduplicating people across model groups.

    The default is a compact grouped overview. Block/item views retain their actual
    identifiers. Detailed columns distinguish response statuses and likelihood events;
    an answered administration does not assert that every comparison was answered.
    """
    if by not in {"model", "block", "item"}:
        raise ValueError("by must be 'model', 'block' or 'item'")
    prepared = _prepared_inputs(data)
    constant = prepared.constant_data
    administration = prepared.observed_data
    table = _measurement_table(prepared)
    group_index = table.index.to_numpy()
    if by == "model":
        groups = table.groupby(list(_MEASUREMENT_GROUP), sort=False, dropna=False)
        group_index = groups.ngroup().to_numpy()
        result = groups.head(1).reset_index(drop=True)
        size = groups.size().to_numpy()
        result["n_blocks"] = size * result.channel.eq("fc")
        result["n_items"] = size * result.channel.eq("items")
        columns = [name for name in ("n_blocks", "n_items") if result[name].any()] + ["size"]
    else:
        channel = "fc" if by == "block" else "items"
        result = table.loc[table.channel == channel].copy()
        columns = ["model_response_format", *(["size"] if by == "block" else [])]
    admin_group = group_index[administration.administration_unit.values]
    people = pd.Series(administration.administration_person.values).groupby(admin_group, sort=False)
    result["n_persons"] = people.nunique().reindex(result.index, fill_value=0)
    statuses = ["answered", "missing", "refused", "not_administered"]
    status_columns = [f"n_{status}" for status in statuses]
    if prepared.data.attrs["purpose"] == "generate":
        result[status_columns] = pd.NA
        result["n_obs"] = 0
    else:
        counts = pd.Series(admin_group).groupby(
            [admin_group, administration.administration_status.values], sort=False
        )
        result[status_columns] = (
            counts.size()
            .unstack(fill_value=0)
            .reindex(index=result.index, columns=statuses, fill_value=0)
            .to_numpy()
        )
        events = pd.Series(group_index[constant.obs_unit.values]).value_counts()
        result["n_obs"] = events.reindex(result.index, fill_value=0)
    columns += ["n_persons", "n_answered", "n_missing"]
    result = result.set_index("model_response_format" if by == "model" else "unit")
    if by != "model":
        result = result.rename_axis(by)

    # Show response-law columns when they distinguish otherwise identical overview
    # groups. The complete schema remains available explicitly through details=True.
    identity = result.reset_index()
    descriptors = ("link", "n_categories")
    distinct = identity.groupby(["model_response_format", "size"], sort=False, dropna=False)
    for name in descriptors:
        if details or distinct[name].nunique(dropna=False).gt(1).any():
            columns.insert(columns.index("size") if "size" in columns else 1, name)
    if details:
        columns.insert(columns.index("n_categories") + 1, "n_response_categories")
        columns += ["n_refused", "n_not_administered", "n_obs"]
    return result.loc[:, columns]


def _response_label(value, *, separator=" > "):
    """Format a display label while keeping ambiguous typed IDs distinguishable."""
    if not isinstance(value, tuple):
        return str(value)
    identifiers = [part for part in value if part is not None]
    names = list(map(str, identifiers))
    if len(set(names)) != len(set(identifiers)) or any(
        operator in name for name in names for operator in (" > ", " < ")
    ):
        names = list(map(repr, identifiers))
    if None in value:
        return f"{names[0]} (best)" + (f" / {names[-1]} (worst)" if value[-1] is not None else "")
    return separator.join(names)


def _response_labels(values, formatter=str):
    """Create display aliases once, preserving different values with identical text."""
    unique = values.dropna().drop_duplicates().astype(object)
    labels = unique.map(formatter)
    if not labels.is_unique:
        labels = unique.map(repr)
    return values.map(dict(zip(unique, labels, strict=True)))


def _comparison_labels(values, scale):
    """Name the declared scale direction, not each respondent's observed winner."""
    separator = " > " if scale["higher_prefers"] == "first" else " < "
    return _response_labels(values, lambda pair: _response_label(pair, separator=separator))


def response_data(
    data: Model | PreparedInputs,
    *,
    blocks=None,
    items=None,
    representation="auto",
    groups: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Arrange existing responses as a labelled long table without changing fitting data."""
    from pyfcirt.common.validation import identifiers
    from pyfcirt.data.encoding import ResponseEncoding, observed_response_view

    prepared = _prepared_inputs(data)
    constant = prepared.constant_data
    selected = select_units(constant, blocks, items)
    modes = representation if isinstance(representation, Mapping) else {}
    options = modes.values() if isinstance(representation, Mapping) else [representation]
    if set(modes) - set(constant.block.values) or any(
        mode not in {"auto", "categories", "pairwise"} for mode in options
    ):
        raise ValueError("representation must be auto, categories, pairwise, or a block mapping")
    people = pd.DataFrame(
        {name: coord.values for name, coord in constant.coords.items() if coord.dims == ("person",)}
    )
    people["population_group"] = constant.population_group.values[constant.person_group.values]
    if groups is not None:
        identifiers(groups.index, "groups.person")
        if groups.index.name != "person":
            raise ValueError("groups must be a DataFrame indexed by person")
        if set(groups).intersection(
            {
                "block",
                "item",
                "pair",
                "category",
                "model_response_format",
                "representation",
                "count",
                "n_valid",
                "proportion",
            }
        ) or any(name.endswith("_label") for name in groups):
            raise ValueError("Group names must not replace response fields or display labels")
        people = people.join(groups, on="person", validate="many_to_one")
    tables = []
    encoding = ResponseEncoding()
    models = constant.unit_model.values.tolist()
    for index in selected:
        unit, label = constant.attrs["units"][index], constant.unit_id.values[index]
        axis = "block" if unit["channel"] == "fc" else "item"
        if unit["scale"]["kind"] == "unit_interval":
            raise ValueError(
                f"{axis.title()} {label!r}: continuous Beta responses have no categories"
            )
        mode = modes.get(label, "auto") if isinstance(representation, Mapping) else representation
        if axis == "item" and isinstance(representation, Mapping):
            mode = "auto"
        if mode == "auto":
            spec = prepared.measurement_specs[index]
            mode = "pairwise" if spec.comparison_based else "categories"
        if axis == "item" and mode == "pairwise":
            raise ValueError(f"Item {label!r}: pairwise representation is unavailable")
        rows = observed_response_view(prepared, unit, mode, encoding)
        declared = unit["scale"].get("labels")
        formatter = (
            dict(zip(unit["scale"]["categories"], declared, strict=True))
            if declared and (mode == "pairwise" or unit["response_format"] is None)
            else _response_label
        )
        rows["category_label"] = _response_labels(rows.category, formatter)
        rows["pair_label"] = _comparison_labels(rows.pair, unit["scale"])
        # Missing entries from the other channel must not promote integer IDs to floats.
        tables.append(
            rows.assign(
                **{axis: pd.Series(label, index=rows.index, dtype=object)},
                model_response_format=_model_response_format(
                    models[index], unit["response_format"]
                ),
                representation=mode,
            )
        )
    metadata = ["model_response_format", "representation"]
    rows = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(
            columns=[
                "person",
                "block",
                "item",
                "pair",
                "pair_label",
                "category",
                "category_label",
                *metadata,
            ]
        )
    )
    rows = rows.merge(people, on="person", how="left", validate="many_to_one")
    if rows.pair.isna().all():
        rows = rows.drop(columns=["pair", "pair_label"])
    indices = [
        name for name in rows if name not in {"category", "category_label", "pair_label", *metadata}
    ]
    for name in indices:
        if name != "pair":
            rows[f"{name}_label"] = _response_labels(rows[name], _response_label)
    return rows[
        [*indices, "category", "category_label", *[f"{name}_label" for name in indices], *metadata]
    ]


def response_summary(data: pd.DataFrame, *, groupby=None) -> pd.DataFrame:
    """Count and normalize a response_data table; reuse its encoding and labels."""
    rows = data
    grouping = [groupby] if isinstance(groupby, str) else [] if groupby is None else list(groupby)
    if any(
        name
        in {"category", "model_response_format", "representation", "count", "n_valid", "proportion"}
        or name.endswith("_label")
        for name in grouping
    ):
        raise ValueError("groupby must name index or group columns")
    fields = [name for name in grouping if name not in {"person", "block", "item", "pair"}]
    if rows[fields].isna().any().any():
        raise ValueError("Selected grouping labels must not be missing")
    keys = list(
        dict.fromkeys([*[name for name in ("block", "item", "pair") if name in rows], *grouping])
    )
    metadata = [
        "model_response_format",
        "representation",
        "category_label",
        *[f"{name}_label" for name in keys],
    ]
    # Categoricals keep integer group keys exact alongside the other channel's NA.
    groupers = [rows[name].astype("category") for name in keys]
    result = (
        rows.groupby([*groupers, "category"], observed=True, sort=False, dropna=False)
        .agg(count=("person", "size"), **{name: (name, "first") for name in metadata})
        .reset_index()
    )
    result["n_valid"] = result.groupby(keys, observed=True, sort=False, dropna=False)[
        "count"
    ].transform("sum")
    result["proportion"] = result["count"] / result.n_valid
    return result


def _summary_options(summary_kwargs, *, fmt, ci_prob, ci_kind):
    from inspect import signature

    from arviz_stats import summary

    options = dict(summary_kwargs)
    if options.pop("fmt", fmt) != fmt or options.pop("group", "posterior") != "posterior":
        raise ValueError(f"This entrance returns a {fmt} posterior summary")
    dims = options.pop("sample_dims", None)
    if dims is not None and tuple(dims) != ("chain", "draw"):
        raise ValueError("Summaries require sample_dims=['chain', 'draw']")
    options.update(ci_prob=ci_prob, ci_kind=ci_kind)
    options.setdefault("round_to", "none")
    signature(summary).bind(None, fmt=fmt, sample_dims=["chain", "draw"], **options)
    return options, _empty_statistics(options)


def _empty_statistics(options):
    """Provide native column names when selected entries are all fixed or empty."""
    from arviz_base import rcParams

    kind = options.get("kind", "all")
    ci = options.get("ci_kind") or rcParams["stats.ci_kind"]
    probability = options.get("ci_prob")
    probability = rcParams["stats.ci_prob"] if probability is None else probability
    if ci not in {"hdi", "eti"} or not 0 < probability < 1:
        raise ValueError("Invalid credible interval settings")
    percent = int(100 * probability)
    columns = {
        "stats": ["mean", "sd", f"{ci}{percent}_lb", f"{ci}{percent}_ub"],
        "diagnostics": ["ess_bulk", "ess_tail", "r_hat", "mcse_mean", "mcse_sd"],
        "stats_median": ["median", "mad", f"eti{percent}_lb", f"eti{percent}_ub"],
        "diagnostics_median": ["ess_median", "ess_tail", "r_hat", "mcse_median"],
        "mc_diagnostics": ["mcse_mean", "ess_mean", "min_ss"],
    }
    columns["all"] = columns["stats"] + columns["diagnostics"]
    columns["all_median"] = columns["stats_median"] + columns["diagnostics_median"]
    if kind not in columns:
        raise ValueError("Invalid summary kind")
    return columns[kind]


def _fixed_entries(layout, fixed):
    """Read the sole constraint template, overridden by an explicit conditioning point."""
    if fixed is not None and layout.kind in fixed:
        values = fixed[layout.kind].transpose(*layout.dims).values
        return values.astype(layout.fixed.dtype, copy=False), np.ones(layout.shape, dtype=bool)
    return layout.fixed, ~np.isnan(layout.fixed)


def _summary_samples(model, idata):
    """Select the saved samples or validate an external trace against this model."""
    if idata is not None:
        posterior = model._checked_samples(idata)
    else:
        idata = model.idata
        posterior = posterior_samples(idata) if "posterior" in idata else xr.Dataset()
    return posterior, idata.attrs.get("inference_method", "mcmc")


def _entry_tables(prepared, layouts, masks, fields, options, columns, posterior, method):
    """Summarize canonical entries once; labels and usage never identify sample values."""
    from arviz_base import extract
    from arviz_stats import summary

    options = dict(options)
    selection = {key: options.pop(key, None) for key in ("var_names", "filter_vars", "coords")}
    allowed = extract(
        xr.Dataset({name: 0 for name in layouts}),
        var_names=selection["var_names"],
        filter_vars=selection["filter_vars"],
        sample_dims=[],
        combined=False,
        keep_dataset=True,
    ).data_vars
    coords = selection["coords"] or {}
    available = {dim for layout in layouts.values() for dim in layout.dims}
    available.update(prepared.output_coords)
    if set(coords) - available:
        raise ValueError(
            "coords must select existing parameter dimensions or auxiliary coordinates"
        )
    fixed = prepared.fixed_parameters
    auto = options["round_to"] == "auto"
    if auto:
        options["round_to"] = "none"
    tables, samples = {}, {}
    for name, layout in layouts.items():
        if name not in allowed:
            continue
        mask = masks[name]
        if coords:
            marker = xr.DataArray(
                np.arange(mask.size).reshape(layout.shape),
                dims=layout.dims,
                coords={dim: prepared.coords[dim] for dim in layout.dims},
            ).assign_coords(
                {
                    key: value
                    for key, value in prepared.output_coords.items()
                    if key in coords and set(value.dims) <= set(layout.dims)
                }
            )
            for dim, labels in coords.items():
                if dim in marker.dims:
                    marker = marker.sel({dim: labels})
                elif dim in marker.coords:
                    marker = marker.where(marker[dim].isin(labels), drop=True)
            positions = marker.values.ravel()
            positions = positions[np.isfinite(positions)].astype(int)
            positions = positions[mask.ravel()[positions]]
        else:
            positions = np.flatnonzero(mask)
        if not len(positions):
            continue
        values, known = _fixed_entries(layout, fixed)
        values, known = values.ravel(), known.ravel()
        tables[name] = pd.DataFrame(
            {field: layout.entry_data[field].values.ravel()[positions] for field in fields},
            index=pd.Index(positions, name="entry_index"),
        ).assign(
            parameter=name,
            fixed_value=np.where(known[positions], values[positions], np.nan),
        )
        free = positions[~known[positions]]
        if not len(free):
            continue
        if method != "mcmc":
            raise NotImplementedError("Coefficient summaries require MCMC posterior samples")
        if name not in posterior:
            raise ValueError(f"Missing required final posterior variable: {name}")
        value = posterior[name].transpose("chain", "draw", *layout.dims)
        axis = f"{name}_summary_entry"
        # Slice contiguous entries so the native summary can reuse the draw array.
        selection = slice(free[0], free[-1] + 1) if np.all(np.diff(free) == 1) else free
        vector = value.stack({axis: layout.dims}, create_index=False).isel({axis: selection})
        samples[name] = vector.reset_coords(drop=True).assign_coords({axis: free})
    if not tables:
        return pd.DataFrame(), None
    entry_axes = ["kind", "entry_index"]
    entries = pd.concat(tables, names=entry_axes)
    if not samples:
        return entries.assign(**dict.fromkeys(columns, np.nan)), {} if auto else None
    native = summary(xr.Dataset(samples), fmt="xarray", sample_dims=["chain", "draw"], **options)
    stats = pd.concat(
        {
            name: native[name].transpose(value.dims[-1], "summary").to_pandas()
            for name, value in samples.items()
        },
        names=entry_axes,
    ).rename_axis(columns=None)
    formatting = None
    if auto:
        from arviz_base import rcParams
        from arviz_stats.summary import _build_fmt_map

        # Format each canonical estimated entry once before expanding shared memberships.
        formatting = _build_fmt_map(stats, rcParams["stats.round_to"])
    return entries.join(stats), formatting


def _coefficient_result(entries, levels, context, formatting):
    """Associate native display formatting with the final table index."""
    result = entries.set_index(levels)
    if formatting is not None:
        from arviz_stats.summary import SummaryDataFrame

        mapped = {}
        for column, labels in formatting.items():
            values = pd.Series(labels).reindex(entries.index)
            values.index = result.index
            mapped[column] = values.dropna().to_dict()
        result = SummaryDataFrame(result, fmt_map=mapped)
    result.attrs = context
    return result


def _entity_labels(indices, labels):
    """Resolve typed entity references without coercing integer IDs through floats."""
    indices = np.asarray(indices, dtype=int)
    result = np.full(indices.shape, pd.NA, dtype=object)
    active = indices >= 0
    result[active] = np.asarray(labels, dtype=object)[indices[active]]
    return result


def coef(
    data: Model,
    *,
    idata: xr.DataTree | None = None,
    component: Literal["measurement", "regression"] = "measurement",
    blocks: Sequence | None = None,
    items: Sequence | None = None,
    ci_prob: float = 0.95,
    ci_kind: Literal["hdi", "eti"] = "hdi",
    var_names: Sequence[str] | None = None,
    coords: Mapping | None = None,
    kind: str = "all",
    **summary_kwargs,
) -> pd.DataFrame:
    """Summarize canonical samples with native ArviZ and prepared entity memberships."""
    from pyfcirt.model import Model

    if not isinstance(data, Model):
        raise TypeError("coef requires the matching Model and its prepared parameter catalogue")
    if component not in {"measurement", "regression"}:
        raise ValueError("component must be measurement or regression")
    if component == "regression" and (blocks is not None or items is not None):
        raise ValueError("blocks/items do not select regression coefficients")
    posterior, method = _summary_samples(data, idata)
    prepared, constant = data.prepared, data.prepared.constant_data
    options, columns = _summary_options(
        dict(summary_kwargs, var_names=var_names, coords=coords, kind=kind),
        fmt="wide",
        ci_prob=ci_prob,
        ci_kind=ci_kind,
    )
    layouts = prepared.parameter_layouts
    point_scoring = data._point_calibrated
    masks = {}
    for name, layout in layouts.items():
        entries = layout.entry_data
        mask = entries.component.values == component
        if point_scoring:
            mask &= (entries.regression_index.values < 0) | entries.fixed_for_scoring.values
        if var_names is None:
            mask &= entries.summary_default.values
        masks[name] = mask.reshape(layout.shape)
    units = select_units(constant, blocks, items) if component == "measurement" else None
    relations = (
        prepared.parameter_units
        if blocks is None and items is None
        else prepared.parameter_unit_table(units)
    )
    if units is not None:
        selected = relations.index.to_frame(index=False).groupby("kind", sort=False)
        selected = selected.entry_index.unique()
        for name, mask in masks.items():
            included = np.zeros_like(mask)
            included.flat[selected.get(name, [])] = True
            mask &= included
    fields = (
        ("statement_index", "trait_index", "pair_index", "cutpoint_position", "category_index")
        if component == "measurement"
        else (
            "population_group_index",
            "trait_index",
            "target_kind",
            "term",
            "parameter_set_index",
            "scale",
        )
    )
    entries, formatting = _entry_tables(
        prepared, layouts, masks, fields, options, columns, posterior, method
    )
    if entries.empty:
        return pd.DataFrame(
            columns=columns,
            index=pd.MultiIndex.from_tuples([], names=["unit", "parameter"]),
        )
    entries["trait"] = _entity_labels(entries.trait_index, constant.trait.values)
    unit_records = constant.attrs["units"]
    if component == "measurement":
        entries["_entry_order"] = np.arange(len(entries))
        order = relations.unit_index.map(dict(zip(units, range(len(units)), strict=True)))
        # Usage owns unit membership, including shared entries with several units.
        entries = entries.join(relations.assign(_order=order), how="inner")
        entries = entries.sort_values(["_order", "_entry_order"], kind="stable")
        indices = entries.unit_index.to_numpy(dtype=int)
        entries["channel"] = constant.unit_channel.values[indices]
        entries["unit"] = pd.Series(
            constant.unit_id.values[indices], index=entries.index, dtype=object
        )
        entries["model_response_format"] = list(
            map(
                _model_response_format,
                constant.unit_model.values[indices],
                constant.unit_response_format.values[indices],
            )
        )
        entries["statement"] = _entity_labels(
            entries.statement_index, constant.statement_label.values
        )
        pair_labels = np.full(len(entries), pd.NA, dtype=object)
        pair_index = entries.pair_index.to_numpy(dtype=int)
        positions = np.flatnonzero(pair_index >= 0)
        paired_units = pd.Series(indices[positions], index=positions)
        for unit, rows in paired_units.groupby(paired_units, sort=False):
            pairs, inverse = np.unique(pair_index[rows.index], return_inverse=True)
            first = constant.pair_statement_1.values[pairs]
            second = constant.pair_statement_2.values[pairs]
            names = constant.statement_label.values
            labels = pd.Series(list(zip(names[first], names[second], strict=True)))
            formatted = _comparison_labels(labels, unit_records[unit]["scale"])
            pair_labels[rows.index] = formatted.to_numpy()[inverse]
        entries["pair"] = pair_labels
        entries["position"] = pd.Series(
            np.where(entries.cutpoint_position >= 0, entries.cutpoint_position + 1, pd.NA),
            index=entries.index,
            dtype=object,
        )
        entries["category"] = pd.Series(
            [
                unit_records[unit]["scale"]["categories"][category] if category >= 0 else pd.NA
                for unit, category in zip(indices, entries.category_index, strict=True)
            ],
            index=entries.index,
            dtype=object,
        )
        levels = [
            "channel",
            "unit",
            "statement",
            "trait",
            "pair",
            "position",
            "category",
            "parameter",
        ]
        context_fields = ["channel", "model_response_format"]
    else:
        names = constant.unit_model.values[relations.unit_index.values]
        models = (
            relations.assign(model=names)
            .groupby(level=["kind", "entry_index"], sort=False)
            .model.agg(lambda names: tuple(pd.unique(names)))
        )
        entries["model"] = (
            models.reindex(entries.index)
            .map(lambda names: names[0] if isinstance(names, tuple) and len(names) == 1 else names)
            .astype(object)
            .where(entries.target_kind != "theta", "Population")
        )
        entries["population_group"] = _entity_labels(
            entries.population_group_index, constant.population_group.values
        )
        entries["target"] = entries.target_kind.replace("", pd.NA)
        entries["parameter"] = entries.term.where(entries.term != "", entries.parameter)
        levels = ["model", "population_group", "target", "trait", "parameter"]
        if entries.duplicated(subset=levels).any():
            levels.insert(1, "parameter_set_index")
        context_fields = ["model", "scale"]
    levels = [name for name in levels if name == "parameter" or entries[name].notna().any()]
    collapsible = dict.fromkeys((*context_fields, *levels))
    context = {
        name: entries[name].iloc[0]
        for name in collapsible
        if name not in {"unit", "parameter"}
        and name in entries
        and entries[name].nunique(dropna=False) == 1
    }
    levels = [name for name in levels if name not in context]
    extras = [
        name
        for name in (("model_response_format",) if component == "measurement" else ("scale",))
        if name in entries and name not in context
    ]
    if entries.fixed_value.notna().any():
        if "mean" not in columns:
            columns = ["mean", *columns]
            entries["mean"] = entries.fixed_value
        else:
            entries["mean"] = entries["mean"].fillna(entries.fixed_value)
    result = entries[[*levels, *columns, *extras]].copy()
    return _coefficient_result(result, levels, context, formatting)


def _value_text(value, *, details=False):
    """Keep uniform constants short; expand heterogeneous arrays only on request."""
    values = np.asarray(value)
    if values.size and np.all(values == values.flat[0]):
        scalar = values.flat[0].item()
        return (
            str(int(scalar)) if isinstance(scalar, float) and scalar.is_integer() else repr(scalar)
        )
    return (
        repr(values.tolist())
        if details or values.size <= 4
        else f"array(shape={values.shape}; use details=True)"
    )


def _prior_text(prior, *, details=False, multiplier=None):
    """Format saved native arguments without restoring a distribution or sampler transform."""
    from pyfcirt.models.priors import describe_prior

    family, arguments = describe_prior(prior)
    parameters = []
    for label, values in arguments:
        value = _value_text(values, details=details)
        if family in {"Normal", "LogNormal"}:
            label = {"loc": "mu", "scale": "sigma", "mean": "mu"}.get(label, label)
        if multiplier and family == "Normal" and not (label == "mu" and np.all(values == 0)):
            value = multiplier if np.all(values == 1) else f"{value} * {multiplier}"
        parameters.append(f"{label}={value}")
    text = family + "(" + ", ".join(parameters) + ")"
    return f"{multiplier} * {text}" if multiplier and family != "Normal" else text


def _model_laws(prepared, fixed, active, *, details):
    """Describe declared laws for the canonical entries supplied by each binding."""
    population_ids = {block.binding_id for block in prepared.population.blocks}
    unit_membership = {}
    for entry, unit in prepared.parameter_units.unit_index.items():
        unit_membership.setdefault(entry, []).append(unit)
    unit_models = prepared.constant_data.unit_model.values
    conditionals = {
        (target.binding_id, target.prior_role): target
        for _, targets in active
        for target in targets
    }
    laws, regressions, known_population = {}, [], {}
    active_regression_blocks = set()
    for plan, targets in active:
        active_regression_blocks.update(block.binding_id for block in plan.blocks)
        if targets[0].kind == "theta":
            continue
        pools = {pool for target in targets for pool in target.pool_index}
        units = {unit for target in targets for unit in target.unit_indices}
        applicable = dict.fromkeys(unit_models[unit] for unit in sorted(units))
        target_names = tuple(dict.fromkeys(target.kind for target in targets))
        scale = ", ".join(
            dict.fromkeys(
                ("log-" if target.conditional["link"] == "log" else "") + target.kind
                for target in targets
            )
        )
        regressions.append((", ".join(target_names), plan.design, applicable, len(pools), scale))
    entries_by_binding, templates = {}, {}
    for layout in prepared.parameters:
        owners = layout.entry_data.binding_id.values
        entries_by_binding.update(pd.RangeIndex(len(owners)).groupby(owners))
        templates[layout.kind] = _fixed_entries(layout, fixed)

    supplied_covariance = "covariance" in fixed
    for block in prepared.blocks:
        if (positions := entries_by_binding.get(block.binding_id)) is None:
            continue
        entries = positions.to_numpy()
        kind, component = block.kind, block.component
        population = block.binding_id in population_ids
        if (
            component == "regression"
            and not population
            and block.binding_id not in active_regression_blocks
        ):
            continue
        supplied = kind in fixed
        values, is_fixed = templates[kind]
        fixed_entries = is_fixed.ravel()[entries]
        if population:
            fully_fixed = fixed_entries.all()
            if fully_fixed and kind != "regression_coefficient":
                known_population[kind] = values
            if (
                kind == "covariance"
                or supplied_covariance
                and not block.fixed_for_scoring
                or kind != "regression_coefficient"
                and fully_fixed
                and not details
            ):
                continue
        model = "Population" if population else "Regression" if component == "regression" else None

        def add(
            selected, law, *, role=None, is_fixed=False, identity=None, block=block, model=model
        ):
            kind = block.kind
            if not len(selected) or kind == "trait_correlation" and is_fixed:
                return
            memberships = [unit_membership.get((kind, entry), ()) for entry in selected]
            units = {unit for members in memberships for unit in members}
            applicable = model or ", ".join(
                dict.fromkeys(unit_models[unit] for unit in sorted(units))
            )
            label = "R" if kind == "trait_correlation" else kind
            if role is not None and block.orientation is not None:
                label += f" ({role})"
            shared = any(len(members) > 1 for members in memberships)
            laws.setdefault((applicable, label, law, identity, is_fixed), []).append(
                (len(selected), units, shared)
            )

        known = entries[fixed_entries]
        if supplied and not details:
            add(known, "supplied parameter values", is_fixed=True)
        else:
            known_values = values.ravel()[known]
            for value in np.unique(known_values):
                add(known[known_values == value], "fixed=" + _value_text(value), is_fixed=True)
        roles = prepared.parameter_layouts[kind].entry_data.prior_role.values[entries]
        for role in dict.fromkeys(roles[~fixed_entries]):
            if not role:
                continue
            conditional = conditionals.get((block.binding_id, role))
            if conditional is not None:
                scheme = conditional.conditional
                law = (
                    f"conditional {scheme['family']}(mu=X @ regression_coefficient, "
                    "sigma=regression_sd"
                )
                law += f", bounds={scheme['bounds']!r}" if scheme["bounds"] is not None else ""
                law += ")" + ("; log scale" if scheme["link"] == "log" else "")
                law += f"; noncentered={conditional.noncentered}" if details else ""
                identity = None
            else:
                prior = block.priors[str(role)]
                multiplier = None
                if block.scale_kind is not None:
                    multiplier = (
                        f"sqrt({block.scale_kind})"
                        if block.scale_power == 0.5
                        else block.scale_kind
                        if block.scale_power == 1
                        else f"{block.scale_kind}**{block.scale_power}"
                    )
                law = _prior_text(prior, details=details, multiplier=multiplier)
                identity = PriorPreparation.key(prior)
                if block.bounds is not None:
                    lower_kind, _, upper = block.bounds
                    law = f"Truncated({law}, lower={lower_kind}, upper={upper})"
            add(
                entries[(roles == role) & ~fixed_entries],
                law + ("; ordered" if block.ordered else ""),
                role=role,
                identity=identity,
            )
    return laws, regressions, known_population


def _population_text(population, known, regression, *, details):
    """Substitute known population constants while leaving estimable quantities symbolic."""
    mean, sd, correlation, covariance = (
        known.get(key)
        for key in ("population_mean", "population_sd", "trait_correlation", "covariance")
    )
    group = "[group]" if population.groups else ""
    mu = _value_text(mean, details=details) if mean is not None else "population_mean" + group
    if regression is not None:
        mu = (
            "" if mean is not None and np.all(mean == 0) else mu + " + "
        ) + "X @ regression_coefficient"
        mu += group if population.specification.regression_by_group else ""
    if covariance is not None:
        cov = _value_text(covariance, details=details)
    elif correlation is not None and sd is not None:
        cov = _value_text(sd[..., :, None] * correlation * sd[..., None, :], details=details)
    elif sd is not None and np.all(sd == 1):
        cov = "R"
    else:
        scale = _value_text(sd, details=details) if sd is not None else "population_sd" + group
        cov = f"diag({scale}) @ R @ diag({scale})"
    return ["Population:", f"  theta ~ MvNormal(mu={mu}, cov={cov})"]


def model_summary(data: Model | PreparedInputs, *, details: bool = False) -> str:
    """Render saved assumptions directly, without coefficient tables or posterior statistics."""

    from pyfcirt.model import Model

    prepared = _prepared_inputs(data)
    constant = prepared.constant_data
    arrays = constant.variables
    unit_ids = arrays["unit_id"].values
    models = list(dict.fromkeys(arrays["unit_model"].values))
    mixed = len(models) > 1
    counts = [
        f"{constant.sizes[d]:,} {label}"
        for d, label in (
            ("person", "people"),
            ("trait", "traits"),
            ("block", "blocks"),
            ("item", "items"),
        )
        if constant.sizes[d]
    ]
    fc_statements = int((arrays["statement_channel"].values == "fc").sum())
    if fc_statements:
        counts.append(
            f"{fc_statements:,} {'FC statements' if constant.sizes['item'] else 'statements'}"
        )
    lines = [("Models: " if mixed else "Model: ") + ", ".join(models), ", ".join(counts)]
    item_rows = arrays["obs_channel"].values == "items"
    pair_rows = (arrays["obs_pair"].values >= 0) & ~item_rows
    event_counts = [
        (int(pair_rows.sum()), "pair comparisons"),
        (int((~pair_rows & ~item_rows).sum()), "block events"),
        (int(item_rows.sum()), "item responses"),
    ]
    lines.append(
        "Observations: " + ("; ".join(f"{n:,} {label}" for n, label in event_counts if n) or "0")
    )
    lines.append("Measurements:")
    groups = _measurement_table(prepared).groupby(
        list(_MEASUREMENT_GROUP), sort=False, dropna=False
    )
    for (channel, response_format, link, size, categories), group in groups:
        noun = "block" if channel == "fc" else "item"
        unit = group.index[0]
        label = response_format if mixed else arrays["unit_response_format"].values[unit]
        description = f"{len(group)} {noun}{'s' if len(group) != 1 else ''}"
        description += f", size={size}" if noun == "block" else ""
        description += f", link={link}" if pd.notna(link) else ""
        description += (
            f", categories={categories}"
            if isinstance(categories, tuple) or pd.notna(categories) and categories > 2
            else ""
        )
        multiplier = group.link_multiplier.iloc[0]
        description += f", link multiplier={multiplier}" if pd.notna(multiplier) else ""
        lines.append("  " + (str(label) + ": " if label else "") + description)
        if details:
            lines.append("    IDs: " + repr(tuple(group.unit)))
    fixed = prepared.fixed_parameters or {}
    posterior_calibrated = isinstance(data, Model) and data._posterior_calibrated
    if posterior_calibrated:
        lines.append("Calibration: posterior approximation")
    active = []
    for plan in prepared.regressions:
        targets = tuple(
            target for target in plan.targets if target.kind == "theta" or target.kind not in fixed
        )
        if targets:
            active.append((plan, targets))
    theta = next((plan for plan, targets in active if targets[0].kind == "theta"), None)
    population = prepared.population
    with PriorPreparation():
        laws, regressions, known = _model_laws(prepared, fixed, active, details=details)
    lines += _population_text(population, known, theta, details=details)
    if population.groups:
        groups = population.groups if details else len(population.groups)
        lines.append(f"  Groups: {groups!r}; reference={population.reference_group!r}")
    if theta is not None:
        lines.append(f"  theta ~ {theta.design.attrs['formula']}")
    for (model, label, law, _, is_fixed), parts in laws.items():
        if model == "Population" and not is_fixed:
            count = sum(count for count, _, _ in parts)
            if label == "R":
                shape = prepared.parameter_layouts["trait_correlation"].shape
                count = " x ".join(map(str, shape))
            lines.append(f"  {label} [{count}] ~ {law}")
    if details:
        lines.append(
            f"  parameterization={population.theta_parameterization}; "
            f"theta_residual_scale={population.specification.theta_residual_scale!r}"
        )
        for plan, _ in active:
            settings = plan.design.attrs
            lines.append(
                "  "
                + "; ".join(
                    f"{key}={settings[key]}"
                    for key in ("center_predictors", "standardize", "coefficient_noncentered")
                )
            )
    for target, design, applicable, count, scale in regressions:
        lines.append(f"Regression: {target} ~ {design.attrs['formula']}")
        names = ", ".join(applicable) + "; " if mixed else ""
        lines.append(f"  {names}{scale}; {count} independent parameter pool(s)")
    sections = {}
    law_counts = Counter((model, label, fixed) for model, label, _, _, fixed in laws)
    for (model, label, law, _, is_fixed), parts in laws.items():
        if model == "Population" and not is_fixed:
            continue
        heading = "Constraints:" if is_fixed else "Priors:"
        scopes = sorted(set().union(*(ids for _, ids, _ in parts)))
        ids = tuple(np.asarray(unit_ids[i]).item() for i in scopes)
        if ids and law_counts[model, label, is_fixed] > 1:
            law += f"; scope={ids!r}" if len(ids) <= 3 else f"; {len(ids)} blocks/items"
        count = sum(count for count, _, _ in parts)
        if is_fixed:
            law += f" [{count} fixed]"
        else:
            label += f" [{count}]"
        output = [f"    {label}{': ' if is_fixed else ' ~ '}{law}"]
        if details:
            if ids:
                output.append(f"      Blocks/items: {ids!r}")
            if any(shared for _, _, shared in parts):
                output.append("      Parameters shared across blocks/items")
        sections.setdefault(heading, {}).setdefault(model, []).extend(output)
    for heading in ("Priors:", "Constraints:"):
        if heading in sections:
            lines.append(
                "Declared priors:" if posterior_calibrated and heading == "Priors:" else heading
            )
            for model, output in sections[heading].items():
                if mixed:
                    lines.append(f"  {model}:")
                lines.extend(output)
    return "\n".join(lines)
