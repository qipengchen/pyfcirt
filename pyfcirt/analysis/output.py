"""Read labelled samples and publish results against the prepared coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import xarray as xr

from pyfcirt.common.validation import fixed_values_match, has_labelled_axes, identifiers
from pyfcirt.data.prepare import _align_ability, _align_point_axes, _validate_parameter_support

if TYPE_CHECKING:
    from pyfcirt.model import Model

POSTERIOR_DEPENDENT_GROUPS = (
    "posterior_predictive",
    "response_probabilities",
    "response_parameters",
    "log_likelihood",
    "posterior_predictive_checks",
    "discrepancy",
    "information",
)


def prepare_output(prepared, dataset, name, *, attached=False):
    """Validate result labels and attach their prepared identities."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"{name}: output must be an xarray.Dataset")
    positions = {}
    for dim in dataset.dims:
        if dim not in prepared.coords:
            continue
        message = (
            f"{dim}: sample coordinates differ from the current model"
            if attached
            else f"{name}: output coordinate {dim!r} differs from this analysis"
        )
        if dim not in dataset.coords:
            raise ValueError(message)
        if attached:
            identifiers(dataset[dim].values, f"{name}.{dim}")
        subset = (name == "log_likelihood" and dim == "obs") or (
            name in {"posterior_predictive_checks", "discrepancy"}
            and dim in {"person", "block", "item", "pair", "unit", "obs"}
        )
        if subset:
            labels = dataset.get_index(dim)
            expected = prepared.constant_data.get_index(dim)
            matches = expected.equals(labels)
            if not matches:
                positions[dim] = expected.get_indexer(labels)
                matches = labels.is_unique and np.all(positions[dim] >= 0)
        else:
            matches = np.array_equal(dataset[dim], prepared.coords[dim])
        if not matches:
            raise ValueError(message)
    return prepared._attach_coordinates(dataset, positions=positions, require=attached)


def posterior_samples(idata, *, mcmc=False):
    """Read the posterior with its original, unique chain and draw labels."""
    if not isinstance(idata, xr.DataTree):
        raise TypeError("idata must be a DataTree")
    if "posterior" not in idata:
        raise ValueError("This operation requires standard posterior samples")
    if mcmc and idata.attrs.get("inference_method") != "mcmc":
        raise NotImplementedError("This predictive operation currently supports MCMC samples")
    posterior = idata["posterior"].to_dataset()
    if not has_labelled_axes(posterior, ("chain", "draw")):
        raise ValueError("posterior requires nonempty unique chain/draw sample labels")
    return posterior


def parameter_point(model: Model, *, draw=None, include_person: bool = True) -> xr.Dataset:
    """Read canonical final means or one sample using the prepared point policy."""
    if not isinstance(include_person, bool):
        raise TypeError("include_person must be boolean")
    return read_parameter_point(model, draw=draw, include_person=include_person)


def read_parameter_point(
    model, *, posterior=None, names=None, draw=None, include_person=True, theta=None
):
    """Project the requested final variables from fixed values and labelled draws."""
    if posterior is None:
        posterior = posterior_samples(model.idata)
    prepared = model.prepared
    if names is None:
        layouts = {
            name: layout
            for name, layout in prepared.point_layouts.items()
            if include_person or layout.fixed_for_scoring
        }
        if include_person and theta is None:
            layouts["theta"] = None
    else:
        layouts = {name: prepared.parameter_layouts[name] for name in names}
    if draw is not None:
        if not isinstance(draw, tuple) or len(draw) != 2:
            raise ValueError("draw must be a (chain_label, draw_label) pair")
        posterior = posterior.sel(chain=draw[0], draw=draw[1], drop=True)
        if {"chain", "draw"}.intersection(posterior.sizes):
            raise ValueError("draw must select one posterior state")

    shapes = {
        name: layout.dims if layout else ("person", "trait") for name, layout in layouts.items()
    }
    result = {}
    sources = ((model._fixed_reference, ()), (posterior, ("chain", "draw") if draw is None else ()))
    for source, sample_dims in sources:
        if source is None:
            continue
        selected = {
            name: dims for name, dims in shapes.items() if name in source and name not in result
        }
        _align_point_axes(
            prepared.constant_data, source, selected, sample_dims=sample_dims, exact=True
        )
        for name in selected:
            value, layout = source[name], layouts[name]
            fixed_entries = layout is not None and len(layout.fixed_flat)
            if value.dtype.kind not in "iuf":
                raise ValueError(f"{name}: final values must be finite real numbers")
            if fixed_entries and not fixed_values_match(
                value.data, layout.fixed, layout.fixed_flat
            ):
                raise ValueError(f"{name}: final values conflict with fixed entries")
            if sample_dims:
                value = value.mean(sample_dims, skipna=False)
                if fixed_entries:
                    value = value.where(np.isnan(layout.fixed), layout.fixed.astype(value.dtype))
            if not np.isfinite(value.values).all():
                raise ValueError(f"{name}: final values must be finite real numbers")
            result[name] = value

    missing = layouts.keys() - result.keys()
    if unresolved := sorted(
        name for name in missing if layouts[name] is None or len(layouts[name].free_flat)
    ):
        raise ValueError(f"Missing required final posterior nodes: {unresolved}")
    for name in missing:
        dims = shapes[name]
        result[name] = xr.DataArray(
            layouts[name].fixed, dims=dims, coords={dim: prepared.coords[dim] for dim in dims}
        )
    result = {name: result[name] for name in layouts}
    if theta is not None:
        result["theta"] = _align_ability(prepared.constant_data, theta)
    method = model.idata.attrs.get("inference_method", posterior.attrs.get("inference_method"))
    point = xr.Dataset(
        result,
        attrs={
            "inference_method": method or "unspecified",
            "point": "final-node mean" if draw is None else "joint sample",
            "approximate": int(method in {"advi", "fullrank_advi", "map_laplace"}),
        },
    )
    return prepared._attach_coordinates(point, positions={})


def validate_attached_samples(prepared, idata, *, fixed=None):
    """Validate and annotate externally supplied posterior arrays once."""
    posterior = posterior_samples(idata)
    layouts = prepared.parameter_layouts
    dimensions = {name: layout.dims for name, layout in layouts.items()}
    dimensions["theta"] = ("person", "trait")
    for name, dims in dimensions.items():
        if (
            dims == (f"{name}_id",)
            and dims[0] in posterior.dims
            and name not in posterior.data_vars
        ):
            raise ValueError(f"{dims[0]} requires its canonical parameter variable {name!r}")
    for name, value in posterior.data_vars.items():
        canonical = name in dimensions
        if canonical and value.dims != ("chain", "draw", *dimensions[name]):
            raise ValueError(f"{name}: sample dimensions differ from the current model")
        kind = value.dtype.kind
        numeric = "iuf" if canonical else "biufc"
        if kind not in numeric or (kind in "fc" and not np.isfinite(value.data).all()):
            expected = "real" if canonical else "numeric"
            raise ValueError(f"{name}: posterior draws must contain only finite {expected} values")
    posterior = prepare_output(prepared, posterior, "posterior", attached=True)
    for name in posterior.data_vars.keys() & layouts.keys():
        layout = layouts[name]
        if len(layout.fixed_flat) and not fixed_values_match(
            posterior[name].data, layout.fixed, layout.fixed_flat
        ):
            raise ValueError(f"{name}: sample values conflict with fixed entries")
    if fixed is not None:
        for name in sorted(posterior.data_vars.keys() & fixed.data_vars.keys()):
            if not fixed_values_match(posterior[name].data, fixed[name].data):
                raise ValueError(f"{name}: sample values conflict with fixed calibration values")
    _validate_parameter_support(posterior, layouts, fixed=fixed)
    if "sample_stats" in idata:
        stats = idata["sample_stats"].to_dataset()
        for axis in ("chain", "draw"):
            if axis not in stats.coords or not stats[axis].equals(posterior[axis]):
                raise ValueError("sample_stats must use the posterior chain/draw coordinates")
    return posterior


def replace_output_groups(idata, groups, *, remove=(), attrs=None):
    """Build the final tree once, retaining local coordinates and replacing whole subtrees."""
    replacements = {f"/{name.lstrip('/')}": dataset for name, dataset in groups.items()}
    replaced = replacements.keys() | {f"/{name.lstrip('/')}" for name in remove}
    nodes = {}
    for node in idata.subtree:
        path = node.path
        # DataTree visits parents first, so excluded branches need only one lookup.
        if path in replaced or path.rpartition("/")[0] in replaced:
            replaced.add(path)
        else:
            nodes[path] = node.to_dataset(inherit=node is idata)
    if attrs is not None:
        nodes["/"].attrs = attrs
    return xr.DataTree.from_dict(nodes | replacements, name=idata.name)


def prepare_outputs(
    prepared,
    idata: xr.DataTree,
    groups: Mapping[str, xr.Dataset],
    *,
    remove: Sequence[str] = (),
    attrs: Mapping | None = None,
    posterior_batch_verified: bool = False,
    verified_posterior: xr.Dataset | None = None,
    variable_attrs: Mapping | None = None,
) -> xr.DataTree:
    """Prepare the complete next result tree before publication."""
    candidates = {} if verified_posterior is None else {"posterior": verified_posterior}
    candidates.update(
        (name, prepare_output(prepared, dataset, name)) for name, dataset in groups.items()
    )
    if variable_attrs:
        for name, candidate in candidates.items():
            candidates[name] = candidate.assign(
                {
                    variable: candidate[variable].assign_attrs(variable_attrs[variable])
                    for variable in candidate.data_vars.keys() & variable_attrs.keys()
                }
            )
    if "posterior" in candidates:
        attrs = dict(idata.attrs if attrs is None else attrs)
        if posterior_batch_verified:
            attrs["scoring_batch_identity"] = prepared.scoring_batch_identity
        else:
            attrs.pop("scoring_batch_identity", None)
    return replace_output_groups(idata, candidates, remove=remove, attrs=attrs)


def pool_log_likelihood(prepared, values, *, cached=None, positions=None):
    """Pool native batches and retained spans of the canonical observation axis."""
    arrays = []
    start = 0

    def retain(stop):
        if cached is not None and start < stop:
            indices = positions[start:stop]
            if np.all(np.diff(indices) == 1):
                indices = slice(indices[0], indices[-1] + 1)
            arrays.append(cached.isel(obs=indices))

    for batch in prepared.batches:
        if batch.response_name in values:
            retain(batch.obs_slice.start)
            arrays.append(values[batch.response_name].rename({batch.obs_dim: "obs"}))
            start = batch.obs_slice.stop
    size = prepared.constant_data.sizes["obs"]
    retain(size)
    if not arrays:
        return None
    result = (
        arrays[0]
        if len(arrays) == 1
        else xr.concat(arrays, dim="obs", join="exact", coords="minimal", compat="equals")
    )
    result = result.rename("response").transpose("chain", "draw", "obs")
    result.attrs = dict(
        inference_method="mcmc",
        observation_unit="prepared actual events",
        complete_observations=int(result.sizes["obs"] == size),
    )
    return result


def finalize_samples(
    prepared, idata, sampled: xr.DataTree, *, sampling_attrs: Mapping, variable_attrs=None
) -> xr.DataTree:
    """Apply one posterior replacement policy to ordinary and special samplers."""
    outputs = {
        name: child.to_dataset()
        for name, child in sampled.children.items()
        if name not in {"constant_data", "observed_data"}
    }
    if "log_likelihood" in outputs:
        pooled = pool_log_likelihood(prepared, outputs.pop("log_likelihood"))
        if pooled is not None:
            outputs["log_likelihood"] = pooled.to_dataset()
    attrs = {
        key: value
        for key, value in idata.attrs.items()
        if key != "sampling" and not key.startswith("sampling_")
    }
    # Saved predictions use this identity to reject draws from a replaced posterior.
    attrs.update(inference_method="mcmc", posterior_run_id=uuid4().hex, **sampling_attrs)
    return prepare_outputs(
        prepared,
        idata,
        outputs,
        remove=(
            "posterior",
            "sample_stats",
            "warmup_posterior",
            "warmup_sample_stats",
            *POSTERIOR_DEPENDENT_GROUPS,
            "estimates",
            "optimizer_result",
        ),
        attrs=attrs,
        posterior_batch_verified=True,
        variable_attrs=variable_attrs,
    )
