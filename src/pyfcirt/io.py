"""Portable JSON metadata and native NetCDF draws; graphs rebuild explicitly."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ._model import ModelSpec
from .data import FCData, FCDesign
from .model import Model
from .models import MGPPM
from .priors import Prior
from .results import FitResult


def _json_value(value):
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Portable metadata dictionaries require string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"Unsupported portable value {type(value).__name__}; save custom code separately"
    )


def _frame_value(value):
    from .data._validation import is_missing

    if isinstance(value, (list, tuple, np.ndarray)):
        return [_frame_value(item) for item in value]
    return None if is_missing(value) else _json_value(value)


def _pack_frame(frame):
    if frame is None:
        return None
    return {
        "columns": frame.columns.tolist(),
        "rows": [
            [_frame_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
    }


def _unpack_frame(payload):
    if payload is None:
        return None
    frame = pd.DataFrame(payload["rows"], columns=payload["columns"], dtype=object)
    for column, dtype in zip(payload["columns"], payload["dtypes"], strict=True):
        frame[column] = frame[column].astype(dtype)
    return frame


def _pack_data(data):
    design = data.design
    return {
        "design": _pack_frame(design.to_frame()),
        "block_formats": list(design.response_formats.items()),
        "allow_variable_block_size": design.allow_variable_block_size,
        "graded_scale": design.graded_scale,
        "statement_features": _pack_frame(design.statement_features),
        "original_responses": _pack_frame(data.original_responses),
        "input_encoding": data.input_encoding,
        "administration": _pack_frame(data.administration),
    }


def _unpack_data(payload):
    design = FCDesign.from_frame(
        _unpack_frame(payload["design"]),
        block_response_formats=dict(payload["block_formats"]),
        allow_variable_block_size=payload["allow_variable_block_size"],
        graded_scale=payload["graded_scale"],
        statement_features=_unpack_frame(payload["statement_features"]),
    )
    return FCData.from_frames(
        _unpack_frame(payload["original_responses"]),
        design,
        encoding=payload["input_encoding"],
        administration=_unpack_frame(payload["administration"]),
    )


def _netcdf_tree(tree):
    """Encode only object coordinates that NetCDF cannot preserve natively."""
    nodes, encoded = {}, {}
    for node in tree.subtree:
        dataset = node.to_dataset(inherit=False).copy(deep=False)
        for name, coordinate in list(dataset.coords.items()):
            if isinstance(dataset.indexes.get(name), pd.MultiIndex):
                raise ValueError("Reset MultiIndex coordinates before portable NetCDF saving")
            values = coordinate.values
            if values.dtype.kind != "O" or all(isinstance(v, str) for v in values.ravel()):
                continue
            tokens = [
                json.dumps(_json_value(v), ensure_ascii=False, allow_nan=False)
                for v in values.ravel()
            ]
            dataset = dataset.assign_coords(
                {
                    name: xr.DataArray(
                        np.asarray(tokens).reshape(values.shape),
                        dims=coordinate.dims,
                        attrs=coordinate.attrs,
                    )
                }
            )
            encoded.setdefault(node.path, []).append(name)
        nodes[node.path] = dataset
    return xr.DataTree.from_dict(nodes), encoded


def _decode_tree(tree, encoded):
    nodes = {}
    for node in tree.subtree:
        dataset = node.to_dataset(inherit=False)
        for name in encoded.get(node.path, []):
            coordinate = dataset.coords[name]
            values = np.empty(coordinate.size, dtype=object)
            values[:] = [json.loads(value) for value in coordinate.values.ravel()]
            dataset = dataset.assign_coords(
                {
                    name: xr.DataArray(
                        values.reshape(coordinate.shape),
                        dims=coordinate.dims,
                        attrs=coordinate.attrs,
                    )
                }
            )
        nodes[node.path] = dataset
    return xr.DataTree.from_dict(nodes)


def _rebuild(spec, data):
    if spec.model_id != "mgppm" or spec.model_version != "2":
        raise ValueError("Unsupported saved model or statistical model version")
    declarations = spec.settings["priors"]
    family = MGPPM(
        priors={
            name: Prior(value["name"], **value["parameters"])
            for name, value in declarations.items()
            if name != "trait_correlation"
        }
    )
    correlation = declarations["trait_correlation"]
    built = Model(
        data,
        family=family,
        priors={"trait_correlation": Prior(correlation["name"], **correlation["parameters"])},
    ).build()
    expected, actual = spec.to_dict(), built.spec.to_dict()
    # Provenance refers to original defaults, not the explicit restoration call.
    expected["settings"].pop("prior_sources", None)
    actual["settings"].pop("prior_sources", None)
    if actual != expected:
        raise ValueError("Saved specification and reconstructed statistical model do not match")
    return replace(built, spec=spec)


@dataclass(frozen=True)
class SavedFit:
    """Loaded draws/data/specification; no executable model is created on load."""

    idata: xr.DataTree
    data: FCData
    spec: ModelSpec
    sampling: dict

    def rebuild(self) -> FitResult:
        """Explicitly reconstruct the declared PyMC graph without sampling."""
        return FitResult(
            idata=self.idata, built=_rebuild(self.spec, self.data), sampling=self.sampling
        )


def _graph_signature(model):
    return {
        name: [
            (variable.name, type(variable.owner.op).__name__ if variable.owner else "Constant")
            for variable in getattr(model, name)
        ]
        for name in ["free_RVs", "observed_RVs", "deterministics", "potentials"]
    }


def _same_declarative_graph(expected, actual):
    """Compare declared graphs, including density, derived values and shared sizes."""
    from pytensor.compile.sharedvalue import SharedVariable
    from pytensor.graph.basic import equal_computations
    from pytensor.graph.traversal import graph_inputs

    if _graph_signature(expected) != _graph_signature(actual) or expected.coords != actual.coords:
        return False
    if [v.name for v in expected.value_vars] != [v.name for v in actual.value_vars]:
        return False
    left = [expected.logp(), *expected.replace_rvs_by_values(expected.deterministics)]
    right = [actual.logp(), *actual.replace_rvs_by_values(actual.deterministics)]
    shared = []
    for outputs in [left, right]:
        variables = [v for v in graph_inputs(outputs) if isinstance(v, SharedVariable)]
        named = {v.name: v for v in variables}
        if None in named or len(named) != len(variables):
            return False
        shared.append(named)
    if shared[0].keys() != shared[1].keys():
        return False
    for name, variable in shared[0].items():
        other = shared[1][name]
        if variable.dtype != other.dtype or not np.array_equal(
            variable.get_value(), other.get_value()
        ):
            return False
    return equal_computations(
        left,
        right,
        [*expected.value_vars, *shared[0].values()],
        [*actual.value_vars, *[shared[1][name] for name in shared[0]]],
    )


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(result, path, *, include_log_likelihood: bool = False) -> Path:
    """Write a new analysis directory without overwriting an existing result.

    Only supported declarative models can be reconstructed. Arbitrary native
    graph edits and executable callbacks are not a portable configuration.
    """
    from .scoring import ScoreResult

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Result directory already exists: {path}")
    if importlib.util.find_spec("h5netcdf") is None:
        raise ImportError("Install pyfcirt[io] to save or load native NetCDF results")
    if isinstance(result, FitResult):
        canonical = _rebuild(result.spec, result.built.data)
        if not _same_declarative_graph(canonical.model, result.built.model):
            raise ValueError(
                "Modified native graphs require separately saved custom construction code"
            )
        kind = "fit"
        tree = result.idata.copy(deep=False)
        if include_log_likelihood:
            tree["log_likelihood"] = result.log_likelihood(progressbar=False).to_dataset()
        metadata = {
            "data": _pack_data(result.built.data),
            "spec": result.spec.to_dict(),
            "sampling": result.sampling,
        }
    elif isinstance(result, ScoreResult):
        if include_log_likelihood:
            raise ValueError("include_log_likelihood applies to calibration FitResult only")
        kind, tree = "score", result.idata
        metadata = {
            "score_format_version": 2,
            "data": _pack_data(result.data),
            "settings": result.settings,
            "sampling": result.sampling,
        }
    else:
        raise TypeError("save requires FitResult or ScoreResult")
    portable, encoded = _netcdf_tree(tree)
    metadata.update(
        {"coordinate_codec": encoded, "pyfcirt_version": importlib.metadata.version("pyfcirt")}
    )
    text = json.dumps(_json_value(metadata), ensure_ascii=False, indent=2, allow_nan=False)
    path.mkdir(parents=True, exist_ok=False)
    (path / "metadata.json").write_text(text, encoding="utf8")
    portable.to_netcdf(path / "posterior.nc", engine="h5netcdf")
    manifest = {
        "format": "pyfcirt",
        "format_version": 1,
        "kind": kind,
        "sha256": {name: _sha256(path / name) for name in ["metadata.json", "posterior.nc"]},
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return path


def load(path):
    """Read portable data and draws; call SavedFit.rebuild for graph operations."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "pyfcirt"
        or type(manifest.get("format_version")) is not int
        or manifest["format_version"] != 1
    ):
        raise ValueError("Unsupported PyFCIRT result format")
    if manifest.get("kind") not in {"fit", "score"}:
        raise ValueError("Unknown saved result kind")
    if set(manifest["sha256"]) != {"metadata.json", "posterior.nc"}:
        raise ValueError("Invalid result file manifest")
    for name in ["metadata.json", "posterior.nc"]:
        if _sha256(path / name) != manifest["sha256"][name]:
            raise ValueError(f"Checksum mismatch for {name}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf8"))
    if manifest["kind"] == "score" and metadata.get("score_format_version") != 2:
        raise ValueError(
            "Unsupported scoring bundle: only current fixed-calibration results are supported"
        )
    data = _unpack_data(metadata["data"])
    if importlib.util.find_spec("h5netcdf") is None:
        raise ImportError("Install pyfcirt[io] to save or load native NetCDF results")
    with xr.open_datatree(path / "posterior.nc", engine="h5netcdf") as source:
        tree = _decode_tree(source.load(), metadata["coordinate_codec"])
    if manifest["kind"] == "fit":
        return SavedFit(
            idata=tree, data=data, spec=ModelSpec(**metadata["spec"]), sampling=metadata["sampling"]
        )
    if manifest["kind"] == "score":
        from .scoring import ScoreResult

        return ScoreResult(
            idata=tree, data=data, settings=metadata["settings"], sampling=metadata["sampling"]
        )
    raise ValueError("Unknown saved result kind")
