"""Runtime Formulae encoders, design matrices and training transformations."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, prod
from numbers import Real

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from jax import devices, enable_x64

from pyfcirt.common.validation import align_rows, require_frame
from pyfcirt.covariates.specification import Regression, _formula_columns


def _feature_rows(data, features, positions, identities, context):
    require_frame(data, features, f"{context}, IDs {identities[:10].tolist()!r}")
    return data[list(features)].iloc[positions].reset_index(drop=True)


def _feature_error(context, name, keys, invalid, message):
    affected = [key for key, bad in zip(keys, invalid, strict=True) if bad][:10]
    raise ValueError(f"{context}, column {name!r}, IDs {affected!r}: {message}")


def _category_values(values, context, name, keys):
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif isinstance(value, str):
            result.append(str(value))
        elif isinstance(value, (int, np.integer)):
            result.append(int(value))
        else:
            _feature_error(
                context, name, keys, [True] * len(keys), "requires str/int/bool categories"
            )
    if len({type(value) for value in result}) != 1:
        _feature_error(context, name, keys, [True] * len(keys), "mixes category types")
    return result


def _numeric(values, context, name, keys):
    if (
        not pd.api.types.is_numeric_dtype(values)
        or pd.api.types.is_bool_dtype(values)
        or pd.api.types.is_complex_dtype(values)
    ):
        _feature_error(context, name, keys, [True] * len(keys), "requires real numeric values")
    # Formulae consumes host tables; frozen transforms must not inherit the model's device.
    array = jnp.asarray(
        values.to_numpy(dtype="float64"), dtype=jnp.float64, device=devices("cpu")[0]
    )
    invalid = ~jnp.isfinite(array)
    if bool(invalid.any()):
        _feature_error(context, name, keys, invalid.tolist(), "contains nonfinite values")
    return array


def _rescale(values, record):
    return np.asarray((values - record["center"]) / record["scale"])


def _training_features(frame, keys, reference, spec, context):
    records, prior_scales = [], {}
    for name in frame:
        values = frame[name]
        if values.isna().any():
            _feature_error(context, name, keys, values.isna(), "contains missing values")
        numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        if numeric and not isinstance(values.dtype, pd.CategoricalDtype):
            array = _numeric(values, context, name, keys)
            selected = array[reference]
            deviation = float(selected.std(ddof=0))
            if deviation == 0 or not bool(jnp.isfinite(deviation)):
                _feature_error(
                    context, name, keys, reference.tolist(), "has zero reference variation"
                )
            record = dict(
                name=name,
                kind="numeric",
                dtype="float64",
                levels=[],
                center=float(selected.mean()) if spec.center_predictors else 0.0,
                scale=deviation if spec.standardize_predictors else 1.0,
            )
            prior_scales[name] = 1.0 if spec.standardize_predictors else deviation
            frame[name] = _rescale(array, record)
        else:
            observed = _category_values(values, context, name, keys)
            if isinstance(values.dtype, pd.CategoricalDtype):
                levels = [value for value in values.cat.categories if value in observed]
                levels = _category_values(levels, context, name, keys)
            else:
                levels = list(dict.fromkeys(observed))
            dtype = {str: "str", int: "int", bool: "bool"}[type(levels[0])]
            record = dict(
                name=name, kind="category", dtype=dtype, levels=levels, center=0.0, scale=1.0
            )
            frame[name] = pd.Categorical(observed, categories=levels, ordered=True)
        records.append(record)
    return records, prior_scales


def _transformed(frame, records, keys, context):
    result = pd.DataFrame(index=frame.index)
    for record in records:
        name = record["name"]
        values = frame[name]
        if values.isna().any():
            _feature_error(context, name, keys, values.isna(), "contains missing values")
        if record["kind"] == "numeric":
            array = _numeric(values, context, name, keys)
            result[name] = _rescale(array, record)
        else:
            observed = _category_values(values, context, name, keys)
            levels = record["levels"]
            invalid = [
                not any(type(value) is type(level) and value == level for level in levels)
                for value in observed
            ]
            if any(invalid):
                _feature_error(context, name, keys, invalid, "contains unknown training categories")
            result[name] = pd.Categorical(observed, categories=levels, ordered=True)
    return result


def _compile(formula, frame, include_intercept):
    try:
        import formulae
    except ModuleNotFoundError as error:
        if error.name != "formulae":
            raise
        raise ImportError("Regression formulas require pyfcirt[covariates]") from error
    common = formulae.design_matrices(formula, frame, na_action="error").common
    matrix = pd.DataFrame(index=frame.index) if common is None else common.as_dataframe()
    if not include_intercept:
        matrix = matrix.drop(columns="Intercept", errors="ignore")
    return common, matrix, formulae.__version__


def _finite_matrix(matrix, keys, context):
    with enable_x64():
        invalid = ~jnp.isfinite(jnp.asarray(matrix.to_numpy(dtype="float64")))
    for index, name in enumerate(matrix.columns):
        if bool(invalid[:, index].any()):
            _feature_error(
                context, name, keys, invalid[:, index].tolist(), "has nonfinite design values"
            )


def fit_design(
    spec: Regression,
    keys: pd.DataFrame,
    *,
    reference_rows: Sequence[bool] | None = None,
    include_intercept: bool = True,
) -> xr.Dataset:
    """Align entity rows and freeze their Formulae coding and training transforms."""
    if not isinstance(spec, Regression):
        raise TypeError("fit_design requires a Regression declaration")
    if not isinstance(include_intercept, bool):
        raise TypeError("include_intercept must be a bool")
    formula = spec.formula.strip()
    columns = (spec.id,) if isinstance(spec.id, str) else spec.id
    features = _formula_columns(formula)
    if set(features).intersection(columns):
        raise ValueError("Regression formula cannot include its ID columns")
    context = f"Regression {formula!r}"
    positions, identities = align_rows(spec.data, keys, columns, context)
    frame = _feature_rows(spec.data, features, positions, identities, context)
    if reference_rows is None:
        reference = np.ones(len(frame), dtype=bool)
    else:
        reference = np.asarray(reference_rows)
        if reference.dtype.kind != "b":
            raise TypeError("reference_rows must contain bool values")
        if reference.shape != (len(frame),) or not reference.any():
            raise ValueError("reference_rows must match keys and select at least one row")
    with enable_x64():
        records, prior_scales = _training_features(frame, identities, reference, spec, context)
    common, matrix, version = _compile(formula, frame, include_intercept)
    _finite_matrix(matrix, identities, context)
    terms = list(matrix.columns)
    if not terms:
        raise ValueError(f"{context} has no design columns; an empty regression is unnecessary")
    if not include_intercept and "Intercept" in spec.priors:
        raise ValueError("A theta regression cannot declare an Intercept prior")
    if unknown := spec.priors.keys() - {*terms, "common"}:
        raise ValueError(f"{context} has priors for unknown design columns: {sorted(unknown)!r}")
    divisors = {}
    for term in common.terms.values():
        divisor = (
            1.0
            if term.kind == "intercept"
            else float(
                prod(prior_scales[part.name] for part in term.components if part.kind == "numeric")
            )
        )
        if not isfinite(divisor) or divisor <= 0:
            raise ValueError(
                f"{context}, term {term.name!r}: invalid prior scale; adjust feature units"
            )
        divisors.update({label: divisor for label in term.labels if label in terms})
    # Native remaining-SD laws are recorded only in resolved_priors by the binder.
    residual_sd = (
        float(spec.parameter_residual_sd) if isinstance(spec.parameter_residual_sd, Real) else None
    )
    result = xr.Dataset(
        {"X": (("row", "term"), matrix.to_numpy(dtype="float64"))},
        coords=dict(row=np.arange(len(frame)), term=terms),
        attrs=dict(
            formula=formula,
            id_columns=list(columns),
            feature_records=records,
            center_predictors=spec.center_predictors,
            standardize=spec.standardize_predictors,
            include_intercept=include_intercept,
            prior_scale_divisors=divisors,
            resolved_priors={},
            residual_sd=residual_sd,
            noncentered=spec.deviation_noncentered,
            coefficient_noncentered=spec.coefficient_noncentered,
            compiler_version=version,
            reference_rows="all" if reference_rows is None else reference.tolist(),
            reference_group=None,
            encoder=common,
        ),
    )
    for index in range(len(columns)):
        values = identities.get_level_values(index).to_numpy(dtype=object)
        result[f"key_{index}"] = ("row", values)
    return result


def apply_design(
    training: xr.Dataset, data: pd.DataFrame, positions: np.ndarray, identities: pd.Index
) -> xr.DataArray:
    """Apply a frozen encoder to person or entity rows already matched by typed ID."""
    if not isinstance(training, xr.Dataset):
        raise TypeError("apply_design requires a training Dataset")
    attrs = training.attrs
    records = attrs["feature_records"]
    context = f"Regression {attrs['formula']!r}"
    frame = _feature_rows(data, [r["name"] for r in records], positions, identities, context)
    with enable_x64():
        transformed = _transformed(frame, records, identities, context)
    common = attrs["encoder"]
    terms = training.term.values.tolist()
    matrix = common.evaluate_new_data(transformed).as_dataframe()
    if not attrs["include_intercept"]:
        matrix = matrix.drop(columns="Intercept", errors="ignore")
    if list(matrix.columns) != terms:
        raise ValueError(f"{context}: new-data design columns differ from the frozen encoding")
    _finite_matrix(matrix, identities, context)
    return xr.DataArray(
        matrix.to_numpy(dtype="float64"),
        dims=("row", "term"),
        coords=dict(row=np.arange(len(frame)), term=terms),
        name="X",
    )
