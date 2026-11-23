"""Runtime Formulae encoders, design matrices and training transformations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, prod
from numbers import Real

import numpy as np
import pandas as pd
import xarray as xr

from pyfcirt.common.validation import align_rows, require_frame
from pyfcirt.covariates.pairwise import group_basis
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


def _category_key(value):
    """Keep integer, boolean and string labels distinct across encoders."""
    if isinstance(value, np.generic):
        value = value.item()
    return type(value), value


def _numeric(values, context, name, keys):
    if (
        not pd.api.types.is_numeric_dtype(values)
        or pd.api.types.is_bool_dtype(values)
        or pd.api.types.is_complex_dtype(values)
    ):
        _feature_error(context, name, keys, [True] * len(keys), "requires real numeric values")
    # Keep Formulae preprocessing at host float64 precision on every backend.
    array = values.to_numpy(dtype="float64")
    invalid = ~np.isfinite(array)
    if invalid.any():
        _feature_error(context, name, keys, invalid, "contains nonfinite values")
    return array


def _rescale(values, record):
    return (values - record["center"]) / record["scale"]


def _training_features(
    frame, keys, reference, spec, context, reference_levels, population_reference
):
    records, prior_scales = [], {}
    group, group_reference = population_reference
    for name in frame:
        values = frame[name]
        if values.isna().any():
            _feature_error(context, name, keys, values.isna(), "contains missing values")
        numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        if numeric and not isinstance(values.dtype, pd.CategoricalDtype):
            if name in reference_levels:
                raise ValueError(f"{context}: {name!r} must be categorical to set a reference")
            array = _numeric(values, context, name, keys)
            selected = array[reference]
            with np.errstate(over="ignore"):
                deviation = float(selected.std(ddof=0))
            if deviation == 0 or not isfinite(deviation):
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
                present = set(observed)
                levels = [value for value in values.cat.categories if value in present]
                levels = _category_values(levels, context, name, keys)
            else:
                levels = list(dict.fromkeys(observed))
            if name == group:
                reference_levels.setdefault(name, group_reference)
            if name in reference_levels:
                level = reference_levels[name]
                try:
                    index = [_category_key(value) for value in levels].index(_category_key(level))
                except ValueError:
                    raise ValueError(
                        f"{context}: unknown reference level {level!r} for {name!r}"
                    ) from None
                levels.insert(0, levels.pop(index))
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
            allowed = {_category_key(level) for level in levels}
            invalid = [_category_key(value) not in allowed for value in observed]
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


_GROUP_FACTOR = "difGroup"


def _crossed_frame(frame, columns, *, levels=None, reference_levels=None):
    """Code the observed joint categories and retain their typed labels."""
    joint = pd.MultiIndex.from_frame(frame[list(columns)])
    if levels is None:
        levels = tuple(joint.unique().tolist())
        if reference_levels:
            required = [
                (i, _category_key(reference_levels[name]))
                for i, name in enumerate(columns)
                if name in reference_levels
            ]
            reference = next(
                (
                    level
                    for level in levels
                    if all(_category_key(level[i]) == value for i, value in required)
                ),
                None,
            )
            if reference is None:
                raise ValueError(f"Unobserved DIF reference combination: {reference_levels!r}")
            levels = (reference, *(level for level in levels if level != reference))
    codes = pd.MultiIndex.from_tuples(levels).get_indexer(joint)
    if (codes < 0).any():
        bad = joint[codes < 0].unique().tolist()[:5]
        raise ValueError(f"Unknown training combinations for {':'.join(columns)}: {bad}")
    return pd.DataFrame(
        {_GROUP_FACTOR: pd.Categorical.from_codes(codes, range(len(levels)), ordered=True)},
        index=frame.index,
    ), levels


def _finite_matrix(matrix, keys, context):
    values = matrix.to_numpy(dtype="float64")
    invalid = ~np.isfinite(values)
    columns = np.flatnonzero(invalid.any(axis=0))
    if len(columns):
        index = columns[0]
        _feature_error(
            context, matrix.columns[index], keys, invalid[:, index], "has nonfinite design values"
        )
    return values


def fit_design(
    spec: Regression,
    keys: pd.DataFrame | None,
    *,
    reference_rows: Sequence[bool] | None = None,
    include_intercept: bool = True,
    aligned_rows: tuple[np.ndarray, pd.Index] | None = None,
    feature_columns: tuple[str, ...] | None = None,
    group_columns: tuple[str, ...] = (),
    reference_levels: Mapping | None = None,
    population_reference: tuple[str | None, str | int | None] = (None, None),
) -> xr.Dataset:
    """Freeze Formulae coding and training transforms in the requested entity order.

    An internal caller can supply rows already matched against this same source
    table and ID columns, or feature columns from the syntax check.
    """
    if not isinstance(spec, Regression):
        raise TypeError("fit_design requires a Regression declaration")
    if not isinstance(include_intercept, bool):
        raise TypeError("include_intercept must be a bool")
    formula = spec.formula.strip()
    columns = (spec.id,) if isinstance(spec.id, str) else spec.id
    features = _formula_columns(formula) if feature_columns is None else feature_columns
    if set(features).intersection(columns):
        raise ValueError("Regression formula cannot include its ID columns")
    context = f"Regression {formula!r}"
    positions, identities = (
        align_rows(spec.data, keys, columns, context) if aligned_rows is None else aligned_rows
    )
    frame = _feature_rows(spec.data, features, positions, identities, context)
    if reference_rows is None:
        reference = np.ones(len(frame), dtype=bool)
    else:
        reference = np.asarray(reference_rows)
        if reference.dtype.kind != "b":
            raise TypeError("reference_rows must contain bool values")
        if reference.shape != (len(frame),) or not reference.any():
            raise ValueError("reference_rows must match keys and select at least one row")
    reference_levels = dict(reference_levels or {})
    records, prior_scales = _training_features(
        frame, identities, reference, spec, context, reference_levels, population_reference
    )
    levels = None
    if group_columns:
        if any(record["kind"] != "category" for record in records):
            raise ValueError("Crossed DIF groups require categorical columns")
        frame, levels = _crossed_frame(frame, group_columns, reference_levels=reference_levels)
    expression = _GROUP_FACTOR if group_columns else formula
    common, matrix, version = _compile(expression, frame, include_intercept)
    values = _finite_matrix(matrix, identities, context)
    terms = list(matrix.columns)
    if not terms:
        raise ValueError(f"{context} has no design columns; an empty regression is unnecessary")
    if not include_intercept and "Intercept" in spec.priors:
        raise ValueError(
            f"{context}: omit the Intercept prior because the base model provides the intercept"
        )
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
        {"X": (("row", "term"), values)},
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
    if group_columns:
        result.attrs.update(group_columns=group_columns, group_levels=levels)
    for index in range(len(columns)):
        values = identities.get_level_values(index).to_numpy(dtype=object)
        result[f"key_{index}"] = ("row", values)
    return result


def _project_groups(values, basis):
    # Formulae omits the reference level, whose row is supplied by the basis.
    return values @ (basis[1:] - basis[0]) + basis[0]


def pairwise_design(training):
    """Represent group differences in orthonormal contrast coordinates."""
    records = training.attrs["feature_records"]
    levels = training.attrs.get("group_levels")
    if levels is None:
        if (
            len(records) != 1
            or records[0]["kind"] != "category"
            or training.attrs["encoder"].terms.keys() - {"Intercept"} != {records[0]["name"]}
        ):
            raise ValueError("PairwiseL1 requires a single categorical group term")
        levels = records[0]["levels"]
    if len(levels) < 2:
        raise ValueError("PairwiseL1 requires at least two groups")
    if training.sizes["term"] != len(levels) - 1:
        raise ValueError("PairwiseL1 requires a formula with reference-level coding")
    basis = group_basis(len(levels))
    basis = basis - basis[0]
    names = [f"contrast[{index}]" for index in range(1, len(levels))]
    return (
        training.assign_coords(term=names)
        .assign(X=(("row", "term"), _project_groups(training.X.values, basis)))
        .assign_attrs(group_basis=basis.tolist(), encoded_terms=training.term.values.tolist())
    )


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
    transformed = _transformed(frame, records, identities, context)
    if columns := attrs.get("group_columns"):
        transformed, _ = _crossed_frame(transformed, columns, levels=attrs["group_levels"])
    common = attrs["encoder"]
    terms = training.term.values.tolist()
    matrix = common.evaluate_new_data(transformed).as_dataframe()
    if not attrs["include_intercept"]:
        matrix = matrix.drop(columns="Intercept", errors="ignore")
    if list(matrix.columns) != attrs.get("encoded_terms", terms):
        raise ValueError(f"{context}: new-data design columns differ from the frozen encoding")
    values = _finite_matrix(matrix, identities, context)
    if "group_basis" in attrs:
        values = _project_groups(values, np.asarray(attrs["group_basis"]))
    return xr.DataArray(
        values,
        dims=("row", "term"),
        coords=dict(row=np.arange(len(frame)), term=terms),
        name="X",
    )
