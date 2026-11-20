"""Small input primitives shared by data preparation and covariate encoding."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd


def identifier(value, context: str) -> str | int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, Integral)):
        raise ValueError(f"{context} must be a nonempty string or integer; got {value!r}")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{context} must not be empty")
        return str(value)
    return int(value)


def identifiers(values, context: str) -> list[str] | list[int]:
    """Validate one entity's labels without casting across integer/string identities."""
    result = [identifier(value, context) for value in values]
    if len({type(value) for value in result}) > 1:
        raise ValueError(f"{context} mixes integer and string IDs; use one type consistently")
    return result


def require_frame(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context} must be a pandas DataFrame")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing columns: {sorted(missing)!r}")


def _identity_index(frame, columns, context):
    require_frame(frame, columns, context)
    if not frame.columns.is_unique:
        raise ValueError(f"{context} has duplicate column names")
    labels = [identifiers(frame[column], f"{context}.{column}") for column in columns]
    index = (
        pd.Index(labels[0], dtype=object)
        if len(columns) == 1
        else pd.MultiIndex.from_arrays(labels)
    )
    if not index.is_unique:
        duplicates = index[index.duplicated()][:10].tolist()
        raise ValueError(f"{context} has duplicate IDs: {duplicates!r}")
    return index


def align_rows(data, keys, columns, context):
    """Match typed entity keys once and return source positions in requested order."""
    wanted = _identity_index(keys, columns, f"{context} keys")
    if not len(wanted):
        raise ValueError(f"{context} keys must contain at least one entity")
    positions = _identity_index(data, columns, f"{context} data").get_indexer(wanted)
    if (missing := positions < 0).any():
        raise ValueError(
            f"{context} is missing required IDs in {columns!r}: {wanted[missing][:10].tolist()!r}"
        )
    return positions, wanted


def has_labelled_axes(data, axes: Sequence[str]) -> bool:
    """Check that each named axis has nonempty, unique, explicit labels."""
    return all(
        data.sizes.get(axis, 0) > 0
        and axis in data.coords
        and data[axis].dims == (axis,)
        and data[axis].to_index().is_unique
        for axis in axes
    )


def fixed_values_match(values, template, positions=None):
    """Compare supplied values at their stored precision, with leading sample axes."""
    if positions is not None:
        leading = values.shape[: values.ndim - template.ndim]
        values = values.reshape((*leading, -1))[..., positions]
        template = template.reshape(-1)[positions]
    actual, expected = np.asarray(values), np.asarray(template)
    if actual.dtype.kind == "f":
        expected = expected.astype(actual.dtype)
        return np.allclose(actual, expected, rtol=0, atol=8 * np.finfo(actual.dtype).eps)
    return np.array_equal(actual, np.broadcast_to(expected, actual.shape))
