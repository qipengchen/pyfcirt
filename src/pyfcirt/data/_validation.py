"""Validation shared by design and response containers."""

from __future__ import annotations

import json
from copy import deepcopy
from numbers import Integral, Real

import numpy as np
import pandas as pd

FORMATS = frozenset({"rank", "pick", "mole", "graded"})
STATUSES = frozenset({"answered", "missing", "not_administered"})


def copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy nested object cells as well as the DataFrame's arrays."""
    result = frame.copy(deep=True)
    for column in result:
        if pd.api.types.is_object_dtype(result[column].dtype):
            result[column] = pd.Series(
                [deepcopy(value) for value in result[column]], index=result.index, dtype=object
            )
    return result


def require_frame(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate column names")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def is_missing(value: object) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


def identifier(value: object, context: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError(f"{context} must be a nonempty string or integer; got {value!r}")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{context} must not be empty")
        return value
    return int(value)


def integer(value: object, context: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{context} must be an integer; got {value!r}")
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"{context} must be a finite integer; got {value!r}")
    return int(value)


def normalize_response(
    raw: object,
    *,
    size: int,
    task: str,
    categories: tuple[str | int, ...],
    context: str,
    status: str,
) -> object:
    """Normalize one declared event without imputing structural missing ranks."""
    if is_missing(raw):
        if status == "answered":
            raise ValueError(f"{context}: missing response requires an explicit missing status")
        return None if task == "graded" else [None] * size
    if task == "graded":
        if status != "answered":
            raise ValueError(f"{context}: {status} cannot contain a graded response")
        # A numeric column with genuine missing values can promote integer
        # categories to floats. Accept only exact integer-valued numbers.
        if isinstance(raw, Real) and not isinstance(raw, Integral):
            raw = integer(raw, f"{context}: graded category")
        value = identifier(raw, f"{context}: graded category")
        if value not in categories:
            raise ValueError(f"{context}: unknown graded category {value!r}")
        return value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: ranking strings must be JSON arrays") from error
    if isinstance(raw, np.ndarray):
        if raw.ndim != 1:
            raise ValueError(f"{context}: ranking arrays must be one-dimensional")
        raw = raw.tolist()
    if not isinstance(raw, (tuple, list)) or len(raw) != size:
        raise ValueError(f"{context}: expected a preference-order list of length {size}")
    order = [None if is_missing(v) else integer(v, context) for v in raw]
    known = [v for v in order if v is not None]
    if not known:
        if status == "answered":
            raise ValueError(f"{context}: all-NA response requires an explicit missing status")
        return order
    if status != "answered":
        raise ValueError(f"{context}: {status} cannot contain known choices")
    if len(set(known)) != len(known) or any(v < 1 or v > size for v in known):
        raise ValueError(f"{context}: choices must be distinct positions in 1..{size}")
    expected_slots = {
        "rank": set(range(size)),
        "pick": {0},
        "mole": {0, size - 1},
    }[task]
    observed_slots = {i for i, value in enumerate(order) if value is not None}
    if observed_slots != expected_slots:
        raise ValueError(f"{context}: NA positions do not match the declared {task} task")
    return order
