"""Small label/index utilities shared by explicit model adapters."""

import json
from collections.abc import Iterable

from ._validation import identifier


def label_index(values: Iterable[str | int]) -> tuple[tuple[str | int, ...], dict[str | int, int]]:
    """Retain first-seen labels and map them to contiguous integer positions."""
    labels = tuple(dict.fromkeys(values))
    return labels, {label: index for index, label in enumerate(labels)}


def observation_id(person: str | int, block: str | int) -> str:
    """Encode a typed person/block pair without delimiter or string/int collisions."""
    return json.dumps(
        [identifier(person, "person_id"), identifier(block, "block_id")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
