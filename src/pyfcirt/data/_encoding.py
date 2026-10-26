"""Complete-ranking patterns and the events observed by PICK/MOLE tasks."""

from itertools import permutations

import numpy as np


def ranking_events(size: int, task: str) -> tuple[tuple[int | None, ...], ...]:
    """Return one-based preference-order events in a declared task's support."""
    if task == "rank":
        return tuple(permutations(range(1, size + 1)))
    if task == "pick":
        return tuple((first, *([None] * (size - 1))) for first in range(1, size + 1))
    if task == "mole":
        return tuple(
            (first, *([None] * (size - 2)), last)
            for first, last in permutations(range(1, size + 1), 2)
        )
    raise ValueError(f"Not a ranking task: {task!r}")


def compatible_patterns(
    patterns: np.ndarray, events: tuple[tuple[int | None, ...], ...]
) -> np.ndarray:
    """Map each observed event to its compatible complete preference orders."""
    constraints = np.array([[0 if v is None else v for v in event] for event in events])
    return np.all(
        (constraints[:, None, :] == 0) | (constraints[:, None, :] == patterns[None, :, :]), axis=-1
    )
