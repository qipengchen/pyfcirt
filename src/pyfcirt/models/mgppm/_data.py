"""Explicit MGPPM array preparation, retaining original label and event mappings."""

from dataclasses import dataclass

import numpy as np

from ...data import FCData
from ...data._encoding import compatible_patterns, ranking_events
from ...data._indexing import label_index, observation_id


@dataclass(frozen=True)
class MGPPMGroup:
    """One compatible size/task group; one array row per effective person/block."""

    name: str
    task: str
    block_ids: tuple[str | int, ...]
    statement_index: np.ndarray
    person_index: np.ndarray
    block_index: np.ndarray
    response: np.ndarray
    obs_ids: tuple[str, ...]
    events: tuple
    scores: np.ndarray
    membership: np.ndarray


@dataclass(frozen=True)
class MGPPMInputs:
    """Shared coordinates and group-specific arrays, not a second input API."""

    person_ids: tuple[str | int, ...]
    block_ids: tuple[str | int, ...]
    statement_ids: tuple[str | int, ...]
    trait_ids: tuple[str | int, ...]
    statement_trait: np.ndarray
    keying: np.ndarray
    groups: tuple[MGPPMGroup, ...]


def prepare_data(data: FCData) -> MGPPMInputs:
    """Encode currently verified MGPPM sizes and tutorial five-category pairs.

    No values are padded to the largest block. Structural NA is represented by
    event compatibility, never by an observed array containing missing numbers.
    """
    design = data.design
    frame = design.to_frame()
    statements = design.statements
    persons, person_lookup = label_index(data.person_ids)
    statement_ids, statement_lookup = label_index(statements["statement_id"])
    traits, trait_lookup = label_index(statements["trait"])
    block_ids, _ = label_index(frame["block_id"])
    observations = data.to_frame(answered_only=True)
    block_groups = {}
    for block, size in design.block_sizes.items():
        task = design.response_formats[block]
        if size not in {2, 3, 4}:
            raise ValueError(f"Block {block!r}: MGPPM currently supports sizes 2, 3 and 4")
        block_groups.setdefault((task, size), []).append(block)
    groups = []
    for (task, size), blocks in block_groups.items():
        local_blocks = {block: index for index, block in enumerate(blocks)}
        members = [frame.loc[frame["block_id"] == block] for block in blocks]
        statement_index = np.array(
            [[statement_lookup[s] for s in rows["statement_id"]] for rows in members], dtype=int
        )
        if task == "graded":
            scale = design.graded_scale
            events = scale["categories"]
            if len(events) != 5:
                raise ValueError("MGPPM currently requires the tutorial five-category graded scale")
            scores = np.array([[0, 2], [0, 1], [0, 0], [1, 0], [2, 0]], dtype=float)
            if scale["higher_prefers"] == "position_2":
                scores = scores[::-1].copy()
            membership = np.eye(5, dtype=bool)
        else:
            patterns = np.asarray(ranking_events(size, "rank"), dtype=int)
            # argsort maps each statement position to its zero-based rank.
            scores = (size - 1 - np.argsort(patterns, axis=-1)).astype(float)
            events = ranking_events(size, task)
            membership = compatible_patterns(patterns, events)
        event_lookup = {event: index for index, event in enumerate(events)}
        selected = observations.loc[observations["block_id"].isin(blocks)]
        response = [event_lookup[r if task == "graded" else tuple(r)] for r in selected["response"]]
        groups.append(
            MGPPMGroup(
                name=f"mgppm_{task}_k{size}",
                task=task,
                block_ids=tuple(blocks),
                statement_index=statement_index,
                person_index=np.array([person_lookup[p] for p in selected["person_id"]], dtype=int),
                block_index=np.array([local_blocks[b] for b in selected["block_id"]], dtype=int),
                response=np.asarray(response, dtype=int),
                obs_ids=tuple(
                    observation_id(p, b)
                    for p, b in zip(selected["person_id"], selected["block_id"], strict=True)
                ),
                events=events,
                scores=scores,
                membership=membership,
            )
        )
    return MGPPMInputs(
        person_ids=persons,
        block_ids=block_ids,
        statement_ids=statement_ids,
        trait_ids=traits,
        statement_trait=np.array([trait_lookup[t] for t in statements["trait"]], dtype=int),
        keying=statements["keying"].to_numpy(dtype=int, copy=True),
        groups=tuple(groups),
    )
