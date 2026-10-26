"""Labelled design and response containers, independent of a measurement model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pandas as pd

from ._validation import (
    FORMATS,
    STATUSES,
    copy_frame,
    identifier,
    integer,
    is_missing,
    normalize_response,
    require_frame,
)

Label = str | int
DESIGN_COLUMNS = ("block_id", "position", "statement_id", "trait", "keying")
RESPONSE_COLUMNS = ("person_id", "block_id", "response")
RANK_COLUMNS = ("person_id", "block_id", "statement_id", "rank")


def _scale(spec: Mapping[str, Any] | None, needed: bool) -> dict[str, Any] | None:
    if spec is None:
        if needed:
            raise ValueError("graded blocks require an explicit graded_scale")
        return None
    if not isinstance(spec, Mapping):
        raise TypeError("graded_scale must be a mapping")
    unknown = set(spec) - {"categories", "higher_prefers", "labels"}
    if unknown:
        raise ValueError(f"Unknown graded_scale fields: {sorted(unknown)}")
    raw = spec.get("categories")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("graded_scale.categories must contain at least two ordered categories")
    categories = tuple(identifier(v, "graded category") for v in raw)
    if len(set(categories)) != len(categories):
        raise ValueError("graded_scale categories must be unique")
    direction = spec.get("higher_prefers")
    if direction not in {"position_1", "position_2"}:
        raise ValueError("graded_scale.higher_prefers must be position_1 or position_2")
    labels = spec.get("labels")
    if labels is not None:
        if not isinstance(labels, (list, tuple)) or len(labels) != len(categories):
            raise ValueError("graded_scale labels must match the category count")
        if any(not isinstance(v, str) or not v.strip() for v in labels):
            raise ValueError("graded_scale labels must be nonempty strings")
        labels = tuple(labels)
    return {"categories": categories, "higher_prefers": direction, "labels": labels}


class FCDesign:
    """Validated statement membership, trait directions and block response tasks.

    Parameters
    ----------
    frame : pandas.DataFrame
        Columns: block_id, position (one-based), statement_id, trait and keying.
    response_format : str
        Default task: rank, pick, mole or graded.
    block_response_formats : mapping, optional
        Task overrides keyed by existing block IDs.
    allow_variable_block_size : bool
        Explicitly allow blocks of different sizes. Sizes come from membership.
    graded_scale : mapping, optional
        Ordered categories and an explicit higher_prefers direction for pairs.
    statement_features : pandas.DataFrame, optional
        One row per statement_id. Storage does not imply a parameter regression.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        response_format: str = "rank",
        block_response_formats: Mapping[Label, str] | None = None,
        allow_variable_block_size: bool = False,
        graded_scale: Mapping[str, Any] | None = None,
        statement_features: pd.DataFrame | None = None,
    ) -> None:
        require_frame(frame, DESIGN_COLUMNS, "design")
        if frame.empty:
            raise ValueError("design must contain at least one block")
        if response_format not in FORMATS:
            raise ValueError(f"Unknown response_format: {response_format!r}")
        if not isinstance(allow_variable_block_size, bool):
            raise TypeError("allow_variable_block_size must be a bool")
        table = copy_frame(frame).reset_index(drop=True)
        for column in ("block_id", "statement_id", "trait"):
            table[column] = [identifier(v, column) for v in table[column]]
        table["position"] = [integer(v, "position") for v in table["position"]]
        table["keying"] = [integer(v, "keying") for v in table["keying"]]
        if not table["keying"].isin([-1, 1]).all():
            raise ValueError("keying must be -1 or +1")
        if table.duplicated(["block_id", "statement_id"]).any():
            raise ValueError("A statement may appear only once within a block")
        for statement, rows in table.groupby("statement_id", sort=False):
            if rows["trait"].nunique() != 1 or rows["keying"].nunique() != 1:
                raise ValueError(f"Conflicting trait or keying for statement {statement!r}")
            if "text" in rows and rows["text"].nunique(dropna=False) != 1:
                raise ValueError(f"Conflicting text for statement {statement!r}")
        groups = []
        sizes = {}
        for block, rows in table.groupby("block_id", sort=False):
            size = len(rows)
            if size < 2 or sorted(rows["position"]) != list(range(1, size + 1)):
                raise ValueError(f"Block {block!r}: positions must be 1..K with K >= 2")
            groups.append(rows.sort_values("position", kind="stable"))
            sizes[block] = size
        if not allow_variable_block_size and len(set(sizes.values())) != 1:
            raise ValueError("Different block sizes require allow_variable_block_size=True")
        formats = dict.fromkeys(sizes, response_format)
        overrides = dict(block_response_formats or {})
        unknown = set(overrides) - set(sizes)
        if unknown:
            raise ValueError(f"Unknown blocks in block_response_formats: {unknown}")
        for block, task in overrides.items():
            if task not in FORMATS:
                raise ValueError(f"Block {block!r}: unknown response format {task!r}")
            formats[block] = task
        for block, task in formats.items():
            if task == "graded" and sizes[block] != 2:
                raise ValueError(
                    f"Block {block!r}: graded responses require exactly two statements"
                )
        self._graded_scale = _scale(graded_scale, "graded" in formats.values())
        self._frame = pd.concat(groups, ignore_index=True)
        self._block_sizes = sizes
        self._response_formats = formats
        self._allow_variable_block_size = allow_variable_block_size
        self._unused_statement_ids: tuple[Label, ...] = ()
        self._features = None
        if statement_features is not None:
            require_frame(statement_features, ("statement_id",), "statement_features")
            features = copy_frame(statement_features).reset_index(drop=True)
            features["statement_id"] = [
                identifier(v, "feature statement_id") for v in features["statement_id"]
            ]
            if features["statement_id"].duplicated().any():
                raise ValueError("statement_features must have unique statement_id values")
            unknown = set(features["statement_id"]) - set(self._frame["statement_id"])
            if unknown:
                raise ValueError(f"Unknown statement_features IDs: {unknown}")
            self._features = features

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, **kwargs: Any) -> FCDesign:
        """Construct a design from a membership table."""
        return cls(frame, **kwargs)

    @classmethod
    def from_mapping(
        cls,
        *,
        traits: Mapping[Label, Sequence[Label]],
        blocks: Mapping[Label, Sequence[Label]],
        keying: Mapping[Label, int] | None = None,
        **kwargs: Any,
    ) -> FCDesign:
        """Define traits and blocks using statement lists; unspecified keying is +1."""
        assignments = {}
        for trait, statements in traits.items():
            trait = identifier(trait, "trait")
            if isinstance(statements, str) or not isinstance(statements, Sequence):
                raise TypeError(f"Trait {trait!r}: statements must be a sequence")
            for statement in statements:
                statement = identifier(statement, "statement_id")
                if statement in assignments:
                    raise ValueError(f"Statement {statement!r} has multiple trait assignments")
                assignments[statement] = trait
        directions = dict(keying or {})
        if set(directions) - set(assignments):
            raise ValueError("keying contains unknown statement IDs")
        rows = []
        for block, statements in blocks.items():
            block = identifier(block, "block_id")
            if isinstance(statements, str) or not isinstance(statements, Sequence):
                raise TypeError(f"Block {block!r}: statements must be a sequence")
            if len(statements) < 2:
                raise ValueError(f"Block {block!r} needs at least two statements")
            for position, statement in enumerate(statements, start=1):
                statement = identifier(statement, "statement_id")
                if statement not in assignments:
                    raise ValueError(f"Block {block!r}: statement {statement!r} has no trait")
                rows.append(
                    (
                        block,
                        position,
                        statement,
                        assignments[statement],
                        directions.get(statement, 1),
                    )
                )
        design = cls.from_frame(pd.DataFrame(rows, columns=DESIGN_COLUMNS), **kwargs)
        used = set(design._frame["statement_id"])
        design._unused_statement_ids = tuple(s for s in assignments if s not in used)
        return design

    def to_frame(self) -> pd.DataFrame:
        """Return an independent membership table in block/position order."""
        return copy_frame(self._frame)

    @property
    def statements(self) -> pd.DataFrame:
        """Unique statement metadata, excluding membership-specific columns."""
        columns = [c for c in self._frame if c not in {"block_id", "position"}]
        return copy_frame(
            self._frame[columns].drop_duplicates("statement_id").reset_index(drop=True)
        )

    @property
    def block_sizes(self) -> dict[Label, int]:
        return self._block_sizes.copy()

    @property
    def block_size(self) -> int | None:
        sizes = set(self._block_sizes.values())
        return next(iter(sizes)) if len(sizes) == 1 else None

    @property
    def response_formats(self) -> dict[Label, str]:
        return self._response_formats.copy()

    @property
    def allow_variable_block_size(self) -> bool:
        return self._allow_variable_block_size

    @property
    def graded_scale(self) -> dict[str, Any] | None:
        return deepcopy(self._graded_scale)

    @property
    def statement_features(self) -> pd.DataFrame | None:
        return None if self._features is None else copy_frame(self._features)

    def summary(self) -> dict[str, Any]:
        """Describe structure without claiming statistical identification."""
        statements = self.statements
        return {
            "n_blocks": len(self._block_sizes),
            "n_statements": len(statements),
            "n_traits": statements["trait"].nunique(),
            "block_sizes": self.block_sizes,
            "response_formats": self.response_formats,
            "statements_per_trait": statements.groupby("trait", sort=False).size().to_dict(),
            "reused_statement_ids": tuple(
                self._frame.loc[self._frame["statement_id"].duplicated(), "statement_id"].unique()
            ),
            "unused_statement_ids": self._unused_statement_ids,
        }


class FCData:
    """Validated responses combined with a design, retaining original labels.

    Use :meth:`from_frames` with one row per person/block and a response list
    (or a scalar graded category). The explicit statement_rank encoding accepts
    all statement rows of each ranking block, including structural NA rows.
    """

    def __init__(
        self,
        responses: pd.DataFrame,
        design: FCDesign,
        *,
        encoding: str = "selection_order",
        administration: pd.DataFrame | None = None,
    ) -> None:
        if not isinstance(design, FCDesign):
            raise TypeError("design must be an FCDesign")
        if encoding not in {"selection_order", "statement_rank"}:
            raise ValueError(f"Unknown response encoding: {encoding!r}")
        columns = RESPONSE_COLUMNS if encoding == "selection_order" else RANK_COLUMNS
        require_frame(responses, columns, "responses")
        original = copy_frame(responses).reset_index(drop=True)
        for column in ("person_id", "block_id"):
            original[column] = [identifier(v, column) for v in original[column]]
        unknown = set(original["block_id"]) - set(design._block_sizes)
        if unknown:
            raise ValueError(f"Unknown response block IDs: {unknown}")
        self._design = design
        self._encoding = encoding
        self._original_responses = original
        self._administration = self._validate_administration(administration)
        if encoding == "statement_rank":
            table = self._from_rank_frame(original)
        else:
            table = copy_frame(original)
        if table.duplicated(["person_id", "block_id"]).any():
            raise ValueError("Duplicate person/block responses")
        status = {}
        if self._administration is not None:
            for row in self._administration.itertuples(index=False):
                status[(row.person_id, row.block_id)] = row.status
        seen = set()
        normalized = []
        for row in table.itertuples(index=False):
            pair = (row.person_id, row.block_id)
            if self._administration is not None and pair not in status:
                raise ValueError(f"Response {pair!r} is absent from administration")
            state = status.get(pair, "answered")
            task = design._response_formats[row.block_id]
            categories = () if design._graded_scale is None else design._graded_scale["categories"]
            normalized.append(
                normalize_response(
                    row.response,
                    size=design._block_sizes[row.block_id],
                    task=task,
                    categories=categories,
                    context=f"Person {row.person_id!r}, block {row.block_id!r}",
                    status=state,
                )
            )
            seen.add(pair)
        missing_answers = {pair for pair, state in status.items() if state == "answered"} - seen
        if missing_answers:
            raise ValueError(f"Answered administration entries have no response: {missing_answers}")
        table["response"] = pd.Series(normalized, dtype=object)
        self._responses = table
        persons = list(table["person_id"])
        if self._administration is not None:
            persons += list(self._administration["person_id"])
        self._person_ids = tuple(dict.fromkeys(persons))
        self._answered_pairs = frozenset(
            (r.person_id, r.block_id)
            for r in table.itertuples(index=False)
            if status.get((r.person_id, r.block_id), "answered") == "answered"
        )

    @classmethod
    def from_frames(
        cls,
        responses: pd.DataFrame,
        design: FCDesign,
        **kwargs: Any,
    ) -> FCData:
        """Construct data from explicit response/design tables and optional statuses."""
        return cls(responses, design, **kwargs)

    def _validate_administration(self, frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None:
            return None
        require_frame(frame, ("person_id", "block_id", "status"), "administration")
        frame = copy_frame(frame).reset_index(drop=True)
        for column in ("person_id", "block_id"):
            frame[column] = [identifier(v, column) for v in frame[column]]
        if frame.duplicated(["person_id", "block_id"]).any():
            raise ValueError("Duplicate administration person/block entries")
        if set(frame["block_id"]) - set(self._design._block_sizes):
            raise ValueError("administration contains unknown block IDs")
        if not frame["status"].isin(STATUSES).all():
            raise ValueError(f"administration.status must be one of {sorted(STATUSES)}")
        return frame

    def _from_rank_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for pair, values in frame.groupby(["person_id", "block_id"], sort=False):
            person, block = pair
            if self._design._response_formats[block] == "graded":
                raise ValueError(f"Block {block!r}: graded categories cannot use statement_rank")
            members = self._design._frame.loc[self._design._frame["block_id"] == block]
            expected = dict(zip(members["statement_id"], members["position"], strict=True))
            statements = [
                identifier(v, f"Person {person!r}, block {block!r}: statement_id")
                for v in values["statement_id"]
            ]
            if len(set(statements)) != len(statements) or set(statements) != set(expected):
                raise ValueError(
                    f"Person {person!r}, block {block!r}: include every statement once"
                )
            order = [None] * len(expected)
            for statement, raw_rank in zip(statements, values["rank"], strict=True):
                if is_missing(raw_rank):
                    continue
                rank = integer(raw_rank, f"Person {person!r}, block {block!r}: rank")
                if rank < 1 or rank > len(order) or order[rank - 1] is not None:
                    raise ValueError(
                        f"Person {person!r}, block {block!r}: invalid or duplicate rank"
                    )
                order[rank - 1] = expected[statement]
            rows.append((person, block, order))
        return pd.DataFrame(rows, columns=RESPONSE_COLUMNS)

    @property
    def design(self) -> FCDesign:
        return self._design

    @property
    def person_ids(self) -> tuple[Label, ...]:
        return self._person_ids

    @property
    def input_encoding(self) -> str:
        return self._encoding

    @property
    def administration(self) -> pd.DataFrame | None:
        return None if self._administration is None else copy_frame(self._administration)

    @property
    def original_responses(self) -> pd.DataFrame:
        return copy_frame(self._original_responses)

    def to_frame(self, *, answered_only: bool = False) -> pd.DataFrame:
        """Export preference-order lists and scalar graded categories, with copies."""
        frame = self._responses
        if answered_only:
            keep = [
                (p, b) in self._answered_pairs
                for p, b in zip(frame["person_id"], frame["block_id"], strict=True)
            ]
            frame = frame.loc[keep]
        return copy_frame(frame).reset_index(drop=True)

    def to_rank_frame(self) -> pd.DataFrame:
        """Export answered ranking tasks only; graded events are not duplicated."""
        rows = []
        for row in self.to_frame(answered_only=True).itertuples(index=False):
            if self._design._response_formats[row.block_id] == "graded":
                continue
            members = self._design._frame.loc[self._design._frame["block_id"] == row.block_id]
            ranks = {
                position: rank
                for rank, position in enumerate(row.response, 1)
                if position is not None
            }
            for member in members.itertuples(index=False):
                rows.append(
                    (row.person_id, row.block_id, member.statement_id, ranks.get(member.position))
                )
        frame = pd.DataFrame(rows, columns=RANK_COLUMNS)
        frame["rank"] = pd.array(frame["rank"], dtype="Int64")
        return frame

    def to_graded_frame(self) -> pd.DataFrame:
        """Export answered graded events, retaining the declared category labels."""
        frame = self.to_frame(answered_only=True)
        return frame.loc[
            frame["block_id"].map(self._design._response_formats) == "graded"
        ].reset_index(drop=True)

    def summary(self) -> dict[str, int]:
        """Count people and observed person/block events, not statement rows."""
        return {
            "n_persons": len(self._person_ids),
            "n_observations": len(self._answered_pairs),
            "n_response_rows": len(self._responses),
        }
