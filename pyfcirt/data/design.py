"""Input declarations and sectioned design notation."""

import ast
import re
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import StringIO
from types import EllipsisType
from typing import Literal

import pandas as pd

from pyfcirt.models.specification import MGPPM, DichotomousIRT

_LABEL = re.compile(r"[^\s~:=#,\[\]()\"']+")
_RANGE = re.compile(r"(.*?)([0-9]+)-([^0-9\s-]*)([0-9]+)")


@dataclass(frozen=True)
class CategoryScale:
    """Complete categories, in increasing preference for first or second statement."""

    categories: tuple
    higher_prefers: str = field(default="first", kw_only=True)
    labels: tuple | None = field(default=None, kw_only=True)


@dataclass(frozen=True, kw_only=True)
class UnitIntervalScale:
    """Continuous responses strictly between zero and one."""

    higher_prefers: str = "first"


@dataclass(frozen=True)
class ComparisonColumn:
    """Declare the meaning of a wide column without parsing its label."""

    column: str | int
    block_id: str | int
    statement_1: str | int
    statement_2: str | int


@dataclass(frozen=True)
class ResponseTable:
    """Declare response-table columns and their storage layout."""

    frame: pd.DataFrame
    format: str = field(kw_only=True)
    person_id_column: str | int | None = field(default=None, kw_only=True)
    comparison_columns: tuple[ComparisonColumn, ...] = field(default=(), kw_only=True)


@dataclass(frozen=True, kw_only=True)
class FCDesignMapping:
    """Map traits and blocks to their ordered statement lists."""

    traits: Mapping
    blocks: Mapping
    keying: Mapping | None = None


@dataclass(frozen=True, kw_only=True)
class ItemDesignMapping:
    """Map traits to item lists; an item may load on more than one trait."""

    traits: Mapping


@dataclass(frozen=True)
class BlockSpec:
    """Override one block; omitted response_format inherits and explicit None clears it."""

    block_id: str | int
    model: object | None = field(default=None, kw_only=True)
    response_format: str | None | EllipsisType = field(default=..., kw_only=True)
    scale: CategoryScale | UnitIntervalScale | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class ItemOverride:
    """Override the measurement law or complete category scale for one item."""

    item_id: str | int
    model: object | None = field(default=None, kw_only=True)
    categories: tuple | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class FCInput:
    """Declare a test and responses; discrete response_format=None defaults to binary 0/1.

    By default, higher values favor the first statement. Explicit scales override this;
    ordinal and continuous responses require their complete declared scales.
    """

    design: pd.DataFrame | str | FCDesignMapping | Mapping
    responses: ResponseTable
    model: object = field(default_factory=MGPPM, kw_only=True)
    response_format: Literal["rank", "pick", "mole"] | None = field(default="rank", kw_only=True)
    scale: CategoryScale | UnitIntervalScale | None = field(default=None, kw_only=True)
    blocks: Sequence[BlockSpec] | Mapping = field(default=(), kw_only=True)
    administration: pd.DataFrame | None = field(default=None, kw_only=True)
    statement_features: pd.DataFrame | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class ItemInput:
    """Declare item-by-trait membership, response laws and recorded answers."""

    design: pd.DataFrame | str | ItemDesignMapping | Mapping
    responses: ResponseTable
    model: object = field(default_factory=DichotomousIRT, kw_only=True)
    categories: tuple = field(default=(0, 1), kw_only=True)
    items: Mapping[str | int, Mapping[str, object]] | Sequence[ItemOverride] = field(
        default=(), kw_only=True
    )
    administration: pd.DataFrame | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class _ModelDeclaration:
    """Temporary constructor syntax; encoding owns the finite model mapping."""

    name: str
    options: dict
    line: int


def _labels(text, line):
    values = text.split()
    if not values or any(_LABEL.fullmatch(value) is None for value in values):
        raise ValueError(f"Line {line}: use nonempty, whitespace-separated labels")
    return values


def _members(text, line):
    """Expand inclusive ranges; numeric text remains a string label."""
    expanded = []
    for token in _labels(text, line):
        match = _RANGE.fullmatch(token)
        if match is None:
            expanded.append(token)
            continue
        prefix, first, end_prefix, last = match.groups()
        if end_prefix and end_prefix != prefix:
            raise ValueError(f"Line {line}: range endpoints need the same label prefix")
        start, end = int(first), int(last)
        if end < start:
            raise ValueError(f"Line {line}: range start must not exceed its end")
        padded = first.startswith("0") or last.startswith("0")
        if padded and len(first) != len(last):
            raise ValueError(f"Line {line}: range endpoints need matching zero padding")
        width = len(first) if padded else 0
        expanded.extend(f"{prefix}{value:0{width}d}" for value in range(start, end + 1))
    return expanded


def _declarations(text, *, sections):
    if not isinstance(text, str):
        raise TypeError("Design specification must be a string")
    section, seen = None, set()
    for number, raw in enumerate(text.splitlines(), 1):
        try:
            comments = [
                token.start[1]
                for token in tokenize.generate_tokens(StringIO(raw).readline)
                if token.type == tokenize.COMMENT
            ]
        except (tokenize.TokenError, IndentationError) as error:
            raise ValueError(f"Line {number}: invalid design syntax") from error
        line = raw[: comments[0] if comments else len(raw)].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]") and "~" not in line:
            section = line[1:-1].strip()
            if section not in sections or section in seen:
                raise ValueError(f"Line {number}: expected a unique section from {sections!r}")
            seen.add(section)
            continue
        head, separator, body = line.partition("~")
        if section is None or not separator or not head.strip() or not body.strip():
            raise ValueError(f"Line {number}: expected a declaration using '~' in a section")
        yield number, section, head.strip(), body.strip()


def _block_options(text, line):
    """Read names and literal keywords only; never execute a string expression."""
    try:
        values = ast.parse(text, mode="eval").body
        if not isinstance(values, ast.List) or len(values.elts) not in (1, 2):
            raise ValueError("expected [model] or [model, response_format]")
        model = values.elts[0]
        if isinstance(model, ast.Name):
            declaration = _ModelDeclaration(model.id, {}, line)
        elif isinstance(model, ast.Call) and isinstance(model.func, ast.Name) and not model.args:
            options = {}
            for keyword in model.keywords:
                if keyword.arg is None or keyword.arg in options:
                    raise ValueError("constructor keywords must be explicit and unique")
                for node in ast.walk(keyword.value):
                    if isinstance(node, ast.Dict):
                        keys = [ast.literal_eval(key) for key in node.keys]
                        if len(keys) != len(set(keys)):
                            raise ValueError("duplicate literal dictionary key")
                options[keyword.arg] = ast.literal_eval(keyword.value)
            declaration = _ModelDeclaration(model.func.id, options, line)
        else:
            raise ValueError("expected a model name or literal-keyword constructor")
        result = {"model": declaration}
        if len(values.elts) == 2:
            response_format = values.elts[1]
            if isinstance(response_format, ast.Name):
                result["response_format"] = response_format.id
            elif isinstance(response_format, ast.Constant) and (
                response_format.value is None or isinstance(response_format.value, str)
            ):
                result["response_format"] = response_format.value
            else:
                raise ValueError("expected a response_format name or None")
        return result
    except (SyntaxError, ValueError, TypeError) as error:
        raise ValueError(f"Line {line}: invalid block options: {error}") from error


def _direction(head, members, line, signs, sides):
    directive = head.split()
    if len(directive) < 2 or directive[0] not in {"sign", "side"}:
        return False
    allowed = {"+": 1, "-": -1, **({"0": 0} if directive[0] == "side" else {})}
    if len(directive) != 2 or directive[1] not in allowed:
        raise ValueError(f"Line {line}: invalid {directive[0]} directive")
    target = signs if directive[0] == "sign" else sides
    for member in members:
        if member in target:
            raise ValueError(f"Line {line}: duplicate {directive[0]} for {member!r}")
        target[member] = allowed[directive[1]], line
    return True


def _fc_membership_table(blocks, assignments, keying, *, sides=None):
    """Expand ordered block membership from validated labels."""
    rows = []
    for block, members in blocks.items():
        for position, statement in enumerate(members, 1):
            if statement not in assignments:
                raise ValueError(f"Block {block!r} uses unassigned statement {statement!r}")
            rows.append(
                (block, position, statement, assignments[statement], keying.get(statement, 1))
            )
    table = pd.DataFrame(
        rows, columns=["block_id", "position", "statement_id", "trait", "keying"], dtype=object
    )
    if sides:
        table["location_side"] = [sides.get(statement, 0) for statement in table.statement_id]
    return table


def parse_design(text):
    """Return membership and temporary wildcard/inline declarations in source order."""
    assignments, blocks, signs, sides, traits = {}, {}, {}, {}, set()
    defaults, options = {}, {}
    for number, section, head, body in _declarations(text, sections=("traits", "blocks")):
        if head == "*":
            if section != "blocks" or defaults:
                raise ValueError(f"Line {number}: a single wildcard belongs in [blocks]")
            defaults = _block_options(body, number)
            continue
        setting = {}
        if section == "blocks" and "[" in body:
            members, separator, tail = body.partition("[")
            setting = _block_options(separator + tail, number)
            body = members.strip()
        members = _members(body, number)
        if _direction(head, members, number, signs, sides):
            if setting:
                raise ValueError(f"Line {number}: direction declarations have no block options")
            continue
        names = _labels(head, number)
        if len(names) != 1 or names[0] == "*":
            raise ValueError(f"Line {number}: a declaration needs one name")
        name = names[0]
        if section == "traits":
            if name in traits:
                raise ValueError(f"Line {number}: duplicate trait {name!r}")
            traits.add(name)
            for statement in members:
                if statement in assignments:
                    raise ValueError(f"Line {number}: statement {statement!r} has multiple traits")
                assignments[statement] = name
        else:
            if name in blocks:
                raise ValueError(f"Line {number}: duplicate block {name!r}")
            blocks[name] = members
            options[name] = setting
    if not assignments or not blocks:
        raise ValueError("Design needs [traits] and [blocks] declarations")
    for kind, values in (("sign", signs), ("side", sides)):
        for statement, (_, line) in values.items():
            if statement not in assignments:
                raise ValueError(f"Line {line}: {kind} refers to unknown statement {statement!r}")
    table = _fc_membership_table(
        blocks,
        assignments,
        {statement: value for statement, (value, _) in signs.items()},
        sides={statement: value for statement, (value, _) in sides.items()},
    )
    return table, {"defaults": defaults, "blocks": options}


def parse_item_design(text):
    """Return active item/trait rows and optional signed locations."""
    rows, sides, traits = [], {}, set()
    for number, _, head, body in _declarations(text, sections=("traits",)):
        members = _members(body, number)
        directive = head.split(maxsplit=1)
        if len(directive) == 2 and directive[0] == "side":
            _direction(head, members, number, {}, sides)
            continue
        names = _labels(head, number)
        if len(names) != 1 or names[0] in traits or names[0] == "*":
            raise ValueError(f"Line {number}: expected a unique trait name")
        traits.add(names[0])
        rows.extend(dict(item_id=item, trait=names[0]) for item in members)
    if sides:
        items = {row["item_id"] for row in rows}
        for item, (_, line) in sides.items():
            if item not in items:
                raise ValueError(f"Line {line}: side refers to unknown item {item!r}")
        for row in rows:
            row["location_side"] = sides.get(row["item_id"], (None, None))[0]
    return pd.DataFrame(rows, dtype=object)
