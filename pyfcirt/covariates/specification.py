"""Fixed-effect regression declarations, independent of target preparation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import TypedDict

import pandas as pd
from pytensor.tensor.variable import TensorVariable

from pyfcirt.common.validation import require_frame


@dataclass(frozen=True)
class Regression:
    """Declare a fixed-effect RHS and its entity feature table.

    Numeric columns are continuous; boolean, string and categorical columns are
    categorical. Create transformed columns in ``data`` before declaring the
    formula. The Model freezes only the used rows and columns when binding a
    declaration to a target; this record does not prepare or copy the table.

    Automatic slope priors adjust for training continuous-feature SDs while
    preserving the chosen predictor units. Loading slopes use a base SD of 0.25;
    other targets use 1. Explicit priors retain their declared fitted-unit scale.
    Intercept priors use the built-in role's conditional defaults, independently
    of a user-supplied ordinary prior.
    """

    formula: str
    data: pd.DataFrame
    id: str | tuple[str, ...] = field(kw_only=True)
    priors: Mapping[str, TensorVariable] | None = field(default=None, kw_only=True)
    parameter_residual_sd: float | TensorVariable | None = field(default=None, kw_only=True)
    center_predictors: bool = field(default=True, kw_only=True)
    standardize_predictors: bool = field(default=False, kw_only=True)
    coefficient_noncentered: bool = field(default=False, kw_only=True)
    deviation_noncentered: bool | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if not isinstance(self.formula, str):
            raise TypeError("Regression formula must be a fixed-effect RHS string")
        if not self.formula.strip():
            raise ValueError("Regression formula must not be blank")
        if not isinstance(self.id, str | tuple):
            raise TypeError("Regression id must be a column name or tuple of column names")
        columns = (self.id,) if isinstance(self.id, str) else self.id
        if not columns or any(not isinstance(name, str) or not name.strip() for name in columns):
            raise ValueError("Regression id columns must be nonblank strings")
        if len(set(columns)) != len(columns):
            raise ValueError("Regression id columns must be distinct")
        require_frame(self.data, columns, "Regression data")
        for name in ("center_predictors", "standardize_predictors", "coefficient_noncentered"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Regression {name} must be a bool")
        if self.deviation_noncentered is not None and not isinstance(
            self.deviation_noncentered, bool
        ):
            raise TypeError("Regression deviation_noncentered must be a bool or None")
        object.__setattr__(self, "priors", MappingProxyType(_coefficient_priors(self.priors)))
        if self.parameter_residual_sd is not None and not isinstance(
            self.parameter_residual_sd, TensorVariable
        ):
            if isinstance(self.parameter_residual_sd, bool) or not isinstance(
                self.parameter_residual_sd, Real
            ):
                raise TypeError(
                    "Regression parameter_residual_sd must be a positive number, "
                    "native PyMC prior or None"
                )
            if not isfinite(self.parameter_residual_sd) or self.parameter_residual_sd <= 0:
                raise ValueError(
                    "Regression parameter_residual_sd must be finite and strictly positive"
                )


def _coefficient_priors(value):
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("Regression priors must be a mapping or None")
    priors = {} if value is None else dict(value)
    if any(not isinstance(name, str) or not name.strip() for name in priors):
        raise ValueError("Regression prior keys must be nonblank strings")
    if any(not isinstance(prior, TensorVariable) for prior in priors.values()):
        raise TypeError("Regression coefficient priors must be native PyMC templates")
    return priors


class RegressionOptions(TypedDict, total=False):
    """Editor metadata for Regression's keyword options."""

    priors: Mapping[str, TensorVariable] | None
    parameter_residual_sd: float | TensorVariable | None
    center_predictors: bool
    standardize_predictors: bool
    coefficient_noncentered: bool
    deviation_noncentered: bool | None


def _formula_columns(formula):
    """Screen syntax and collect names; Formulae alone parses and expands terms."""
    tokens = re.findall(r"[^\W\d]\w*|\d+|\S", formula)
    names = []
    for index, token in enumerate(tokens):
        if token.isidentifier():
            if index + 1 < len(tokens) and tokens[index + 1] == "(":
                raise ValueError("Regression formulas do not support function calls")
            if not token[0].isalpha() or token in {"True", "False", "None"}:
                raise ValueError(
                    "Regression feature names must be ordinary nonreserved identifiers"
                )
            names.append(token)
        elif token == "-":
            if (
                index + 1 == len(tokens)
                or tokens[index + 1] != "1"
                or index + 2 < len(tokens)
                and tokens[index + 2] not in {"+", ")"}
            ):
                raise ValueError("Regression formulas only support '-' for removing the intercept")
        elif token not in {"0", "1", "+", ":", "*", "(", ")"}:
            raise ValueError("Regression formulas require fixed-effect column names and operators")
        if token == "*" and index + 1 < len(tokens) and tokens[index + 1] == "*":
            raise ValueError("Regression formulas do not support powers; create a feature column")
    return tuple(dict.fromkeys(names))


def validate_formula(formula: str) -> tuple[str, ...]:
    """Check syntax before resolving targets and return the required feature columns."""
    try:
        from formulae.parser import ParseError, Parser
        from formulae.scanner import ScanError, Scanner
    except ModuleNotFoundError as error:
        if error.name != "formulae":
            raise
        raise ImportError("Regression formulas require pyfcirt[covariates]") from error

    features = _formula_columns(formula)
    try:
        parser = Parser(Scanner(formula).scan())
        parser.parse()
    except (ParseError, ScanError) as error:
        raise ValueError(f"Invalid regression formula syntax: {formula!r}") from error
    if not parser.at_end():
        raise ValueError(f"Invalid regression formula syntax: {formula!r}")
    return features


def regression_declarations(
    formulas,
    *,
    person_data=None,
    statement_data=None,
    item_data=None,
    pair_data=None,
    id_columns=None,
    options=None,
):
    """Translate user notation to the same Regression declarations used by the binder."""
    tables = dict(person=person_data, statement=statement_data, item=item_data, pair=pair_data)
    if isinstance(formulas, Mapping):
        if any(value is not None for value in (*tables.values(), id_columns, options)):
            raise ValueError("Regression mappings cannot be combined with tables, IDs or options")
        if not formulas:
            raise ValueError("Supply at least one regression declaration")
        declarations = list(formulas.items())
        for _, declaration in declarations:
            if not isinstance(declaration, Regression):
                raise TypeError("regressions values must be Regression declarations")
        return declarations, {}
    if not isinstance(formulas, str):
        raise TypeError("formulas must be a measurement regression string or Regression mapping")
    id_columns = {} if id_columns is None else id_columns
    options = {} if options is None else options
    if not isinstance(id_columns, Mapping) or not isinstance(options, Mapping):
        raise TypeError("id_columns and options must be mappings or None")
    if unknown := id_columns.keys() - tables.keys():
        raise ValueError(f"Unknown regression ID entities: {sorted(unknown)}")
    defaults = dict(
        person="person_id",
        statement="statement_id",
        item="item_id",
        pair=("block_id", "statement_1", "statement_2"),
    )
    declarations, labels = [], set()
    for number, raw in enumerate(formulas.splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        if line.count("~") != 1:
            raise ValueError(f"Line {number}: expected 'target ~ predictors'")
        label, rhs = (part.strip() for part in line.split("~"))
        parts = label.split(".")
        if label == "theta":
            target, entity = label, "person"
        elif len(parts) == 2 and parts[0] in {"fc", "items"} and parts[1]:
            target = tuple(parts)
            entity = (
                "pair" if parts[1] == "cutpoints" else ("statement" if parts[0] == "fc" else "item")
            )
        else:
            raise ValueError(
                f"Line {number}: target must be theta, fc.parameter or items.parameter"
            )
        if tables[entity] is None:
            raise ValueError(f"Line {number}: {label} requires {entity}_data")
        settings = options.get(label, {})
        if not isinstance(settings, Mapping):
            raise TypeError(f"{label}: regression options must be a mapping")
        if unknown := settings.keys() - (
            Regression.__dataclass_fields__.keys() - {"formula", "data", "id"}
        ):
            raise ValueError(f"{label}: unknown regression options {sorted(unknown)}")
        declaration = Regression(
            rhs, tables[entity], id=id_columns.get(entity, defaults[entity]), **settings
        )
        declarations.append((target, declaration))
        labels.add(label)
    qualified_priors = {}
    for label in options.keys() - labels:
        parts = label.split(".") if isinstance(label, str) else ()
        if len(parts) != 3 or not all(parts) or f"{parts[0]}.{parts[2]}" not in labels:
            raise ValueError(f"Options refer to undeclared regression targets: {label!r}")
        channel, model, role = parts
        settings = options[label]
        if not isinstance(settings, Mapping) or settings.keys() != {"priors"}:
            raise ValueError(f"{label}: model-qualified options accept only coefficient priors")
        qualified_priors.setdefault((channel, role), {})[model] = _coefficient_priors(
            settings["priors"]
        )
    if not declarations:
        raise ValueError("Supply at least one regression declaration")
    return declarations, qualified_priors


def _dif_target(label):
    parts = label.split(".") if isinstance(label, str) else ()
    if len(parts) not in (2, 3) or parts[0] not in {"fc", "items"} or not all(parts):
        raise ValueError(f"Invalid DIF target {label!r}; use fc.parameter or items.parameter")
    return tuple(parts)


def _dif_group_terms(groups):
    if isinstance(groups, str):
        groups = (groups,)
    elif not isinstance(groups, Sequence):
        raise TypeError("DIF groups must be a column name or a sequence of group terms")
    for term in groups:
        columns = (term,) if isinstance(term, str) else term
        if (
            not isinstance(columns, tuple)
            or not columns
            or any(
                not isinstance(name, str) or not name.isidentifier() or name.startswith("_")
                for name in columns
            )
        ):
            raise ValueError("DIF groups require column names or tuples of column names")
        if len(set(columns)) != len(columns):
            raise ValueError("Crossed DIF group columns must be distinct")
        yield ":".join(columns), columns if len(columns) > 1 else ()


def _dif_prior(prior, settings):
    if not isinstance(settings, Mapping):
        raise TypeError("DIF prior_options must be a mapping")
    if prior == "hierarchical":
        allowed = RegressionOptions.__annotations__.keys() - {"deviation_noncentered"}
        defaults = {}
    elif prior == "pairwise_l1":
        defaults = {"strength": 3.0}
        allowed = defaults.keys()
    elif prior == "truncated_l1":
        defaults = {"strength": 1.5, "cutoff": 0.5, "base_sd": 0.55}
        allowed = defaults.keys()
    else:
        raise ValueError("DIF prior must be hierarchical, pairwise_l1 or truncated_l1")
    if unknown := settings.keys() - allowed:
        raise ValueError(f"Unknown {prior} DIF prior options: {sorted(unknown)}")
    result = {**defaults, **settings}
    if prior == "truncated_l1" and result["cutoff"] is None:
        raise ValueError("Truncated L1 requires a finite positive cutoff")
    return result


def dif_declarations(
    targets,
    person_data,
    *,
    formula,
    groups,
    id_column,
    pooling,
    prior,
    prior_options,
    overrides,
):
    """Resolve user selectors into one atomic batch of prepared regression declarations."""
    if isinstance(targets, str):
        targets = (targets,)
    if not isinstance(targets, Sequence) or not targets:
        raise ValueError("Specify one or more DIF targets")
    if not isinstance(id_column, str):
        raise TypeError("DIF id must name one respondent ID column")
    if formula is not None and (not isinstance(formula, str) or not formula.strip()):
        raise ValueError("DIF formula must be a nonempty RHS string")
    group_terms = tuple(_dif_group_terms(groups))
    terms = tuple((rhs, columns, True) for rhs, columns in group_terms)
    if formula is not None:
        terms += ((formula.strip(), (), False),)
    if not terms or len({rhs for rhs, _, _ in terms}) != len(terms):
        raise ValueError("Supply distinct DIF group terms or a formula")
    target_keys = tuple(dict.fromkeys(_dif_target(target) for target in targets))
    configs = {} if overrides is None else overrides
    if not isinstance(configs, Mapping):
        raise TypeError("DIF overrides must be a mapping")
    resolved = {_dif_target(label): config for label, config in configs.items()}
    for key in resolved:
        if key not in target_keys and (key[0], key[-1]) not in target_keys:
            raise ValueError(f"DIF override does not select a requested target: {'.'.join(key)}")
    if prior_options is not None and not isinstance(prior_options, Mapping):
        raise TypeError("DIF prior_options must be a mapping")
    defaults = {} if prior_options is None else prior_options
    declarations, options = [], {}
    ordered = sorted(dict.fromkeys((*target_keys, *resolved)), key=len)
    for target in ordered:
        override = resolved.get(target, {})
        if not isinstance(override, Mapping):
            raise TypeError("Each DIF target override must be a mapping")
        if unknown := override.keys() - {"prior", "prior_options", "pooling"}:
            raise ValueError(f"Unknown DIF target overrides: {sorted(unknown)}")
        law = override.get("prior", prior)
        method = override.get("pooling", pooling)
        if method not in {"unit", "shared"}:
            raise ValueError("DIF pooling must be unit or shared")
        specific = override.get("prior_options", {})
        if not isinstance(specific, Mapping):
            raise TypeError("DIF prior_options must be a mapping")
        inherited = defaults if law == prior else {}
        settings = _dif_prior(law, {**inherited, **specific})
        if law != "hierarchical" and not group_terms:
            raise ValueError("The pairwise DIF priors require a groups declaration")
        for rhs, columns, category in terms:
            pairwise = settings if category and law != "hierarchical" else None
            regression_options = settings if law == "hierarchical" else {}
            declarations.append(
                (target, Regression(rhs, person_data, id=id_column, **regression_options))
            )
            options[target, rhs] = {
                "group_columns": columns,
                "categorical": category,
                "pairwise": pairwise,
                "shared": method == "shared",
            }
    return declarations, options
