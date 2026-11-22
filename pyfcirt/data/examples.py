"""Read current bundled input declarations and their original scientific sources."""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from numbers import Integral
from typing import TYPE_CHECKING

import pandas as pd

from pyfcirt.data.design import (
    BlockSpec,
    CategoryScale,
    FCInput,
    ItemInput,
    ItemOverride,
    ResponseTable,
    UnitIntervalScale,
)
from pyfcirt.data.encoding import _model_declaration

if TYPE_CHECKING:
    import xarray as xr


@dataclass(frozen=True)
class ExampleData:
    """Fresh inputs, matched optional companion tables, and source metadata."""

    data: FCInput | ItemInput | tuple[FCInput | ItemInput, ...]
    truth: xr.Dataset | None
    metadata: Mapping
    person_data: pd.DataFrame | None = None
    response_times: pd.DataFrame | None = None


def _table(record):
    if record is None:
        return None
    return pd.DataFrame(record["rows"], columns=record["columns"], dtype=object)


def _select_people(frame, selected):
    """Keep the original row order for every table in a selected cohort."""
    return frame.loc[frame.person_id.isin(selected)].reset_index(drop=True)


def _companion(root, path, selected):
    with root.joinpath(path).open("r", encoding="utf-8", newline="") as stream:
        # CSV identifiers are literal strings; only measurement columns are inferred.
        frame = pd.read_csv(
            stream, dtype={"person_id": str, "block_id": str}, keep_default_na=False
        )
    return _select_people(frame, selected)


def _scale(record):
    if record is None:
        return None
    if record["kind"] == "unit_interval":
        return UnitIntervalScale(higher_prefers=record["higher_prefers"])
    return CategoryScale(
        tuple(record["categories"]),
        higher_prefers=record["higher_prefers"],
        labels=None if record["labels"] is None else tuple(record["labels"]),
    )


def _input(record, model, selected):
    responses = record["responses"]
    frame = _select_people(_table(responses), selected)
    administration = _table(record["administration"])
    if administration is not None:
        administration = _select_people(administration, selected)
    common = dict(
        design=_table(record["design"]),
        responses=ResponseTable(frame, format=responses["format"]),
        model=model,
        administration=administration,
    )
    if record["kind"] == "items":
        return ItemInput(
            **common,
            categories=tuple(record["items"][0]["categories"]),
            items=tuple(
                ItemOverride(value["item_id"], categories=tuple(value["categories"]))
                for value in record["items"]
            ),
        )
    features = _table(record["statement_features"])
    if features is not None:
        # Infer stored numbers, not numeric-looking strings; identifiers retain
        # their exact types. Formulae and NumPy can then use numeric features directly.
        for column in features.columns.difference(["statement_id"]):
            features[column] = features[column].infer_objects()
    # Original resources keep their historical task field and checksum.
    return FCInput(
        **common,
        response_format=record["blocks"][0]["task"],
        blocks=tuple(
            BlockSpec(
                value["block_id"], response_format=value["task"], scale=_scale(value["scale"])
            )
            for value in record["blocks"]
        ),
        statement_features=features,
    )


def _read_cohort(root, name, n_persons, random_seed):
    """Read the smallest sufficient cohort, or the full sample for seeded draws."""
    cases = {}
    for parent in (root, root.joinpath("studies")):
        cases.update(
            (case.name, case)
            for case in parent.iterdir()
            if case.is_dir() and case.name != "studies"
        )
    if not isinstance(name, str) or name not in cases:
        available = ", ".join(sorted(cases))
        raise ValueError(f"Unknown example {name!r}. Available examples: {available}")

    case = cases[name]
    cohorts = {
        int(file.name.removesuffix(".json.gz")[1:]): file
        for file in case.iterdir()
        if file.name.startswith("n") and file.name.endswith(".json.gz")
    }
    sizes = sorted(cohorts)
    largest = sizes[-1]
    if n_persons is None:
        n_persons = largest
        if len(sizes) > 1:
            metadata = json.loads(case.joinpath("metadata.json").read_text(encoding="utf-8"))
            n_persons = metadata["default_n_persons"]
    if n_persons > largest:
        raise ValueError(f"Example {name!r} contains at most {largest} people")
    stored = (
        largest if random_seed is not None else next(size for size in sizes if size >= n_persons)
    )
    resource = cohorts[stored]
    return json.loads(gzip.decompress(resource.read_bytes())), int(n_persons), resource.name


def _example_models(records, model):
    """Resolve the declared model for every channel in this example."""
    channels = {record["kind"] for record in records}
    if isinstance(model, Mapping):
        if set(model) != channels:
            raise ValueError(f"model mapping keys must match input channels {sorted(channels)!r}")
        declarations = model
    elif len(channels) == 1:
        declarations = dict.fromkeys(channels, model)
    else:
        raise TypeError("Mixed-channel examples require a model mapping for 'fc' and 'items'")
    return {
        channel: _model_declaration(value, channel=channel)
        for channel, value in declarations.items()
    }


def _example_people(records, count, random_seed):
    """Select source people once for all responses, companions and truth."""
    people = []
    for record in records:
        for table in (record["responses"], record["administration"]):
            if table is not None:
                position = table["columns"].index("person_id")
                people.extend(row[position] for row in table["rows"])
    people = list(dict.fromkeys(people))
    selected = people[:count]
    if random_seed is not None:
        selected = (
            pd.Series(people, dtype=object)
            .sample(n=count, replace=False, random_state=int(random_seed))
            .tolist()
        )
    return people, selected


def _example_truth(payload, selected):
    """Subset original generating arrays on their recorded person axes."""
    if payload["truth"] is None:
        return None
    import xarray as xr

    truth = xr.Dataset.from_dict(payload["truth"])
    selected_ids = set(selected)
    truth = truth.isel(
        {
            dimension: [index for index, person in enumerate(values) if person in selected_ids]
            for dimension, values in payload["truth_person_axes"].items()
        }
    )
    truth.attrs.update(
        pyfcirt_truth_source="original_generation",
        pyfcirt_current_fixed_parameters=False,
    )
    return truth


def load_example(
    name: str,
    *,
    model,
    n_persons: int | None = None,
    random_seed: int | None = None,
    include_truth: bool = False,
) -> ExampleData:
    """Read bundled data with a caller-selected current measurement declaration.

    For one channel, ``model`` is its canonical model name or specification
    instance, including explicit link/prior options. Mixed inputs require a
    mapping from each actual channel (``fc`` and ``items``) to its declaration;
    a single-channel mapping must also match exactly. Loading does not prepare
    or fit a model. Historical source-model tags never override this choice.

    Selecting ``n_persons`` retains the complete design, declared
    categories, missingness and administration. Optional person covariates and
    response times use the same selected IDs and retain their source row order.
    By default selection takes the first source people. With explicit
    ``n_persons``, ``random_seed`` selects people without replacement using pandas,
    using the largest stored cohort for a study.
    Each call returns fresh tables; missing companions are None.
    ``include_truth`` returns original generating arrays when available, with their
    original names and constraints;
    they are not current fixed parameters or draws from the new default priors.
    Empirical examples have no generating truth and return None.
    """
    if n_persons is not None and (
        isinstance(n_persons, bool) or not isinstance(n_persons, Integral) or n_persons < 1
    ):
        raise ValueError("n_persons must be a positive integer")
    if random_seed is not None:
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, Integral)
            or not 0 <= random_seed < 2**32
        ):
            raise ValueError("random_seed must be an integer between 0 and 2**32 - 1")
        if n_persons is None:
            raise ValueError("random_seed requires an explicit n_persons")
    if not isinstance(include_truth, bool):
        raise TypeError("include_truth must be a boolean")
    root = files("pyfcirt").joinpath("_example_data")
    payload, count, resource_name = _read_cohort(root, name, n_persons, random_seed)
    if payload["format"] != "pyfcirt-example" or payload["version"] != 1:
        raise ValueError("Unsupported bundled example input format")
    records = payload["inputs"]
    models = _example_models(records, model)
    people, selected = _example_people(records, count, random_seed)
    inputs = tuple(_input(record, models[record["kind"]], selected) for record in payload["inputs"])
    metadata = payload["metadata"]
    companions = {
        field: _companion(root, path, selected)
        for field, path in metadata.get("companion_tables", {}).items()
    }
    metadata["selection"] = {
        "n_persons": len(selected),
        "stored_n_persons": len(people),
        "resource": resource_name,
        "policy": (
            "first people of the stored cohort; complete design retained"
            if random_seed is None
            else "seeded sample without replacement; source row order and complete design retained"
        ),
        "random_seed": None if random_seed is None else int(random_seed),
    }
    metadata["truth_available"] = payload["truth"] is not None
    if payload["generation"] is not None:
        metadata["original_generation"] = payload["generation"]
    truth = _example_truth(payload, selected) if include_truth else None
    return ExampleData(inputs[0] if len(inputs) == 1 else inputs, truth, metadata, **companions)
