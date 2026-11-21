"""Prepare validated runtime inputs for construction, scoring and generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from functools import cached_property
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from pyfcirt.common.validation import align_rows, fixed_values_match, identifiers
from pyfcirt.covariates.specification import regression_declarations
from pyfcirt.data.arrays import LayoutBuilder, freeze_arrays
from pyfcirt.data.constraints import _layout, freeze_layout, freeze_prior_shapes
from pyfcirt.data.coordinates import (
    PARAMETER_INDEX_FIELDS,
    EventLayout,
    ParameterBlock,
    ParameterLayout,
    PopulationLayout,
    PreparedBatch,
    PreparedRegression,
    RegressionTarget,
    SlotRef,
    _identity_digest,
    _parameter_scopes,
    _person_key,
    finalize_parameters,
    prepare_layout,
)
from pyfcirt.data.design import FCInput, ItemInput, ResponseTable
from pyfcirt.data.encoding import (
    OBSERVED_STATUSES,
    UNIT_COLUMNS,
    NormalizedChannel,
    _model_declaration,
    encode_batches,
    frozen_statement_order,
    generated_rows,
    has_statement_rows,
    index_response_rows,
    normalize_administration,
    normalize_channel,
    normalize_indexed_administration,
    normalize_responses,
    response_category_count,
    warn_unobserved_statements,
)
from pyfcirt.data.grouping import theta_target_indices
from pyfcirt.data.regressions import _bind_regressions, _resolve_regressions
from pyfcirt.models.population import PopulationFactory
from pyfcirt.models.priors import PriorPreparation
from pyfcirt.models.specification import PopulationSpec

if TYPE_CHECKING:
    from pyfcirt.model import Model


def _scalar_categories(values, dimension):
    """Project validated integer/string category codes into native storage types."""
    integer = np.asarray([not isinstance(value, str) for value in values], dtype=bool)
    limits = np.iinfo(np.int64)
    native_integer = [
        known and limits.min <= value <= limits.max
        for value, known in zip(values, integer, strict=True)
    ]
    return xr.Dataset(
        {
            "value_is_integer": (dimension, integer),
            "value_integer": (
                dimension,
                np.asarray(
                    [
                        value if known else 0
                        for value, known in zip(values, native_integer, strict=True)
                    ],
                    dtype=np.int64,
                ),
            ),
            "value_text": (
                dimension,
                np.asarray(
                    [
                        "" if known else str(value)
                        for value, known in zip(values, native_integer, strict=True)
                    ],
                    dtype=str,
                ),
            ),
        },
        coords={dimension: np.arange(len(values), dtype=np.int64)},
    )


def _channel_units(constant):
    units = {}
    for label, record in zip(constant.unit_id.values, constant.attrs["units"], strict=True):
        units.setdefault(record["channel"], {})[label] = record
    return units


@dataclass(frozen=True)
class PreparedInputs:
    """Validated runtime inputs; native declarations and encoders are never persisted.

    The labelled arrays remain authoritative. Cached views bind their existing
    indices once, without restoring specifications or validating saved records.
    """

    data: xr.DataTree
    _response_projection: Mapping[str, xr.Dataset] | None = dataclass_field(
        default=None, repr=False, compare=False, kw_only=True
    )
    intervention_values: Mapping[str, np.ndarray] = dataclass_field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        fixed = self.fixed_parameters
        values = {name: value.values for name, value in fixed.data_vars.items()} if fixed else {}
        if "covariance" in values:
            template = self.constant_data
            values.update(
                PopulationFactory.fixed_covariance_values(
                    values["covariance"],
                    shared_correlation="population_group"
                    not in template["trait_correlation_fixed_value"].dims,
                    sd_shape=template["population_sd_fixed_value"].shape,
                )
            )
        object.__setattr__(self, "intervention_values", MappingProxyType(values))
        for node in self.data.subtree:
            freeze_arrays(node.dataset)

    @classmethod
    def from_inputs(
        cls,
        data: FCInput | ItemInput | tuple[FCInput | ItemInput, ...],
        *,
        population: PopulationSpec | None = None,
        person_data: pd.DataFrame | None = None,
        regressions: str | Mapping | None = None,
        statement_data: pd.DataFrame | None = None,
        item_data: pd.DataFrame | None = None,
        pair_data: pd.DataFrame | None = None,
        id_columns: Mapping | None = None,
        regression_options: Mapping | None = None,
        purpose: Literal["observed", "generate"] = "observed",
        theta_parameterization: Literal["centered", "noncentered", "conditional"] = "centered",
        centered_traits: tuple | list | None = None,
        fixed_parameters: xr.Dataset | None = None,
    ) -> PreparedInputs:
        """Prepare declarations, coordinates and optional regressions before graph creation.

        Data preparation resolves response encoding and identification; the
        covariate modules fit formulas using those coordinates. No probability
        graph or posterior is created. Fixed values are checked against the
        complete layout after regressions have introduced their parameter nodes.
        """
        if purpose not in {"observed", "generate"}:
            raise ValueError("purpose must be observed or generate")
        if theta_parameterization not in {"centered", "noncentered", "conditional"}:
            raise ValueError("theta_parameterization must be centered, noncentered or conditional")
        if centered_traits is not None and theta_parameterization != "conditional":
            raise ValueError("centered_traits requires theta_parameterization='conditional'")
        if isinstance(centered_traits, (str, bytes)):
            raise TypeError("centered_traits must be a sequence of trait IDs")
        if regressions is None and any(
            value is not None
            for value in (statement_data, item_data, pair_data, id_columns, regression_options)
        ):
            raise ValueError("Regression tables and options require regressions")
        sources = data if isinstance(data, tuple) else (data,)
        if not sources or any(not isinstance(source, (FCInput, ItemInput)) for source in sources):
            raise TypeError("data must contain FCInput or ItemInput declarations")
        if len({type(source) for source in sources}) != len(sources):
            raise ValueError("Declare at most one FC channel and one item channel")
        if population is not None and not isinstance(population, PopulationSpec):
            raise TypeError("population must be a PopulationSpec")
        declarations = None
        population_data = person_data
        person_id = "person_id"
        if regressions is not None:
            declarations, qualified_priors = regression_declarations(
                regressions,
                person_data=person_data if isinstance(regressions, str) else None,
                statement_data=statement_data,
                item_data=item_data,
                pair_data=pair_data,
                id_columns=id_columns,
                options=regression_options,
            )
            if isinstance(regressions, str) and person_data is not None and id_columns:
                # Population grouping uses the canonical person ID, formula tables retain theirs.
                person_id = id_columns.get("person", "person_id")
                if isinstance(person_id, tuple):
                    if len(person_id) != 1:
                        raise ValueError("Population person ID requires one column")
                    (person_id,) = person_id
                population_data = person_data.rename(columns={person_id: "person_id"})
        with PriorPreparation():
            channels = tuple(normalize_channel(source, purpose=purpose) for source in sources)
            layout, person_alignment = prepare_layout(
                channels,
                population=PopulationSpec() if population is None else population,
                person_data=population_data,
                theta_parameterization=theta_parameterization,
                centered_traits=centered_traits,
            )
            layout = freeze_layout(layout)
            layout.attrs.update(regression_scopes=[], marginal_correlation=False)
            groups = {}
            constant = layout
            if declarations:
                constant = layout.build()
                active, replaced = _resolve_regressions(constant, declarations)
                theta = next((spec for target, spec, _, _ in active if target == "theta"), None)
                # Renaming an ID for population grouping preserves source row positions.
                same_person_table = (
                    theta is not None
                    and theta.data is person_data
                    and theta.id in (person_id, (person_id,))
                )
                constant, _ = _bind_regressions(
                    constant,
                    groups,
                    active,
                    replaced,
                    qualified_priors=qualified_priors,
                    theta_alignment=person_alignment if same_person_table else None,
                )
            constant = groups["/constant_data"] = finalize_parameters(constant)
            if fixed_parameters is not None:
                # Validate external values against the completed parameter catalogue.
                design = cls(xr.DataTree.from_dict(groups))
                groups["/fixed_parameters"] = _calibration_values(design, fixed_parameters)
            rows = index_response_rows(constant.person.values, _channel_units(constant), channels)
            prepared = cls._from_rows(groups, rows, purpose=purpose)
            if purpose == "observed" and fixed_parameters is None:
                warn_unobserved_statements(prepared.constant_data)
            return prepared

    @classmethod
    def _from_rows(cls, groups, rows, *, purpose):
        """Publish one complete candidate using its prepared design and indexed rows."""
        constant, observed = encode_batches(groups["/constant_data"], *rows, purpose=purpose)
        root = groups.get("/")
        groups.update({"/constant_data": constant, "/observed_data": observed})
        groups["/"] = (xr.Dataset() if root is None else root).assign_attrs(
            purpose=purpose,
            theta_parameterization=constant.attrs["population"]["theta_parameterization"],
        )
        return cls(xr.DataTree.from_dict(groups))

    def _with_parameter_updates(self, groups):
        """Reuse the response projection only when keeping the encoded rows."""
        projection = self.response_metadata if "observed_data" in self.data else None
        return type(self)(
            xr.DataTree.from_dict(groups, name=self.data.name), _response_projection=projection
        )

    def _design_groups(self):
        """Borrow the frozen design and fitted covariate encodings for a new response set."""
        constant = self.constant_data
        attrs = dict(constant.attrs, batches=[])
        design = constant.drop_vars(attrs.pop("observation_variables"))
        design.attrs = attrs
        return {
            "/constant_data": design,
            **{
                f"/{record['path']}": self.data[record["path"]].dataset
                for record in self.constant_data.attrs["regression_scopes"]
            },
        }

    @cached_property
    def constant_data(self):
        return self.data["constant_data"].dataset

    @cached_property
    def _units_by_channel(self):
        return _channel_units(self.constant_data)

    @cached_property
    def observed_data(self):
        return self.data["observed_data"].dataset

    @cached_property
    def fixed_parameters(self) -> xr.Dataset | None:
        """The validated conditioning values owned by this prepared analysis."""
        node = self.data.get("fixed_parameters")
        return None if node is None else node.dataset

    @cached_property
    def coords(self):
        return MappingProxyType(
            {
                name: self._array(name)
                for name, value in self.constant_data.coords.variables.items()
                if value.dims == (name,)
            }
        )

    def _array(self, name):
        if name is None:
            return None
        values = self.constant_data.variables[name].values.view()
        values.flags.writeable = False
        return values

    def _build_blocks(self, scope):
        priors = (
            scope["specification"].priors
            if "specification" in scope
            else self.data[scope["path"]].attrs["resolved_priors"]
        )
        arrays = {
            "free_flat",
            "entry_index",
            "orientation",
            "prior_role_index",
            "scale_entry_index",
        }
        result = []
        for record in scope["blocks"]:
            if "scope" in record:
                continue
            values = {
                key: self._array(value) if key in arrays else value
                for key, value in record.items()
                if key in ParameterBlock.__dataclass_fields__ and key != "bounds"
            }
            if record["key"] == "B":
                regression = self.constant_data.attrs["population"]["recipe"]["regression"]
                priors = self.data[f"covariates/{regression}"].attrs["resolved_priors"]
            groups = []
            for role, positions in record["prior_groups"].items():
                if isinstance(positions, range):
                    (dim,) = self.constant_data.variables[record["free_flat"]].dims
                    indices = self._array(dim)
                else:
                    indices = np.asarray(positions, dtype=np.int64)
                    indices.flags.writeable = False
                groups.append((role, indices))
            values.update(
                dims=tuple(record["dims"]),
                shape=tuple(record["shape"]),
                priors=priors,
                prior_groups=tuple(groups),
            )
            if record.get("bounds") is not None:
                # A bounded block has one lower kind and limit for all entries.
                entry_data = self.parameter_layouts[record["kind"]].entry_data
                entries = values["entry_index"].ravel()
                lower = entry_data.lower_entry_index.values[entries[values["free_flat"]]]
                lower.flags.writeable = False
                values["bounds"] = (
                    entry_data.lower_kind.values[entries[0]],
                    lower,
                    entry_data.upper_limit.values[entries[0]],
                )
            result.append(ParameterBlock(**values))
        return tuple(result)

    @cached_property
    def parameters(self) -> tuple[ParameterLayout, ...]:
        """Final parameter layouts in preparation order."""
        return tuple(self.parameter_layouts.values())

    @cached_property
    def parameter_layouts(self) -> Mapping[str, ParameterLayout]:
        """The prepared final parameter layouts keyed by kind."""
        result = {}
        for record in self.constant_data.attrs["parameters"]:
            kind, dims = record["kind"], tuple(record["dims"])
            variables = {
                field: self.constant_data.variables[f"{kind}_{field}"]
                for field in record["columns"]
            }
            if dims != (f"{kind}_id",):
                entry_dim = f"{kind}_entry"
                entry_data = xr.Dataset(
                    {
                        name: (entry_dim, value.values.reshape(-1))
                        for name, value in variables.items()
                    },
                    coords={entry_dim: np.arange(int(np.prod(record["shape"])))},
                )
                entry_data.attrs.update(parameter_dims=list(dims), parameter_shape=record["shape"])
            else:
                entry_data = xr.Dataset(
                    variables, coords={dim: self.constant_data[dim] for dim in dims}
                )
            fixed = self._array(f"{kind}_fixed_value")
            free_mask = np.isnan(fixed)
            free, bound = np.flatnonzero(free_mask), np.flatnonzero(~free_mask)
            free.flags.writeable = bound.flags.writeable = False
            result[kind] = ParameterLayout(
                kind=kind,
                dims=dims,
                shape=tuple(record["shape"]),
                fixed=fixed,
                free_flat=free,
                fixed_flat=bound,
                entry_data=entry_data,
                # A supplied scoring point retains the canonical axes of a kind.
                fixed_for_scoring=bool(entry_data.fixed_for_scoring.any()),
                point_parameter=bool(entry_data.point_parameter.any()),
            )
        return MappingProxyType(result)

    @cached_property
    def _blocks_by_scope(self):
        return {
            scope["name"]: self._build_blocks(scope)
            for scope in _parameter_scopes(self.constant_data)
        }

    @cached_property
    def blocks(self):
        return tuple(block for blocks in self._blocks_by_scope.values() for block in blocks)

    @cached_property
    def blocks_by_kind(self):
        groups = defaultdict(list)
        for block in self.blocks:
            groups[block.kind].append(block)
        return groups

    @cached_property
    def parameter_uses(self):
        return xr.Dataset(
            {
                name.removeprefix("parameter_use_"): value
                for name, value in self.constant_data.variables.items()
                if name.startswith("parameter_use_")
            }
        )

    @cached_property
    def parameter_units(self) -> pd.DataFrame:
        """Reuse the full projection across model and coefficient summaries."""
        return self.parameter_unit_table()

    def parameter_unit_table(self, units=None) -> pd.DataFrame:
        """Project distinct entry memberships for the requested unit positions."""
        uses = self.parameter_uses[["kind", "entry_index", "unit_index"]]
        if units is not None:
            uses = uses.isel(parameter_use=np.flatnonzero(np.isin(uses.unit_index.values, units)))
        return uses.to_dataframe().drop_duplicates().set_index(["kind", "entry_index"])

    @cached_property
    def output_coords(self) -> xr.Dataset:
        """Label the prepared identity columns on their authoritative axes."""
        result = {}
        for record in self.constant_data.attrs["parameters"]:
            kind, dims = record["kind"], tuple(record["dims"])
            # Population matrices already carry their mathematical identity axes.
            if dims != (f"{kind}_id",):
                continue
            for field in ("parameter_set_index", *PARAMETER_INDEX_FIELDS):
                name = f"{kind}_{field}"
                value = self.constant_data.variables[name]
                if np.any(value.values >= 0):
                    result[name] = value
        for name, value in self.constant_data.variables.items():
            if name.startswith("obs_") and value.dims == ("obs",):
                field = name.removeprefix("obs_")
                public = (
                    f"obs_{field}_index" if field in {"person", "unit", "pair", "batch"} else name
                )
                result[public] = value
        axes = {dim for variable in result.values() for dim in variable.dims}
        return xr.Dataset(result, coords={dim: self.constant_data.variables[dim] for dim in axes})

    def annotate_coordinates(self, dataset: xr.Dataset, *, require: bool = False) -> xr.Dataset:
        """Attach prepared identities using explicit dimension labels."""
        return self._attach_coordinates(dataset, require=require)

    def _attach_coordinates(self, dataset, *, positions=None, require=False):
        identities = self.output_coords
        names = [name for name in identities if identities.variables[name].dims[0] in dataset.dims]
        if require:
            for name in names:
                if name not in dataset.coords:
                    raise ValueError(f"Missing prepared parameter identity coordinate: {name}")
        selected = identities[names]
        if positions is None:
            if missing := selected.sizes.keys() - dataset.coords.keys():
                raise ValueError(
                    f"Result labels must match the frozen design: missing {sorted(missing)}"
                )
            try:
                selected = selected.sel({dim: dataset[dim].values for dim in selected.dims})
            except KeyError as error:
                raise ValueError(
                    f"Result labels must match the frozen design: {error.args[0]}"
                ) from None
        elif positions:
            selected = selected.isel(
                {dim: positions[dim] for dim in positions if dim in selected.dims}
            )
        coordinates = {name: selected.variables[name] for name in names}
        for name, expected in coordinates.items():
            if name in dataset.coords and not dataset.variables[name].equals(expected):
                raise ValueError(
                    f"{name}: coordinate conflicts with the prepared parameter contract"
                )
        return dataset.assign_coords(coordinates)

    @cached_property
    def person_keys(self) -> tuple[str, ...]:
        """Stable typed identities for the recursive scoring evidence history."""
        return tuple(_person_key(label) for label in self.coords["person"])

    @cached_property
    def observed_person_keys(self) -> frozenset[str]:
        """People contributing actual likelihood factors in this prepared batch."""
        if "obs_person" not in self.constant_data or self.data.attrs["purpose"] != "observed":
            return frozenset()
        return frozenset(
            self.person_keys[position] for position in np.unique(self.constant_data.obs_person)
        )

    @cached_property
    def point_layouts(self) -> Mapping[str, ParameterLayout]:
        """Canonical final parameters for posterior point extraction and simulation truth."""
        return MappingProxyType(
            {layout.kind: layout for layout in self.parameters if layout.point_parameter}
        )

    @cached_property
    def shared_parameters(self) -> tuple[ParameterLayout, ...]:
        """Calibration entries with identity independent of the current people."""
        return tuple(
            layout
            for layout in self.parameters
            if layout.fixed.size and not (layout.entry_data.person_index.values >= 0).any()
        )

    @cached_property
    def scoring_batch_identity(self) -> str:
        """Identify the prepared likelihood, person inputs and fixed calibration."""
        batches = [
            (
                batch.unit_index,
                self.coords[batch.obs_dim],
                batch.person,
                batch.statement,
                batch.trait,
                # Keep the implicit positive direction in the existing batch-v2 digest.
                np.broadcast_to(1.0, batch.person.shape),
                batch.category_order,
                batch.administration_index,
                tuple(getattr(batch.events, field.name) for field in fields(EventLayout)),
                batch.y,
            )
            for batch in self.batches
        ]
        return _identity_digest(
            (
                "batch-v2",
                self.calibration_identity,
                self.person_keys,
                self.population.person_group,
                self.population.predictors,
                batches,
                self.fixed_parameters,
            )
        )

    @cached_property
    def calibration_identity(self) -> str:
        """Identify the frozen statistical design independently of the current responses."""
        design, categories = self._measurement_catalog
        catalog = dict(design.variables)
        catalog.update(
            {
                name: self.constant_data.variables[name]
                for name in ("unit_id", "item_statement", "item_location_side")
                if name in self.constant_data
            }
        )
        specifications = [
            (
                type(spec).__name__,
                {
                    field.name: getattr(spec, field.name)
                    for field in fields(spec)
                    if field.name not in {"priors", "statement_effect_parameterization"}
                },
            )
            for spec in self.measurement_specs
        ]
        recipe = self.constant_data.attrs["population"]["recipe"]
        population = {
            name: recipe[name]
            for name in (
                "group",
                "groups",
                "reference_group",
                "regression_by_group",
                "theta_residual_scale",
            )
        }
        predictors = []
        for regression in self.regressions:
            design = regression.design
            encoding = {
                name: design.attrs[name]
                for name in (
                    "formula",
                    "id_columns",
                    "feature_records",
                    "center_predictors",
                    "standardize",
                    "include_intercept",
                    "compiler_version",
                )
            }
            if regression.targets[0].kind != "theta":
                encoding["rows"] = design[
                    ["X", *(f"key_{i}" for i in range(len(encoding["id_columns"])))]
                ]
                targets = []
                for target in regression.targets:
                    block = target.block
                    shape = block.shape[:-1] if block.ordered else block.shape
                    # Hash the logical layout so packing a runtime binding preserves saved identity.
                    indices = np.full((2, *shape), -1, dtype=np.int64)
                    indices.reshape(2, -1)[:, target.local_position] = (
                        target.row_index,
                        target.pool_index,
                    )
                    targets.append(
                        (
                            target.kind,
                            target.prior_role,
                            block.entry_index,
                            *indices,
                            target.noncentered,
                        )
                    )
                encoding["targets"] = targets
            predictors.append((design.term.values, encoding))
        entry_fields = [
            "unit_index",
            "block_index",
            "statement_index",
            "item_index",
            "trait_index",
            "trait_aux_index",
            "pair_index",
            "category_index",
            "cutpoint_position",
            "population_group_index",
            "term",
            "target_kind",
            "scale",
            "pool_index",
            "fixed_value",
            "shared",
            "ordered_group_index",
            "lower_kind",
            "lower_entry_index",
            "upper_limit",
            "point_parameter",
            "fixed_for_scoring",
        ]
        parameters = {
            layout.kind: (
                {dim: self.coords[dim] for dim in layout.dims},
                layout.entry_data[entry_fields],
            )
            for layout in self.shared_parameters
        }
        uses = self.parameter_uses
        if "kind" in uses:
            uses = uses.isel(parameter_use=uses.kind.isin(list(parameters)))
        return _identity_digest(
            (
                "calibration-v2",
                catalog,
                categories,
                specifications,
                [unit["scale"] for unit in self.constant_data.attrs["units"]],
                population,
                predictors,
                parameters,
                uses,
            )
        )

    @cached_property
    def _measurement_catalog(self):
        """Freeze the measurement variables used by identity and publication."""
        constant = self.constant_data
        names = [
            name
            for name, value in constant.variables.items()
            if (
                name.startswith("unit_")
                and value.dims == ("unit",)
                or name
                in {
                    "statement_label",
                    "statement_channel",
                    "statement_trait",
                    "statement_keying",
                    "statement_location_side",
                }
                or name.startswith("pair_")
                and value.dims == ("pair",)
            )
            and name != "unit_id"
        ]
        names.extend(
            name for name in ("fc_statement", "fc_size", "item_q") if name in constant.variables
        )
        design = xr.Dataset(
            {name: constant.variables[name] for name in names}, attrs={"parameter_contract": 1}
        )
        entity_dims = (
            "trait",
            "trait_aux",
            "block",
            "item",
            "statement",
            "unit",
            "population_group",
        )
        design = design.assign_coords(
            {dim: constant[dim].variable for dim in entity_dims if dim in constant.coords}
        )
        design = design.set_coords([name for name in names if name in constant.coords])
        # Public entity domains stay separate; the internal kernel axis also has items.
        design = design.isel(statement=np.flatnonzero(constant.statement_channel.values == "fc"))
        units = constant.attrs["units"]
        for field in ("kind", "higher_prefers"):
            design[f"unit_scale_{field}"] = (
                "unit",
                np.asarray([unit["scale"][field] for unit in units], dtype=str),
            )
        categories = [
            (index, position, value)
            for index, unit in enumerate(units)
            for position, value in enumerate(unit["scale"]["categories"])
        ]
        scalar = _scalar_categories([row[2] for row in categories], "category_entry")
        for position, field in enumerate(("unit_index", "position")):
            scalar[field] = (
                "category_entry",
                np.asarray([row[position] for row in categories], dtype=np.int64),
            )
        return design, scalar

    @cached_property
    def result_metadata(self):
        """Publish the current parameter groups alongside the frozen response design."""
        result = {
            "parameter_uses": self.parameter_uses,
            **{
                f"parameter_data/{layout.kind}": layout.entry_data.copy(deep=False)
                for layout in self.parameters
            },
        }
        if self.fixed_parameters is not None:
            result["fixed_parameters"] = self.annotate_coordinates(self.fixed_parameters)
        for dataset in result.values():
            freeze_arrays(dataset)
        return self.response_metadata | result

    @cached_property
    def response_metadata(self):
        if self._response_projection is not None:
            return self._response_projection
        constant = self.constant_data
        design, categories = self._measurement_catalog
        observations = {
            name: value
            for name, value in constant.variables.items()
            if name == "person_group"
            or name.startswith("obs_")
            and value.dims == ("obs",)
            or name.startswith("observation_source_")
        }
        design = design.assign(observations).assign_coords(
            person=constant.person.variable, obs=constant.obs.variable
        )
        design = design.set_coords([name for name in observations if name in constant.coords])
        result = {"design": design, "categories": categories.copy(deep=False)}
        models = constant.unit_model.values.tolist()
        for batch_index, batch in enumerate(self.batches):
            events, dimension = batch.events, batch.category_dim
            pairs = batch.projected_pairs
            if events.orders is None:
                support = _scalar_categories(events.labels, dimension)
            else:
                component = pairs.dims[0] if pairs is not None else f"{batch.name}_component"
                pattern = [
                    [-1 if value is None else value for value in row] for row in events.labels
                ]
                support = xr.Dataset(
                    {"pattern": ((dimension, component), np.asarray(pattern, dtype=np.int64))},
                    coords={dimension: constant[dimension]},
                )
                if pairs is not None:
                    support["pairs"] = pairs
                support.attrs["pattern_kind"] = (
                    "ranking_positions" if pairs is None else "comparison_signature"
                )
            unit = constant.attrs["units"][batch.unit_index]
            kind = unit["scale"]["kind"]
            support.attrs.update(
                batch_index=batch_index,
                model=models[batch.unit_index],
                response_format=events.response_format or "",
                response_kind=kind,
                likelihood_kind=batch.likelihood_kind,
                response_name=batch.response_name,
                obs_dim=batch.obs_dim,
                category_dim=dimension,
            )
            result[f"response_support/{batch.name}"] = support
        for dataset in result.values():
            freeze_arrays(dataset)
        return result

    @cached_property
    def batches(self) -> tuple[PreparedBatch, ...]:
        """Native batches in global observation order."""
        batches = []
        constant = self.constant_data
        keying, people, pairs, administrations = (
            self._array(name)
            for name in ("statement_keying", "obs_person", "obs_pair", "obs_input_event_index")
        )
        for batch in constant.attrs["batches"]:
            events = batch["events"]
            event = EventLayout(
                **{
                    key: value if key in {"response_format", "labels"} else self._array(value)
                    for key, value in events.items()
                    if key in EventLayout.__dataclass_fields__
                }
            )
            obs = batch["obs_slice"]
            unit_index = batch["unit_index"]
            value = PreparedBatch(
                likelihood_kind=constant.attrs["units"][unit_index]["likelihood_kind"],
                obs_slice=obs,
                person=people[obs],
                pair_index=pairs[obs],
                administration_index=administrations[obs],
                projected_pairs=constant.variables.get(events.get("pairs")),
                statement=self._array(batch["statement"]),
                trait=self._array(batch["trait"]),
                category_order=self._array(batch["category_order"]),
                name=batch["name"],
                response_name=batch["response_name"],
                unit_index=unit_index,
                specification=self.measurement_specs[unit_index],
                events=event,
                obs_dim=batch["obs_dim"],
                category_dim=batch["category_dim"],
                statement_keying=keying,
                y=(
                    self.observed_data.variables[batch["response_name"]].values
                    if self.data.attrs["purpose"] == "observed"
                    else None
                ),
                slots=tuple(
                    SlotRef(
                        key=slot["key"],
                        kind=slot["kind"],
                        entry_index=self._array(slot["entry_index"]),
                    )
                    for slot in batch["slots"]
                ),
            )
            batches.append(value)
        return tuple(batches)

    @cached_property
    def batches_by_unit(self) -> tuple[tuple[PreparedBatch, ...], ...]:
        """Group the existing batches in measurement and observation order."""
        by_unit = [[] for _ in range(self.constant_data.sizes["unit"])]
        for batch in self.batches:
            by_unit[batch.unit_index].append(batch)
        return tuple(map(tuple, by_unit))

    @cached_property
    def measurement_specs(self):
        """Declared measurement laws in unit order, including unanswered units."""
        return tuple(scope["specification"] for scope in self.constant_data.attrs["measurements"])

    @cached_property
    def measurement_overview(self):
        """Return unit size, declared link and both support counts in unit order."""
        constant = self.constant_data
        rows = []
        for unit, spec, batches in zip(
            constant.attrs["units"], self.measurement_specs, self.batches_by_unit, strict=True
        ):
            declared = response_category_count(unit)
            nominal = 2 if spec.comparison_based and unit["response_format"] else declared
            actual = sorted({len(batch.events.labels) for batch in batches})
            if declared is None or not actual:
                categories = nominal
            else:
                categories = actual[0] if len(actual) == 1 else tuple(actual)
            link = getattr(spec, "link", spec.response_link)
            multiplier = spec.link_multipliers.get(link)
            if link in {"probit", "logit"} and len(unit["scale"]["categories"]) > 2:
                link = f"ordered_{link}"
            rows.append((unit["size"], link, multiplier, categories or pd.NA, declared or pd.NA))
        return tuple(rows)

    @cached_property
    def regressions(self):
        bindings = {block.binding_id: block for block in self.blocks}
        plans = []
        for scope in self.constant_data.attrs["regression_scopes"]:
            theta = scope["target"] == "theta"
            blocks = (
                (bindings[scope["blocks"][0]["binding_id"]],)
                if theta
                else self._blocks_by_scope[scope["name"]]
            )
            positions = self._array(scope["coefficient_positions"])
            coefficient = blocks[0]
            entries = coefficient.entry_index.ravel()[positions]
            entries.flags.writeable = False
            targets = []
            for record in scope["targets"]:
                indices = record["indices"]
                rows, pools = self._array(indices)
                binding = record["binding_id"]
                targets.append(
                    RegressionTarget(
                        block=None if theta else bindings[binding],
                        prior_role=record["prior_role"],
                        row_index=rows,
                        pool_index=pools,
                        conditional=record["conditional"],
                        noncentered=record["noncentered"],
                        local_position=(
                            None if theta else self._array(self.constant_data[indices].dims[-1])
                        ),
                        unit_indices=tuple(sorted({unit for unit, _ in record["unit_pools"]})),
                    )
                )
            design = self.data[scope["path"]].to_dataset()
            plans.append(
                PreparedRegression(
                    name=scope["name"],
                    design=design,
                    blocks=blocks,
                    priors=design.attrs["resolved_priors"],
                    coefficient_positions=positions,
                    coefficient_entries=entries,
                    targets=tuple(targets),
                )
            )
        return tuple(plans)

    @cached_property
    def theta_regression(self):
        name = self.constant_data.attrs["population"]["recipe"]["regression"]
        return next((plan for plan in self.regressions if plan.name == name), None)

    @cached_property
    def population(self):
        scope = self.constant_data.attrs["population"]
        regression = self.theta_regression
        return PopulationLayout(
            specification=scope["specification"],
            blocks=self._blocks_by_scope[scope["name"]],
            recipe=scope["recipe"],
            theta_parameterization=scope["theta_parameterization"],
            person_group=self._array("person_group"),
            predictors=(
                self._array("predictors") if regression is None else regression.design.X.values
            ),
            centered_index=self._array("centered_index"),
            noncentered_index=self._array("noncentered_index"),
        )


def _align_point_axes(constant, point, shapes, *, sample_dims=(), exact=False):
    """Check each shared physical axis once per source, aligning explicit points only."""
    axes, reorder = {}, {}
    for name, dims in shapes.items():
        actual, expected = point[name].dims, (*sample_dims, *dims)
        if actual != expected:
            can_transpose = not exact and len(actual) == len(dims) and set(actual) == set(dims)
            if not can_transpose:
                raise ValueError(f"{name}: expected final dimensions {expected}, received {actual}")
        axes.update(dict.fromkeys(dims, name))
    for dim, name in axes.items():
        actual, expected = point.get_index(dim), constant.get_index(dim)
        if dim not in point.coords or not actual.is_unique or len(actual) != len(expected):
            raise ValueError(f"{name}: final coordinates for {dim!r} must match the frozen design")
        if not exact:
            identifiers(actual, f"{name}.{dim}")
        if not actual.equals(expected):
            if exact or np.any((positions := actual.get_indexer(expected)) < 0):
                raise ValueError(f"{name}: final labels must match the frozen design")
            reorder[dim] = positions
    return point.isel(reorder) if reorder else point


def _align_ability(constant, theta):
    if not isinstance(theta, xr.DataArray):
        raise TypeError("theta must be a labelled DataArray")
    dims = ("person", "trait")
    theta = _align_point_axes(constant, theta.to_dataset(name="theta"), {"theta": dims}).theta
    if theta.dtype.kind not in "iuf" or not np.isfinite(theta.values).all():
        raise ValueError("theta: fixed values must be finite real numbers")
    return theta.transpose(*dims)


def _validate_parameters(
    prepared: PreparedInputs,
    point: xr.Dataset,
    *,
    scoring: bool = False,
    fixed_parameters: xr.Dataset | None = None,
    include_theta: bool = False,
    theta: xr.DataArray | None = None,
) -> xr.Dataset:
    """Validate one final parameter point against the prepared axes and constraints."""
    if not isinstance(point, xr.Dataset) or {"chain", "draw"}.intersection(point.dims):
        raise TypeError("parameters must be a final-parameter Dataset without chain/draw")
    constant = prepared.constant_data
    required = {
        layout.kind: layout
        for layout in prepared.parameters
        if not scoring or layout.fixed_for_scoring
    }
    if scoring:
        needed = {
            name
            for name, layout in required.items()
            if layout.entry_data.fixed_for_scoring.values.ravel()[layout.free_flat].any()
        }
        if missing := needed - point.data_vars.keys():
            raise ValueError(f"Missing final fixed parameters: {sorted(missing)}")
    values, shapes = {}, {}
    if include_theta:
        if theta is None:
            if "theta" not in point:
                raise ValueError(
                    "Provide theta(person, trait), or include final theta in parameters"
                )
            shapes["theta"] = ("person", "trait")
        else:
            values["theta"] = _align_ability(constant, theta)
    shapes.update({name: layout.dims for name, layout in required.items() if name in point})
    aligned = _align_point_axes(constant, point, shapes)
    for name, dims in shapes.items():
        value = aligned[name].transpose(*dims)
        layout = required.get(name)
        if value.dtype.kind not in "iuf" or not np.isfinite(value.values).all():
            raise ValueError(f"{name}: fixed values must be finite real numbers")
        if (
            layout is not None
            and len(layout.fixed_flat)
            and not fixed_values_match(value.values, layout.fixed, layout.fixed_flat)
        ):
            raise ValueError(f"{name}: fixed values conflict with structural constraints")
        values[name] = value
    result = xr.Dataset(values, attrs=dict(point.attrs))
    if prepared.output_coords.keys() & result.coords.keys():
        # Verify supplied identities before discarding auxiliary coordinates.
        prepared._attach_coordinates(result, positions={})
    return _constrain_parameter_point(result, required, fixed_parameters)


def _constrain_parameter_point(point, layouts, fixed_parameters):
    """Check final values after their parameter axes and fixed entries are verified."""
    result = point.reset_coords(drop=True)
    if fixed_parameters is not None:
        result = xr.merge(
            (result, fixed_parameters), compat="equals", join="exact", combine_attrs="override"
        )
    _validate_parameter_support(point, layouts, fixed=fixed_parameters)
    return result


def _validate_parameter_support(parameters, layouts, *, fixed=None):
    """Check prepared constraints on aligned final values, per point or sample state."""
    checked = {}
    for name, layout in layouts.items():
        if name in parameters:
            values = parameters[name].values
        elif fixed is not None and name in fixed:
            values = fixed[name].values
        elif not len(layout.free_flat):
            values = layout.fixed
        else:
            continue
        leading = values.shape[: values.ndim - len(layout.shape)]
        checked[name] = values.reshape((*leading, -1))

    for name, values in checked.items():
        layout = layouts[name]
        entries = layout.entry_data
        if name in parameters:
            domain = entries.domain.values
            positive, unit = domain == "positive", domain == "unit"
            if positive.any() and np.any(values[..., positive] <= 0):
                raise ValueError(f"{name}: values must be positive")
            if unit.any():
                probabilities = values[..., unit]
                if np.any((probabilities < 0) | (probabilities > 1)):
                    raise ValueError(f"{name}: values must lie inside [0, 1]")
            sides = entries.orientation.values
            above = entries.strict_positive.values & (sides > 0)
            below = entries.strict_negative.values & (sides < 0)
            if (above.any() and np.any(values[..., above] < 0)) or (
                below.any() and np.any(values[..., below] > 0)
            ):
                raise ValueError(f"{name}: values violate frozen location-side constraints")
            if name == "covariance":
                covariance = values.reshape((*values.shape[:-1], *layout.shape))
                eps = 8 * np.finfo(values.dtype).eps if values.dtype.kind == "f" else 0
                if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=eps, atol=eps):
                    raise ValueError("covariance: covariance must be symmetric")
                chol = jnp.linalg.cholesky(covariance, symmetrize_input=False)
                if not bool(jnp.isfinite(chol).all()):
                    raise ValueError("covariance: covariance must be positive definite")
        groups = entries.ordered_group_index.values
        # The catalogue packs each ordered group contiguously in its native order.
        consecutive = np.flatnonzero((groups[:-1] >= 0) & (groups[:-1] == groups[1:]))
        if len(consecutive) and np.any(values[..., consecutive + 1] <= values[..., consecutive]):
            raise ValueError(f"{name}: cutpoints must be strictly increasing")

        upper = entries.upper_limit.values
        bounded = ~np.isnan(upper)
        if bounded.any() and np.any(values[..., bounded] > upper[bounded]):
            raise ValueError(f"{name}: values violate declared bounds")
        lower_kinds = entries.lower_kind.values
        lower_indices = entries.lower_entry_index.values
        for lower in set(lower_kinds) & checked.keys():
            linked = lower_kinds == lower
            if np.any(checked[lower][..., lower_indices[linked]] >= values[..., linked]):
                raise ValueError(f"{name}: lower asymptotes must be below upper asymptotes")


def _labelled_administration(value, channels):
    if value is None:
        return {}
    if isinstance(value, pd.DataFrame):
        if "channel" in value:
            if set(value.channel) - set(channels):
                raise ValueError("administration contains an unknown channel")
            return {
                name: rows.drop(columns="channel")
                for name, rows in value.groupby("channel", sort=False)
            }
        if len(channels) != 1:
            raise ValueError("Mixed administration requires a channel column or fc/items mapping")
        return {channels[0]: value}
    if not isinstance(value, Mapping) or set(value) - set(channels):
        raise TypeError("administration must be a Dataset, DataFrame or fc/items table mapping")
    return value


def _new_channels(reference, responses, administration):
    constant, units = reference.constant_data, reference._units_by_channel
    names = tuple(units)
    if isinstance(responses, ResponseTable):
        if len(names) != 1:
            raise ValueError("Mixed new responses require an fc/items mapping")
        responses = {names[0]: responses}
    if not isinstance(responses, Mapping) or set(responses) - set(names):
        raise TypeError("new responses must be ResponseTable or an fc/items mapping")
    if any(not isinstance(value, ResponseTable) for value in responses.values()):
        raise TypeError("Each response channel must be a ResponseTable")
    indexed = isinstance(administration, xr.Dataset)
    if indexed:
        rows, people, _ = normalize_indexed_administration(
            administration, constant, purpose="observed"
        )
        # Labels match raw answers. The validated positions stay attached to each row.
        rows = rows.assign(
            person_id=np.asarray(people, dtype=object)[rows.person],
            unit_id=constant.unit_id.values[rows.unit],
        )
        channels = constant.unit_channel.values[rows.unit]
        administration = {name: rows.loc[channels == name] for name in names}
    else:
        administration = {
            name: normalize_administration(
                rows,
                units[name],
                unit_column=UNIT_COLUMNS[name],
                allowed_statuses=OBSERVED_STATUSES,
            )
            for name, rows in _labelled_administration(administration, names).items()
        }
        people = ()
    people = dict.fromkeys(people)
    channels = []
    for name in names:
        fc = name == "fc"
        source = responses.get(name)
        input_format = None if source is None else source.format
        order = (
            frozen_statement_order(constant) if fc and has_statement_rows(input_format) else None
        )
        rows, admin, channel_people = normalize_responses(
            source,
            units[name],
            unit_column=UNIT_COLUMNS[name],
            purpose="observed",
            order=order,
            administration=administration.get(name),
        )
        people.update(dict.fromkeys(channel_people))
        channels.append(
            NormalizedChannel(
                channel=name,
                tables=dict(responses=rows, administration=admin),
                input_format=input_format,
                people=channel_people,
            )
        )
    if not people:
        raise ValueError("New responses or administration must identify at least one person")
    return channels, tuple(people), units.keys() - responses.keys()


def _population_inputs(reference: PreparedInputs, people, *, person_data=None, administration=None):
    """Align replacement population inputs, or return None to reuse the frozen rows."""
    from pyfcirt.covariates.design import apply_design
    from pyfcirt.data.grouping import apply_groups

    constant = reference.constant_data
    recipe = constant.attrs["population"]["recipe"]
    name = recipe["regression"]
    design = None if name is None else reference.data[f"covariates/{name}"].dataset
    if (
        person_data is None
        and isinstance(administration, xr.Dataset)
        and "predictors" in administration
    ):
        if administration.attrs.get("population_recipe") != recipe:
            raise ValueError("The generation schedule must use the frozen population recipe")
        if tuple(administration.person.values) != tuple(people):
            raise ValueError("Schedule person order must match the new response people")
        predictors, groups = administration.predictors, administration.person_group
        if (
            predictors.dims != ("person", "theta_predictor")
            or not predictors.theta_predictor.equals(constant.theta_predictor)
            or groups.dims != ("person",)
            or not np.issubdtype(groups.dtype, np.integer)
            or not np.issubdtype(predictors.dtype, np.number)
            or not np.isrealobj(predictors.values)
            or not np.isfinite(predictors.values).all()
            or np.any((groups.values < 0) | (groups.values >= constant.sizes["population_group"]))
        ):
            raise ValueError("The generation schedule has invalid frozen population arrays")
        return predictors.values.copy(), groups.values.copy()
    if person_data is None and tuple(people) == tuple(constant.person.values):
        return None
    grouped = recipe["group"] is not None
    if not grouped and design is None:
        return np.empty((len(people), 0)), np.zeros(len(people), dtype=np.int64)
    if grouped and person_data is None:
        raise ValueError("person_data is required for the population group column")
    column = "person_id" if design is None else design.attrs["id_columns"][0]
    keys = pd.DataFrame({column: pd.Series(people, dtype=object)})
    positions, identities = align_rows(person_data, keys, (column,), "person_data")
    groups = (
        apply_groups(recipe, person_data, positions, identities)
        if grouped
        else np.zeros(len(people), dtype=np.int64)
    )
    predictors = (
        np.empty((len(people), 0))
        if design is None
        else apply_design(design, person_data, positions, identities).values
    )
    return predictors, groups


def _replace_people(
    reference: PreparedInputs, people, *, person_data=None, administration=None
) -> dict[str, xr.Dataset]:
    """Bind a replacement person axis in the candidate design groups."""
    datasets = reference._design_groups()
    population_inputs = _population_inputs(
        reference, people, person_data=person_data, administration=administration
    )
    if population_inputs is None:
        return datasets
    predictors, person_groups = population_inputs
    constant = datasets["/constant_data"]
    resize = len(people) != constant.sizes["person"]
    drop = {"person"}
    person_blocks = []
    scopes = constant.attrs["measurements"] if resize else ()
    for scope_index, scope in enumerate(scopes):
        for index, layout in enumerate(scope["blocks"]):
            if "person" in layout["dims"]:
                bindings = [slot["binding_id"] for slot in scope["slots"]]
                slot_index = bindings.index(layout["binding_id"])
                positions = scope["slots"][slot_index]["positions"]
                person_blocks.append((scope_index, index, slot_index, layout))
                for name in (layout["free_flat"], layout["fixed_flat"], positions):
                    drop.update(constant[name].dims)
    labels = np.asarray(people, dtype=object)
    result = (constant.drop_dims(drop) if resize else constant).assign_coords(person=labels)
    attrs = result.attrs
    if person_blocks:
        # Only measurement scopes own person-local blocks and their mutable gathers.
        attrs["measurements"] = deepcopy(attrs["measurements"])
    theta_regression = attrs["population"]["recipe"]["regression"]
    if theta_regression is None:
        result["predictors"] = ("person", "theta_predictor"), predictors
    else:
        path = f"/covariates/{theta_regression}"
        design = datasets[path].drop_dims("row")
        design.coords["row"] = np.arange(len(people), dtype=np.int64)
        design["X"] = ("row", "term"), predictors
        design["key_0"] = "row", labels
        datasets[path] = design
        scope = next(
            scope for scope in attrs["regression_scopes"] if scope["name"] == theta_regression
        )
        target = scope["targets"][0]
        rows, pools = theta_target_indices(
            person_groups,
            result.sizes["trait"],
            by_group=attrs["population"]["recipe"]["regression_by_group"],
        )
        name = target["indices"]
        result[name] = ("regression_target_field", "person", "trait"), np.stack((rows, pools))
    result["person_group"] = "person", person_groups
    if person_blocks:
        arrays = LayoutBuilder(coords={}, attrs=attrs)
        arrays.update(result)
        with PriorPreparation():
            for scope_index, index, slot_index, old in person_blocks:
                scope = attrs["measurements"][scope_index]
                slot = scope["slots"][slot_index]
                layout = _layout(
                    arrays,
                    scope["name"],
                    old["key"],
                    tuple(old["dims"]),
                    old["prior_role"],
                    fixed_for_scoring=old["fixed_for_scoring"],
                    domain=old["domain"],
                    kind=old["kind"],
                    component=old["component"],
                    summary_default=old["summary_default"],
                )
                # The nuisance law stays frozen; only its independent person batch changes.
                freeze_prior_shapes(arrays, layout, scope["specification"].priors)
                scope["blocks"][index] = layout
                slot["binding_id"] = layout["binding_id"]
                arrays[slot["positions"]] = "person", np.arange(len(people), dtype=np.int64)
        result = finalize_parameters(
            arrays.build(), changed={block["kind"] for _, _, _, block in person_blocks}
        )
    datasets["/constant_data"] = result
    return datasets


def _calibration_values(prepared, parameters, *, partial=False, existing=None):
    """Validate and detach calibration values, optionally retaining a fixed snapshot."""
    if existing is not None:
        names = {layout.kind for layout in prepared.parameters if layout.fixed_for_scoring}
        existing = existing[list(names & existing.data_vars.keys())]
    point = _validate_parameters(
        prepared, parameters, scoring=not partial, fixed_parameters=existing
    )
    if partial:
        # A partly updated final node can require an actual native RV as a constant.
        native = parameters.drop_vars(point.data_vars.keys(), errors="ignore")
        if any(not np.isfinite(value.values).all() for value in native.data_vars.values()):
            raise ValueError("Fixed native calibration values must be finite")
        point = xr.merge((point, native), compat="equals", join="exact")
    return point.copy(deep=True)


def _prepare_scoring(
    reference: PreparedInputs,
    new_responses: pd.DataFrame | ResponseTable | Mapping[str, ResponseTable],
    *,
    format: str | None = None,
    fixed_parameters: xr.Dataset | None = None,
    partial: bool = False,
    excluded_people: frozenset[str] = frozenset(),
    person_data=None,
    administration=None,
) -> PreparedInputs:
    """Prepare an independent scoring analysis against the exact frozen measurement design."""
    if isinstance(new_responses, pd.DataFrame):
        if format is None:
            raise TypeError("format is required for a new response DataFrame")
        new_responses = ResponseTable(new_responses, format=format)
    elif format is not None:
        raise TypeError("format applies to a new response DataFrame")
    channels, people, omitted = _new_channels(reference, new_responses, administration)
    if excluded_people and any(_person_key(label) in excluded_people for label in people):
        raise ValueError("Posterior scoring requires respondents absent from prior evidence")
    if fixed_parameters is not None and (
        partial or fixed_parameters is not reference.fixed_parameters
    ):
        fixed_parameters = _calibration_values(reference, fixed_parameters, partial=partial)
    groups = _replace_people(
        reference, people, person_data=person_data, administration=administration
    )
    if fixed_parameters is not None:
        groups["/fixed_parameters"] = fixed_parameters
    rows = index_response_rows(people, reference._units_by_channel, channels, omitted=omitted)
    return PreparedInputs._from_rows(groups, rows, purpose="observed")


def generation_schedule(model: Model, person_ids: Sequence, *, person_data=None) -> xr.Dataset:
    """Return complete unobserved events and frozen population inputs for new people.

    ``schedule_person`` and ``schedule_unit`` index the labelled coordinates.
    Comparison events additionally carry global statement indices; other events use -1.
    Status is scheduled throughout, and the dataset contains no answer values.
    Its mutable coordinates, population inputs and recipe are detached from the model.
    """
    from pyfcirt.common.validation import identifier

    if isinstance(person_ids, (str, bytes)):
        raise TypeError("person_ids must be a sequence of string or integer labels")
    people = tuple(identifier(person, "person_ids") for person in person_ids)
    if not people or len(set(people)) != len(people):
        raise ValueError("person_ids must be nonempty and unique")
    constant = model.prepared.constant_data
    recipe = constant.attrs["population"]["recipe"]
    population = _population_inputs(model.prepared, people, person_data=person_data)
    if population is None:
        frozen = model.prepared.population
        population = frozen.predictors, frozen.person_group
    predictors, groups = population
    pair_units = constant.pair_unit.values
    pairs = np.column_stack(
        (pair_units, constant.pair_statement_1.values, constant.pair_statement_2.values)
    )
    whole_units = np.flatnonzero(~np.isin(constant.unit.values, pair_units))
    whole = np.column_stack((whole_units, np.full((len(whole_units), 2), -1, dtype=np.int64)))
    events = np.concatenate((pairs, whole))
    events = events[np.argsort(events[:, 0], kind="stable")]
    schedule = xr.Dataset(
        coords={
            "person": np.asarray(people, dtype=object),
            **{
                name: constant.variables[name].copy(deep=True)
                for name in ("unit", "unit_id", "unit_channel", "theta_predictor")
            },
        },
        attrs=dict(
            schema="pyfcirt-generation-schedule",
            schema_version=1,
            population_recipe=deepcopy(recipe),
        ),
    )
    return schedule.assign(
        schedule_person=(
            "schedule",
            np.repeat(np.arange(len(people), dtype=np.int64), len(events)),
        ),
        **{
            f"schedule_{key}": ("schedule", np.tile(values, len(people)))
            for key, values in zip(("unit", "statement_1", "statement_2"), events.T, strict=True)
        },
        schedule_status=("schedule", np.full(len(people) * len(events), "scheduled")),
        predictors=(("person", "theta_predictor"), predictors.copy()),
        person_group=("person", groups.copy()),
    )


def _prepare_automatic_generation(
    specification, *, n_persons, n_traits, n_blocks, block_size, response_format, seed
):
    """Generate ordinary FC declarations and prepare their unobserved response rows."""
    counts = (n_persons, n_traits, n_blocks, block_size)
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1
        for value in counts
    ):
        raise ValueError("n_persons, n_traits, n_blocks and block_size must be positive integers")
    if not 2 <= block_size <= n_traits <= n_blocks * block_size:
        raise ValueError("Require 2 <= block_size <= n_traits <= n_blocks * block_size")

    spec = _model_declaration("MGPPM" if specification is None else specification, channel="fc")
    rng = np.random.default_rng(seed)
    usage = np.zeros(n_traits, dtype=np.int64)
    members = np.empty((n_blocks, block_size), dtype=np.int64)
    for row in members:
        # Use the least represented traits, with independent tie and position randomization.
        row[:] = np.lexsort((rng.random(n_traits), usage))[:block_size]
        rng.shuffle(row)
        usage[row] += 1

    blocks = [f"B{i + 1}" for i in range(n_blocks)]
    statements = np.array([f"S{i + 1}" for i in range(n_blocks * block_size)]).reshape(
        members.shape
    )
    people = pd.DataFrame({"person_id": [f"P{i + 1}" for i in range(n_persons)]})
    design = pd.DataFrame(
        {
            "block_id": np.repeat(blocks, block_size),
            "position": np.tile(np.arange(1, block_size + 1), n_blocks),
            "statement_id": statements.ravel(),
            "trait": [f"T{i + 1}" for i in members.ravel()],
            "keying": 1,
        }
    )
    if any(parameter.key == "location" for parameter in spec.parameter_declarations()):
        design["location_side"] = 1
    if n_traits == block_size == 2 and spec.loading_identification in {"automatic", "required"}:
        anchors = spec.loading_anchors
        if not any(all(label in anchors for label in row) for row in statements):
            design.loc[:1, "fix_loading"] = [anchors.get(label, 1.0) for label in statements[0]]

    if spec.comparison_based:
        first, second = np.triu_indices(block_size, k=1)
        events = pd.DataFrame(
            {
                "block_id": np.repeat(blocks, len(first)),
                "statement_1": statements[:, first].ravel(),
                "statement_2": statements[:, second].ravel(),
            }
        )
        storage = "comparison_schedule"
    else:
        events = pd.DataFrame({"block_id": blocks})
        storage = "block_schedule"
        if response_format is None:
            response_format = "rank"
    inputs = FCInput(
        design,
        ResponseTable(people.merge(events, how="cross"), format=storage),
        model=spec,
        response_format=response_format,
    )
    return PreparedInputs.from_inputs(inputs, purpose="generate")


def _simulation_parameter_point(prepared, parameters):
    """Label positional inputs with the same axes as an explicit simulation point."""
    if isinstance(parameters, xr.Dataset):
        return parameters
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a labelled Dataset or a parameter mapping")
    shapes = {name: layout.dims for name, layout in prepared.point_layouts.items()}
    shapes["theta"] = ("person", "trait")
    if unknown := parameters.keys() - shapes.keys():
        raise ValueError(f"Unknown final simulation parameters: {sorted(map(repr, unknown))}")
    values, coords = {}, {}
    labels = prepared.coords
    for name, value in parameters.items():
        dims = shapes[name]
        shape = tuple(len(labels[dim]) for dim in dims)
        value = np.asarray(value)
        if value.shape != shape:
            raise ValueError(f"{name}: expected shape {shape}, received {value.shape}")
        values[name] = dims, value
        coords.update((dim, labels[dim]) for dim in dims)
    return xr.Dataset(values, coords=coords)


def _prepare_generated(template: PreparedInputs, responses: xr.Dataset) -> PreparedInputs:
    """Encode the generated answers without fixing a future fit."""
    return PreparedInputs._from_rows(
        template._design_groups(), generated_rows(template, responses), purpose="observed"
    )


def _prepare_generation(
    reference: PreparedInputs,
    administration: xr.Dataset,
    *,
    fixed_parameters: xr.Dataset | None = None,
) -> PreparedInputs:
    """Normalize actual unobserved schedules against the already frozen design.

    Each unit keeps its native block, comparison or item event representation.
    Only person-dependent layouts and input indices change; no defaults, priors
    or identification constraints are resolved again, and no answers are invented.
    Carry any validated shared calibration point unchanged into the new state.
    """
    if not isinstance(administration, xr.Dataset) or (
        administration.attrs.get("schema"),
        administration.attrs.get("schema_version"),
    ) != ("pyfcirt-generation-schedule", 1):
        raise ValueError("administration must be a current generation_schedule Dataset")
    indexed, people, scheduled = normalize_indexed_administration(
        administration, reference.constant_data, purpose="generate"
    )
    groups = _replace_people(reference, people, administration=administration)
    if fixed_parameters is None:
        fixed_parameters = reference.fixed_parameters
    if fixed_parameters is not None:
        groups["/fixed_parameters"] = fixed_parameters
    return PreparedInputs._from_rows(groups, (indexed, scheduled), purpose="generate")
