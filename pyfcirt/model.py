"""Runtime preparation and one native graph, separate from persisted result arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Self
from uuid import uuid4

import numpy as np
import pymc as pm
import xarray as xr

from pyfcirt import simulation
from pyfcirt.analysis import checks, information, prediction, reliability, summary, traits
from pyfcirt.analysis.output import (
    POSTERIOR_DEPENDENT_GROUPS,
    finalize_samples,
    parameter_point,
    posterior_samples,
    prepare_outputs,
    replace_output_groups,
    validate_attached_samples,
)
from pyfcirt.data import prepare as preparation
from pyfcirt.data.design import (
    ComparisonColumn,
    FCDesignMapping,
    FCInput,
    ItemDesignMapping,
    ItemInput,
    ResponseTable,
)
from pyfcirt.data.prepare import PreparedInputs
from pyfcirt.models.assembly import ModelGraph
from pyfcirt.models.population import population_output_attrs
from pyfcirt.models.priors import PriorPreparation
from pyfcirt.models.specification import PopulationSpec
from pyfcirt.scoring import scoring

if TYPE_CHECKING:
    import pandas as pd
    from pytensor.tensor.variable import TensorVariable

    from pyfcirt.covariates.specification import Regression, RegressionOptions


_EVIDENCE_DIMS = {"people": "evidence_person", "batches": "evidence_batch"}


def _checked_evidence_ids(people, batches) -> tuple[frozenset[str], frozenset[str]]:
    for name, values in (("people", people), ("batches", batches)):
        if any(not isinstance(key, str) or not key for key in values):
            raise ValueError(f"calibration_evidence requires valid {name} identities")
    if bool(people) != bool(batches):
        raise ValueError("calibration_evidence must track both people and batches")
    return frozenset(people), frozenset(batches)


def _read_evidence(evidence: xr.DataTree) -> tuple[frozenset[str], frozenset[str]]:
    for name, dim in _EVIDENCE_DIMS.items():
        if name not in evidence or evidence[name].dims != (dim,):
            raise ValueError(f"calibration_evidence requires valid {name} identities")
    return _checked_evidence_ids(
        evidence["people"].values.tolist(), evidence["batches"].values.tolist()
    )


class Model:
    """Own runtime preparation and build its native graph only when requested.

    Register response tables and prepare them together once. Existing
    ``PreparedInputs`` are reused.
    A different design or new respondents belong to another Model.
    Posterior arrays and computed results live only in ``idata``. Graph objects
    and evaluation caches are runtime conveniences and are never serialized.
    """

    def __init__(
        self,
        prepared: PreparedInputs
        | FCInput
        | ItemInput
        | tuple[FCInput | ItemInput, ...]
        | None = None,
        *,
        marginal_correlation: bool | None = None,
        _approximation=None,
    ):
        self._pending_inputs: dict[type[FCInput] | type[ItemInput], FCInput | ItemInput] = {}
        self._pending_population: tuple[PopulationSpec, pd.DataFrame | None] | None = None
        self._person_data: pd.DataFrame | None = None
        self._prepared = None
        self._built = None
        self._calibration_prior = _approximation
        if prepared is not None:
            self.prepare(prepared, marginal_correlation=marginal_correlation)
        elif marginal_correlation is not None or _approximation is not None:
            raise ValueError("Pass prepared inputs with constructor options, or use prepare()")
        else:
            self._idata = xr.DataTree()

    @property
    def prepared(self) -> PreparedInputs:
        if self._prepared is None:
            raise ValueError("Prepare the model's data first")
        return self._prepared

    def prepare(
        self,
        inputs: PreparedInputs
        | FCInput
        | ItemInput
        | tuple[FCInput | ItemInput, ...]
        | None = None,
        *,
        marginal_correlation: bool | None = None,
        **preparation_options,
    ) -> Self:
        """Prepare registered tables or explicit inputs and return this model."""
        if self._prepared is not None:
            raise ValueError(
                "This model is already prepared; create a new Model for different data"
            )
        if inputs is None:
            if not self._pending_inputs:
                raise ValueError("Add response tables or pass input declarations to prepare()")
            inputs = tuple(self._pending_inputs.values())
        elif self._pending_inputs:
            raise ValueError("Call prepare() without inputs after adding response tables")
        if self._pending_population is not None:
            population, registered_people = self._pending_population
            if "population" in preparation_options or (
                registered_people is not None and "person_data" in preparation_options
            ):
                raise TypeError(
                    "Supply each population option to set_population() or prepare() once"
                )
            preparation_options["population"] = population
            if registered_people is not None:
                preparation_options["person_data"] = registered_people
        person_data = preparation_options.get("person_data")
        if isinstance(inputs, PreparedInputs):
            if preparation_options:
                raise TypeError("Pass preparation options with input declarations")
            prepared = inputs
        else:
            prepared = PreparedInputs.from_inputs(inputs, **preparation_options)
        if marginal_correlation is not None:
            from pyfcirt.data.regressions import prepare_regressions

            prepared = prepare_regressions(prepared, marginal_correlation=marginal_correlation)
        candidate = prepared.data
        groups = dict(prepared.result_metadata)
        if (approximation := self._calibration_prior) is not None:
            groups["calibration_evidence"] = xr.Dataset(
                {
                    name: (_EVIDENCE_DIMS[name], np.asarray(sorted(ids), dtype=str))
                    for name, ids in (
                        ("people", approximation.evidence_people),
                        ("batches", approximation.evidence_batches),
                    )
                },
                attrs={"incoming_approximation_id": approximation._fit_id},
            )
        idata = xr.DataTree.from_dict(groups)
        for name in ("calibration_uncertainty", "point_source"):
            if name in candidate.attrs:
                idata.attrs[name] = candidate.attrs[name]
        person_snapshot = None if person_data is None else person_data.copy(deep=True)
        self._output_attrs = population_output_attrs(prepared.constant_data.sizes["person"])
        self._idata, self._prepared, self._person_data = idata, prepared, person_snapshot
        self._pending_inputs.clear()
        self._pending_population = None
        return self

    def _add_input(self, declaration: FCInput | ItemInput) -> Self:
        if self._prepared is not None:
            raise ValueError(
                "This model is already prepared; create a new Model for different data"
            )
        self._pending_inputs[type(declaration)] = declaration
        return self

    def add_fc(
        self,
        design: pd.DataFrame | str | FCDesignMapping | Mapping,
        responses: pd.DataFrame,
        *,
        format: str,
        person_id_column: str | int | None = None,
        comparison_columns: tuple[ComparisonColumn, ...] = (),
        **input_options,
    ) -> Self:
        """Add or replace the forced-choice tables before preparation.

        ``format`` specifies the table layout. ``response_format`` is an
        FCInput option that specifies the questionnaire task.
        """
        table = ResponseTable(
            responses,
            format=format,
            person_id_column=person_id_column,
            comparison_columns=comparison_columns,
        )
        return self._add_input(FCInput(design, table, **input_options))

    def set_population(
        self,
        *,
        group: str | None = None,
        reference_group: str | int | None = None,
        person_data: pd.DataFrame | None = None,
        theta_residual_scale: Literal["unit", "group"] | None = None,
        theta_correlation: Literal["group", "shared"] = "group",
        regression_by_group: bool = False,
        priors: Mapping[str, TensorVariable] | None = None,
    ) -> Self:
        """Register a population and optional respondent table before preparation."""
        if self._prepared is not None:
            raise ValueError("Set the population before preparing the model")
        population = PopulationSpec(
            group=group,
            reference_group=reference_group,
            theta_residual_scale=theta_residual_scale,
            theta_correlation=theta_correlation,
            regression_by_group=regression_by_group,
            priors=priors,
        )
        self._pending_population = population, person_data
        return self

    def add_items(
        self,
        design: pd.DataFrame | str | ItemDesignMapping | Mapping,
        responses: pd.DataFrame,
        *,
        format: str,
        person_id_column: str | int | None = None,
        **input_options,
    ) -> Self:
        """Add or replace the individual-item tables before preparation."""
        table = ResponseTable(responses, format=format, person_id_column=person_id_column)
        return self._add_input(ItemInput(design, table, **input_options))

    @property
    def idata(self) -> xr.DataTree:
        return self._idata

    @property
    def _bindings(self):
        if self._built is None:
            self._built = ModelGraph.build(self.prepared, approximation=self._calibration_prior)
        return self._built

    @property
    def _posterior_calibrated(self) -> bool:
        return (
            self.prepared.data.attrs.get("calibration_uncertainty") == scoring.POSTERIOR_CALIBRATION
            or "calibration_evidence" in self.idata
        )

    @property
    def _fixed_reference(self) -> xr.Dataset | None:
        """Use the prepared point or an externally supplied reference for analysis."""
        if (fixed := self.prepared.fixed_parameters) is not None:
            return fixed
        exported = self.idata.get("fixed_parameters")
        return None if exported is None else exported.dataset

    @property
    def _point_calibrated(self) -> bool:
        return self._fixed_reference is not None and not self._posterior_calibrated

    def _prior_conditioning(self) -> str:
        """Report the runtime prior and require it for a reconstructed scoring round."""
        if self._calibration_prior is not None:
            return scoring.POSTERIOR_CALIBRATION
        if self._posterior_calibrated:
            raise ValueError(
                "This scoring round requires its original incoming approximation. "
                "Reconstruct it with score_model(..., approximation=...)."
            )
        return "fixed_parameters" if self.prepared.fixed_parameters is not None else "full prior"

    def _fit_bindings(self, overwrite):
        if self.prepared.data.attrs["purpose"] != "observed":
            raise ValueError("Generate real responses before fitting a generation schedule")
        if "posterior" in self.idata and not overwrite:
            raise ValueError("A posterior already exists; pass overwrite=True to replace it")
        self._prior_conditioning()
        return self._bindings

    def _require_mutable(self):
        if (
            self._calibration_prior is not None
            or self._fixed_reference is not None
            or any(name in self.idata for name in ("posterior", "estimates"))
        ):
            raise ValueError("Complete model declarations before fitting or attaching samples")

    def _checked_samples(self, idata, *, assume_same_batch=False):
        """Return an aligned posterior after checking the frozen sample contract."""
        if idata is self.idata:
            return posterior_samples(idata)
        fixed = self.prepared.result_metadata.get("fixed_parameters")
        posterior = validate_attached_samples(self.prepared, idata, fixed=fixed)
        if "fixed_parameters" in idata and (
            fixed is None or not fixed.equals(idata["fixed_parameters"].dataset)
        ):
            raise ValueError("Samples use a different fixed calibration snapshot")
        identity = idata.attrs.get("scoring_batch_identity")
        if identity is not None and (
            not isinstance(identity, str) or identity != self.prepared.scoring_batch_identity
        ):
            raise ValueError("Samples belong to a different prepared response batch")
        evidence = idata.get("calibration_evidence")
        if (approximation := self._calibration_prior) is not None:
            incoming = None if evidence is None else evidence.attrs.get("incoming_approximation_id")
            if incoming is not None and (
                not isinstance(incoming, str) or incoming != approximation._fit_id
            ):
                raise ValueError("Scoring samples used a different incoming approximation")
            if (identity is None or incoming is None) and not assume_same_batch:
                raise ValueError(
                    "Scoring samples lack a batch or incoming approximation identity; use "
                    "assume_same_batch=True only when both match"
                )
            if evidence is not None and _read_evidence(evidence) != (
                approximation.evidence_people,
                approximation.evidence_batches,
            ):
                raise ValueError("Scoring samples contain different incoming evidence")
        elif evidence is not None:
            _read_evidence(evidence)
        elif (
            self._posterior_calibrated
            or idata.attrs.get("calibration_uncertainty") == scoring.POSTERIOR_CALIBRATION
        ):
            raise ValueError("Recursive scoring samples require the calibration_evidence group")
        return posterior

    def attach_samples(self, idata: xr.DataTree, *, assume_same_batch: bool = False) -> None:
        """Attach samples, optionally affirming the untagged batch and incoming prior."""
        if not isinstance(assume_same_batch, bool):
            raise TypeError("assume_same_batch must be boolean")
        if "posterior" in self.idata:
            raise ValueError("This model already has samples; create a fresh Model")
        posterior = self._checked_samples(idata, assume_same_batch=assume_same_batch)
        # Saved predictions/checks may describe different response data. Attach
        # posterior draws and sampler statistics; evaluate analyses on this model.
        attrs = dict(self.idata.attrs)
        attrs.update(
            {
                key: value
                for key, value in idata.attrs.items()
                if key.startswith("sampling_") or key == "sampling"
            }
        )
        coordinates = scoring.CALIBRATION_COORDINATES
        if coordinates in idata.attrs:
            attrs[coordinates] = idata.attrs[coordinates]
        elif assume_same_batch:
            attrs[coordinates] = ModelGraph.calibration_coordinates(self.prepared)
        attrs.update(
            inference_method=idata.attrs.get("inference_method", "mcmc"),
            posterior_run_id=idata.attrs.get("posterior_run_id") or uuid4().hex,
        )
        self._publish_outputs(
            {
                name: idata[name].to_dataset()
                for name in ("sample_stats", "calibration_evidence")
                if name in idata and (name == "sample_stats" or name not in self.idata)
            },
            attrs=attrs,
            verified_posterior=posterior,
            posterior_batch_verified=(
                assume_same_batch or idata.attrs.get("scoring_batch_identity") is not None
            ),
        )

    def _publish_outputs(self, groups, **options):
        self._idata = prepare_outputs(
            self.prepared, self.idata, groups, variable_attrs=self._output_attrs, **options
        )

    def _publish_samples(self, sampled, *, sampling_attrs):
        self._idata = finalize_samples(
            self.prepared,
            self.idata,
            sampled,
            sampling_attrs={
                **sampling_attrs,
                scoring.CALIBRATION_COORDINATES: ModelGraph.calibration_coordinates(self.prepared),
            },
            variable_attrs=self._output_attrs,
        )
        return self.idata

    def build(self) -> pm.Model:
        """Return the cached native model; a scoring graph requires its runtime prior."""
        self._prior_conditioning()
        return self._bindings.model

    def _committed_evidence(self) -> tuple[frozenset[str], frozenset[str]]:
        people, batches = frozenset(), frozenset()
        if (approximation := self._calibration_prior) is not None:
            people, batches = approximation.evidence_people, approximation.evidence_batches
        elif (evidence := self.idata.get("calibration_evidence")) is not None:
            people, batches = _read_evidence(evidence)
        elif self._posterior_calibrated:
            raise ValueError("Restore calibration_evidence before continuing recursive scoring")
        if "posterior" in self.idata and (observed := self.prepared.observed_person_keys):
            batch = self.prepared.scoring_batch_identity
            if self.idata.attrs.get("scoring_batch_identity") != batch:
                raise ValueError(
                    "Posterior samples lack a verified batch identity for observed evidence; "
                    "recreate the model and attach_samples(..., assume_same_batch=True) "
                    "only when the samples used these prepared answers and predictors"
                )
            return people | observed, batches | {batch}
        return people, batches

    def _scoring_exclusions(self, approximation) -> frozenset[str]:
        """Return the verified incoming respondent IDs for new data preparation."""
        people, batches = approximation.evidence_people, approximation.evidence_batches
        if not isinstance(people, frozenset) or not isinstance(batches, frozenset):
            raise ValueError("Approximation requires immutable person and batch identities")
        people, batches = _checked_evidence_ids(people, batches)
        committed_people, committed_batches = self._committed_evidence()
        if not (committed_people <= people and committed_batches <= batches):
            raise ValueError("Approximation omits evidence already committed by the reference")
        return people

    def available_covariate_targets(
        self,
        level: Literal["person", "statement", "pair", "item", "measurement"] | None = None,
        *,
        kind: Literal["regression", "dif"] | None = None,
    ) -> pd.DataFrame:
        """List eligible covariate targets and their required predictor table.

        Filter by the level of the predicted parameter or by regression kind.
        Prepare the model before calling this method.
        """
        from pyfcirt.data.regressions import available_covariate_targets

        return available_covariate_targets(self.prepared, level=level, kind=kind)

    def add_regressions(
        self,
        formulas: str | Mapping[str | tuple[str, str] | tuple[str, str, str], Regression],
        *,
        person_data: pd.DataFrame | None = None,
        statement_data: pd.DataFrame | None = None,
        item_data: pd.DataFrame | None = None,
        pair_data: pd.DataFrame | None = None,
        id_columns: Mapping[str, str | tuple[str, ...]] | None = None,
        options: Mapping[str, RegressionOptions] | None = None,
    ) -> None:
        """Replace selected role formulas atomically before estimation."""
        from pyfcirt.covariates.specification import regression_declarations
        from pyfcirt.data.regressions import prepare_regressions

        declarations, qualified_priors = regression_declarations(
            formulas,
            person_data=person_data,
            statement_data=statement_data,
            item_data=item_data,
            pair_data=pair_data,
            id_columns=id_columns,
            options=options,
        )
        self._update_parameters(
            prepare_regressions, declarations, qualified_priors=qualified_priors
        )

    def add_dif_terms(
        self,
        targets: str | Sequence[str] | Mapping[str, Mapping],
        *,
        person_data: pd.DataFrame | None = None,
        reference_levels: Mapping[str, str | int | bool] | None = None,
        formula: str | None = None,
        groups: str | Sequence[str | tuple[str, ...]] = (),
        id: str = "person_id",
        pooling: Literal["unit", "shared"] = "unit",
        prior: Literal["laplace", "hierarchical", "pairwise_l1", "truncated_l1"]
        | TensorVariable = "laplace",
        prior_options: Mapping | None = None,
    ) -> None:
        """Add respondent-level deviations to selected free measurement parameters.

        Provide one row per respondent keyed by `id`. Call after preparation and
        before fitting or fixing parameters.
        """
        from pyfcirt.covariates.specification import dif_declarations
        from pyfcirt.data.regressions import prepare_regressions

        if person_data is None:
            person_data = self._person_data
        if person_data is None:
            raise ValueError("Supply person_data or register it before preparation")
        declarations, options = dif_declarations(
            targets,
            person_data,
            formula=formula,
            groups=groups,
            id_column=id,
            pooling=pooling,
            prior=prior,
            prior_options=prior_options,
            reference_levels=reference_levels,
        )
        self._update_parameters(
            prepare_regressions, declarations, person_effects=True, dif_options=options
        )

    def dif_summary(
        self,
        contrasts: Mapping | None = None,
        *,
        at: Mapping | None = None,
        prob: float = 0.95,
        significant_only: bool = False,
    ) -> pd.DataFrame:
        """Summarize posterior contrasts on the original measurement parameter scale.

        Each row identifies a target and predictor contrast and reports the mean
        difference, equal-tailed interval and whether it excludes zero.
        """
        from pyfcirt.analysis.dif import summarize_dif

        return summarize_dif(
            self, contrasts=contrasts, at=at, prob=prob, significant_only=significant_only
        )

    def set_priors(self, priors: Mapping[str, TensorVariable]) -> None:
        """Replace selected ordinary priors or baseline regressions.

        Unrelated roles and respondent-level DIF terms retain their declarations.
        Complete updates before fitting or fixing.
        """
        from pyfcirt.data.regressions import prepare_priors

        self._update_parameters(prepare_priors, priors)

    def _update_parameters(self, prepare, declarations, **options):
        """Prepare a candidate with one prior cache, then publish after validation."""
        self._require_mutable()
        with PriorPreparation():
            candidate = prepare(self.prepared, declarations, **options)
        self._publish(candidate)

    def _publish(self, candidate):
        """Invalidate derived results only after a specification update succeeds."""
        self._idata = replace_output_groups(
            self.idata,
            candidate.result_metadata,
            remove=(
                *self.prepared.result_metadata,
                "prior",
                "prior_predictive",
                "observed_data",
                *POSTERIOR_DEPENDENT_GROUPS,
            ),
        )
        self._prepared = candidate
        self._built = None

    # Bound methods expose the existing workflows without duplicating their logic.
    data_coverage = summary.data_coverage
    response_data = summary.response_data
    coef = summary.coef
    summary = summary.model_summary
    trait_summary = traits.trait_summary
    parameter_point = parameter_point
    prior_predictive = prediction.prior_predictive
    posterior_predictive = prediction.posterior_predictive
    log_likelihood = prediction.log_likelihood
    chi_ppp = checks.chi_ppp
    residuals = checks.residuals
    information_reliability = reliability.information_reliability
    information = information.information
    reliability = reliability.reliability
    generation_schedule = preparation.generation_schedule
    simulate = simulation.simulate
    posterior_approximation = scoring.posterior_approximation
    score_model = scoring.score_model
    score = scoring.score

    def _sampling_names(self, var_names=None) -> list[str]:
        """Select outputs and retain final calibration parameters needed by consumers."""
        if isinstance(var_names, str):
            raise TypeError("var_names must be a sequence of variable names")
        if var_names is None:
            from pytensor.graph.traversal import ancestors

            model = self._bindings.model
            free = set(model.free_RVs)
            selected = [
                variable.name
                for variable in model.unobserved_RVs
                if not variable.name.endswith("_raw") and not free.isdisjoint(ancestors([variable]))
            ]
        else:
            selected = list(var_names)
        if self._calibration_prior is not None:
            selected.extend(
                name for name in self._calibration_prior.var_names if name not in selected
            )
        return selected

    def _final_sampling_names(self, var_names=None) -> list[str]:
        """Apply the prepared final-node storage policy for specialized samplers."""
        selected = self._sampling_names(var_names)
        required = {"theta"} | {
            layout.kind
            for layout in self.prepared.parameters
            if len(layout.free_flat) and layout.point_parameter
        }
        if (fixed := self.prepared.fixed_parameters) is not None:
            required -= fixed.data_vars.keys()
        if missing := required - set(selected):
            raise ValueError(f"Missing required final storage nodes: {sorted(missing)}")
        return selected

    def fit(
        self,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int | None = None,
        target_accept: float = 0.8,
        nuts_sampler: str | None = "blackjax",
        random_seed=None,
        var_names: Sequence[str] | None = None,
        retain_calibration: bool = False,
        overwrite: bool = False,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Call native PyMC sampling and replace results only after success.

        BlackJAX is the default sampler; explicit choices and options go to PyMC.
        Pass nuts_sampler=None for native automatic selection. By default, final
        parameter nodes are retained. Set retain_calibration=True to keep the
        shared native draws needed for posterior scoring. Use var_names to
        retain other native draws. Native free-RV checks default off.
        """
        if sample_kwargs.get("return_inferencedata") is False:
            raise ValueError("Model.fit requires native InferenceData output")
        bindings = self._fit_bindings(overwrite)
        selected = self._sampling_names(var_names)
        if retain_calibration and self._calibration_prior is None:
            native, dependencies = bindings.calibration_source()
            shared = set().union(*dependencies.values())
            selected.extend(
                rv.name for rv in native.free_RVs if rv in shared and rv.name not in selected
            )
        if nuts_sampler in {"blackjax", "numpyro"}:
            import jax

            # Use the caller's native option name; PyMC validates conflicting options.
            option = (
                "nuts_sampler_kwargs"
                if sample_kwargs.get("nuts_sampler_kwargs") is not None
                else "nuts"
            )
            nuts = dict(sample_kwargs.get(option) or {})
            if "chain_method" not in nuts:
                nuts["chain_method"] = (
                    "vectorized" if jax.default_backend() == "gpu" else "parallel"
                )
            sample_kwargs[option] = nuts
            if nuts_sampler == "blackjax" and nuts["chain_method"] == "vectorized":
                sample_kwargs.setdefault("progressbar", False)
        sample_kwargs.update(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            random_seed=random_seed,
        )
        sample_kwargs.setdefault("compute_convergence_checks", False)
        idata_kwargs = bindings.idata_kwargs(sample_kwargs.pop("idata_kwargs", None))
        response_logp = ()
        if log_likelihood := idata_kwargs.get("log_likelihood"):
            # PyMC's converter requires all free RVs, including omitted raw values.
            # Evaluate the ordinary factors from the retained final nodes instead.
            ordinary = {
                batch.response_name: batch
                for batch in self.prepared.batches
                if batch.likelihood_kind != "rank"
            }
            requested = ordinary if log_likelihood is True else tuple(log_likelihood)
            if unknown := set(requested) - ordinary.keys():
                raise ValueError(f"No ordinary response likelihood for {sorted(unknown)}")
            response_logp = tuple(ordinary[name] for name in requested)
            idata_kwargs["log_likelihood"] = False
        with bindings.model:
            sampled = pm.sample(var_names=selected, idata_kwargs=idata_kwargs, **sample_kwargs)
        if response_logp:
            sampled["log_likelihood"] = xr.DataTree(
                prediction._response_log_densities(
                    bindings,
                    sampled["posterior"].to_dataset(),
                    response_logp,
                    progressbar=sample_kwargs.get("progressbar", True),
                )
            )
        sampling_attrs = {}
        for key in (
            "draws",
            "tune",
            "chains",
            "cores",
            "nuts_sampler",
            "target_accept",
            "random_seed",
            "compute_convergence_checks",
        ):
            value = sample_kwargs.get(key)
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (str, int, float)):
                sampling_attrs[f"sampling_{key}"] = int(value) if isinstance(value, bool) else value
        return self._publish_samples(sampled, sampling_attrs=sampling_attrs)
