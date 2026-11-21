"""Independent new-person models with point or posterior calibration priors."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from pyfcirt.analysis.output import parameter_point, posterior_samples, read_parameter_point
from pyfcirt.data.prepare import _prepare_scoring
from pyfcirt.models.assembly import ModelGraph
from pyfcirt.scoring.approximation import PosteriorApproximation

if TYPE_CHECKING:
    from pyfcirt.model import Model

POSTERIOR_CALIBRATION = "parameter_kind_posterior_approximation"
CALIBRATION_COORDINATES = "calibration_coordinate_identity"


def posterior_approximation(
    model: Model,
    *,
    var_names=None,
    joint: bool | None = None,
    method="auto",
    components=(2, 3),
    random_seed=None,
    transforms=None,
):
    """Fit a calibration prior, pooling the automatic selection by default."""
    if joint is not None and not isinstance(joint, bool):
        raise TypeError("joint must be a bool or None")
    if joint and var_names is not None:
        raise ValueError("joint=True requires automatic selection; omit var_names")
    previous = model._calibration_prior
    if joint is None:
        joint = var_names is None and previous is None
    if var_names is None and previous is not None:
        var_names = previous.groups
        if transforms is None:
            transforms = previous.transforms
    people, batches = model._committed_evidence()
    coordinates = ModelGraph.calibration_coordinates(model.prepared)
    if coordinates != model.idata.attrs.get(CALIBRATION_COORDINATES):
        raise ValueError("Posterior draws require matching native calibration coordinates")
    bindings = model._bindings
    native, dependencies = bindings.calibration_source()
    posterior = posterior_samples(model.idata)
    if var_names is None:
        var_names = bindings.calibration_groups(native, dependencies)
        if not var_names:
            raise ValueError("No supported free shared calibration variables")
        if missing := {name for group in var_names.values() for name in group} - posterior.keys():
            raise ValueError(
                f"Missing calibration draws for {sorted(missing)}. "
                "Use Model.fit(retain_calibration=True) when fitting the initial calibration."
            )
    if joint:
        var_names = {"joint": tuple(name for group in var_names.values() for name in group)}
    approximation = PosteriorApproximation._from_native(
        posterior,
        model=native,
        var_names=var_names,
        method=method,
        components=components,
        random_seed=random_seed,
        transforms=transforms,
        eligible=set().union(*dependencies.values()),
    )
    finals, free = bindings.calibration_complement(native, dependencies, approximation.var_names)
    point = read_parameter_point(model, posterior=posterior, names=sorted(finals))
    fixed = model._fixed_reference
    if fixed is not None:
        free -= fixed.data_vars.keys()
    if missing := free - posterior.data_vars.keys():
        raise ValueError(f"Missing calibration draws to fix native RV {min(missing)!r}")
    means = {name: posterior[name].mean(("chain", "draw"), skipna=False) for name in sorted(free)}
    return replace(
        approximation,
        evidence_people=people,
        evidence_batches=batches,
        design_identity=model.prepared.calibration_identity,
        coordinate_identity=coordinates,
        fixed_parameters=xr.merge(
            (point, fixed if fixed is not None else xr.Dataset(), means),
            compat="equals",
            join="exact",
            combine_attrs="override",
        ),
    )


def score_model(
    reference: Model,
    new_responses,
    *,
    format: str | None = None,
    person_data: pd.DataFrame | None = None,
    administration=None,
    parameters: xr.Dataset | None = None,
    calibration="point",
    approximation=None,
) -> Model:
    """Construct an independent scoring graph without running an estimator.

    Point mode uses explicit parameters, an existing fixed snapshot, or final
    posterior means, in that order. Posterior mode supplies calibration priors
    estimated from the reference posterior by default.
    """
    from pyfcirt.model import Model

    if not isinstance(reference, Model):
        raise TypeError("The scoring reference must be a Model")
    excluded_people = frozenset()
    if calibration == "point":
        if approximation is not None:
            raise ValueError("approximation requires posterior calibration")
        if parameters is not None:
            fixed, point_source = parameters, "explicit_parameters"
        elif (
            not reference._posterior_calibrated
            and (fixed := reference._fixed_reference) is not None
        ):
            point_source = "fixed_parameters"
        else:
            fixed, point_source = parameter_point(reference, include_person=False), "posterior_mean"
        attrs = {"calibration_uncertainty": "fixed_final_point", "point_source": point_source}
    elif calibration == "posterior":
        if parameters is not None:
            raise ValueError("parameters cannot be supplied with posterior calibration")
        if approximation is None:
            approximation = reference.posterior_approximation()
            # The fit already used this design, native coordinates and committed evidence.
            excluded_people = approximation.evidence_people
        else:
            if not isinstance(approximation, PosteriorApproximation):
                raise TypeError("Posterior scoring requires a runtime PosteriorApproximation")
            if approximation.design_identity != reference.prepared.calibration_identity:
                raise ValueError("Approximation belongs to a different calibration design")
            if approximation.coordinate_identity != ModelGraph.calibration_coordinates(
                reference.prepared
            ):
                raise ValueError("Approximation requires matching native calibration coordinates")
            excluded_people = reference._scoring_exclusions(approximation)
        fixed = approximation.fixed_parameters
        attrs = {"calibration_uncertainty": POSTERIOR_CALIBRATION}
    else:
        raise ValueError("calibration must be 'point' or 'posterior'")
    state = _prepare_scoring(
        reference.prepared,
        new_responses,
        format=format,
        fixed_parameters=fixed,
        partial=approximation is not None,
        excluded_people=excluded_people,
        person_data=person_data,
        administration=administration,
    )
    state.data.attrs.update(attrs)
    return Model(state, _approximation=approximation)


def score(
    model: Model,
    new_responses,
    *,
    format: str | None = None,
    person_data=None,
    administration=None,
    parameters: xr.Dataset | None = None,
    calibration="point",
    approximation=None,
    var_names=None,
    **sample_kwargs,
) -> Model:
    """Fit new abilities with native sampling and the selected calibration law.

    Both modes call Model.fit. Posterior mode uses the feedback target: the new
    responses can update shared calibration quantities. It is not an equal-weight
    average of separately normalized conditional scoring posteriors.
    """
    scored = score_model(
        model,
        new_responses,
        format=format,
        person_data=person_data,
        administration=administration,
        parameters=parameters,
        calibration=calibration,
        approximation=approximation,
    )
    scored.fit(var_names=var_names, **sample_kwargs)
    return scored
