"""Explicit teaching-sized inputs and labelled supplied arrays; never inference."""

from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

import pyfcirt as fc


def generated_pair_frame(prepared, values):
    """Read decoded values and verify their published comparison provenance."""
    from pyfcirt.data.encoding import encode_batches, generated_rows

    design = prepared._design_groups()["/constant_data"]
    admin, responses = generated_rows(prepared, values)
    _, observed = encode_batches(design, admin, responses, purpose="observed")
    for row in responses.index[responses.native_response.to_numpy(dtype=bool)]:
        unit = design.attrs["units"][observed.response_unit.values[row]]
        scale = unit["scale"]
        native = responses.at[row, "response"]
        if scale["kind"] == "unit_interval":
            response = 1 - native if scale["higher_prefers"] == "second" else native
        else:
            labels = (0, 1) if unit["response_format"] is not None else scale["categories"]
            response = labels[native]
        responses.at[row, "response"] = response
    return responses.assign(
        person_id=design.person.values[observed.response_person.values],
        unit_id=design.unit_id.values[observed.response_unit.values],
        **{
            col: design.statement_label.values[observed[f"response_{col}"].values]
            for col in ("statement_1", "statement_2", "source_statement_1", "source_statement_2")
        },
    )


def replay_design(training, data, keys):
    from pyfcirt.common.validation import align_rows
    from pyfcirt.covariates.design import apply_design

    return apply_design(
        training, data, *align_rows(data, keys, training.attrs["id_columns"], "covariates")
    )


def runtime_snapshot(prepared):
    """Snapshot arrays/metadata while retaining opaque templates and trained encoders."""
    objects = [
        value
        for plan in prepared.regressions
        for value in (plan.design.attrs["encoder"], *plan.priors.values())
    ]
    return deepcopy(prepared.data, memo={id(value): value for value in objects})


def mixed_inputs():
    design = fc.FCDesignMapping(
        traits={"A": ["a", "c", "e", "g", "i"], "B": ["b", "d", "f", "h"]},
        blocks={"pair": ["a", "b"], "7": ["c", "d", "e"], "quad": ["f", "g", "h", "i"]},
        keying={"b": -1},
    )
    answers = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2],
            "block_id": ["pair", "7", "quad", "pair"],
            "response": [[1, 2], [2, None, None], [1, None, None, 4], [2, 1]],
        }
    )
    inputs = fc.FCInput(
        design,
        fc.ResponseTable(answers, format="block_long"),
        blocks=(
            fc.BlockSpec("7", response_format="pick"),
            fc.BlockSpec("quad", response_format="mole"),
        ),
    )
    items = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["pair"], "B": ["ordinal"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": [1, 2, 1],
                    "item_id": ["pair", "pair", "ordinal"],
                    "response": [1, 0, 0],
                }
            ),
            format="item_long",
        ),
        items=(fc.ItemOverride("ordinal", model=fc.GPCM(), categories=(0, 1, 2)),),
    )
    return inputs, items


def mixed_model():
    return fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs()))


def labelled_point(model, values):
    graph = model.build()
    return xr.Dataset(
        {
            name: xr.DataArray(
                np.asarray(value),
                dims=graph.named_vars_to_dims[name],
                coords={
                    dim: np.asarray(graph.coords[dim], dtype=object)
                    for dim in graph.named_vars_to_dims[name]
                },
            )
            for name, value in values.items()
        }
    )


def supplied_posterior(model, values, *, inference_method, chains=("supplied",), draws=(3, 9)):
    posterior = values.expand_dims(
        chain=np.asarray(tuple(chains), dtype=object), draw=np.asarray(tuple(draws), dtype=object)
    )
    samples = xr.DataTree.from_dict({"posterior": model.prepared.annotate_coordinates(posterior)})
    samples.attrs.update(
        inference_method=inference_method, sampling_test_provenance="supplied arrays; no inference"
    )
    model.attach_samples(samples)
    return model


def workflow_point(model):
    values = {
        "population_mean": np.zeros((1, 2)),
        "covariance": np.eye(2),
        "loading": np.ones(11),
        "statement_intercept": [0.0, 0.1, 0.0, 0.3, 0.4, 0.0, 0.6, 0.7, 0.8],
        "item_intercept": [0.2],
        "lower_asymptote": [0.0],
        "upper_asymptote": [1.0],
        "category_intercept": [0.0, -0.3, -0.8],
        "theta": [[-0.4, 0.2], [0.6, -0.1]],
    }
    for layout in model.prepared.parameters:
        if layout.kind in {"regression_coefficient", "regression_sd"}:
            values[layout.kind] = np.where(
                np.isfinite(layout.fixed),
                layout.fixed,
                0.0 if layout.kind == "regression_coefficient" else 0.2,
            )
    return labelled_point(model, values)


def assert_same_preparation(actual, expected):
    """Compare arrays and native statistical declarations, not opaque object identities."""
    from pyfcirt.models.priors import PriorPreparation

    xr.testing.assert_equal(actual.constant_data, expected.constant_data)
    xr.testing.assert_equal(actual.observed_data, expected.observed_data)
    for a, b in zip(
        (
            *(scope["specification"] for scope in actual.constant_data.attrs["owners"]),
            *actual.measurement_specs,
            actual.population.specification,
        ),
        (
            *(scope["specification"] for scope in expected.constant_data.attrs["owners"]),
            *expected.measurement_specs,
            expected.population.specification,
        ),
        strict=True,
    ):
        assert type(a) is type(b)
        assert PriorPreparation.key(vars(a)) == PriorPreparation.key(vars(b))
    assert actual.population.theta_parameterization == expected.population.theta_parameterization
    assert actual.data.attrs["purpose"] == expected.data.attrs["purpose"]
    assert [(b.events.response_format, b.events.labels) for b in actual.batches] == [
        (b.events.response_format, b.events.labels) for b in expected.batches
    ]
