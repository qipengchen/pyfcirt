"""Prior prediction is independent of posterior fitting and observed values."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from pyfcirt import FCData, FCDesign, Model, Prior
from pyfcirt.models import MGPPM


def case_data(task, changed=False):
    if task == "graded":
        design = FCDesign.from_mapping(
            traits={"T1": ["A"], "T2": ["B"]},
            blocks={"pair": ["A", "B"]},
            response_format="graded",
            graded_scale={"categories": ["a", "b", "c", "d", "e"], "higher_prefers": "position_1"},
        )
        responses = pd.DataFrame(
            {
                "person_id": ["P", "Q"],
                "block_id": ["pair", "pair"],
                "response": ["b", "d"] if changed else ["a", "c"],
            }
        )
    else:
        design = FCDesign.from_mapping(
            traits={"T1": ["S1", "S4", "S7"], "T2": ["S2", "S5", "S8"], "T3": ["S3", "S6", "S9"]},
            blocks={
                "rank": ["S1", "S2", "S3"],
                "pick": ["S4", "S5", "S6"],
                "mole": ["S7", "S8", "S9"],
            },
            block_response_formats={"pick": "pick", "mole": "mole"},
        )
        responses = pd.DataFrame(
            {
                "person_id": ["P"] * 3,
                "block_id": ["rank", "pick", "mole"],
                "response": [[2, 3, 1], [2, None, None], [3, None, 1]]
                if changed
                else [[1, 2, 3], [1, None, None], [1, None, 3]],
            }
        )
    return FCData.from_frames(responses, design)


@pytest.fixture(scope="module", params=["rank", "graded"])
def prior_case(request):
    built = Model(case_data(request.param)).build()

    def forbid_posterior(*args, **kwargs):
        raise AssertionError("Prior prediction must not invoke posterior MCMC")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pm, "sample", forbid_posterior)
        draws = built.prior_predictive(draws=4, random_seed=73)
    return request.param, built, draws


def test_native_prior_groups_labels_and_constraints(prior_case):
    task, built, draws = prior_case
    assert isinstance(draws, xr.DataTree)
    assert {"prior", "prior_predictive", "observed_data"} <= set(draws.children)
    prior = draws["prior"]
    assert prior["theta"].dims == ("chain", "draw", "person", "trait")
    assert prior["theta"].coords["person"].values.tolist() == list(built.inputs.person_ids)
    assert prior["intercept"].coords["statement"].values.tolist() == list(
        built.inputs.statement_ids
    )
    assert (prior["loading_magnitude"].values > 0).all()
    correlation = prior["trait_correlation"].values
    assert_allclose(np.diagonal(correlation, axis1=-2, axis2=-1), 1, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(correlation) > 0)
    assert "theta" in {rv.name for rv in built.model.free_RVs}
    assert "theta_raw" not in prior
    assert built.spec.settings["parameterization"] == "centered"
    if task == "rank":
        for component in built.inputs.groups:
            beta = prior["intercept"].values[..., component.statement_index]
            assert_allclose(beta.sum(axis=-1), 0, atol=1e-12)
    else:
        assert built.spec.settings["priors"]["graded_intercept"]["name"] == "Normal"
        assert not np.allclose(prior["intercept"].values.sum(axis=-1), 0)


def test_decoded_predictions_preserve_task_support_and_structural_na(prior_case):
    _, built, draws = prior_case
    for index in range(4):
        frame = built.response_frame(draws, draw=index)
        restored = FCData.from_frames(frame, built.data.design)
        assert restored.summary()["n_observations"] == built.data.summary()["n_observations"]
        assert set(zip(frame.person_id, frame.block_id, strict=True)) == {
            (r["person_id"], r["block_id"]) for r in built.spec.observation_map
        }
        for row in frame.itertuples(index=False):
            task = built.data.design.response_formats[row.block_id]
            if task == "pick":
                assert row.response[1:] == [None, None]
            elif task == "mole":
                assert row.response[1] is None
            elif task == "graded":
                assert row.response in built.data.design.graded_scale["categories"]


def test_observed_value_changes_do_not_condition_the_prior(prior_case):
    task, _, first = prior_case
    changed = Model(case_data(task, changed=True)).prior_predictive(draws=4, random_seed=73)
    for group in ["prior", "prior_predictive"]:
        for name in first[group].data_vars:
            assert_array_equal(first[group][name].values, changed[group][name].values)


def test_custom_prior_is_used_by_forward_generation():
    configured = Model(
        case_data("graded"),
        family=MGPPM(priors={"loading_magnitude": Prior("Uniform", lower=5, upper=6)}),
    )
    draws = configured.prior_predictive(draws=3, random_seed=81)
    assert (draws["prior"]["loading_magnitude"].values >= 5).all()
    assert (draws["prior"]["loading_magnitude"].values <= 6).all()


def test_response_decoding_rejects_mismatched_coordinates(prior_case):
    _, built, draws = prior_case
    modified = draws.copy(deep=True)
    component = built.spec.components[0]
    dimension = f"{component['id']}_obs"
    modified["prior_predictive"] = (
        modified["prior_predictive"]
        .to_dataset()
        .assign_coords(
            {dimension: [f"wrong-{i}" for i in range(len(built.inputs.groups[0].response))]}
        )
    )
    with pytest.raises(ValueError, match="coordinates do not match"):
        built.response_frame(modified)


@pytest.mark.parametrize("draws", [0, -1, True, 1.5])
def test_invalid_draw_counts_are_not_passed_to_pymc(draws):
    built = Model(case_data("graded")).build()
    with pytest.raises(ValueError, match="positive integer"):
        built.prior_predictive(draws=draws)
