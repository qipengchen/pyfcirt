"""Calibration constraints, conditional targets and modular scoring wiring."""

from math import log, pi

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyfcirt import FCData, FCDesign
from pyfcirt._model import BuiltModel
from pyfcirt.results import FitResult
from pyfcirt.scoring import _interventions, _new_data, _scoring_graph, _validate_calibration, score
from tests._reference_mgppm import category_logps, partial_logp
from tests.test_model_comparison import comparison_case as comparison_case


@pytest.fixture(scope="module")
def scoring_case(comparison_case):
    frame = comparison_case.built.data.to_frame(answered_only=True)
    frame["person_id"] = [f"new-{p}" for p in frame.person_id]
    administration = frame[["person_id", "block_id"]].assign(status="answered")
    administration = pd.concat(
        [
            administration,
            pd.DataFrame(
                {
                    "person_id": ["no-response"],
                    "block_id": [comparison_case.built.inputs.block_ids[0]],
                    "status": ["missing"],
                }
            ),
        ],
        ignore_index=True,
    )
    data = FCData.from_frames(
        frame, comparison_case.built.data.design, administration=administration
    )
    point = comparison_case.calibration()
    return comparison_case, data, point


def test_calibration_mean_recomputes_derived_values_and_preserves_scale(scoring_case):
    fitted, _, point = scoring_case
    posterior = fitted.idata["posterior"]
    assert_allclose(
        point.loading_magnitude, posterior["loading_magnitude"].mean(("chain", "draw")), atol=1e-12
    )
    assert_allclose(point.intercept, posterior["intercept"].mean(("chain", "draw")), atol=1e-12)
    assert_allclose(
        point.trait_correlation, posterior["trait_correlation"].mean(("chain", "draw")), atol=1e-12
    )
    factor = _interventions(fitted.built, point)["trait_cholesky"]
    assert_allclose(np.diag(factor @ factor.T), 1, atol=1e-12)
    assert not np.allclose(factor, posterior["trait_cholesky"].mean(("chain", "draw")))
    shuffled = point.sel(
        statement=point.statement.values[::-1],
        trait=point.trait.values[::-1],
        trait_aux=point.trait_aux.values[::-1],
    )
    xr.testing.assert_identical(point, _validate_calibration(shuffled, fitted.built))


def test_selected_joint_draw_and_stale_deterministics(scoring_case):
    fitted, _, _ = scoring_case
    original = fitted.idata.copy(deep=True)
    altered = fitted.idata.copy(deep=True)
    altered["posterior"]["intercept"].values[...] = 999
    altered["posterior"]["trait_correlation"].values[...] = 999
    changed = FitResult(idata=altered, built=fitted.built, sampling={})
    expected = fitted.calibration(draw=(0, 2))
    actual = changed.calibration(draw=(0, 2))
    xr.testing.assert_identical(actual, expected)
    assert actual.attrs["source_chain"] == 0
    assert actual.attrs["source_draw"] == 2
    xr.testing.assert_identical(original, fitted.idata)
    with pytest.raises(ValueError, match="Unknown calibration draw"):
        fitted.calibration(draw=(999, 2))


def test_scoring_joint_density_and_gradient_match_independent_target(scoring_case):
    fitted, data, parameters = scoring_case
    canonical = _new_data(fitted.built, data)
    graph = _scoring_graph(fitted.built, canonical, parameters)
    assert [rv.name for rv in graph.model.free_RVs] == ["theta"]
    point = graph.model.initial_point()
    raw = np.linspace(-0.5, 0.7, len(data.person_ids) * len(graph.inputs.trait_ids)).reshape(
        len(data.person_ids), -1
    )
    point["theta"] = raw
    factor = np.linalg.cholesky(parameters.trait_correlation.values)

    def reference(values):
        theta = values
        whitened = np.linalg.solve(factor, theta.T).T
        total = (
            -0.5 * np.sum(whitened**2)
            - values.size * log(2 * pi) / 2
            - len(values) * np.log(np.diag(factor)).sum()
        )
        for group in graph.inputs.groups:
            for index, code in enumerate(group.response):
                statement = group.statement_index[group.block_index[index]]
                utility = (
                    parameters.loading_magnitude.values[statement]
                    * graph.inputs.keying[statement]
                    * theta[group.person_index[index], graph.inputs.statement_trait[statement]]
                    - parameters.intercept.values[statement]
                )
                total += (
                    category_logps(utility, group.scores)[code]
                    if group.task == "graded"
                    else partial_logp(utility, group.events[code])
                )
        return total

    assert_allclose(graph.model.compile_logp()(point), reference(raw), rtol=1e-11, atol=1e-11)
    actual_gradient = graph.model.compile_dlogp()(point)
    expected_gradient = []
    for index in range(raw.size):
        high, low = raw.copy(), raw.copy()
        high.flat[index] += 1e-5
        low.flat[index] -= 1e-5
        expected_gradient.append((reference(high) - reference(low)) / 2e-5)
    assert_allclose(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "problem",
    [
        "zero_loading",
        "nonfinite",
        "bad_diagonal",
        "indefinite",
        "unknown_label",
        "missing_role",
        "complex",
    ],
)
def test_invalid_calibration_is_rejected_before_sampling(scoring_case, problem):
    fitted, _, parameters = scoring_case
    changed = parameters.copy(deep=True)
    if problem == "zero_loading":
        changed.loading_magnitude.values[0] = 0
    elif problem == "nonfinite":
        changed.intercept.values[0] = np.nan
    elif problem == "bad_diagonal":
        changed.trait_correlation.values[0, 0] = 0.5
    elif problem == "indefinite":
        changed.trait_correlation.values[0, 1] = changed.trait_correlation.values[1, 0] = 2
    elif problem == "unknown_label":
        changed = changed.assign_coords(
            statement=[f"wrong{i}" for i in range(changed.sizes["statement"])]
        )
    elif problem == "missing_role":
        changed = changed.drop_vars("intercept")
    else:
        changed["loading_magnitude"] = changed.loading_magnitude.astype(complex)
    with pytest.raises(ValueError):
        _validate_calibration(changed, fitted.built)


def test_fixed_scoring_calls_sampler_once_and_preserves_calibration(scoring_case, monkeypatch):
    fitted, data, point = scoring_case
    original = fitted.idata.copy(deep=True)
    calls = []

    def sample(graph, **kwargs):
        calls.append(kwargs)
        assert_allclose(graph.model["loading_magnitude"].get_value(), point.loading_magnitude)
        factor = graph.model["trait_cholesky"].get_value()
        assert_allclose(factor @ factor.T, point.trait_correlation, atol=1e-12)
        posterior = xr.Dataset(
            {
                "theta": (
                    ("chain", "draw", "person", "trait"),
                    np.zeros(
                        (
                            kwargs["chains"],
                            kwargs["draws"],
                            len(graph.inputs.person_ids),
                            len(graph.inputs.trait_ids),
                        )
                    ),
                )
            },
            coords={
                "chain": range(kwargs["chains"]),
                "draw": range(kwargs["draws"]),
                "person": list(graph.inputs.person_ids),
                "trait": list(graph.inputs.trait_ids),
            },
        )
        return FitResult(
            idata=xr.DataTree.from_dict({"posterior": posterior}),
            built=graph,
            sampling={"purpose": "mock_wiring", "random_seed": kwargs["random_seed"]},
        )

    monkeypatch.setattr(BuiltModel, "fit", sample)
    scored = fitted.score(data, draws=8, chains=2, random_seed=53)
    assert len(calls) == 1
    assert calls[0]["random_seed"] == 53
    assert scored.settings["target"] == "fixed_calibration"
    assert scored.settings["calibration_uncertainty"] == "not_propagated"
    assert scored.samples.sizes["sample"] == 16
    assert scored.idata["posterior"]["theta"].sizes["chain"] == 2
    assert scored.sampling["purpose"] == "mock_wiring"
    assert "conditional_0" not in scored.idata and "score_samples" not in scored.idata
    xr.testing.assert_equal(scored.idata["calibration"].to_dataset(), point)
    assert scored.person_status.sel(person="no-response")["status"].item() == "prior_only"
    xr.testing.assert_identical(original, fitted.idata)


def test_all_unanswered_returns_exact_prior_and_explicit_status(scoring_case):
    fitted, _, _ = scoring_case
    empty = pd.DataFrame(columns=["person_id", "block_id", "response"])
    administration = pd.DataFrame(
        {
            "person_id": ["prior-A", "prior-B"],
            "block_id": [fitted.built.inputs.block_ids[0]] * 2,
            "status": ["missing", "not_administered"],
        }
    )
    data = FCData.from_frames(empty, fitted.built.data.design, administration=administration)
    kwargs = {
        "draws": 20,
        "chains": 2,
        "random_seed": 54,
    }
    scored = fitted.score(data, **kwargs)
    repeated = fitted.score(data, **kwargs)
    assert scored.sampling["method"] == "exact_prior"
    assert scored.samples.sizes["sample"] == 40
    assert (scored.person_status["status"].values == "prior_only").all()
    assert (scored.person_status.n_observations.values == 0).all()
    xr.testing.assert_identical(scored.idata, repeated.idata)
    assert np.isfinite(scored.summary()["mean"]).all()
    with pytest.raises(ValueError, match="Exact prior draws"):
        scored.diagnostics()


def test_unknown_statements_and_changed_keying_are_rejected(scoring_case):
    fitted, data, _ = scoring_case
    reference = data.design
    frame = reference.to_frame()
    first = frame.index[0]
    frame.at[first, "statement_id"] = "unknown-statement"
    wrong = FCDesign.from_frame(
        frame,
        block_response_formats=reference.response_formats,
        graded_scale=reference.graded_scale,
        allow_variable_block_size=True,
    )
    unknown = FCData.from_frames(data.to_frame(), wrong, administration=data.administration)
    with pytest.raises(ValueError, match="unknown calibration statements"):
        _new_data(fitted.built, unknown)
    frame = reference.to_frame()
    frame.at[first, "keying"] *= -1
    wrong = FCDesign.from_frame(
        frame,
        block_response_formats=reference.response_formats,
        graded_scale=reference.graded_scale,
        allow_variable_block_size=True,
    )
    incompatible = FCData.from_frames(data.to_frame(), wrong, administration=data.administration)
    with pytest.raises(ValueError, match="keying differ"):
        _new_data(fitted.built, incompatible)


def test_incomplete_scoring_draws_are_rejected(scoring_case, monkeypatch):
    fitted, data, _ = scoring_case

    def interrupted(graph, **kwargs):
        posterior = xr.Dataset(
            {
                "theta": (
                    ("chain", "draw", "person", "trait"),
                    np.zeros(
                        (
                            kwargs["chains"],
                            1,
                            len(graph.inputs.person_ids),
                            len(graph.inputs.trait_ids),
                        )
                    ),
                )
            }
        )
        return FitResult(
            idata=xr.DataTree.from_dict({"posterior": posterior}), built=graph, sampling={}
        )

    monkeypatch.setattr(BuiltModel, "fit", interrupted)
    with pytest.raises(ValueError, match="different chain/draw counts"):
        fitted.score(data, draws=4, chains=2, random_seed=56)


@pytest.mark.parametrize("options", [{"mode": "posterior"}, {"calibration_draws": 4}])
def test_removed_posterior_scoring_is_rejected_before_sampling(options):
    with pytest.raises(ValueError, match="fixed-calibration scoring"):
        score(None, None, **options)
