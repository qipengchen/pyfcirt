"""Prior declarations, role support and visible configuration provenance."""

import numpy as np
import pandas as pd
import pytest

from pyfcirt import FCData, Model, Prior
from pyfcirt.models import MGPPM


def test_prior_declaration_is_independent_of_graph_and_external_mutation():
    prior = Prior("LogNormal", mu=0, sigma=0.5)
    prior.parameters["sigma"] = 99
    assert prior.to_dict() == {"name": "LogNormal", "parameters": {"mu": 0, "sigma": 0.5}}
    assert repr(prior) == "Prior('LogNormal', mu=0, sigma=0.5)"


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("__import__", {}),
        ("os.system", {}),
        ("", {}),
        ("Normal", {"mu": np.nan}),
        ("Normal", {"sigma": np.inf}),
        ("Normal", {"factory": lambda: None}),
    ],
)
def test_invalid_prior_declarations_are_not_executable(name, parameters):
    with pytest.raises(ValueError):
        Prior(name, **parameters)


@pytest.mark.parametrize(
    "prior",
    [
        Prior("LogNormal", mu=0, sigma=0.5),
        Prior("HalfNormal", sigma=1),
        Prior("Gamma", alpha=2, beta=1),
        Prior("Exponential", lam=1),
        Prior("Beta", alpha=2, beta=2),
        Prior("Uniform", lower=1, upper=2),
    ],
)
def test_positive_prior_families_can_be_selected(design, prior):
    frame = pd.DataFrame({"person_id": ["P"], "block_id": ["B1"], "response": [[1, 2, 3]]})
    source = {"loading_magnitude": prior}
    model = Model(FCData.from_frames(frame, design), family=MGPPM(priors=source))
    source.clear()
    summary = model.prior_summary().set_index("role")
    assert summary.loc["loading_magnitude", "distribution"] == prior.name
    assert summary.loc["loading_magnitude", "source"] == "user"
    assert summary.loc["ranking_intercept", "source"] == "default"
    assert summary.loc["loading_magnitude", "labels"] == ["S1", "S2", "S3", "S4", "S5", "S6"]


@pytest.mark.parametrize(
    "role,prior,match",
    [
        ("loading_magnitude", Prior("Normal", mu=10, sigma=0.1), "positive support"),
        ("loading_magnitude", Prior("LogNormal", sigma=-1), "strictly positive"),
        ("loading_magnitude", Prior("LogNormal", sigmaa=1), "invalid parameters"),
        ("loading_magnitude", Prior("HalfNormal", shape=3), "managed by Model"),
        ("loading_magnitude", Prior("HalfNormal", sigma=[1, 2, 3]), "scalar numeric"),
        ("loading_magnitude", Prior("Uniform", lower=2, upper=1), "upper must exceed"),
        ("ranking_intercept", Prior("Normal", sigma=1), "requires ZeroSumNormal"),
        ("ranking_intercept", Prior("ZeroSumNormal", sigma=1, mu=0), "requires ZeroSumNormal"),
    ],
)
def test_invalid_role_priors_fail_before_graph(design, role, prior, match):
    frame = pd.DataFrame({"person_id": ["P"], "block_id": ["B1"], "response": [[1, 2, 3]]})
    model = Model(FCData.from_frames(frame, design), family=MGPPM(priors={role: prior}))
    with pytest.raises(ValueError, match=match):
        model.prior_summary()


def test_global_correlation_prior_does_not_override_item_priors(design):
    frame = pd.DataFrame({"person_id": ["P"], "block_id": ["B1"], "response": [[1, 2, 3]]})
    data = FCData.from_frames(frame, design)
    model = Model(data, priors={"trait_correlation": Prior("LKJCorr", eta=3)})
    summary = model.prior_summary().set_index("role")
    assert summary.loc["trait_correlation", "parameters"] == {"eta": 3}
    assert summary.loc["loading_magnitude", "parameters"] == {"mu": 0.0, "sigma": 2.0}
    with pytest.raises(ValueError, match="item priors belong"):
        Model(data, priors={"loading_magnitude": Prior("HalfNormal", sigma=1)})
    with pytest.raises(ValueError, match="Unknown MGPPM prior roles"):
        MGPPM(priors={"loading_typo": Prior("HalfNormal", sigma=1)})


def test_graded_intercepts_are_independent_scalar_priors(graded_design):
    frame = pd.DataFrame({"person_id": ["P"], "block_id": ["B1"], "response": [3]})
    data = FCData.from_frames(frame, graded_design)
    model = Model(
        data, family=MGPPM(priors={"graded_intercept": Prior("StudentT", nu=5, mu=0, sigma=1)})
    )
    assert (
        model.prior_summary().set_index("role").loc["graded_intercept", "distribution"]
        == "StudentT"
    )
    with pytest.raises(ValueError, match="independent scalar"):
        Model(
            data, family=MGPPM(priors={"graded_intercept": Prior("ZeroSumNormal", sigma=1)})
        ).build()
