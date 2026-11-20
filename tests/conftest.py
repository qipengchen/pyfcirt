"""Ordinary tests never execute posterior MCMC."""

import pytest

from tests.helpers import mixed_model, supplied_posterior, workflow_point


@pytest.fixture(autouse=True)
def forbid_posterior_mcmc(monkeypatch):
    import pymc as pm

    def forbidden(*args, **kwargs):
        raise AssertionError("Posterior MCMC is forbidden in ordinary tests")

    monkeypatch.setattr(pm, "sample", forbidden)


@pytest.fixture
def supplied_model():
    model = mixed_model()
    return supplied_posterior(model, workflow_point(model), inference_method="mcmc")
