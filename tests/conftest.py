"""Small deterministic designs; no sampler runs in data-layer fixtures."""

import pandas as pd
import pytest

from pyfcirt import FCDesign


@pytest.fixture
def design_frame():
    return pd.DataFrame(
        [
            ("B1", 1, "S1", "T1", 1),
            ("B1", 2, "S2", "T2", -1),
            ("B1", 3, "S3", "T3", 1),
            ("B2", 1, "S4", "T1", 1),
            ("B2", 2, "S5", "T2", 1),
            ("B2", 3, "S6", "T3", -1),
        ],
        columns=["block_id", "position", "statement_id", "trait", "keying"],
    )


@pytest.fixture
def design(design_frame):
    return FCDesign.from_frame(design_frame)


@pytest.fixture
def graded_design():
    return FCDesign.from_mapping(
        traits={"T1": ["S1"], "T2": ["S2"]},
        blocks={"B1": ["S1", "S2"]},
        response_format="graded",
        graded_scale={
            "categories": [1, 2, 3, 4, 5],
            "higher_prefers": "position_1",
            "labels": ["strong_second", "second", "equal", "first", "strong_first"],
        },
    )


@pytest.fixture(autouse=True)
def forbid_posterior_mcmc(monkeypatch):
    """Formal tests may mock sampling but must never launch posterior MCMC."""
    import pymc as pm

    def forbidden(*args, **kwargs):
        raise AssertionError("Posterior MCMC is manual-only; use scripts/check_mgppm_sampling.py")

    monkeypatch.setattr(pm, "sample", forbidden)
