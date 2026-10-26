"""Native PyMC graph, identification and metadata checks without posterior fitting."""

import json
from math import log

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose

from pyfcirt import FCData, FCDesign, Model
from pyfcirt.models import MGPPM
from tests._reference_mgppm import category_logps, partial_logp


@pytest.fixture(scope="module", params=["rank", "graded"])
def built(request):
    task = request.param
    size = 3 if task == "rank" else 2
    kwargs = {"graded_scale": {"categories": [0, 1, 2, 3, 4], "higher_prefers": "position_1"}}
    design = FCDesign.from_mapping(
        traits={f"T{i}": [f"S{i}"] for i in range(size)},
        blocks={"B": [f"S{i}" for i in range(size)]},
        response_format=task,
        keying={"S1": -1},
        **(kwargs if task == "graded" else {}),
    )
    responses = [[2, 3, 1], [3, 1, 2]] if task == "rank" else [4, 1]
    frame = pd.DataFrame({"person_id": ["P", "Q"], "block_id": ["B", "B"], "response": responses})
    wrapper = Model(FCData.from_frames(frame, design))
    return wrapper, wrapper.build()


def test_one_shared_trait_structure_and_serializable_metadata(built):
    wrapper, result = built
    assert wrapper.build() is result
    assert list(result.model.named_vars).count("theta") == 1
    assert list(result.model.named_vars).count("loading_magnitude") == 1
    assert result.spec.observation_unit == "person-block"
    assert result.spec.likelihood_kind == "conditional_response"
    assert len(result.spec.observation_map) == 2
    assert len(result.spec.data_fingerprint) == 64
    assert result.spec.settings["parameterization"] == "centered"
    json.dumps(result.spec.to_dict(), allow_nan=False)
    copy = result.spec.to_dict()
    copy["settings"]["parameterization"] = "changed"
    assert result.spec.settings["parameterization"] == "centered"


def test_fixed_native_response_logp_matches_independent_reference(built):
    _, result = built
    native = result.model
    point = native.initial_point()
    n_traits = len(result.inputs.trait_ids)
    theta = np.array([np.linspace(-0.4, 0.7, n_traits), np.linspace(0.3, -0.6, n_traits)])
    point[native.rvs_to_values[native["theta"]].name] = theta
    magnitude = np.linspace(0.7, 1.3, len(result.inputs.statement_ids))
    rv = native["loading_magnitude"]
    transform = native.rvs_to_transforms[rv]
    transformed = pytensor.function(
        [], transform.forward(pt.constant(magnitude), *rv.owner.inputs), mode="NUMBA"
    )()
    point[native.rvs_to_values[rv].name] = transformed
    expected = 0.0
    for group in result.inputs.groups:
        for i, code in enumerate(group.response):
            positions = group.statement_index[group.block_index[i]]
            utility = (
                magnitude[positions]
                * result.inputs.keying[positions]
                * theta[group.person_index[i], result.inputs.statement_trait[positions]]
            )
            if group.task == "graded":
                expected += category_logps(utility, group.scores)[code]
            else:
                expected += partial_logp(utility, group.events[code])
    actual = native.compile_logp(vars=native.observed_RVs)(point)
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_native_initial_joint_density_and_gradient_are_finite(built):
    _, result = built
    point = result.model.initial_point()
    assert np.isfinite(result.model.compile_logp()(point))
    assert np.isfinite(result.model.compile_dlogp()(point)).all()


def test_variable_sizes_share_parameters_without_padding(design_frame):
    design = FCDesign.from_frame(design_frame.iloc[:-1], allow_variable_block_size=True)
    frame = pd.DataFrame(
        {"person_id": ["P", "P"], "block_id": ["B1", "B2"], "response": [[2, 3, 1], [2, 1]]}
    )
    result = Model(FCData.from_frames(frame, design), block_families={"B2": MGPPM()}).build()
    assert len(result.inputs.groups) == 2
    assert len(result.inputs.statement_ids) == 5
    assert list(result.model.named_vars).count("theta") == 1
    assert len(result.model.observed_RVs) == 2
    point = result.model.initial_point()
    actual = result.model.compile_logp(vars=result.model.observed_RVs)(point)
    assert_allclose(actual, -log(6) - log(2), atol=1e-12)


def test_unsupported_shared_and_single_trait_designs_fail_explicitly(design_frame):
    design_frame.at[3, "statement_id"] = "S1"
    design = FCDesign.from_frame(design_frame)
    frame = pd.DataFrame({"person_id": ["P"], "block_id": ["B1"], "response": [[1, 2, 3]]})
    with pytest.raises(NotImplementedError, match="Shared statements"):
        Model(FCData.from_frames(frame, design)).build()
    single = FCDesign.from_mapping(traits={"T": ["A", "B", "C"]}, blocks={"B1": ["A", "B", "C"]})
    with pytest.raises(ValueError, match="cannot identify all free slopes"):
        Model(FCData.from_frames(frame, single)).build()


def test_no_unknown_block_family_fallback(design):
    frame = pd.DataFrame({"person_id": ["P"], "block_id": ["B1"], "response": [[1, 2, 3]]})
    data = FCData.from_frames(frame, design)
    with pytest.raises(ValueError, match="unknown block IDs"):
        Model(data, block_families={"typo": MGPPM()})
    with pytest.raises(TypeError, match="only MGPPM"):
        Model(data, block_families={"B1": "other"})


def test_empty_data_requires_explicit_people(design):
    data = FCData.from_frames(pd.DataFrame(columns=["person_id", "block_id", "response"]), design)
    with pytest.raises(ValueError, match="At least one person"):
        Model(data).build()


@pytest.mark.parametrize(
    "option,value",
    [
        ("draws", 0),
        ("tune", -1),
        ("chains", True),
        ("cores", 1.5),
        ("random_seed", -1),
        ("nuts_sampler", "unknown"),
        ("return_inferencedata", False),
        ("idata_kwargs", {"coords": {"person": ["wrong", "Q"]}}),
        ("idata_kwargs", {"dims": {"theta": ["person", "wrong"]}}),
    ],
)
def test_fit_rejects_invalid_settings_before_sampling(built, monkeypatch, option, value):
    import pymc as pm

    def forbidden(**kwargs):
        raise AssertionError("Invalid settings must not start sampling")

    monkeypatch.setattr(pm, "sample", forbidden)
    with pytest.raises(ValueError):
        built[0].fit(**{option: value})


def test_fit_forwards_settings_and_retains_executed_metadata(built, monkeypatch):
    import pymc as pm
    import xarray as xr

    wrapper, graph = built
    observed = {}
    fake = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({"a": (("chain", "draw"), np.ones((2, 3)))}),
            "sample_stats": xr.Dataset(
                {"diverging": (("chain", "draw"), np.zeros((2, 3), dtype=bool))}
            ),
        }
    )

    sampler_options = {"max_treedepth": 8}

    def sample(**kwargs):
        observed.update(kwargs)
        assert pm.modelcontext(None) is graph.model
        # Match PyMC's in-place normalization without changing caller-owned settings.
        kwargs["nuts"]["target_accept"] = kwargs["target_accept"]
        return fake

    monkeypatch.setattr(pm, "sample", sample)
    result = wrapper.fit(
        draws=3,
        tune=4,
        chains=2,
        cores=1,
        random_seed=82,
        target_accept=0.91,
        nuts=sampler_options,
    )
    assert sampler_options == {"max_treedepth": 8}
    converted = observed.pop("idata_kwargs")["coords"]
    for dimension, labels in graph.model.coords.items():
        assert converted[dimension].tolist() == list(labels)
    assert observed == {
        "draws": 3,
        "tune": 4,
        "chains": 2,
        "cores": 1,
        "random_seed": 82,
        "nuts_sampler": "pymc",
        "target_accept": 0.91,
        "nuts": {"max_treedepth": 8, "target_accept": 0.91},
    }
    assert result.idata is fake
    assert result.built is graph
    assert result.sampling["random_seed"] == 82
    assert result.sampling["actual_draws"] == 3
    assert result.sampling["extra_settings"] == {
        "target_accept": 0.91,
        "nuts": {"max_treedepth": 8},
    }
    assert result.sampling["elapsed_seconds"] >= 0
    assert result.sampling["versions"]["pymc"] == pm.__version__
    assert result.idata.attrs["pyfcirt_data_fingerprint"] == graph.spec.data_fingerprint
