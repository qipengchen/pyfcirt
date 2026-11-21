"""Native GTUM factorization and the shared-runner publication boundary."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose


def gtum_model(*, block_effect=False, fixed_parameters=None):
    import pandas as pd

    import pyfcirt as fc

    rows = [("block", 1, "a", "A", 1, 1), ("block", 2, "b", "B", -1, 1)]
    if block_effect:
        rows.append(("block", 3, "c", "A", 1, 1))
    design = pd.DataFrame(
        rows,
        columns=["block_id", "position", "statement_id", "trait", "location_side", "keying"],
    )
    source = fc.FCInput(
        design,
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "block_id": ["block"] * 2,
                    "response": [[1, 2, 3], [2, 1, 3]] if block_effect else [[1, 2], [2, 1]],
                }
            ),
            format="block_long",
        ),
        model=fc.GTUM(block_effect=block_effect),
    )
    return fc.Model(fc.PreparedInputs.from_inputs(source, fixed_parameters=fixed_parameters))


@pytest.mark.parametrize("regression", [False, True])
def test_conditional_target_population_guard_reads_the_current_frozen_recipe(regression):
    import pandas as pd

    import pyfcirt as fc
    from pyfcirt.inference.gtum import _require_gtum

    design = pd.DataFrame(
        [("block", 1, "a", "A", 1, 1), ("block", 2, "b", "B", -1, 1)],
        columns=["block_id", "position", "statement_id", "trait", "location_side", "keying"],
    )
    source = fc.FCInput(
        design,
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": [1, 2], "block_id": ["block"] * 2, "response": [[1, 2], [2, 1]]}
            ),
            format="block_long",
        ),
        model=fc.GTUM(),
    )
    formulas = (
        {
            "theta": fc.Regression(
                "x", pd.DataFrame({"person": [1, 2], "x": [-1.0, 1.0]}), id="person"
            )
        }
        if regression
        else None
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    if formulas is not None:
        model.add_regressions(formulas)
    if regression:
        with pytest.raises(NotImplementedError, match="covariates/groups"):
            _require_gtum(model._bindings)
    else:
        _require_gtum(model._bindings)


@pytest.mark.parametrize(
    "modification", ["transform", "potential", "binding", "free", "parameter", "unobserved"]
)
def test_conditional_target_rejects_native_graph_modifications(modification):
    from dataclasses import replace

    import pymc as pm

    from pyfcirt.inference.gtum import _require_gtum

    model = gtum_model()
    bindings = model._bindings
    native = model.build()
    theta = bindings.parameters["theta"]
    if modification == "transform":
        native.rvs_to_transforms[theta] = object()
    elif modification == "potential":
        with native:
            pm.Potential("extra_likelihood", -(theta[0, 0] ** 2))
    elif modification == "binding":
        native.named_vars[theta.name] = theta + 0
    elif modification == "free":
        with native:
            pm.Normal("unrelated")
    elif modification == "parameter":
        response = bindings.responses[0]
        bindings.responses = (replace(response, parameters=(*response.parameters, theta)),)
    else:
        response = bindings.responses[0]
        bindings.responses = (replace(response, batch=replace(response.batch, y=None)),)
    with pytest.raises(ValueError, match="Conditional MH"):
        _require_gtum(bindings)


def test_conditional_target_rejects_non_gtum_measurements():
    from pyfcirt.inference.gtum import _require_gtum
    from tests.helpers import mixed_model

    with pytest.raises(ValueError, match="original logistic GTUM"):
        _require_gtum(mixed_model()._bindings)


@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_conditional_target_rejects_an_unanswered_non_gtum_unit():
    import pandas as pd

    import pyfcirt as fc
    from pyfcirt.inference.gtum import _require_gtum

    inputs = fc.FCInput(
        "[traits]\nA ~ a b c d\n[blocks]\nobserved ~ a b\nunanswered ~ c d\nside + ~ a b",
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1], "block_id": ["observed"], "response": [[1, 2]]}),
            format="block_long",
        ),
        model=fc.GTUM(),
        blocks=(fc.BlockSpec("unanswered", model=fc.TIRT()),),
    )
    prepared = fc.PreparedInputs.from_inputs(inputs)
    assert tuple(type(spec) for spec in prepared.measurement_specs) == (fc.GTUM, fc.TIRT)
    assert [batch.unit_index for batch in prepared.batches] == [0]
    with pytest.raises(ValueError, match="original logistic GTUM"):
        _require_gtum(fc.Model(prepared)._bindings)


def test_native_person_factors_equal_the_actual_full_task_up_to_global_prior_constants():
    import jax

    from pyfcirt.inference.gtum import _conditional_task

    model = gtum_model(block_effect=True)
    graph = _conditional_task(model._bindings)
    (initial,) = graph.initial_positions([5])
    theta = initial["theta"]

    def person(value):
        return graph.outputs({**initial, "theta": value})["__mode_person_logp"]

    def full(value):
        return graph.logdensity({**initial, "theta": value})

    changed = theta.at[0, 0].add(0.4)
    difference = person(changed) - person(theta)
    assert_allclose(difference[1], 0, atol=1e-12)
    assert_allclose(difference.sum(), full(changed) - full(theta), atol=1e-12)
    assert_allclose(
        jax.grad(lambda x: person(x).sum())(changed), jax.grad(full)(changed), atol=1e-12
    )
    assert not hasattr(graph, "bank") and not hasattr(graph, "row_name")


def test_gtum_proposal_context_ignores_old_trait_and_preserves_other_parameters(monkeypatch):
    import pyfcirt.inference.gtum as gtum

    model = gtum_model()
    graph = gtum._conditional_task(model._bindings)
    (position,) = graph.initial_positions([6])
    initial = {name: np.array(value) for name, value in position.items()}
    move = gtum._mode_update(graph, model._bindings)
    import jax

    updated, stats = jax.jit(move)(jax.random.key(7), position)
    assert updated["theta"].shape == initial["theta"].shape
    assert 0 <= stats["imh_acceptance"] <= 1
    for name in initial:
        assert_allclose(position[name], initial[name])
        if name != "theta":
            assert_allclose(updated[name], initial[name])
    from pyfcirt.inference.conditional_mode import ModeInfo

    def proposed_only(key, value, *, logdensity, proposal):
        # Test proposal independence separately from its legitimate dependence
        # on the current value in the MH correction.
        yes = jnp.ones_like(value, dtype=bool)
        no = jnp.zeros_like(value, dtype=bool)
        return proposal.sample(key), ModeInfo(
            yes, jnp.ones_like(value), no, no, jnp.zeros_like(value)
        )

    monkeypatch.setattr("pyfcirt.inference.conditional_mode.independence_step", proposed_only)
    independent = gtum._mode_update(graph, model._bindings)
    low = {**position, "theta": position["theta"].at[:, 0].set(-100.0)}
    high = {**position, "theta": position["theta"].at[:, 0].set(100.0)}
    left, _ = jax.jit(independent)(jax.random.key(8), low)
    right, _ = jax.jit(independent)(jax.random.key(8), high)
    assert_allclose(left["theta"], right["theta"], atol=0)


def test_shared_runner_result_publication_is_atomic_and_preserves_native_outputs(monkeypatch):
    import pyfcirt.inference.gtum as gtum
    import pyfcirt.inference.runner as runner

    model = gtum_model()
    model.idata["information"] = xr.Dataset({"old": 1})
    model.idata.attrs.update(study="kept", sampling_nuts_sampler="old")
    calls = []
    posterior = xr.Dataset(
        {"theta": (("chain", "draw", "person", "trait"), np.zeros((1, 2, 2, 2)))},
        coords={"chain": [0], "draw": [2, 5], "person": [1, 2], "trait": ["A", "B"]},
    )

    def controlled(graph, **settings):
        calls.append((graph, settings))
        return xr.DataTree.from_dict({"posterior": posterior}), {"controlled_run": True}

    monkeypatch.setattr(runner, "sample_task", controlled)
    result = gtum.sample_gtum(
        model, draws=2, tune=1, chains=1, random_seed=7, compute_convergence_checks=False
    )
    assert result is model.idata and "information" not in result
    assert result.attrs["study"] == "kept" and "sampling_nuts_sampler" not in result.attrs
    assert json.loads(result.attrs["sampling"])["controlled_run"]
    assert calls[0][1]["target_accept"] == 0.8 and calls[0][1]["random_seed"] == 7
    assert calls[0][0].model is not model.build()
    assert callable(calls[0][1]["extra_step"])
    before = result.copy(deep=True)
    with pytest.raises(ValueError, match="overwrite"):
        gtum.sample_gtum(model)

    def failure(*args, **kwargs):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(runner, "sample_task", failure)
    with pytest.raises(RuntimeError, match="controlled failure"):
        gtum.sample_gtum(model, overwrite=True, compute_convergence_checks=False)
    xr.testing.assert_identical(model.idata, before)
    monkeypatch.setattr(
        runner,
        "sample_task",
        lambda *a, **k: (xr.DataTree.from_dict({"posterior": posterior}), {"invalid": object()}),
    )
    with pytest.raises(TypeError):
        gtum.sample_gtum(model, overwrite=True, compute_convergence_checks=False)
    xr.testing.assert_identical(model.idata, before)


def test_removed_pymc_engine_has_no_compatibility_dispatch():
    from pyfcirt.inference.gtum import sample_gtum

    with pytest.raises(TypeError, match="engine"):
        sample_gtum(None, engine="pymc")


def test_required_final_storage_policy_is_owned_by_model():
    model = gtum_model()
    with pytest.raises(TypeError, match="sequence"):
        model._final_sampling_names("theta")
    with pytest.raises(ValueError, match="required final"):
        model._final_sampling_names(["theta"])
    assert "theta" in model._final_sampling_names()
