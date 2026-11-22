"""Actual graph/output conversion with controlled transitions, never posterior MCMC."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytest
from numpy.testing import assert_allclose

from pyfcirt.inference.runner import sample_task
from pyfcirt.inference.task import TaskGraph, TaskInfo


class NutsInfo(NamedTuple):
    is_divergent: object
    energy: object
    num_trajectory_expansions: object
    num_integration_steps: object
    acceptance_rate: object


@pytest.fixture
def controlled_task(monkeypatch):
    import pyfcirt.inference.runner as runner

    labels = np.array([1, "1"], dtype=object)
    with pm.Model(coords={"person": labels, "item": labels}) as model:
        a = pm.Normal("a", shape=2, dims="item")
        x = pm.Normal("x", shape=2, dims="person")
        pm.Normal("response", a + x, observed=np.zeros(2), dims="person")
    graph = TaskGraph.from_model(model, extra_outputs={"runtime": a + x})

    def adapt(key, position, **settings):
        import blackjax

        state = blackjax.nuts.init(jax.tree.map(lambda x: x + 1, position), graph.logdensity)
        return state, dict(
            step_size=jnp.asarray(0.2),
            inverse_mass_matrix=jnp.ones(4),
        )

    def retained(key, state, **settings):
        draws = settings["draws"]
        positions = jax.tree.map(
            lambda x: (
                jnp.broadcast_to(x, (draws, *x.shape))
                + jnp.arange(draws).reshape((draws,) + (1,) * x.ndim)
            ),
            state.position,
        )
        samples = state._replace(
            position=positions,
            logdensity=jnp.zeros(draws),
            logdensity_grad=jax.tree.map(jnp.zeros_like, positions),
        )
        info = TaskInfo(
            NutsInfo(
                jnp.zeros(draws, dtype=bool),
                jnp.ones(draws) * 3,
                jnp.ones(draws, dtype=int),
                jnp.ones(draws, dtype=int) * 4,
                jnp.ones(draws) * 0.8,
            ),
            None,
        )
        return state, (samples, info)

    monkeypatch.setattr(runner, "warmup", adapt)
    monkeypatch.setattr(runner, "run_chain", retained)
    return graph


@pytest.mark.parametrize("parallel", [False, True])
def test_common_output_retains_typed_coordinates_and_actual_final_logp(
    controlled_task, monkeypatch, parallel
):
    import pyfcirt.inference.runner as runner

    monkeypatch.setattr(runner.jax, "local_device_count", lambda: 2 if parallel else 1)
    graph = controlled_task
    compile_initial_point = type(graph.model)._make_initial_point
    compilations = []

    def track_initializer(model, **options):
        compilations.append(options["overrides"])
        return compile_initial_point(model, **options)

    monkeypatch.setattr(type(graph.model), "_make_initial_point", track_initializer)
    result, metadata = sample_task(
        graph, draws=np.int64(3), tune=2, chains=2, random_seed=123, var_names=["x", "a"]
    )
    posterior = result["posterior"].to_dataset()
    statistics = result["sample_stats"].to_dataset()
    assert posterior.person.values.tolist() == [1, "1"]
    assert posterior.item.values.tolist() == [1, "1"]
    assert set(posterior.data_vars) == {"x", "a"} and "runtime" not in posterior
    assert "nuts_energy" in statistics and "energy" not in statistics
    assert metadata["chain_method"] == ("parallel" if parallel else "vectorized")
    for name in ["compile_seconds", "warmup_seconds", "sampling_seconds", "postprocessing_seconds"]:
        assert metadata[name] >= 0
    assert metadata["random_seed"] == 123
    assert compilations == [None]
    assert type(metadata["draws"]) is int and metadata["draws"] == posterior.sizes["draw"]
    assert metadata["chains"] == posterior.sizes["chain"]
    assert metadata["target_accept"] == 0.8
    assert "calibration_bank" not in result
    assert "bank_row" not in statistics
    for chain in range(2):
        for draw in range(3):
            point = {name: jnp.asarray(posterior[name][chain, draw].values) for name in ("a", "x")}
            assert_allclose(statistics.lp[chain, draw], graph.logdensity(point), atol=1e-12)


@pytest.mark.parametrize(
    "settings",
    [
        {"draws": True},
        {"chains": 0},
        {"tune": 0},
        {"target_accept": float("nan")},
        {"initial_step_size": float("inf")},
        {"random_seed": True},
        {"dense_mass": None},
        {"progressbar": True},
        {"var_names": "x"},
    ],
)
def test_controls_fail_before_any_graph_or_sampler_access(settings):
    with pytest.raises((ValueError, TypeError)):
        sample_task(None, **settings)


def test_storage_and_initial_values_fail_before_sampling(controlled_task):
    for names in [[], ["runtime"], ["x", "x"]]:
        with pytest.raises(ValueError, match="Storage"):
            sample_task(controlled_task, var_names=names)
    with pytest.raises(ValueError, match="initial-value"):
        sample_task(controlled_task, chains=2, initvals=[{}])


def test_native_initializers_honor_distinct_chain_values(controlled_task):
    starts = [{"a": [1.0, 2.0]}, {"a": [3.0, 4.0]}]
    positions = controlled_task.initial_positions([11, 12], initvals=starts)
    assert_allclose(positions[0]["a"], starts[0]["a"])
    assert_allclose(positions[1]["a"], starts[1]["a"])


@pytest.mark.parametrize("nested", [False, True])
def test_native_parameter_support_survives_jax_translation(nested):
    import pytensor.tensor as pt
    from pytensor.compile.builders import OpFromGraph

    with pm.Model() as model:
        pm.Normal("a")
        x = pm.HalfNormal("x", sigma=1)
        if nested:
            p = pt.scalar("p")
            inner = OpFromGraph([p], [pm.logp(pm.Bernoulli.dist(p=p), 1)], inline=False)
            pm.Potential("y", OpFromGraph([p], [inner(p)], inline=False)(x))
        else:
            pm.Bernoulli("y", p=x, observed=1)
    graph = TaskGraph.from_model(model)
    assert np.isfinite(
        graph.logdensity({"a": jnp.asarray(0.0), "x_log__": jnp.asarray(np.log(0.5))})
    )
    assert np.isneginf(
        graph.logdensity({"a": jnp.asarray(0.0), "x_log__": jnp.asarray(np.log(1.2))})
    )


@pytest.mark.parametrize("nested", [False, True])
def test_untranslatable_native_assertions_are_rejected_instead_of_ignored(nested):
    import pytensor.tensor as pt
    from pytensor.compile.builders import OpFromGraph
    from pytensor.raise_op import Assert

    with pm.Model() as model:
        pm.Normal("a")
        x = pm.Normal("x")
        p = pt.scalar("p")
        expression = Assert("p < 1")(-(p**2), p < 1)
        expression = (
            OpFromGraph([p], [expression], inline=False)(x)
            if nested
            else Assert("x < 1")(-(x**2), x < 1)
        )
        pm.Potential("check", expression)
    with pytest.raises(ValueError, match="assertion"):
        TaskGraph.from_model(model)


@pytest.mark.parametrize("registered", [False, True])
def test_mutable_external_inputs_require_native_registration_and_are_isolated(registered):
    import pytensor

    with pm.Model() as model:
        mu = pm.Data("mu", np.zeros(2)) if registered else pytensor.shared(np.zeros(2))
        pm.Normal("a")
        pm.Normal("x", mu=mu, shape=mu.shape)
    if not registered:
        with pytest.raises(ValueError, match="pm.Data"):
            TaskGraph.from_model(model)
        return
    graph = TaskGraph.from_model(model)
    mu.set_value(np.ones(3))
    (position,) = graph.initial_positions([11])
    assert position["x"].shape == (2,)
    assert np.isfinite(graph.logdensity(position))


def test_observed_shared_inputs_and_output_assertions_follow_the_same_boundary():
    import pytensor
    from pytensor.raise_op import Assert

    with pm.Model() as model:
        pm.Normal("a")
        x = pm.Normal("x")
        pm.Normal("y", x, observed=pytensor.shared(np.array(0.2)))
    with pytest.raises(ValueError, match="pm.Data"):
        TaskGraph.from_model(model)
    with pm.Model() as output_model:
        x = pm.Normal("x")
        pm.Deterministic("checked", Assert("x<1")(x, x < 1))
    with pytest.raises(ValueError, match="assertion"):
        TaskGraph.from_model(output_model)


def test_native_task_and_runtime_outputs_reuse_rv_coordinates_without_a_bank():
    with pm.Model() as model:
        a = pm.Normal("a")
        x = pm.LogNormal("x", mu=a, sigma=0.7)
    graph = TaskGraph.from_model(model, extra_outputs={"prior": pm.logp(x, x)})
    (position,) = graph.initial_positions([10], initvals={"a": 0.2, "x": 1.1})
    assert not hasattr(graph, "bank")
    outputs = graph.outputs(position)
    assert_allclose(outputs["a"], 0.2)
    assert_allclose(outputs["x"], 1.1)
    expected = -0.5 * ((np.log(1.1) - 0.2) / 0.7) ** 2 - np.log(0.7 * 1.1) - 0.5 * np.log(2 * np.pi)
    assert_allclose(outputs["prior"], expected, atol=1e-12)
