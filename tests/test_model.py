"""Analysis-state and native sampler wiring without posterior MCMC."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymc as pm
import pytest
import xarray as xr

import pyfcirt as fc
from tests.helpers import mixed_inputs, mixed_model, workflow_point


@pytest.mark.parametrize("channels", ["fc", "items", "mixed"])
def test_model_accepts_raw_declarations_and_reuses_prepared_inputs(channels):
    forced_choice, items = mixed_inputs()
    data = {"fc": forced_choice, "items": items, "mixed": (forced_choice, items)}[channels]
    prepared = fc.PreparedInputs.from_inputs(data)
    actual = fc.Model(data)
    assert actual._built is None
    assert fc.Model(prepared=prepared).prepared is prepared
    xr.testing.assert_equal(actual.prepared.constant_data, prepared.constant_data)
    xr.testing.assert_identical(actual.prepared.observed_data, prepared.observed_data)


def test_models_share_read_only_metadata_and_isolate_edits():
    prepared = mixed_model().prepared
    first, second = fc.Model(prepared), fc.Model(prepared)
    graph = first.build()
    identity = prepared.scoring_batch_identity
    loading = prepared.parameter_layouts["loading"].entry_data
    with pytest.raises((AttributeError, TypeError)):
        prepared.constant_data["obs_person"] = 100
    with pytest.raises(TypeError):
        prepared.coords["person"] = np.array([100])
    for group, column, authority in (
        ("design", "obs_person", prepared.constant_data["obs_person"]),
        ("parameter_data/loading", "trait_index", loading["trait_index"]),
        ("categories", "position", prepared.result_metadata["categories"]["position"]),
    ):
        published = first.idata[group][column]
        assert np.shares_memory(published.values, authority.values)
        with pytest.raises(ValueError, match="read-only"):
            published.values.flat[0] = 100
        editable = first.idata[group].to_dataset().copy(deep=True)
        editable[column].values.flat[0] = 100
        first.idata[group] = editable
        xr.testing.assert_identical(second.idata[group][column].variable, authority.variable)
    assert first.build() is graph
    assert second.prepared.scoring_batch_identity == identity

    editable = first.idata["design"].to_dataset().copy(deep=True)
    editable.obs_batch.values[0] = 100
    first.idata["design"] = editable
    prediction = first.prior_predictive(draws=2, random_seed=42, store=False)
    xr.testing.assert_identical(
        prediction["design"]["obs_batch"].variable,
        prepared.response_metadata["design"].obs_batch.variable,
    )
    first.set_priors({"items.intercept": pm.Normal.dist(mu=0.2, sigma=1)})
    xr.testing.assert_identical(
        first.idata["design"].to_dataset(), second.idata["design"].to_dataset()
    )
    assert not first.idata["design"]["obs_person"].values.flags.writeable
    prediction = second.prior_predictive(draws=2, random_seed=42, store=False)
    assert not prediction["design"]["obs_batch"].values.flags.writeable
    assert not next(iter(prediction["observed_data"].data_vars.values())).values.flags.writeable


def test_prepare_updates_the_same_model_only_after_success():
    inputs = mixed_inputs()
    model = fc.Model()
    empty = model.idata
    with pytest.raises(ValueError, match="Prepare the model"):
        model.build()
    with pytest.raises(ValueError, match="theta_parameterization"):
        model.prepare(inputs, theta_parameterization="invalid")
    assert model.idata is empty and not empty.children
    assert model.prepare(inputs, theta_parameterization="noncentered") is model
    assert model._built is None
    assert model.prepared.population.theta_parameterization == "noncentered"
    assert model.data_summary().equals(fc.data_summary(model.prepared))
    assert isinstance(model.build(), pm.Model)
    prepared, results, graph = model.prepared, model.idata, model.build()
    with pytest.raises(ValueError, match="already prepared"):
        model.prepare(inputs)
    assert model.prepared is prepared and model.idata is results and model.build() is graph


def test_add_items_keeps_local_item_declared_options():
    _, item_input = mixed_inputs()
    model = (
        fc.Model()
        .add_items(
            item_input.design,
            item_input.responses.frame,
            format="item_long",
            model="DichotomousIRT",
            categories=(0, 1),
            items=item_input.items,
        )
        .prepare()
    )
    reference = fc.Model(item_input)
    xr.testing.assert_identical(model.prepared.observed_data, reference.prepared.observed_data)
    assert model._built is None


def test_add_fc_uses_declared_table_layout_and_response_format():
    import pandas as pd

    design = "[traits]\nA ~ a\nB ~ b\n[blocks]\npair ~ a b"
    responses = pd.DataFrame({"respondent": ["a", "b"], "choice": [0, 1]})
    columns = (fc.ComparisonColumn("choice", "pair", "a", "b"),)
    tirt = fc.TIRT(loading_anchors={"a": 1.0, "b": 1.0})
    actual = (
        fc.Model()
        .add_fc(
            design,
            responses,
            format="comparison_wide",
            person_id_column="respondent",
            comparison_columns=columns,
            model=tirt,
            response_format=None,
        )
        .prepare()
    )
    table = fc.ResponseTable(
        responses,
        format="comparison_wide",
        person_id_column="respondent",
        comparison_columns=columns,
    )
    expected = fc.Model(fc.FCInput(design, table, model=tirt, response_format=None))
    assert actual._built is None
    assert actual.prepared.constant_data.attrs["units"][0]["response_format"] is None
    xr.testing.assert_identical(actual.prepared.observed_data, expected.prepared.observed_data)


def test_registered_channels_prepare_together_and_can_be_corrected():
    forced_choice, items = mixed_inputs()
    model = fc.Model()
    empty = model.idata
    with pytest.raises(ValueError, match="Add response tables"):
        model.prepare()
    assert (
        model.add_fc(
            forced_choice.design,
            forced_choice.responses.frame,
            format="block_long",
            blocks=forced_choice.blocks,
        )
        is model
    )
    model.add_items(
        items.design, items.responses.frame.drop(columns="response"), format="item_long"
    )
    with pytest.raises(ValueError, match="after adding response tables"):
        model.prepare(items)
    with pytest.raises(ValueError, match="response"):
        model.prepare()
    assert model.idata is empty
    model.add_items(
        items.design,
        items.responses.frame,
        format="item_long",
        items=items.items,
    )
    assert model.prepare(theta_parameterization="noncentered") is model
    expected = fc.PreparedInputs.from_inputs(
        (forced_choice, items),
        theta_parameterization="noncentered",
    )
    xr.testing.assert_equal(model.prepared.constant_data, expected.constant_data)
    assert model.prepared.calibration_identity == expected.calibration_identity
    assert model.prepared.scoring_batch_identity == expected.scoring_batch_identity
    xr.testing.assert_identical(model.prepared.observed_data, expected.observed_data)
    assert model.data_summary().equals(fc.data_summary(expected))
    with pytest.raises(ValueError, match="already prepared"):
        model.add_items(items.design, items.responses.frame, format="item_long")


def test_model_coef_binds_current_results(supplied_model):
    result = supplied_model.coef(blocks=["pair"], kind="stats")
    assert len(result) > 0
    assert result.equals(fc.coef(supplied_model, blocks=["pair"], kind="stats"))
    for name in (
        "coef",
        "posterior_predictive",
        "information",
        "reliability",
        "score",
        "score_model",
        "simulate",
    ):
        assert getattr(fc.Model, name) is getattr(fc, name)


def test_incremental_updates_scope_existing_prior_cache_and_restore_it_after_failure(monkeypatch):
    import pandas as pd

    from pyfcirt.data import regressions
    from pyfcirt.models.priors import PriorPreparation

    model = mixed_model()
    templates = []
    prepare_regressions, prepare_priors = (
        regressions.prepare_regressions,
        regressions.prepare_priors,
    )

    def template():
        return PriorPreparation.default(pm.Normal.dist, mu=2.75, sigma=1.5)

    def remember_template():
        current = template()
        assert template() is current
        templates.append(current)

    def regression(*args, **kwargs):
        remember_template()
        return prepare_regressions(*args, **kwargs)

    def prior(*args, **kwargs):
        remember_template()
        return prepare_priors(*args, **kwargs)

    monkeypatch.setattr(regressions, "prepare_regressions", regression)
    monkeypatch.setattr(regressions, "prepare_priors", prior)
    assert template() is not template()
    with PriorPreparation():
        outer = template()
        people = pd.DataFrame({"person_id": [1, 2], "x": [-1.0, 1.0]})
        model.add_regressions("theta ~ x", person_data=people)
        assert template() is outer
        model.set_priors({"items.intercept": pm.Normal.dist(mu=0.2, sigma=1)})
        assert template() is outer
        before = model.prepared
        with pytest.raises(ValueError):
            model.add_regressions("theta ~ missing", person_data=people)
        assert model.prepared is before and template() is outer
        assert len(templates) == 3 and all(value is not outer for value in templates)
        assert len({id(value) for value in templates}) == 3
    assert template() is not template()


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("offset", [-8, -2, 2, 8])
@pytest.mark.parametrize("point_scoring", [False, True], ids=["structural", "point"])
def test_attachment_accepts_native_precision_roundoff_without_repair(
    precision, offset, point_scoring
):
    from dataclasses import replace

    import jax
    import pandas as pd
    import pytensor

    from tests.data.test_schema import pair_input

    source = replace(
        pair_input(),
        design=pd.DataFrame(
            dict(
                block_id=["pair", "pair"],
                position=[1, 2],
                statement_id=["a", "b"],
                trait=["A", "B"],
                keying=[1, 1],
                fix_loading=[0.7, 1.1],
            )
        ),
    )
    with pytensor.config.change_flags(floatX=precision):
        if point_scoring:
            reference = mixed_model()
            responses = {
                channel: data.responses
                for channel, data in zip(("fc", "items"), mixed_inputs(), strict=True)
            }
            point = workflow_point(reference).drop_dims("person").astype(precision)
            model = fc.score_model(reference, responses, parameters=point)
            node = model.build()["item_intercept"]
            message = "item_intercept: sample values conflict with fixed calibration values"
        else:
            model = fc.Model(fc.PreparedInputs.from_inputs(source))
            node = model.build()["loading"]
            message = "loading: sample values conflict with fixed entries"
        assert node.dtype == precision
        values = node.eval().copy()
        (entry_dim,) = model.build().named_vars_to_dims[node.name]
    posterior = xr.Dataset(
        {node.name: (("chain", "draw", entry_dim), values[None, None])},
        coords={"chain": [0], "draw": [0], entry_dim: model.prepared.coords[entry_dim]},
    )
    posterior = model.prepared.annotate_coordinates(posterior)
    epsilon = np.finfo(values.dtype).eps
    posterior[node.name].values += offset * epsilon
    before = posterior.copy(deep=True)
    model.attach_samples(xr.DataTree.from_dict({"posterior": posterior}))
    xr.testing.assert_equal(model.idata["posterior"].to_dataset(), posterior)
    xr.testing.assert_identical(posterior, before)
    invalid = posterior.copy(deep=True)
    invalid[node.name].values += np.copysign(16 * epsilon, offset)
    fresh = fc.Model(model.prepared)
    with jax.enable_x64(False), pytest.raises(ValueError, match=message):
        fresh.attach_samples(xr.DataTree.from_dict({"posterior": invalid}))
    assert "posterior" not in fresh.idata
    wrong_labels = posterior.assign_coords({entry_dim: posterior[entry_dim] + 100})
    with pytest.raises(ValueError, match=f"{entry_dim}: sample coordinates differ"):
        fresh.attach_samples(xr.DataTree.from_dict({"posterior": wrong_labels}))
    assert "posterior" not in fresh.idata


@pytest.mark.parametrize("common", [1.2, -0.7])
def test_attached_one_pl_samples_preserve_shared_identity_without_changing_prior_support(
    common, monkeypatch
):
    import pandas as pd

    source = fc.ItemInput(
        {"traits": {"A": ["x", "y"]}},
        fc.ResponseTable(
            pd.DataFrame(dict(person_id=[1, 1], item_id=["x", "y"], response=[0, 1])),
            format="item_long",
        ),
        model=fc.DichotomousIRT(kind="1PL", priors={"loading_magnitude": pm.Normal.dist()}),
    )
    with pytest.warns(fc.IdentificationWarning, match="orientation"):
        prepared = fc.PreparedInputs.from_inputs(source)
    model = fc.Model(prepared)
    values = {
        "loading": [common],
        "item_intercept": [0.2, -0.3],
        "lower_asymptote": [0.0, 0.0],
        "upper_asymptote": [1.0, 1.0],
        "population_mean": [[0.0]],
        "covariance": [[1.0]],
    }
    point = xr.Dataset(
        {
            layout.kind: xr.DataArray(
                values[layout.kind],
                dims=layout.dims,
                coords={dim: prepared.coords[dim] for dim in layout.dims},
            )
            for layout in prepared.parameters
            if layout.kind in values
        }
    )
    point["theta"] = xr.DataArray(
        [[0.4]],
        dims=("person", "trait"),
        coords={dim: prepared.coords[dim] for dim in ("person", "trait")},
    )
    posterior = point.expand_dims(chain=[0], draw=[0, 1])
    posterior = model.prepared.annotate_coordinates(posterior)
    conflict = posterior.reindex(loading_id=[0, 1], fill_value=common)
    conflict["loading"][..., 1] += 0.2
    with pytest.raises(ValueError, match="coordinate"):
        model.attach_samples(xr.DataTree.from_dict({"posterior": conflict}))
    assert "posterior" not in model.idata and model._built is None
    supplied = xr.DataTree.from_dict({"posterior": posterior})
    supplied.attrs.update(inference_method="supplied", test_provenance="final arrays; no inference")
    with monkeypatch.context() as patch:
        annotate = type(prepared)._attach_coordinates
        checks = []

        def count_annotation(self, dataset, **kwargs):
            checks.append(kwargs)
            return annotate(self, dataset, **kwargs)

        patch.setattr(type(prepared), "_attach_coordinates", count_annotation)
        model.attach_samples(supplied)
    assert checks == [{"positions": {}, "require": True}]
    xr.testing.assert_equal(model.idata["posterior"].to_dataset(), posterior)
    assert model._built is None
    before = model.idata.copy(deep=True)
    point = fc.parameter_point(model)
    assert point.loading.item() == common
    np.testing.assert_array_equal(point.item_intercept, values["item_intercept"])
    answers = fc.ResponseTable(
        pd.DataFrame(dict(person_id=[9, 9], item_id=["x", "y"], response=[0, 1])),
        format="item_long",
    )
    scored = fc.score_model(model, answers)
    assert scored.prepared.coords["person"].tolist() == [9]
    assert scored.idata["fixed_parameters"]["loading"].item() == common
    assert {rv.name for rv in scored.build().free_RVs} == {"theta"}
    np.testing.assert_array_equal(scored.build()["loading"].eval(), [common])

    generated = fc.simulate(model, source="fixed", parameters=point, random_seed=17)
    sampled = fc.simulate(model, source="posterior", draw=(0, 1), random_seed=17)
    assert generated.idata["parameters/generation"]["loading"].item() == common
    xr.testing.assert_equal(generated.prepared.observed_data, sampled.prepared.observed_data)
    explicit = fc.information(model, parameters=point, store=False, progressbar=False)
    implicit = fc.information(model, store=False, progressbar=False)
    eta = common * 0.4 - np.asarray(values["item_intercept"])
    probability = 1 / (1 + np.exp(-eta))
    np.testing.assert_allclose(
        explicit.item_information.values[0, :, 0, 0],
        common**2 * probability * (1 - probability),
        atol=1e-12,
    )
    xr.testing.assert_equal(explicit.test_information, implicit.test_information)
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize("precision", [None, "float32", "float64"])
def test_platform_configures_real_jax_and_pytensor_precision_in_a_fresh_session(
    tmp_path, precision
):
    environment = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="false",
        PYTENSOR_FLAGS="floatX=float32,cxx=,mode=FAST_COMPILE",
    )
    code = f"""
import jax
import pytensor
assert not jax.config.x64_enabled
assert pytensor.config.floatX == 'float32'
import pyfcirt as fc
assert jax.config.x64_enabled
assert pytensor.config.floatX == 'float64'
assert jax.config.jax_platforms == 'cpu'
assert 'set_platform' in fc.__all__
precision = {precision!r}
if precision is not None:
    fc.set_platform('cpu', precision=precision)
import jax.numpy as jnp
import pymc as pm
expected = precision or 'float64'
assert str(jnp.ones(1).dtype) == expected
assert jax.default_backend() == 'cpu'
assert jnp.ones(1).device == jax.devices('cpu')[0]
assert pm.Normal.dist().dtype == expected
assert pytensor.config.floatX == expected
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("platform", [None, "cpu", "gpu", "cuda", "rocm", "tpu"])
def test_platform_preserves_native_device_selection_and_keeps_cpu_for_accelerators(
    monkeypatch, platform
):
    import jax

    calls = []
    monkeypatch.setattr(jax.config, "update", lambda name, value: calls.append((name, value)))
    monkeypatch.setattr(jax, "devices", lambda backend: calls.append(("devices", backend)))
    fc.set_platform(platform)
    if platform is None:
        expected = []
    elif platform == "gpu":
        expected = [("jax_platforms", None), ("jax_platform_name", "gpu")]
    else:
        expected = [
            ("jax_platform_name", ""),
            ("jax_platforms", "cpu" if platform == "cpu" else f"{platform},cpu"),
        ]
    if platform is not None:
        expected.append(("devices", platform))
    expected.append(("jax_enable_x64", True))
    assert calls == expected


@pytest.mark.parametrize(
    ("platform", "available"),
    [
        ("gpu", "cuda"),
        ("gpu", "rocm"),
        ("cpu", "cuda"),
        ("cuda", "cuda"),
        ("rocm", "rocm"),
        ("tpu", "tpu"),
        ("gpu", "tpu"),
        ("rocm", "cuda"),
        ("not_a_backend", "cuda"),
    ],
)
def test_platform_native_backend_initialization_and_parallel_devices(tmp_path, platform, available):
    environment = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        JAX_PLATFORMS="",
        JAX_PLATFORM_NAME="",
        PYTENSOR_FLAGS="cxx=,mode=FAST_COMPILE",
    )
    code = f"""
from types import SimpleNamespace
from unittest.mock import patch
import jax
import pytensor
import pyfcirt as fc
from jax._src import xla_bridge as xb
from jax._src.pmap import _get_mesh_devices

# Mock only the hardware clients; leave JAX initialization, aliases and default
# backend selection intact, including the mesh-selection path used by pmap.
def client(platform):
    device = SimpleNamespace(process_index=0, platform=platform)
    return SimpleNamespace(
        platform=platform, process_index=lambda: 0, device_count=lambda: 1,
        local_devices=lambda: [device], devices=lambda: [device],
    )

xb._discover_and_register_pjrt_plugins()
assert not xb._backends
platform, available = {platform!r}, {available!r}
cpu = client("cpu")
accelerator = client("tpu" if available == "tpu" else "gpu")
with patch.dict(xb._backend_factories, {{}}, clear=True), patch.object(
    xb.hardware_utils, "has_visible_nvidia_gpu", return_value=True
):
    xb.register_backend_factory("cpu", lambda: cpu, priority=0, fail_quietly=False)
    xb.register_backend_factory(available, lambda: accelerator, priority=200)
    if available != "tpu":
        # GPU preference must also win when a higher-priority TPU is present.
        xb.register_backend_factory("tpu", lambda: client("tpu"), priority=300)
    missing = platform not in {{"cpu", available}} and not (
        platform == "gpu" and available in {{"cuda", "rocm"}}
    )
    if missing:
        jax.config.update("jax_platforms", "cpu")
        keys = ("jax_platforms", "jax_platform_name", "jax_enable_x64")
        previous = [jax.config.values[name] for name in keys]
        precision = pytensor.config.floatX
        try:
            fc.set_platform(platform, precision="float32")
        except RuntimeError:
            pass
        else:
            raise AssertionError("An unavailable explicit accelerator must raise")
        assert [jax.config.values[name] for name in keys] == previous
        assert pytensor.config.floatX == precision
        assert jax.default_backend() == "cpu"
        assert jax.devices() == cpu.devices()
        with jax.enable_x64(False):
            try:
                fc.set_platform(platform, precision="float32")
            except RuntimeError:
                pass
            else:
                raise AssertionError("An unavailable initialized backend must raise")
        assert [jax.config.values[name] for name in keys] == previous
        assert pytensor.config.floatX == precision
        assert xb.get_backend() is cpu
    else:
        fc.set_platform(platform)
        fc.set_platform()  # Preserve the explicit choice when only resetting precision.
        selected = cpu if platform == "cpu" else accelerator
        assert jax.default_backend() == selected.platform
        assert jax.devices() == selected.devices()
        assert jax.devices("cpu") == cpu.devices()
        assert _get_mesh_devices(None, None, 1, None, True) == tuple(selected.devices())
        if platform == "gpu":
            # The absent GPU vendor must not become a required backend.
            assert set(xb.backends()) == {{"cpu", available, "tpu"}}
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_platform_validates_cuda_with_native_hardware_visibility(tmp_path):
    environment = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        JAX_PLATFORMS="",
        JAX_PLATFORM_NAME="",
        PYTENSOR_FLAGS="cxx=,mode=FAST_COMPILE",
    )
    code = """
import jax
import pyfcirt as fc
from jax._src import xla_bridge as xb

assert not xb._backends
fc.set_platform()
assert not xb._backends
visible_gpu = xb.hardware_utils.has_visible_nvidia_gpu()
try:
    fc.set_platform("cuda")
except RuntimeError as error:
    assert "cuda" in str(error)
    if not visible_gpu:
        assert set(xb.backends()) == {"cpu"}
        assert jax.devices("cpu")
else:
    assert visible_gpu, "Explicit CUDA silently selected CPU on a CPU-only host"
    assert jax.default_backend() == "gpu"
    assert jax.devices("cuda")
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_platform_rejects_invalid_precision_before_changing_settings(monkeypatch):
    import jax

    monkeypatch.setattr(jax.config, "update", lambda *args: pytest.fail("settings changed"))
    with pytest.raises(ValueError, match="precision"):
        fc.set_platform("cpu", precision="float16")


def test_platform_restores_settings_after_interrupted_probe(monkeypatch):
    import jax
    import pytensor

    previous = jax.config.values
    precision = pytensor.config.floatX

    def interrupt(platform):
        raise KeyboardInterrupt(platform)

    monkeypatch.setattr(jax, "devices", interrupt)
    with pytest.raises(KeyboardInterrupt, match="gpu"):
        fc.set_platform("gpu", precision="float32")

    assert jax.config.values == previous
    assert pytensor.config.floatX == precision


def test_build_is_lazy_cached_native_and_shares_one_theta(monkeypatch):
    from pyfcirt.models import assembly

    def unexpected_graph_replace(*args, **kwargs):
        pytest.fail("The forward clone already owns these observed responses")

    monkeypatch.setattr(assembly, "graph_replace", unexpected_graph_replace)
    model = mixed_model()
    assert model._built is None
    graph = model.build()
    assert isinstance(graph, pm.Model) and model.build() is graph
    assert [rv.name for rv in graph.free_RVs].count("theta") == 1
    assert model._bindings.evaluation().model is not graph
    view = model._bindings.evaluation()
    (name,) = view.responses[0].batch.distribution_names
    assert view is model._bindings.evaluation(var_names=[name])
    assert name in view.model.named_vars
    assert all(binding is view.model[binding.name] for binding in view.parameters.values())
    assert view.parameters["theta"] is view.model["theta"]
    assert all(r.distribution is view.model[r.batch.response_name] for r in view.responses)
    assert isinstance(model.idata, xr.DataTree)
    assert "MGPPM" in model.summary()


@pytest.mark.parametrize("ranking", [False, True])
@pytest.mark.parametrize("generation", [False, True])
def test_native_response_axes_and_plates_preserve_observed_forward_lifecycles(ranking, generation):
    import pandas as pd
    from pymc.model_graph import ModelGraph

    from pyfcirt.data.prepare import _prepare_generation

    model = mixed_model()
    if ranking:
        inputs = fc.FCInput(
            {"traits": {"A": ["a", "c"], "B": ["b"]}, "blocks": {"triple": ["a", "b", "c"]}},
            fc.ResponseTable(
                pd.DataFrame({"person_id": ["p"], "block_id": ["triple"], "response": [[1, 2, 3]]}),
                format="block_long",
            ),
            model=fc.TIRTRank(),
        )
        model = fc.Model(fc.PreparedInputs.from_inputs(inputs))
    if generation:
        schedule = fc.generation_schedule(model, person_ids=["new", "new2"])
        model = fc.Model(_prepare_generation(model.prepared, schedule))

    native = model.build()
    assert bool(native.observed_RVs) == (not generation)
    assert not native.potentials
    for response in model._bindings.responses:
        batch = response.batch
        if generation:
            assert response.distribution not in native.basic_RVs and not response.augmentation
        elif ranking:
            contrast, indicator = response.augmentation
            assert indicator is native[f"{batch.name}_event_in_region"]
            assert native.named_vars_to_dims[indicator.name] == (batch.obs_dim,)
            assert contrast in native.free_RVs and response.distribution not in native.basic_RVs
        else:
            assert response.distribution is native[batch.response_name]
            assert native.named_vars_to_dims[batch.response_name] == (batch.obs_dim,)
    forward = model._bindings.evaluation()
    for graph in (native, forward.model):
        assert set(graph.coords) == set(model.prepared.coords)
        for dim, labels in model.prepared.coords.items():
            np.testing.assert_array_equal(graph.coords[dim], labels)
        ModelGraph(graph).get_plates(var_names=[rv.name for rv in graph.observed_RVs] or ["theta"])
    from pytensor.graph.traversal import ancestors

    for fitting, response in zip(model._bindings.responses, forward.responses, strict=True):
        fitting_names = {rv.name for rv in fitting.augmentation}
        assert fitting_names.isdisjoint(forward.model.named_vars)
        assert fitting_names.isdisjoint(node.name for node in ancestors([response.distribution]))
        assert response.distribution is forward.model[response.batch.response_name]
        assert forward.model.named_vars_to_dims[response.batch.response_name] == (
            response.batch.obs_dim,
        )
        assert (response.distribution in forward.model.observed_RVs) == (not generation)
        assert (response.distribution in forward.model.free_RVs) == generation


@pytest.mark.parametrize("invalid", [{}, xr.DataTree()])
def test_model_rejects_old_or_incomplete_transport(invalid):
    with pytest.raises(TypeError, match="FCInput or ItemInput"):
        fc.Model(invalid)


@pytest.mark.parametrize("update", ["prior", "regression"])
def test_successful_specification_change_invalidates_prior_results_atomically(update):
    import pandas as pd

    model = mixed_model()
    fc.prior_predictive(model, draws=1, random_seed=60, save_responses=True)
    previous, graph = model.idata, model.build()
    before = previous.copy(deep=True)
    with pytest.raises(ValueError):
        model.set_priors({"fc.unknown": pm.Normal.dist()})
    assert model.idata is previous and model.build() is graph
    xr.testing.assert_identical(model.idata, before)
    if update == "prior":
        model.set_priors({"fc.loading_magnitude": pm.LogNormal.dist(mu=0.2, sigma=0.3)})
    else:
        model.add_regressions(
            "theta ~ x", person_data=pd.DataFrame({"person_id": [1, 2], "x": [0.0, 1.0]})
        )
    assert model.idata is not previous and model._built is None
    assert not {"prior", "prior_predictive"}.intersection(model.idata.children)
    assert {"prior", "prior_predictive"}.issubset(previous.children)
    xr.testing.assert_identical(previous, before)
    for group in ("design", "categories", "response_support"):
        xr.testing.assert_identical(model.idata[group], previous[group])
    for name, support in previous["response_support"].children.items():
        for variable in support.data_vars:
            assert np.shares_memory(
                model.idata[f"response_support/{name}"][variable].data, support[variable].data
            )


@pytest.mark.parametrize("native_checks", [None, True])
@pytest.mark.parametrize("sampler", ["default", None, "pymc", "blackjax", "numpyro", "nutpie"])
def test_fit_forwards_native_options_and_invalidates_only_after_success(
    monkeypatch, native_checks, sampler
):
    import pandas as pd

    model = mixed_model()
    model.idata["information"] = xr.Dataset({"old": 1})
    model.idata.attrs.update(sampling="old GTUM record", sampling_previous=True, study="kept")
    graph = model.build()
    calls = []
    posterior = workflow_point(model).expand_dims(chain=["mock"], draw=[0, 1])
    prepared = model.prepared
    prepared_before = prepared.data.copy(deep=True)
    bindings = dict(model._bindings.parameters)

    def capture(**kwargs):
        assert pm.modelcontext(None) is graph
        calls.append(kwargs)
        return xr.DataTree.from_dict(
            {"posterior": posterior, "sample_stats": xr.Dataset({"ok": 1})}
        )

    monkeypatch.setattr(pm, "sample", capture)
    conversion = {"log_likelihood": False, "coords": {"person": ["wrong"], "extra": ["kept"]}}
    returned = model.fit(
        draws=2,
        tune=0,
        chains=1,
        nuts={"chain_method": "vectorized"},
        progressbar=False,
        random_seed=17,
        idata_kwargs=conversion,
        **({} if sampler == "default" else {"nuts_sampler": sampler}),
        **({} if native_checks is None else {"compute_convergence_checks": native_checks}),
    )
    assert conversion == {
        "log_likelihood": False,
        "coords": {"person": ["wrong"], "extra": ["kept"]},
    }
    received = calls[0]["idata_kwargs"]
    assert received["log_likelihood"] is False
    assert received["coords"]["extra"] == ["kept"]
    assert received["coords"]["person"].tolist() == [1, 2]
    assert returned is model.idata
    expected_sampler = "blackjax" if sampler == "default" else sampler
    assert calls[0]["nuts_sampler"] == expected_sampler
    if expected_sampler is None:
        assert "sampling_nuts_sampler" not in model.idata.attrs
    else:
        assert model.idata.attrs["sampling_nuts_sampler"] == expected_sampler
    assert calls[0]["nuts"] == {"chain_method": "vectorized"}
    assert calls[0]["progressbar"] is False
    assert calls[0]["compute_convergence_checks"] is bool(native_checks)
    assert model.idata.attrs["sampling_draws"] == 2
    assert all(not name.endswith("_raw") for name in calls[0]["var_names"])
    assert model.idata.attrs["inference_method"] == "mcmc"
    assert "sampling" not in model.idata.attrs and "sampling_previous" not in model.idata.attrs
    assert model.idata.attrs["study"] == "kept"
    assert "information" not in model.idata
    assert model.idata.attrs["posterior_run_id"]
    with pytest.raises(ValueError, match="overwrite"):
        model.fit()
    coefficients, point = fc.coef(model, kind="stats"), fc.parameter_point(model)
    # The native sampler boundary supplies the same final arrays for each backend.
    # Downstream analysis must not need backend-specific naming or reconstruction.
    model.fit(nuts_sampler="pymc", draws=2, tune=0, chains=1, progressbar=False, overwrite=True)
    assert calls[-1]["nuts_sampler"] == "pymc"
    pd.testing.assert_frame_equal(fc.coef(model, kind="stats"), coefficients)
    xr.testing.assert_identical(fc.parameter_point(model), point)
    xr.testing.assert_identical(
        model.idata["posterior"].to_dataset(), prepared.annotate_coordinates(posterior)
    )
    assert model.prepared is prepared and model.build() is graph
    xr.testing.assert_identical(prepared.data, prepared_before)
    assert model._bindings.parameters.keys() == bindings.keys()
    for key, binding in bindings.items():
        assert model._bindings.parameters[key] is binding
        assert binding is graph[binding.name]
    before = model.idata.copy(deep=True)

    def failure(**kwargs):
        raise RuntimeError("captured sampler failure")

    monkeypatch.setattr(pm, "sample", failure)
    with pytest.raises(RuntimeError, match="captured"):
        model.fit(overwrite=True)
    xr.testing.assert_identical(model.idata, before)


def test_output_publication_checks_the_complete_tree_before_replacing_results():
    model = mixed_model()
    model.idata["information"] = xr.Dataset({"old": 1})
    model.idata["information/stale/detail"] = xr.Dataset({"old_detail": 2})
    model.idata["report"] = xr.Dataset(coords={"entry": ["a"]}, attrs={"source": "retained"})
    model.idata["report/detail"] = xr.Dataset({"report_score": ("entry", [3.0])})
    previous = model.idata
    before = previous.copy(deep=True)
    parent = xr.Dataset({"score": ("entry", [1.0])}, coords={"entry": ["a"]})
    child = xr.Dataset({"point": ("entry", [2.0])}, coords={"entry": ["b"]})
    with pytest.raises(ValueError):
        model._publish_outputs(
            {"information": parent, "information/parameters": child},
            remove=("information",),
        )
    assert model.idata is previous
    xr.testing.assert_identical(previous, before)

    detail = xr.Dataset({"point": ("entry", [2.0])})
    model._publish_outputs(
        {"information/parameters": detail, "information": parent}, remove=("information",)
    )
    assert model.idata is not previous
    xr.testing.assert_identical(previous, before)
    xr.testing.assert_equal(model.idata["information"].to_dataset(), parent)
    assert set(model.idata["information"].children) == {"parameters"}
    xr.testing.assert_equal(
        model.idata["information/parameters"].to_dataset(),
        detail.assign_coords(entry=["a"]),
    )
    xr.testing.assert_identical(model.idata["report"], before["report"])
    assert "entry" not in model.idata["report/detail"].to_dataset(inherit=False).coords
    for name in model.prepared.result_metadata:
        xr.testing.assert_identical(model.idata[name].to_dataset(), before[name].to_dataset())


def test_refit_replaces_optional_groups_and_rejects_late_coordinate_failures(monkeypatch):
    model = mixed_model()
    metadata_groups = set(model.idata.children)
    posterior = workflow_point(model).expand_dims(chain=[0], draw=[0, 1])
    model.idata["prior"] = xr.Dataset({"retained": 1.0})
    native = xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "sample_stats": xr.Dataset({"old": 1.0}),
            "warmup_posterior": posterior,
            "warmup_sample_stats": xr.Dataset({"old": 1.0}),
        }
    )
    monkeypatch.setattr(pm, "sample", lambda **kwargs: native)
    first = model.fit()
    assert "warmup_posterior" in first
    first_run = first.attrs["posterior_run_id"]

    native = xr.DataTree.from_dict({"posterior": posterior})
    second = model.fit(overwrite=True)
    assert set(second.children) == {"posterior", "prior"} | metadata_groups
    assert "warmup_posterior" in first
    assert second.attrs["posterior_run_id"] != first_run
    before = second.copy(deep=True)

    native = xr.DataTree.from_dict({"posterior": posterior.assign_coords(person=[99, 98])})
    with pytest.raises(ValueError, match="coordinate"):
        model.fit(overwrite=True)
    assert model.idata is second
    xr.testing.assert_identical(second, before)


def test_fit_requires_native_tree_output():
    with pytest.raises(ValueError, match="InferenceData"):
        mixed_model().fit(return_inferencedata=False)


@pytest.mark.parametrize("seed", [None, np.int64(17), [3, 8], np.random.default_rng(4)])
def test_fit_records_serializable_native_seed_configuration_and_log_units(monkeypatch, seed):
    model = mixed_model()
    bindings = model._bindings
    posterior = workflow_point(model).expand_dims(chain=[0], draw=[0, 1])
    likelihood = xr.Dataset(
        {
            response.batch.response_name: xr.DataArray(
                np.full((1, 2, len(response.batch.y)), -0.5),
                dims=("chain", "draw", response.batch.obs_dim),
                coords={
                    "chain": [0],
                    "draw": [0, 1],
                    response.batch.obs_dim: model.prepared.coords[response.batch.obs_dim],
                },
            )
            for response in bindings.responses
        }
    )
    selected = ["theta"]

    def capture(**kwargs):
        assert kwargs["var_names"] == selected
        assert kwargs["random_seed"] is seed
        return xr.DataTree.from_dict(
            {"posterior": posterior, "log_likelihood": likelihood, "observed_data": xr.Dataset()}
        )

    monkeypatch.setattr(pm, "sample", capture)
    result = model.fit(var_names=selected, random_seed=seed)
    assert selected == ["theta"]
    assert result.attrs["sampling_draws"] == 1000
    if isinstance(seed, np.integer):
        assert type(result.attrs["sampling_random_seed"]) is int
        assert result.attrs["sampling_random_seed"] == int(seed)
    assert result["log_likelihood"].response.sizes["obs"] == 7
    assert result["log_likelihood"].response.attrs["complete_observations"]


@pytest.mark.parametrize("platform,expected", [("gpu", "vectorized"), ("cpu", "parallel")])
@pytest.mark.parametrize("sampler", ["blackjax", "numpyro", "pymc", "nutpie", None])
def test_fit_selects_only_jax_chain_defaults_and_preserves_core_detection(
    monkeypatch, platform, expected, sampler
):
    import jax

    model = mixed_model()
    calls = []
    monkeypatch.setattr(jax, "default_backend", lambda: platform)
    monkeypatch.setattr(pm, "sample", lambda **kwargs: calls.append(kwargs) or xr.DataTree())
    options = {"target_accept": 0.8}
    model.fit(nuts_sampler=sampler, nuts=options, nuts_sampler_kwargs=None)
    assert options == {"target_accept": 0.8}
    assert calls[0]["nuts_sampler_kwargs"] is None
    assert "cores" not in calls[0]
    if sampler == "blackjax" and platform == "gpu":
        assert calls[0]["progressbar"] is False
    else:
        assert "progressbar" not in calls[0]
    assert calls[0]["nuts"] == dict(
        options, **({"chain_method": expected} if sampler in {"blackjax", "numpyro"} else {})
    )


@pytest.mark.parametrize("given", [{}, {"chain_method": "vectorized"}])
def test_fit_preserves_native_legacy_sampler_options(monkeypatch, given):
    import jax

    model = mixed_model()
    calls = []
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(pm, "sample", lambda **kwargs: calls.append(kwargs) or xr.DataTree())
    original = dict(given)
    model.fit(nuts_sampler_kwargs=given, progressbar=True)
    assert given == original
    assert calls[0]["progressbar"] is True
    assert "nuts" not in calls[0]
    assert calls[0]["nuts_sampler_kwargs"] == {
        "chain_method": given.get("chain_method", "parallel")
    }


def test_summary_explains_actual_regression_transforms_and_parameterizations(monkeypatch):
    import pandas as pd

    from tests.helpers import mixed_inputs

    model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs()))
    model.add_regressions(
        {
            "theta": fc.Regression(
                "age",
                pd.DataFrame({"person": [1, 2], "age": [20, 40]}),
                id="person",
                coefficient_noncentered=True,
            ),
            ("fc", "loading_magnitude"): fc.Regression(
                "length + family",
                pd.DataFrame(
                    {
                        "statement": list("abcdefghi"),
                        "length": range(9),
                        "family": ["one", "two", "three"] * 3,
                    }
                ),
                id="statement",
            ),
        }
    )
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("summary built a graph"))
    text = model.summary(details=True)
    assert "theta ~ age" in text and "coefficient_noncentered=True" in text
    assert "conditional LogNormal" in text and "noncentered=True" in text
    assert "log scale" in text and "theta_residual_scale='unit'" in text
    assert "loading ~ length + family" in text and "log-loading" in text
    assert "::" not in text


def test_add_regressions_preserves_cache_on_failure_and_replaces_only_after_success():
    import pandas as pd

    from tests.helpers import runtime_snapshot

    prepared = fc.PreparedInputs.from_inputs(mixed_inputs())
    original = runtime_snapshot(prepared)
    model = fc.Model(prepared)
    graph = model.build()
    bindings = model._bindings
    with pytest.raises(ValueError, match="missing"):
        model.add_regressions(
            "theta ~ age",
            person_data=pd.DataFrame({"person_id": [1], "age": [20.0]}),
        )
    assert model._built is bindings and model.build() is graph and model.prepared is prepared
    xr.testing.assert_identical(prepared.data, original)

    assert (
        model.add_regressions(
            "theta ~ age",
            person_data=pd.DataFrame({"person_id": [1, 2], "age": [20.0, 40.0]}),
        )
        is None
    )
    assert model._built is None
    assert "regression_coefficient" not in graph
    assert "regression_coefficient" in model.build()
    xr.testing.assert_identical(prepared.data, original)
    with pytest.raises(TypeError, match="regressions"):
        fc.Model(prepared, regressions={})
