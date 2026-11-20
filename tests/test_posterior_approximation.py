"""Runtime approximation uses PyMC transforms and ordinary labelled results."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc.distributions.transforms import LogTransform
from pytensor.graph.traversal import ancestors
from scipy.special import expit
from scipy.stats import multivariate_normal, norm

import pyfcirt as fc
from tests.data.test_schema import pair_input


def test_native_correlation_transform_preserves_matrix_support():
    with pm.Model() as model:
        correlation = pm.LKJCorr("correlation", n=3, eta=2)
    samples = pm.draw(correlation, draws=24, random_seed=31)
    posterior = xr.Dataset(
        {"correlation": (("chain", "draw", "row", "column"), samples.reshape(2, 12, 3, 3))},
        coords={"chain": [3, 5], "draw": np.arange(12) + 7},
    )
    approximation = fc.PosteriorApproximation.from_idata(
        xr.DataTree.from_dict({"posterior": posterior}), model=model, var_names=["correlation"]
    )

    inputs = correlation.type()
    transform = model.rvs_to_transforms[correlation]
    forward = pm.compile([inputs], transform.forward(inputs, *correlation.owner.inputs))
    expected = np.stack([forward(value) for value in samples])
    _, mean, _ = approximation.densities["correlation"]
    np.testing.assert_allclose(mean[0], expected.mean(axis=0))

    updated = approximation.apply(model)
    factors = pm.draw(updated["correlation"], draws=20, random_seed=32)
    matrices = factors @ factors.swapaxes(-2, -1)
    np.testing.assert_allclose(np.diagonal(matrices, axis1=-2, axis2=-1), 1)
    np.testing.assert_allclose(matrices, matrices.swapaxes(-2, -1))
    assert np.all(np.linalg.eigvalsh(matrices) > 0)


def test_dependent_bounds_follow_the_native_graph():
    with pm.Model() as model:
        lower = pm.Beta("lower", 1, 4, shape=2)
        upper = pm.Truncated("upper", pm.Beta.dist(4, 1, shape=2), lower=lower, upper=1)
        theta = pm.Normal("theta")
        pm.Normal("y", (upper - lower) * theta, observed=[1.0, -0.5])
    generator = np.random.default_rng(92)
    c = generator.uniform(0.02, 0.4, (2, 30, 2))
    d = c + (1 - c) * generator.uniform(0.1, 0.9, c.shape)
    dims = ("chain", "draw", "pair")
    posterior = xr.Dataset(
        {"lower": (dims, c), "upper": (dims, d)},
        coords={"chain": [3, 5], "draw": np.arange(30) + 7},
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    with pytest.raises(ValueError, match="Missing posterior draws for transform input 'lower'"):
        fc.PosteriorApproximation.from_idata(
            xr.DataTree.from_dict({"posterior": posterior.drop_vars("lower")}),
            model=model,
            var_names=["upper"],
        )
    upper_only = fc.PosteriorApproximation.from_idata(idata, model=model, var_names=["upper"])
    assert upper_only.var_names == ("upper",)
    _, upper_mean, _ = upper_only.densities["upper"]
    np.testing.assert_allclose(upper_mean[0], np.log((d - c) / (1 - d)).mean(axis=(0, 1)))
    upper_model = upper_only.apply(model)
    c_only, d_only = pm.draw([upper_model["lower"], upper_model["upper"]], draws=40, random_seed=7)
    assert upper_model["lower"] in upper_model.free_RVs
    assert np.all((c_only > 0) & (d_only > c_only) & (d_only < 1))
    approximation = fc.PosteriorApproximation.from_idata(
        idata,
        model=model,
        var_names={"asymptotes": ("lower", "upper")},
        random_seed=2,
    )
    updated = approximation.apply(model)
    lower_samples, upper_samples = pm.draw(
        [updated["lower"], updated["upper"]], draws=50, random_seed=41
    )
    assert np.all((lower_samples > 0) & (upper_samples > lower_samples) & (upper_samples < 1))
    assert [rv.name for rv in model.free_RVs] == ["lower", "upper", "theta"]
    assert updated["lower"] in updated.deterministics
    (latent,) = (rv for rv in updated.free_RVs if rv.name != "theta")
    raw = np.array([-0.4, -1.2, 0.7, -0.3])
    ability = 0.3
    point = {updated.rvs_to_values[latent].name: raw, "theta": ability}
    c = expit(raw[:2])
    d = c + (1 - c) * expit(raw[2:])
    _, means, cholesky = approximation.densities["asymptotes"]
    expected = (
        multivariate_normal.logpdf(raw, mean=means[0], cov=cholesky[0] @ cholesky[0].T)
        + norm.logpdf(ability)
        + norm.logpdf([1.0, -0.5], loc=(d - c) * ability).sum()
    )
    np.testing.assert_allclose(updated.compile_logp(mode="JAX")(point), expected, atol=1e-11)


def test_scoring_graph_applies_fixed_input_inside_native_dependent_transform(monkeypatch):
    source = fc.load_example("single_binary", model=fc.DichotomousIRT(kind="4PL")).data
    table = pd.DataFrame({"person_id": "old", "item_id": source.design.item_id, "response": 1})
    source = replace(
        source, responses=fc.ResponseTable(table, format="item_long"), administration=None
    )
    reference = fc.Model(fc.PreparedInputs.from_inputs(source))
    native = reference.build()
    lower, upper = native["lower_asymptote"], native["upper_asymptote"]
    free = set(native.free_RVs)
    lower_rvs = free.intersection(ancestors([lower]))
    (upper_rv,) = free.intersection(ancestors([upper])) - lower_rvs
    (lower_rv,) = lower_rvs
    draws = pm.sample_prior_predictive(
        model=native,
        var_names=[lower_rv.name, upper_rv.name, lower.name, "loading", "item_intercept"],
        draws=24,
        random_seed=51,
    )["prior"].to_dataset()
    labelled = reference.prepared.annotate_coordinates(draws)
    reference.attach_samples(xr.DataTree.from_dict({"posterior": labelled}), assume_same_batch=True)
    fitted = reference.posterior_approximation(var_names=[upper_rv.name])
    c, d = draws[lower_rv.name].values, draws[upper_rv.name].values
    _, means, _ = fitted.densities[upper_rv.name]
    np.testing.assert_allclose(means[0], np.log((d - c) / (1 - d)).mean(axis=(0, 1)))

    new = fc.ResponseTable(
        source.responses.frame.iloc[:1].assign(person_id="new"), format=source.responses.format
    )
    with pytest.raises(TypeError, match="format is required"):
        reference.score_model(new.frame, calibration="posterior", approximation=fitted)
    model = reference.score_model(
        new.frame, format=new.format, calibration="posterior", approximation=fitted
    )
    scored = model.build()
    fixed = model.idata.fixed_parameters[lower.name]
    np.testing.assert_allclose(fixed, draws[lower.name].mean(("chain", "draw")))
    assert lower_rv.name not in scored.named_vars
    (latent,) = set(scored.free_RVs).intersection(ancestors([scored[upper.name]]))
    assert {rv.name for rv in scored.free_RVs} == {latent.name, "theta"}
    actual_lower, actual_upper, raw_draw = pm.draw(
        [scored[lower.name], scored[upper.name], latent], draws=20, random_seed=56
    )
    np.testing.assert_allclose(actual_lower, np.broadcast_to(fixed.values, actual_lower.shape))
    np.testing.assert_allclose(
        actual_upper, fixed.values + (1 - fixed.values) * expit(raw_draw), atol=1e-12
    )
    assert set(native.free_RVs) == free
    raw = np.linspace(-0.3, 0.3, len(fixed))
    theta = 0.2
    state = {scored.rvs_to_values[latent].name: raw, "theta": np.array([[theta]])}
    c = fixed.values[0]
    d = c + (1 - c) * expit(raw[0])
    loading = model.idata.fixed_parameters["loading"].values[0]
    intercept = model.idata.fixed_parameters["item_intercept"].values[0]
    probability = c + (d - c) * expit(loading * theta - intercept)
    _, mean, cholesky = fitted.densities[upper_rv.name]
    logp = multivariate_normal.logpdf(raw, mean=mean[0], cov=cholesky[0] @ cholesky[0].T)
    np.testing.assert_allclose(
        scored.compile_logp(mode="JAX")(state),
        logp + norm.logpdf(theta) + np.log(probability),
        atol=1e-10,
    )
    _attach_draws(model, upper_rv.name, seed=57)
    different = replace(
        fitted,
        fixed_parameters=fitted.fixed_parameters.assign(
            item_intercept=fitted.fixed_parameters.item_intercept + 0.2
        ),
    )
    other = reference.score_model(new, calibration="posterior", approximation=different)
    assert other.prepared.calibration_identity == model.prepared.calibration_identity
    trace = xr.DataTree.from_dict({"posterior": model.idata.posterior.to_dataset()})
    trace.attrs["scoring_batch_identity"] = model.idata.attrs["scoring_batch_identity"]
    with pytest.raises(ValueError, match="different prepared response batch"):
        other.attach_samples(trace)
    assert "posterior" not in other.idata
    native_do = pm.do
    conditioned = []

    def track_fixed_source(source, interventions, **kwargs):
        if lower.name in interventions:
            conditioned.append(source)
        return native_do(source, interventions, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pm, "do", track_fixed_source)
        next_prior = model.posterior_approximation(random_seed=58)
        repeated = model.posterior_approximation(random_seed=58)
    assert len(conditioned) == 1
    assert next_prior.var_names == fitted.var_names
    xr.testing.assert_identical(next_prior.fixed_parameters, repeated.fixed_parameters)
    xr.testing.assert_identical(next_prior.fixed_parameters[lower.name], fixed)
    for left, right in zip(
        next_prior.densities[upper_rv.name], repeated.densities[upper_rv.name], strict=True
    ):
        np.testing.assert_array_equal(left, right)

    lower_only = reference.posterior_approximation(var_names=[lower_rv.name])
    incompatible = fc.score_model(reference, new, calibration="posterior", approximation=lower_only)
    with pytest.raises(ValueError, match="Fixed parameters depend on selected native RVs"):
        incompatible.build()


@pytest.mark.parametrize("default", [None, LogTransform()])
def test_native_override_and_label_alignment(default):
    with pm.Model(coords={"item": ["a", "b"]}) as model:
        pm.HalfNormal("scale", dims="item", default_transform=default)
    rng = np.random.default_rng(93)
    posterior = xr.Dataset(
        {"scale": (("chain", "draw", "item"), rng.lognormal(size=(2, 32, 2)))},
        coords={"chain": [1, 5], "draw": np.arange(32), "item": ["a", "b"]},
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    with pytest.raises(ValueError, match="posterior labels differ"):
        fc.PosteriorApproximation.from_idata(
            idata.sel(item=["b", "a"]), model=model, var_names=["scale"]
        )
    with pytest.raises(ValueError, match="constrained native RV cannot disable"):
        fc.PosteriorApproximation.from_idata(
            idata, model=model, var_names=["scale"], transforms={"scale": None}
        )
    approximation = fc.PosteriorApproximation.from_idata(
        idata,
        model=model,
        var_names=["scale"],
        method="mean_field",
        transforms={"scale": LogTransform()},
    )
    assert approximation.diagnostics["kinds"]["scale"]["family"] == "mean_field"
    automatic = fc.PosteriorApproximation.from_idata(
        idata, model=model, var_names=["scale"], method="mean_field"
    )
    with pm.Model(coords=model.coords) as destination:
        pm.HalfNormal("scale", dims="item", default_transform=None)
    for fitted in (approximation, automatic):
        original = pm.draw(fitted.apply(model)["scale"], draws=15, random_seed=43)
        transferred = pm.draw(fitted.apply(destination)["scale"], draws=15, random_seed=43)
        np.testing.assert_array_equal(original, transferred)
        assert np.all(transferred > 0)
    assert destination.rvs_to_transforms[destination["scale"]] is None


def test_named_posterior_axes_are_read_in_native_order():
    coords = {"row": ["a", "b"], "column": ["c", "d"]}
    with pm.Model(coords=coords) as model:
        pm.Normal("weight", dims=("row", "column"))
    rng = np.random.default_rng(94)
    values = rng.normal(size=(2, 30, 2, 2)) + [[1, 2], [3, 4]]
    posterior = xr.Dataset(
        {"weight": (("chain", "draw", "row", "column"), values)},
        coords={"chain": [0, 1], "draw": np.arange(30), **coords},
    ).transpose("column", "draw", "chain", "row")
    fit = fc.PosteriorApproximation.from_idata(
        xr.DataTree.from_dict({"posterior": posterior}), model=model, var_names=["weight"]
    )
    np.testing.assert_allclose(fit.densities["weight"][1][0], values.mean(axis=(0, 1)).ravel())


def test_fitted_simplex_keeps_its_native_shape_and_support():
    with pm.Model() as source:
        pm.Dirichlet("probability", np.ones(3))
    with pm.Model() as destination:
        pm.Dirichlet("probability", np.ones(3), default_transform=None)
    rng = np.random.default_rng(44)
    values = rng.dirichlet([2, 3, 4], size=(2, 30))
    posterior = xr.Dataset(
        {"probability": (("chain", "draw", "category"), values)},
        coords={"chain": [0, 1], "draw": np.arange(30)},
    )
    fitted = fc.PosteriorApproximation.from_idata(
        xr.DataTree.from_dict({"posterior": posterior}), model=source, var_names=["probability"]
    )
    ((_, means, _),) = fitted.densities.values()
    log_values = np.log(values)
    centered = log_values[..., :2] - log_values.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(means[0], centered.mean(axis=(0, 1)))
    updated = fitted.apply(destination)
    draws = pm.draw(updated["probability"], draws=20, random_seed=45)
    assert draws.shape == (20, 3)
    assert np.all(draws > 0)
    np.testing.assert_allclose(draws.sum(axis=-1), 1)


def test_explicit_auto_compares_mixture_directly_on_untransformed_draws(monkeypatch):
    def unexpected_conversion(*args, **kwargs):
        pytest.fail("Untransformed variables require no conversion graph")

    monkeypatch.setattr("pyfcirt.scoring.approximation.vectorize_graph", unexpected_conversion)
    rng = np.random.default_rng(146)
    states = rng.normal(0, 0.12, (4, 128)) + np.where(np.arange(128) % 2, 4, -4)
    with pm.Model() as model:
        pm.Normal("x")
        pm.Normal("unrelated", shape=2)
    posterior = xr.Dataset(
        {"x": (("draw", "chain"), states.T)},
        coords={"chain": np.arange(4), "draw": np.arange(128)},
    )
    tree = xr.DataTree.from_dict({"posterior": posterior})
    default = fc.PosteriorApproximation.from_idata(tree, model=model, var_names=["x"])
    mixed = fc.PosteriorApproximation.from_idata(
        tree, model=model, var_names=["x"], method="auto", random_seed=32
    )
    assert default.diagnostics["requested_method"] == "mvn"
    assert not default.diagnostics["kinds"]["x"]["comparison_performed"]
    np.testing.assert_allclose(default.densities["x"][1][0], states.mean())
    assert mixed.diagnostics["kinds"]["x"]["family"] == "gmm"
    updated = mixed.apply(model)
    assert np.isfinite(updated.compile_logp()(updated.initial_point()))


def _responses(person):
    return fc.ResponseTable(
        pd.DataFrame({"person_id": [person], "block_id": ["pair"], "response": [[1, 2]]}),
        format="block_long",
    )


def _attach_draws(model, native, seed):
    # Native draws exercise labelled handoff without running calibration MCMC.
    graph = model.build()
    native = [native] if isinstance(native, str) else native
    names = [*native, "loading", "theta"]
    sampled = pm.sample_prior_predictive(draws=20, var_names=names, model=graph, random_seed=seed)[
        "prior"
    ].to_dataset()
    posterior = model.prepared.annotate_coordinates(sampled)
    model.attach_samples(xr.DataTree.from_dict({"posterior": posterior}), assume_same_batch=True)


def test_default_posterior_scoring_updates_jointly_and_reuses_subset(monkeypatch):
    example = fc.load_example("simulated_mgppm", model="MGPPM", n_persons=4)
    responses = example.data.responses.frame

    def response(person):
        return fc.ResponseTable(responses.loc[responses.person_id == person], format="block_long")

    def sample_fixture(*, var_names, **_):
        # Exercise Model.fit's public retention path with native draws.
        prior = pm.sample_prior_predictive(draws=160, var_names=var_names, random_seed=42)
        return xr.DataTree.from_dict({"posterior": prior.prior.to_dataset()})

    monkeypatch.setattr(pm, "sample", sample_fixture)
    initial = replace(example.data, responses=response(1))
    calibration = fc.Model().prepare(initial, theta_parameterization="noncentered")
    calibration.fit()
    with pytest.raises(ValueError, match=r"retain_calibration=True"):
        calibration.score_model(response(2), calibration="posterior")

    calibration.fit(retain_calibration=True, overwrite=True)
    assert "theta_raw" not in calibration.idata.posterior
    prior = calibration.posterior_approximation(joint=False)
    assert set(prior.groups) == {"loading", "statement_intercept", "trait_correlation"}
    assert sum(map(len, prior.groups.values())) == 3
    assert not prior.fixed_parameters.data_vars

    scored = calibration.score(response(2), calibration="posterior")
    assert scored.build().coords["person"] == (2,)
    continued = scored.posterior_approximation()
    assert continued.groups == {"joint": prior.var_names}
    assert continued.diagnostics["requested_method"] == "auto"
    with pytest.raises(ValueError, match="absent from prior evidence"):
        scored.score_model(response(1), calibration="posterior")

    loading_prior = calibration.posterior_approximation(
        var_names={"loading": prior.groups["loading"]}
    )
    selected = calibration.score(response(2), calibration="posterior", approximation=loading_prior)
    assert set(selected.posterior_approximation().groups) == {"loading"}
    fixed = selected.score_model(response(3), calibration="posterior").idata.fixed_parameters
    assert {"statement_intercept", "covariance"}.issubset(fixed.data_vars)

    for selection in (prior.var_names, prior.groups):
        with pytest.raises(ValueError, match="omit var_names"):
            calibration.posterior_approximation(joint=True, var_names=selection)
    with pytest.raises(TypeError, match="joint must be a bool"):
        calibration.posterior_approximation(joint=1)

    joint = calibration.posterior_approximation()
    assert joint.groups == {"joint": prior.var_names}
    assert joint.diagnostics["requested_method"] == "auto"
    assert not joint.fixed_parameters.data_vars
    dimension = sum(prior.densities[kind][1].shape[-1] for kind in prior.groups)
    weights, mean, factor = joint.densities["joint"]
    assert weights.shape == (1,)
    assert mean.shape == (1, dimension)
    assert factor.shape == (1, dimension, dimension)

    incoming = scored.posterior_approximation(joint=True, random_seed=43)
    assert incoming.groups == joint.groups
    assert incoming.evidence_people > joint.evidence_people
    assert incoming.diagnostics["requested_method"] == "auto"
    assert scored.score_model(
        response(3), calibration="posterior", approximation=incoming
    ).build().coords["person"] == (3,)


def test_default_groups_dependent_asymptotes_by_native_owner():
    example = fc.load_example("single_binary", model=fc.DichotomousIRT(kind="4PL"))
    calibration = fc.Model().prepare(example.data)
    native = calibration.build()
    # This univariate model has exactly one person-trait RV outside the calibration.
    names = [rv.name for rv in native.free_RVs if rv.name != "theta"]
    _attach_draws(calibration, names, seed=31)
    prior = calibration.posterior_approximation(joint=False)
    assert set(prior.groups) == {"loading", "item_intercept", "lower_asymptote", "upper_asymptote"}
    for kind in ("lower_asymptote", "upper_asymptote"):
        assert len(prior.groups[kind]) == 1
    assert set(prior.var_names) == set(names)


def test_pyfcirt_rejects_person_rvs_before_reading_draws():
    source = pair_input(model=fc.TIRTRank(loading_anchors={"a": 1.0, "b": 0.8}))
    reference = fc.Model(fc.PreparedInputs.from_inputs(source))
    graph = reference.build()
    (theta,) = (
        rv for rv in graph.free_RVs if "person" in graph.named_vars_to_dims.get(rv.name, ())
    )
    draws = pm.sample_prior_predictive(
        model=graph, var_names=["statement_intercept"], draws=3, random_seed=15
    )["prior"].to_dataset()
    posterior = reference.prepared.annotate_coordinates(draws)
    reference.attach_samples(
        xr.DataTree.from_dict({"posterior": posterior}), assume_same_batch=True
    )
    assert theta.name not in reference.idata.posterior
    with pytest.raises(ValueError, match="Select shared calibration RVs"):
        reference.posterior_approximation(var_names={"person": [theta.name]})


def test_runtime_approximation_reuses_rank_scoring_graph_and_saves_only_evidence(
    tmp_path, monkeypatch
):
    from pyfcirt.data import prepare

    pytest.importorskip("h5netcdf")
    source = pair_input(model=fc.TIRTRank(loading_anchors={"a": 1.0, "b": 0.8}))
    reference = fc.Model(fc.PreparedInputs.from_inputs(source))
    graph = reference.build()
    selected = [
        rv.name for rv in graph.free_RVs if rv in set(ancestors([graph["statement_intercept"]]))
    ]
    (intercept,) = selected
    _attach_draws(reference, [intercept, "covariance"], seed=61)
    approximation = reference.posterior_approximation(var_names=selected, random_seed=62)
    scored = fc.score_model(
        reference, _responses("first"), calibration="posterior", approximation=approximation
    )
    scoring_graph = scored.build()
    response = scored._bindings.responses[0]
    assert scoring_graph.coords["person"] == ("first",)
    contrast, indicator = response.augmentation
    assert indicator in scoring_graph.observed_RVs
    assert contrast in scoring_graph.free_RVs
    roots = set(scoring_graph.free_RVs)
    assert roots == {
        scoring_graph["theta"],
        scoring_graph["calibrated_prior_0_raw"],
        contrast,
    }
    np.testing.assert_allclose(
        scored.idata.fixed_parameters["covariance"],
        reference.idata.posterior["covariance"].mean(("chain", "draw")),
    )
    assert scoring_graph[intercept] not in roots
    assert np.isfinite(scoring_graph.compile_logp()(scoring_graph.initial_point()))
    schedule = fc.generation_schedule(scored, ["simulated"])
    result = fc.simulate(scored, administration=schedule, random_seed=34)
    assert result.prepared.coords["person"].tolist() == ["simulated"]
    _attach_draws(scored, intercept, seed=63)
    assert "posterior_approximation" not in scored.idata
    next_prior = scored.posterior_approximation(random_seed=64)
    assert next_prior.evidence_people == (
        reference.prepared.observed_person_keys | scored.prepared.observed_person_keys
    )
    assert next_prior.evidence_batches == {
        reference.prepared.scoring_batch_identity,
        scored.prepared.scoring_batch_identity,
    }
    before = scored.idata

    def replace_people(*args, **kwargs):
        pytest.fail("Evidence validation must precede rebuilding the person axis")

    with monkeypatch.context() as patch:
        patch.setattr(prepare, "_replace_people", replace_people)
        for person in ("p", "first"):
            with pytest.raises(ValueError, match="respondents absent"):
                scored.score_model(
                    _responses(person), calibration="posterior", approximation=next_prior
                )
        for missing_history in (
            replace(next_prior, evidence_people=approximation.evidence_people),
            replace(next_prior, evidence_batches=approximation.evidence_batches),
        ):
            with pytest.raises(ValueError, match="omits evidence already committed"):
                scored.score_model(
                    _responses("second"), calibration="posterior", approximation=missing_history
                )
        unpaired_history = replace(next_prior, evidence_batches=frozenset())
        with pytest.raises(ValueError, match="must track both people and batches"):
            scored.score_model(
                _responses("second"), calibration="posterior", approximation=unpaired_history
            )
    assert scored.idata is before and scored.build() is scoring_graph
    next_model = scored.score_model(
        _responses("second"), calibration="posterior", approximation=next_prior
    )
    assert next_model.build().coords["person"] == ("second",)
    path = tmp_path / "scores.nc"
    scored.idata.to_netcdf(path, engine="h5netcdf")
    restored = fc.Model(scored.prepared)
    with pytest.raises(ValueError, match="original incoming approximation"):
        restored.build()
    with pytest.raises(ValueError, match="original incoming approximation"):
        restored.fit()
    with xr.open_datatree(path, engine="h5netcdf") as saved:
        saved.load()

        def scoring(prior):
            return reference.score_model(
                _responses("first"), calibration="posterior", approximation=prior
            )

        weights, means, scales = approximation.densities[intercept]
        shifted = replace(approximation, densities={intercept: (weights, means + 2.0, scales)})
        mismatched = scoring(shifted)
        assert mismatched.prepared.scoring_batch_identity == scored.prepared.scoring_batch_identity
        assert mismatched.idata.calibration_evidence.equals(scored.idata.calibration_evidence)
        before_mismatch = mismatched.idata
        for assume in (False, True):
            with pytest.raises(ValueError, match="different incoming approximation"):
                mismatched.attach_samples(saved, assume_same_batch=assume)
        assert mismatched.idata is before_mismatch

        replayed = scoring(approximation)
        different_history = saved.copy(deep=True)
        different_history["calibration_evidence"]["people"] = xr.DataArray(
            ["different-respondent"], dims=saved.calibration_evidence.people.dims
        )
        before_replay = replayed.idata
        with pytest.raises(ValueError, match="different incoming evidence"):
            replayed.attach_samples(different_history, assume_same_batch=True)
        assert replayed.idata is before_replay
        replayed.attach_samples(saved)
        xr.testing.assert_identical(replayed.idata.posterior, scored.idata.posterior)
        assert replayed._committed_evidence() == scored._committed_evidence()

        legacy = saved.copy(deep=True)
        del legacy.calibration_evidence.attrs["incoming_approximation_id"]
        confirmed = scoring(approximation)
        with pytest.raises(ValueError, match="lack a batch or incoming approximation identity"):
            confirmed.attach_samples(legacy)
        confirmed.attach_samples(legacy, assume_same_batch=True)
        xr.testing.assert_identical(
            confirmed.idata.calibration_evidence, replayed.idata.calibration_evidence
        )

        # A current-batch match does not identify the evidence in the incoming prior.
        missing = saved.drop_nodes("calibration_evidence")
        malformed = saved.copy(deep=True)
        del malformed["calibration_evidence"]["people"]
        before_restore = restored.idata
        for invalid in (missing, malformed):
            for assume in (False, True):
                with pytest.raises(ValueError, match="calibration_evidence"):
                    restored.attach_samples(invalid, assume_same_batch=assume)
                assert restored.idata is before_restore

        with pytest.raises(ValueError, match="Restore calibration_evidence"):
            restored.score_model(
                _responses("second"), calibration="posterior", approximation=next_prior
            )

        active = scoring(approximation)
        active.attach_samples(missing, assume_same_batch=True)
        xr.testing.assert_identical(active.idata.calibration_evidence, saved.calibration_evidence)

        changed = saved.copy(deep=True)
        changed["fixed_parameters"]["covariance"] = changed["fixed_parameters"]["covariance"] + 0.1
        with pytest.raises(ValueError, match="different fixed calibration snapshot"):
            restored.attach_samples(changed)
        assert "posterior" not in restored.idata
        restored.attach_samples(saved)
    assert "calibration_evidence" in restored.idata
    attached = restored.idata
    with pytest.raises(ValueError, match="original incoming approximation"):
        restored.fit(overwrite=True)
    with pytest.raises(ValueError, match="original incoming approximation"):
        fc.prior_predictive(restored, draws=1)
    with pytest.raises(ValueError, match="original incoming approximation"):
        fc.simulate(restored, administration=schedule)
    assert restored.idata is attached
    rebuilt = restored.posterior_approximation(var_names=selected, random_seed=65)
    assert rebuilt.evidence_people == next_prior.evidence_people
    assert rebuilt.evidence_batches == next_prior.evidence_batches
    with pytest.raises(ValueError, match="respondents absent"):
        restored.score_model(_responses("p"), calibration="posterior", approximation=rebuilt)
    assert restored.idata is attached
    continued = restored.score_model(
        _responses("second"), calibration="posterior", approximation=rebuilt
    )
    assert continued.build().coords["person"] == ("second",)
    assert reference.build() is graph
    assert "calibration_evidence" not in reference.idata


def test_native_draw_reuse_checks_noncentered_coordinates(tmp_path, monkeypatch):
    pytest.importorskip("formulae")
    pytest.importorskip("h5netcdf")
    items = ["a", "b", "c"]
    responses = pd.DataFrame({"person_id": "old", "item_id": items, "response": [0, 1, 0]})
    source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": items}),
        fc.ResponseTable(responses, format="item_long"),
    )
    features = pd.DataFrame({"item_id": items, "x": [0.0, 1.0, 2.0]})

    def configured(mean, *, scale=0.7):
        model = fc.Model(source)
        model.add_regressions(
            {
                ("items", "loading_magnitude"): fc.Regression(
                    "0 + x",
                    features,
                    id="item_id",
                    priors={"common": pm.Normal.dist(mu=mean, sigma=scale)},
                    parameter_residual_sd=0.6,
                    coefficient_noncentered=True,
                ),
            }
        )
        return model

    original = configured(0.4)
    changed = configured(1.4)
    assert original.prepared.scoring_batch_identity == changed.prepared.scoring_batch_identity
    native = original.build()
    altered = changed.build()
    (coefficient,) = set(native.free_RVs) & set(ancestors([native["regression_coefficient"]]))
    assert coefficient.name.removesuffix("_raw") not in native.named_vars
    assert native.eval_rv_shapes()[coefficient.name] == altered.eval_rv_shapes()[coefficient.name]
    for graph, location in ((native, 0.4), (altered, 1.4)):
        raw, final = pm.draw(
            [graph[coefficient.name], graph["regression_coefficient"]], random_seed=7
        )
        np.testing.assert_allclose(final.ravel(), location + 0.7 * raw.ravel())
    names = [*(rv.name for rv in native.free_RVs), "loading", "item_intercept"]
    samples = pm.sample_prior_predictive(
        model=native, var_names=names, draws=20, random_seed=8
    ).prior.to_dataset()
    monkeypatch.setattr(
        pm, "sample", lambda **kwargs: xr.DataTree.from_dict({"posterior": samples})
    )
    original.fit(var_names=names)
    approximation = original.posterior_approximation(var_names=[coefficient.name])
    new = fc.ResponseTable(responses.assign(person_id="next"), format="item_long")
    assert original.score_model(new, calibration="posterior", approximation=approximation).build()
    for candidate in (changed, configured(0.4, scale=1.4)):
        with pytest.raises(ValueError, match="native calibration coordinates"):
            candidate.score_model(new, calibration="posterior", approximation=approximation)

    # A prior on an ordinary centered variable can change its density without
    # changing the natural coordinates retained by the selected approximation.
    same_coordinates = configured(0.4)
    same_coordinates.set_priors({"items.intercept": pm.Normal.dist(mu=2.0, sigma=1.0)})
    same_coordinates.score_model(new, calibration="posterior", approximation=approximation).build()

    path = tmp_path / "native-draws.nc"
    original.idata.to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as saved:
        matching = configured(0.4)
        matching.attach_samples(saved.load())
        matching.posterior_approximation(var_names=[coefficient.name])
        changed.attach_samples(saved, assume_same_batch=True)
        assert "loading" in changed.idata.posterior
        with pytest.raises(ValueError, match="native calibration coordinates"):
            changed.posterior_approximation(var_names=[coefficient.name])


def test_calibration_reuse_checks_frozen_item_predictors():
    pytest.importorskip("formulae")
    items = list("abcd")
    responses = pd.DataFrame({"person_id": "old", "item_id": items, "response": [0, 1, 0, 1]})
    source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": items}),
        fc.ResponseTable(responses, format="item_long"),
        model="DichotomousIRT",
    )
    features = pd.DataFrame({"item_id": items, "x": [0.0, 1.0, 3.0, 6.0]})

    def reference(frame):
        model = fc.Model(fc.PreparedInputs.from_inputs(source))
        model.add_regressions("items.loading_magnitude ~ x", item_data=frame)
        return model

    training = reference(features)
    graph = training.build()
    ancestors_of_coefficients = set(ancestors([graph["regression_coefficient"]]))
    coefficient = next(rv.name for rv in graph.free_RVs if rv in ancestors_of_coefficients)
    loading_rvs = set(graph.free_RVs).intersection(ancestors([graph["loading"]]))
    residual = {rv.name for rv in loading_rvs if rv.name != coefficient}
    _attach_draws(training, [coefficient, "regression_sd", "item_intercept"], seed=54)
    with pytest.raises(ValueError, match="Missing calibration draws to fix native RV"):
        training.posterior_approximation(var_names=[coefficient])
    training = reference(features)
    _attach_draws(
        training, [coefficient, *sorted(residual), "regression_sd", "item_intercept"], seed=55
    )
    approximation = training.posterior_approximation(var_names=[coefficient], random_seed=56)
    new = fc.ResponseTable(responses.assign(person_id="next"), format="item_long")
    matched = fc.score_model(training, new, calibration="posterior", approximation=approximation)
    assert matched.prepared.calibration_identity == training.prepared.calibration_identity
    model = matched.build()
    assert {rv.name for rv in model.free_RVs} == {"theta", "calibrated_prior_0_raw"}
    assert {"regression_coefficient", "regression_sd"} <= model.named_vars.keys()
    assert set(matched.idata.fixed_parameters.data_vars) >= {"item_intercept"}
    reordered = reference(features.iloc[::-1])
    assert reordered.prepared.calibration_identity == training.prepared.calibration_identity

    changed = reference(features.assign(x=[0.0, 2.0, 2.0, 6.0]))
    np.testing.assert_array_equal(
        changed.prepared.regressions[0].design.term, training.prepared.regressions[0].design.term
    )
    with pytest.raises(ValueError, match="different calibration design"):
        fc.score_model(changed, new, calibration="posterior", approximation=approximation)
    with pytest.raises(ValueError, match="different prepared response batch"):
        changed.attach_samples(training.idata)
    assert "posterior" not in changed.idata
