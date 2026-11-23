"""Frozen population encodings and independent Gaussian moment identities."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pyfcirt as fc
from pyfcirt.common.validation import align_rows
from pyfcirt.covariates.design import fit_design
from pyfcirt.data.grouping import apply_groups, fit_groups
from pyfcirt.models.population import PopulationFactory
from tests.helpers import mixed_inputs, replay_design


def test_person_predictors_align_by_typed_id_and_center_from_training_data():
    people = pd.DataFrame({"person_id": [2, 1], "age": [30.0, 10.0]})
    model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs()[1]))
    model.add_regressions({"theta": fc.Regression("age", people, id="person_id")})
    design = model.prepared.data["covariates/r0"]
    np.testing.assert_allclose(design.X.values.ravel(), [-10.0, 10.0])
    assert design.attrs["feature_records"][0]["center"] == 20.0
    assert design.attrs["feature_records"][0]["scale"] == 1.0
    assert "predictors" not in model.prepared.data["constant_data"]
    people.loc[:, "age"] = 999
    np.testing.assert_allclose(design.X.values.ravel(), [-10.0, 10.0])


@pytest.mark.parametrize("formula", ["age ** 2", "log(age)", "age - age", "person_id"])
def test_unsupported_regression_formulas_fail_at_preparation(formula):
    with pytest.raises(ValueError):
        model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs()[1]))
        model.add_regressions(
            {
                "theta": fc.Regression(
                    formula,
                    pd.DataFrame({"person_id": [1, 2], "age": [10.0, 30.0]}),
                    id="person_id",
                )
            }
        )


def test_missing_person_covariates_are_not_imputed():
    with pytest.raises(ValueError, match="missing"):
        model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs()[1]))
        model.add_regressions(
            {
                "theta": fc.Regression(
                    "age", pd.DataFrame({"person_id": [1], "age": [10.0]}), id="person_id"
                )
            }
        )


def test_covariance_and_conditional_theta_match_schur_complement():
    covariance = np.array([[1.0, 0.4], [0.4, 2.0]])
    correlation, sd = PopulationFactory._from_covariance(covariance)
    np.testing.assert_allclose(sd, np.sqrt([1.0, 2.0]))
    np.testing.assert_allclose(PopulationFactory._covariance(correlation, sd), covariance)
    theta = PopulationFactory.conditional_theta(
        np.array([[0.5]]),
        np.array([[1.2]]),
        np.array([[0.1, -0.2]]),
        covariance[None],
        np.array([0]),
        np.array([1]),
    )
    expected = -0.2 + 0.4 * (0.5 - 0.1) + np.sqrt(2.0 - 0.4**2) * 1.2
    np.testing.assert_allclose(theta, [[0.5, expected]], rtol=1e-12)


def test_conditional_parameterization_tracks_fc_and_noncentered_traits():
    state = fc.PreparedInputs.from_inputs(mixed_inputs(), theta_parameterization="conditional")
    c = state.data["constant_data"]
    assert c.centered_trait.values.tolist() == ["A", "B"]
    assert c.noncentered_trait.size == 0
    assert c.attrs["population"]["theta_parameterization"] == "conditional"


def test_centered_ability_parameterization_remains_the_default():
    model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs()))
    assert (
        model.prepared.data["constant_data"].attrs["population"]["theta_parameterization"]
        == "centered"
    )
    graph = model.build()
    assert graph["theta"] in graph.free_RVs
    assert "theta_raw" not in graph


def test_optional_noncentered_theta_matches_regression_covariance_and_centered_law():
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace

    people = pd.DataFrame({"person_id": [1, 2], "age": [10.0, 30.0]})
    models = {}
    for mode in ("centered", "noncentered"):
        model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs(), theta_parameterization=mode))
        model.add_regressions({"theta": fc.Regression("age", people, id="person_id")})
        models[mode] = model
    coefficients = np.array([[0.15, -0.25]])
    covariance = np.array([[1.0, 0.45], [0.45, 1.0]])
    factor = np.linalg.cholesky(covariance)
    mean = np.array([[-10.0], [10.0]]) @ coefficients
    z = np.array([[0.4, -0.7], [-0.2, 0.6]])
    theta = mean + z @ factor.T
    raw, final = pt.dmatrices("raw", "final")
    graph = models["noncentered"].build()
    assert graph["theta_raw"] in graph.free_RVs and graph["theta"] in graph.deterministics
    assert "theta" in models["noncentered"]._sampling_names()
    assert "theta_raw" not in models["noncentered"]._sampling_names()

    def fixed_population(native):
        return {
            native["regression_coefficient"]: pt.as_tensor_variable(coefficients.ravel()),
            native["population_mean"]: pt.zeros((1, 2)),
            native["covariance"]: pt.as_tensor_variable(covariance),
        }

    transform = graph_replace(
        graph["theta"],
        replace=fixed_population(graph) | {graph["theta_raw"]: raw},
        strict=False,
    )
    evaluate = pytensor.function([raw], transform, mode="JAX")
    np.testing.assert_allclose(evaluate(z), theta, atol=1e-12)
    # Symmetric standard-normal cubature has exact first and second moments.
    points = np.sqrt(2.0) * np.concatenate([np.eye(2), -np.eye(2)])
    values = np.stack([evaluate(np.broadcast_to(point, z.shape)) for point in points])
    np.testing.assert_allclose(values.mean(axis=0), mean, atol=1e-12)
    for person in range(2):
        centered = values[:, person] - mean[person]
        np.testing.assert_allclose(centered.T @ centered / len(points), covariance, atol=1e-12)

    centered_graph = models["centered"].build()
    density = graph_replace(
        pm.logp(centered_graph["theta"], final).sum(),
        replace=fixed_population(centered_graph),
        strict=False,
    )
    centered_logp = pytensor.function([final], density, mode="JAX")(theta)
    raw_logp = pytensor.function([], pm.logp(graph["theta_raw"], z).sum(), mode="JAX")()
    independent_raw_logp = -0.5 * np.sum(z**2) - z.size / 2 * np.log(2 * np.pi)
    np.testing.assert_allclose(raw_logp, independent_raw_logp, atol=1e-12)
    # Density equivalence includes the affine change-of-variables determinant.
    np.testing.assert_allclose(
        raw_logp, centered_logp + len(z) * np.log(np.diag(factor)).sum(), atol=1e-12
    )


def test_group_specific_regression_and_covariance_match_explicit_person_rows():
    groups = np.array([1, 0])
    means = np.array([[0.0, 0.0], [1.0, -2.0]])
    # The unique regression builder supplies already selected X B_g rows.
    regression_mean = np.array([[4.0, -1.0], [-0.5, -1.0]])
    covariances = np.array([np.eye(2), [[2.0, 0.3], [0.3, 1.0]]])
    mean, cov = PopulationFactory.moments(
        groups, means, covariances, regression_mean=regression_mean
    )
    np.testing.assert_allclose(mean, [[5.0, -3.0], [-0.5, -1.0]])
    np.testing.assert_allclose(cov, covariances[[1, 0]])


def test_frozen_person_recipe_reuses_training_scale_and_rejects_new_levels():
    training = pd.DataFrame(
        {"person_id": [1, 2, 3, 4], "age": [10.0, 20.0, 30.0, 40.0], "school": ["A", "B", "A", "B"]}
    )
    design = fit_design(
        fc.Regression("age + school", training, id="person_id", standardize_predictors=True),
        training[["person_id"]],
        include_intercept=False,
    )
    new = pd.DataFrame({"person_id": [9], "age": [50.0], "school": ["A"]})
    result = replay_design(design, new, new[["person_id"]])
    np.testing.assert_allclose(result, [[25.0 / np.sqrt(125.0), 0.0]])
    assert result.term.values.tolist() == design.term.values.tolist()
    new.loc[0, "school"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        replay_design(design, new, new[["person_id"]])


@pytest.mark.parametrize("theta_residual_scale", ["unit", "group"])
@pytest.mark.parametrize("theta_correlation", ["group", "shared"])
def test_group_scale_defaults_and_only_nonreference_sd_is_free(
    theta_residual_scale, theta_correlation
):

    people = pd.DataFrame({"person_id": [1, 2], "school": ["A", "B"]})
    options = {}
    options["theta_residual_scale"] = theta_residual_scale
    if theta_correlation == "shared":
        options["theta_correlation"] = "shared"
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            mixed_inputs(),
            person_data=people,
            population=fc.PopulationSpec(group="school", reference_group="A", **options),
        )
    )
    layouts = {layout.kind: layout for layout in model.prepared.parameters}
    sd = layouts["population_sd"]
    expected_free = [] if theta_residual_scale == "unit" else [2, 3]
    assert sd.free_flat.tolist() == expected_free
    np.testing.assert_allclose(sd.fixed_value, np.ones(len(sd.fixed_flat)))
    # All reference variances are fixed to one, including in explicit group mode.
    covariance = layouts["covariance"]
    fixed = dict(zip(covariance.fixed_flat, covariance.fixed_value, strict=True))
    assert fixed[0] == fixed[3] == 1.0
    if theta_residual_scale == "unit":
        assert fixed[4] == fixed[7] == 1.0

    correlation_shape = (2, 2) if theta_correlation == "shared" else (2, 2, 2)
    assert layouts["trait_correlation"].shape == correlation_shape
    graph = model.build()
    correlation, sd, sigma, mean = pm.draw(
        [
            graph[name]
            for name in ("trait_correlation", "population_sd", "covariance", "population_mean")
        ],
        random_seed=37,
    )
    np.testing.assert_allclose(sigma, correlation * sd[..., :, None] * sd[..., None, :])
    np.testing.assert_array_equal(mean[0], [0, 0])
    if theta_residual_scale == "unit":
        np.testing.assert_array_equal(sd, np.ones((2, 2)))


def test_fixed_scoring_binds_population_reports_from_shared_correlation():
    from pytensor.tensor import TensorConstant

    from tests.helpers import labelled_point, supplied_posterior

    source = mixed_inputs()[1]
    people = pd.DataFrame({"person_id": [1, 2], "school": ["A", "B"]})
    reference = fc.Model(
        fc.PreparedInputs.from_inputs(
            source,
            person_data=people,
            population=fc.PopulationSpec(
                group="school",
                reference_group="A",
                theta_residual_scale="group",
                theta_correlation="shared",
            ),
        )
    )
    point = labelled_point(
        reference,
        {
            "population_mean": [[0, 0], [0.1, -0.2]],
            "covariance": [[[1, 0.35], [0.35, 1]], [[4, 2.1], [2.1, 9]]],
            "loading": [1, 1],
            "item_intercept": [0.2],
            "lower_asymptote": [0],
            "upper_asymptote": [1],
            "category_intercept": [0, -0.3, -0.8],
        },
    )
    scored = fc.score_model(reference, source.responses, person_data=people, parameters=point)
    graph = scored.build()
    assert [rv.name for rv in graph.free_RVs] == ["theta"]
    assert {"trait_correlation", "population_sd"}.isdisjoint(scored.idata.fixed_parameters)
    for name, dims, expected in (
        ("trait_correlation", ("trait", "trait_aux"), [[1, 0.35], [0.35, 1]]),
        ("population_sd", ("population_group", "trait"), [[1, 1], [2, 3]]),
    ):
        assert graph.named_vars_to_dims[name] == dims
        assert isinstance(graph[name], TensorConstant)
        np.testing.assert_allclose(graph[name].eval(), expected)

    original_results = reference.idata
    incompatible = point.copy(deep=True)
    incompatible["covariance"].values[1, 0, 1] = incompatible["covariance"].values[1, 1, 0] = 0
    with pytest.raises(ValueError, match="declared shared R"):
        fc.score_model(reference, source.responses, person_data=people, parameters=incompatible)
    assert reference.idata is original_results
    with pytest.raises(ValueError, match="declared shared R"):
        supplied_posterior(reference, incompatible, inference_method="mcmc")
    assert reference.idata is original_results
    supplied_posterior(reference, point, inference_method="mcmc")


def test_marginal_moments_match_independent_total_covariance():
    means = np.array([[0.0, 1.0], [2.0, -1.0], [3.0, 2.0]])
    covariances = np.array(
        [[[1.0, 0.2], [0.2, 2.0]], [[2.0, -0.3], [-0.3, 1.0]], [[1.5, 0.1], [0.1, 0.5]]]
    )
    average = means.mean(axis=0)
    total = sum(
        covariance + np.outer(mean - average, mean - average)
        for mean, covariance in zip(means, covariances, strict=True)
    ) / len(means)
    sd = np.sqrt(total.diagonal())
    expected = total / np.outer(sd, sd)
    mean, actual_sd, correlation = PopulationFactory.marginal_moments(means, covariances)
    np.testing.assert_allclose(mean, average, atol=1e-12)
    np.testing.assert_allclose(actual_sd, sd, atol=1e-12)
    np.testing.assert_allclose(correlation, expected, atol=1e-12)
    # A one-person scoring population has no between-person covariance contribution.
    single = covariances[1]
    sd = np.sqrt(single.diagonal())
    mean, actual_sd, correlation = PopulationFactory.marginal_moments(means[1:2], covariances[1:2])
    np.testing.assert_allclose(mean, means[1], atol=1e-12)
    np.testing.assert_allclose(actual_sd, sd, atol=1e-12)
    np.testing.assert_allclose(correlation, single / np.outer(sd, sd), atol=1e-12)


@pytest.mark.parametrize("by_group", [False, True])
def test_population_consumes_the_unique_B_with_group_and_trait_order_preserved(by_group):
    from dataclasses import replace

    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace

    source = mixed_inputs()[1]
    answers = source.responses.frame.loc[source.responses.frame.person_id.eq(1)]
    source = replace(
        source,
        responses=fc.ResponseTable(
            pd.concat([answers.assign(person_id=person) for person in range(4)], ignore_index=True),
            format="item_long",
        ),
    )
    people = pd.DataFrame(
        {"person_id": range(4), "group": ["ref", "ref", "other", "other"], "x": [10, 20, 30, 40]}
    )
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            source,
            person_data=people,
            population=fc.PopulationSpec(group="group", regression_by_group=by_group),
        )
    )
    model.add_regressions({"theta": fc.Regression("x", people, id="person_id")})
    graph = model.build()
    binding = model._bindings.parameters["regression_coefficient"]
    assert binding is graph["regression_coefficient"]
    coefficients = np.array([[[0.4, -0.2]], [[0.1, 0.3]]])
    group_means = np.array([[0.0, 0.0], [1.0, -1.0]])
    B = coefficients if by_group else coefficients[0]
    node = graph["theta"].owner
    mean, _ = node.op.dist_params(node)
    expression = graph_replace(
        mean,
        replace={
            graph["regression_coefficient"]: pt.as_tensor_variable(B.ravel()),
            graph["population_mean"]: pt.as_tensor_variable(group_means),
            # The shared moments kernel also returns covariance; fix every input.
            graph["covariance"]: pt.as_tensor_variable(np.stack([np.eye(2)] * 2)),
        },
    )
    x = np.array([-5, 5, 15, 25])
    groups = np.array([0, 0, 1, 1])
    slopes = coefficients[groups, 0] if by_group else coefficients[0, 0]
    expected = group_means[groups] + x[:, None] * slopes
    np.testing.assert_allclose(pytensor.function([], expression, mode="JAX")(), expected)


@pytest.mark.parametrize("channels", ["items", "fc", "mixed_extra_trait"])
def test_conditional_ability_parameterization_builds_each_actual_trait_partition(channels):
    from dataclasses import replace

    fc_input, item_input = mixed_inputs()
    if channels == "items":
        inputs = item_input
    elif channels == "fc":
        inputs = fc_input
    else:
        item_input = replace(
            item_input, design=fc.ItemDesignMapping(traits={"C": ["pair", "ordinal"]})
        )
        inputs = (fc_input, item_input)
    model = fc.Model(fc.PreparedInputs.from_inputs(inputs, theta_parameterization="conditional"))
    graph = model.build()
    assert any(variable.name == "theta" for variable in graph.deterministics)
    assert ("theta_centered_raw" in graph) is (channels != "items")
    assert ("theta_noncentered_raw" in graph) is (channels != "fc")
    initial = graph.initial_point(random_seed=52)
    assert np.isfinite(graph.compile_logp(mode="JAX")(initial))


@pytest.mark.parametrize(
    "options, error",
    [
        ({"group": "person_id"}, ValueError),
        ({"regression_by_group": 1}, TypeError),
        ({"regression_by_group": True}, ValueError),
        ({"reference_group": "A"}, ValueError),
        ({"theta_residual_scale": "free"}, ValueError),
        ({"theta_residual_scale": "group"}, ValueError),
        ({"group": "school", "reference_group": "absent"}, ValueError),
    ],
)
def test_group_recipe_options_fail_before_encoding(options, error):
    people = pd.DataFrame({"person_id": [1, 2], "school": ["A", "B"]})
    with pytest.raises(error):
        fit_groups(people, [1, 2], fc.PopulationSpec(**options))


@pytest.mark.parametrize("problem", ["empty", "duplicate", "no_data", "ids", "missing", "group"])
def test_group_alignment_does_not_invent_people_or_group_labels(problem):
    people = pd.DataFrame({"person_id": [1, 2], "school": ["A", "B"]})
    ids = [1, 2]
    if problem == "empty":
        ids = []
    elif problem == "duplicate":
        ids = [1, 1]
    elif problem == "no_data":
        people = None
    elif problem == "ids":
        people.loc[1, "person_id"] = 1
    elif problem == "missing":
        ids = [1, 3]
    else:
        people.loc[1, "school"] = None
    with pytest.raises(ValueError):
        fit_groups(people, ids, fc.PopulationSpec(group="school"))


@pytest.mark.parametrize("groups", [("A", "B", "unused"), (2**80, 2**80 + 1, 2**80 + 2)])
def test_group_recipe_uses_training_order_and_replays_custom_ids(groups):
    first, reference, unseen = groups
    people = pd.DataFrame({"person_id": [2, 1, 9], "school": [reference, first, unseen]})
    population = fc.PopulationSpec(group="school", reference_group=reference)
    recipe, index, _ = fit_groups(people, [1, 2], population)
    assert recipe["groups"] == [first, reference]
    assert recipe["reference_group"] == reference
    np.testing.assert_array_equal(index, [0, 1])
    new = pd.DataFrame({"pid": [7], "school": [reference]})
    positions, identities = align_rows(new, new[["pid"]], ("pid",), "person_data")
    np.testing.assert_array_equal(apply_groups(recipe, new, positions, identities), [1])
    with pytest.raises(ValueError, match="Unknown population groups"):
        apply_groups(recipe, new.assign(school=unseen), positions, identities)
    ungrouped, index, _ = fit_groups(
        pd.DataFrame({"person_id": [2, 1]}), [1, 2], fc.PopulationSpec()
    )
    assert ungrouped["groups"] == []
    np.testing.assert_array_equal(index, [0, 0])


@pytest.mark.parametrize("case", ["no_regression", "item_only", "group_predictor"])
def test_group_regression_requires_theta_without_repeating_group_indicators(case):
    people = pd.DataFrame(
        {"person_id": range(4), "school": ["A", "A", "B", "B"], "x": [0.0, 1.0, 2.0, 3.0]}
    )
    data = fc.ItemInput(
        fc.ItemDesignMapping(traits={"T": ["a", "b"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": range(4), "item_id": ["a"] * 4, "response": [0, 1, 0, 1]}),
            format="item_long",
        ),
    )
    state = fc.PreparedInputs.from_inputs(
        data,
        person_data=people,
        population=fc.PopulationSpec(group="school", regression_by_group=True),
    )
    regressions = None
    if case == "item_only":
        regressions = {
            ("items", "intercept"): fc.Regression("1", pd.DataFrame({"id": ["a", "b"]}), id="id")
        }
    elif case == "group_predictor":
        regressions = {"theta": fc.Regression("x + school", people, id="person_id")}
    model = fc.Model(state)
    if case == "group_predictor":
        with pytest.raises(ValueError, match="repeat.*group"):
            model.add_regressions(regressions)
    else:
        if regressions is not None:
            model.add_regressions(regressions)
        with pytest.raises(ValueError, match="requires a theta"):
            model.build()
        # A later theta declaration completes the same prepared population.
        model.add_regressions({"theta": fc.Regression("x", people, id="person_id")})
        assert "regression_coefficient" in model.build()


@pytest.mark.parametrize("centered", [[], ["A"], ["B"], ["B", "A"]])
def test_explicit_trait_partition_builds_and_survives_scoring_and_generation(centered):
    from tests.helpers import workflow_point

    inputs = mixed_inputs()
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            inputs, theta_parameterization="conditional", centered_traits=centered
        )
    )
    expected = [trait for trait in ("A", "B") if trait in centered]
    graph = model.build()
    assert ("theta_centered_raw" in graph) == bool(centered)
    assert ("theta_noncentered_raw" in graph) == (len(centered) < 2)
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=31)))
    point = workflow_point(model)
    responses = {
        channel: source.responses for channel, source in zip(("fc", "items"), inputs, strict=True)
    }
    scored = fc.score_model(model, responses, parameters=point)
    generated = fc.simulate(model, source="fixed", parameters=point, random_seed=31)
    for candidate in (model, scored, generated):
        constant = candidate.prepared.constant_data
        assert constant.centered_trait.values.tolist() == expected
        assert constant.noncentered_trait.values.tolist() == [
            trait for trait in ("A", "B") if trait not in centered
        ]
        for kind in ("centered", "noncentered"):
            np.testing.assert_array_equal(
                constant.trait.values[constant[f"{kind}_index"].values],
                constant[f"{kind}_trait"].values,
            )


@pytest.mark.parametrize(
    "selection,mode,message",
    [
        (["missing"], "conditional", "unknown"),
        (["A", "A"], "conditional", "duplicate"),
        (["A", 1], "conditional", "mixes"),
        (["A"], "centered", "requires"),
        ([], "noncentered", "requires"),
    ],
)
def test_invalid_centered_trait_selection_fails_during_preparation(selection, mode, message):
    with pytest.raises(ValueError, match=message):
        fc.PreparedInputs.from_inputs(
            mixed_inputs(), theta_parameterization=mode, centered_traits=selection
        )


def test_centered_traits_rejects_a_bare_string():
    with pytest.raises(TypeError, match="sequence"):
        fc.PreparedInputs.from_inputs(
            mixed_inputs(), theta_parameterization="conditional", centered_traits="A"
        )


@pytest.mark.parametrize("centered", [[], [0], [1], [0, 2], [0, 1, 2]])
def test_selected_conditional_partition_preserves_joint_gaussian_and_gradients(centered):
    import jax
    import jax.numpy as jnp
    from scipy.stats import multivariate_normal, norm

    c = np.array(centered, dtype=int)
    n = np.array([i for i in range(3) if i not in centered], dtype=int)
    covariance = np.array([[1.0, 0.4, -0.2], [0.4, 2.0, 0.3], [-0.2, 0.3, 1.5]])
    means = np.array([[0.2, -0.5, 0.8], [-0.3, 0.4, 0.1]])
    # Two group-specific covariances and distinct regression means use the same kernel.
    covariances = np.stack([covariance, covariance * 1.3])

    def transform(z):
        theta_c = means[:, c] + np.linalg.cholesky(covariances[:, c[:, None], c]) @ z[c]
        return PopulationFactory.conditional_theta(
            theta_c, jnp.broadcast_to(z[n], (2, len(n))), means, covariances, c, n
        )

    points = np.sqrt(3.0) * np.concatenate([np.eye(3), -np.eye(3)])
    values = np.array([transform(z) for z in points])
    np.testing.assert_allclose(values.mean(axis=0), means, atol=1e-12)
    residual = values - means
    np.testing.assert_allclose(
        np.einsum("spi,spj->pij", residual, residual) / 6, covariances, atol=1e-12
    )
    # Change of variables: joint theta density plus Jacobian equals centered
    # marginal density times independent standardized conditional residuals.
    z = np.array([0.3, -0.7, 0.5])
    jacobian = np.asarray(jax.jacfwd(transform)(jnp.asarray(z)))
    theta = np.asarray(transform(z))
    for i in range(2):
        theta_c = theta[i, c]
        centered_logp = (
            multivariate_normal.logpdf(theta_c, mean=means[i, c], cov=covariances[i][np.ix_(c, c)])
            if len(c)
            else 0.0
        )
        # The transform also whitens the centered draw for exact cubature.
        # Subtract its marginal Cholesky determinant from the full Jacobian.
        centered_logdet = (
            np.linalg.slogdet(np.linalg.cholesky(covariances[i][np.ix_(c, c)]))[1]
            if len(c)
            else 0.0
        )
        actual = multivariate_normal.logpdf(theta[i], mean=means[i], cov=covariances[i])
        actual += np.linalg.slogdet(jacobian[i])[1] - centered_logdet
        np.testing.assert_allclose(actual, centered_logp + norm.logpdf(z[n]).sum(), atol=1e-12)


@pytest.mark.parametrize("mode,expected", [("centered", ["A", "B"]), ("noncentered", [])])
def test_partition_coordinates_describe_uniform_modes(mode, expected):
    prepared = fc.PreparedInputs.from_inputs(mixed_inputs(), theta_parameterization=mode)
    assert prepared.constant_data.centered_trait.values.tolist() == expected


def test_theta_regression_reuses_explicit_centered_trait_partition():
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            mixed_inputs(), theta_parameterization="conditional", centered_traits=["B"]
        )
    )
    model.add_regressions(
        {
            "theta": fc.Regression(
                "age", pd.DataFrame({"person_id": [1, 2], "age": [10.0, 30.0]}), id="person_id"
            )
        }
    )
    np.testing.assert_array_equal(model.prepared.population.centered_index, [1])
    np.testing.assert_array_equal(model.prepared.population.noncentered_index, [0])
    graph = model.build()
    assert "theta_centered_raw" in graph and "theta_noncentered_raw" in graph
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=15)))
