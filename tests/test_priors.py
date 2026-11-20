"""Native prior declarations, faithful records and independent graph ownership."""

from dataclasses import replace

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pymc.distributions.shape_utils import rv_size_is_none
from pytensor.graph.traversal import ancestors
from scipy.stats import multivariate_normal, norm, t

import pyfcirt as fc
from pyfcirt.models.priors import (
    bind_prior,
    prior_shape,
    prior_with_parameters,
    require_proper,
    same_prior,
    subset_prior,
    validate_prior,
)
from tests.data.test_schema import graded_input
from tests.helpers import mixed_inputs


def _callback_dist(prior):
    """Exercise symbolic rebinding using the original operator."""
    op, node = prior.owner.op, prior.owner
    size = op.size_param(node)
    return prior_with_parameters(
        prior, *op.dist_params(node), size=None if rv_size_is_none(size) else size
    )


@pytest.mark.parametrize("law", ["Normal", "StudentT", "HalfNormal", "HalfStudentT"])
def test_native_array_parameters_survive_records_and_match_density_and_gradient(law):
    mu, sigma = np.array([-0.3, 0.7]), np.array([0.4, 1.2])
    kwargs = {"sigma": sigma}
    if not law.startswith("Half"):
        kwargs["mu"] = mu
    if "StudentT" in law:
        kwargs["nu"] = np.array([4.0, 7.0])
    prior = getattr(pm, law).dist(**kwargs)
    validate_prior(prior, role="coefficient_or_scale", domain="real", shape=(2,))
    restored = _callback_dist(prior)
    assert same_prior(prior, restored)
    assert type(restored.owner.op) is type(prior.owner.op)
    assert prior_shape(prior) == ((2,), ())
    x = pt.dvector("x")
    density = pm.logp(restored, x).sum()
    evaluate = pytensor.function([x], [density, pt.grad(density, x)], mode="FAST_COMPILE")
    point = np.array([0.2, 1.1])
    center = 0 if law.startswith("Half") else mu
    delta = point - center
    if "StudentT" in law:
        nu = kwargs["nu"]
        expected = t.logpdf(point, nu, loc=center, scale=sigma)
        gradient = -(nu + 1) * delta / (nu * sigma**2 + delta**2)
    else:
        expected = norm.logpdf(point, loc=center, scale=sigma)
        gradient = -delta / sigma**2
    if law.startswith("Half"):
        expected += np.log(2)
    actual, actual_gradient = evaluate(point)
    assert_allclose(actual, expected.sum(), atol=1e-12)
    assert_allclose(actual_gradient, gradient, atol=1e-12)


@pytest.mark.parametrize(
    "prior, domain",
    [
        (pm.Normal.dist(), "unit"),
        (pm.Normal.dist(), "correlation"),
        (pm.Bernoulli.dist(p=0.5), "real"),
    ],
)
def test_native_domain_validation_never_samples(prior, domain, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Support must be established without drawing samples")

    monkeypatch.setattr(pm, "draw", forbidden)
    with pytest.raises(ValueError):
        validate_prior(prior, role="test", domain=domain, shape=())


@pytest.mark.parametrize("sigma", [-1.0, np.nan])
def test_ordinary_prior_arguments_are_validated_by_native_logp(sigma):
    source = fc.load_example("single_binary", model="DichotomousIRT", n_persons=1).data
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    model.set_priors({"items.intercept": pm.Normal.dist(sigma=sigma)})
    native = model.build()
    assert np.isneginf(native.compile_logp()(native.initial_point()))


def test_native_prior_folds_a_constant_graph_input_and_binds_its_original_law(monkeypatch):
    from pytensor.compile.builders import OpFromGraph

    from pyfcirt.data import regressions

    input_value = pt.dscalar()
    double = OpFromGraph([input_value], [2 * input_value])
    prior = pm.Normal.dist(mu=double(pt.constant(0.3, dtype="float64")), sigma=0.8)
    source = fc.load_example("single_binary", model="DichotomousIRT", n_persons=1).data
    model = fc.Model(source)
    previous = model.prepared

    def unexpected_finalization(*args, **kwargs):
        pytest.fail("An ordinary prior replacement rebuilt the parameter catalogue")

    monkeypatch.setattr(regressions, "finalize_parameters", unexpected_finalization)
    model.set_priors({"items.intercept": prior})
    assert model.prepared.parameter_uses.identical(previous.parameter_uses)
    assert model.prepared.observed_data.identical(previous.observed_data)
    assert "mu=0.6" in model.summary()
    native = model.build()
    (intercept,) = set(native.free_RVs).intersection(ancestors([native["item_intercept"]]))
    density = pm.logp(intercept, pt.full(intercept.type.shape, 0.4)).eval()
    assert_allclose(density, norm.logpdf(0.4, loc=0.6, scale=0.8), atol=1e-12)


def test_direct_precision_prior_is_used_without_support_preflight():
    source = replace(
        graded_input(0.25, beta=True),
        design="[traits]\nA~ a\nB~ b\n[blocks]\ngraded ~ a b\nside +~ a\nside -~ b",
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    model.set_priors({"fc.response_precision": pm.Normal.dist(mu=3, sigma=1)})
    native = model.build()
    (precision,) = set(native.free_RVs).intersection(ancestors([native["response_precision"]]))
    logp = native.compile_logp()
    point = native.initial_point()
    assert np.isfinite(logp(point))
    point[precision.name] = np.asarray(-1.0)
    assert np.isneginf(logp(point))


@pytest.mark.parametrize("law", [pm.Flat, pm.HalfFlat])
def test_improper_laws_are_rejected_for_proper_roles(law):
    with pytest.raises(ValueError, match="proper"):
        require_proper(law.dist(), role="scale")


@pytest.mark.parametrize(
    "prior", [pm.Beta.dist(alpha=2, beta=3), pm.Uniform.dist(lower=0.1, upper=0.9)]
)
def test_unit_support_accepts_native_bounded_priors(prior):
    validate_prior(prior, role="asymptote", domain="unit", shape=(2,))


def test_native_templates_are_unnamed_and_have_constant_parameters():
    with pm.Model():
        named = pm.Normal("named")
        dependent = pm.Normal.dist(mu=named)
    with pytest.raises(ValueError, match="unnamed"):
        validate_prior(named, role="coefficient", domain="real", shape=())
    with pytest.raises(ValueError, match="random ancestors"):
        validate_prior(dependent, role="coefficient", domain="real", shape=())
    with pytest.raises(TypeError, match="native"):
        validate_prior("Normal", role="coefficient", domain="real", shape=())
    shared = pytensor.shared(np.arange(3.0), shape=(None,))
    with pytest.raises(ValueError, match="requires constant native parameters"):
        validate_prior(pm.Normal.dist(mu=shared), role="coefficient", domain="real", shape=(3,))


def test_joint_event_retains_covariance_and_rejects_partial_replacement():
    mean = np.array([0.2, -0.4])
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    prior = pm.MvNormal.dist(mu=mean, cov=covariance)
    validate_prior(prior, role="joint", domain="real", shape=(3, 2))
    assert prior_shape(prior) == ((), (2,))
    restored = _callback_dist(prior)
    point = np.array([[0.3, -0.2], [1.0, 0.7], [-0.5, 0.4]])
    actual = pytensor.function([], pm.logp(restored, point), mode="FAST_COMPILE")()
    assert_allclose(actual, multivariate_normal.logpdf(point, mean, covariance), atol=1e-12)
    with pytest.raises(ValueError, match="joint event"):
        validate_prior(prior, role="joint", domain="real", shape=(3,))
    with pytest.raises(ValueError, match="whole block"):
        subset_prior(prior, [0], (2,))
    assert subset_prior(prior, [0, 1], (2,)) is prior


def test_native_binding_has_independent_rngs_without_mutating_template():
    template = pm.Normal.dist(mu=np.array([0.1, 0.5]), sigma=0.7)
    before = _callback_dist(template)
    with pm.Model():
        left = bind_prior(template, name="left", shape=(2,))
        right = bind_prior(template, name="right", shape=(2,))
    assert template.name is None and same_prior(template, before)
    assert left is not right and left is not template
    rngs = [rv.owner.inputs[0] for rv in (template, left, right)]
    assert len({id(rng) for rng in rngs}) == 3
    left_draws, right_draws = pm.draw([left, right], draws=5, random_seed=93)
    assert left_draws.shape == right_draws.shape == (5, 2)
    assert not np.array_equal(left_draws, right_draws)


def test_symbolic_law_restriction_comes_from_the_native_graph():
    prior = pm.Interpolated.dist(
        x_points=np.array([-2.0, 0.0, 2.0]), pdf_points=np.array([0.1, 0.4, 0.1])
    )
    validate_prior(prior, role="ordinary", domain="real", shape=(2,))
    assert same_prior(prior, _callback_dist(prior))
    source = replace(
        graded_input("equal"),
        model=fc.TIRT(loading_anchors={"a": 1.0, "b": 1.0}),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    model.set_priors({"fc.cutpoints": prior})
    with pytest.raises(AttributeError, match="object has no attribute 'data'"):
        model.build().logp()


def test_unknown_roles_and_non_native_values_fail_at_declaration():
    with pytest.raises(ValueError, match="Unknown prior"):
        fc.MGPPM(priors={"nonexistent": pm.Normal.dist()})
    with pytest.raises(TypeError):
        fc.MGPPM(priors={"ranking_intercept": "Normal"})
    assert not hasattr(fc, "Prior")


def test_signed_loading_warns_and_preserves_the_declared_density():
    source = mixed_inputs()[0]
    template = pm.Normal.dist(sigma=0.7)
    spec = fc.MGPPM(priors={"loading_magnitude": template})
    with pytest.warns(fc.IdentificationWarning):
        state = fc.PreparedInputs.from_inputs(replace(source, model=spec))
    record = (
        state.data["constant_data"].attrs["owners"][0]["specification"].priors["loading_magnitude"]
    )
    assert same_prior(record, template)


def test_model_summary_exposes_native_prior_and_reference_without_refreezing():
    model = fc.Model(fc.PreparedInputs.from_inputs(mixed_inputs()[0]))
    text = model.summary()
    assert isinstance(text, str)
    assert "Normal" in text and "MGPPM" in text


@pytest.mark.parametrize(
    "constructor, options, error",
    [
        (fc.TIRT, {"link": "identity"}, ValueError),
        (fc.TIRT, {"residuals": "none"}, ValueError),
        (fc.TIRT, {"statement_effect_parameterization": "other"}, ValueError),
        (fc.GTUM, {"link": "probit"}, ValueError),
        (fc.GTUM, {"block_effect": 1}, TypeError),
        (fc.GTUM, {"link": "beta_logit", "block_effect": True}, ValueError),
        (fc.DichotomousIRT, {"kind": "5PL"}, ValueError),
        (fc.PopulationSpec, {"regression_by_group": 1}, TypeError),
        (fc.PopulationSpec, {"theta_residual_scale": "free"}, ValueError),
    ],
)
def test_model_declarations_reject_unsupported_options(constructor, options, error):
    with pytest.raises(error):
        constructor(**options)


@pytest.mark.parametrize(
    "spec, active, message",
    [
        (
            fc.MGPPM(priors={"graded_intercept": pm.Normal.dist()}),
            {"ranking_intercept"},
            "inactive roles",
        ),
        (fc.TIRT(link="beta_probit"), {"response_precision"}, "Explicit priors required"),
    ],
)
def test_resolved_priors_require_exactly_the_active_roles(spec, active, message):
    from pyfcirt.models.specification import resolve_spec

    with pytest.raises(ValueError, match=message):
        resolve_spec(spec, active_roles=frozenset(active))


def test_beta_gtum_uses_its_documented_loading_prior_scale():
    from pyfcirt.models.specification import resolve_spec

    resolved = resolve_spec(
        fc.GTUM(link="beta_logit"), active_roles=frozenset({"loading_magnitude"})
    )
    assert same_prior(resolved.priors["loading_magnitude"], pm.LogNormal.dist(mu=0.0, sigma=0.25))


@pytest.mark.parametrize(
    "lower, upper, valid",
    [
        (pm.Uniform.dist(lower=0.05, upper=0.2), pm.Uniform.dist(lower=0.4, upper=0.9), True),
        (pm.Beta.dist(alpha=1, beta=4), pm.Beta.dist(alpha=4, beta=1), True),
        (pm.Uniform.dist(lower=0.1, upper=0.9), pm.Uniform.dist(lower=0.2, upper=0.5), False),
        (pm.Normal.dist(mu=0.2, sigma=0.1), pm.Beta.dist(alpha=8, beta=2), False),
    ],
)
def test_four_pl_priors_require_unit_support_and_nonempty_conditional_domain(lower, upper, valid):
    import pandas as pd

    source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["a"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1, 2], "item_id": ["a", "a"], "response": [0, 1]}),
            format="item_long",
        ),
        model=fc.DichotomousIRT(
            kind="4PL", priors={"lower_asymptote": lower, "upper_asymptote": upper}
        ),
    )
    if valid:
        model = fc.Model(fc.PreparedInputs.from_inputs(source))
        layouts = {layout.kind: layout for layout in model.prepared.parameters}
        graph = model.build()
        lower_name, upper_name = (
            layouts[key].kind for key in ("lower_asymptote", "upper_asymptote")
        )
        draws = pm.draw([graph[lower_name], graph[upper_name]], draws=8, random_seed=42)
        assert np.all(draws[0] < draws[1])
        text = model.summary()
        assert "lower=lower_asymptote, upper=1.0" in text
    else:
        with pytest.raises(ValueError, match="support|4PL"):
            fc.PreparedInputs.from_inputs(source)


@pytest.mark.parametrize("family", ["beta", "uniform"])
def test_four_pl_native_truncation_preserves_itemwise_density_and_gradient(family):
    from scipy.stats import beta

    source = fc.load_example("single_binary", model=fc.DichotomousIRT(kind="4PL")).data
    model = fc.Model(source)
    count = model.prepared.constant_data.sizes["item"]
    lower = np.linspace(0.12, 0.32, count)
    if family == "beta":
        alpha = np.linspace(2.0, 4.0, count)
        shape_b = np.linspace(1.0, 2.0, count)
        model.set_priors({"items.upper_asymptote": pm.Beta.dist(alpha=alpha, beta=shape_b)})
        point = lower + (1 - lower) * 0.45
        expected = beta.logpdf(point, alpha, shape_b) - beta.logsf(lower, alpha, shape_b)
        derivative = (alpha - 1) / point - (shape_b - 1) / (1 - point)
    else:
        floor, ceiling = np.linspace(0.05, 0.4, count), np.linspace(0.7, 0.9, count)
        model.set_priors(
            {
                "items.lower_asymptote": pm.Uniform.dist(lower=0.01, upper=0.5),
                "items.upper_asymptote": pm.Uniform.dist(lower=floor, upper=ceiling),
            }
        )
        width = ceiling - np.maximum(lower, floor)
        point = ceiling - width * 0.55
        expected, derivative = -np.log(width), np.zeros(count)
    graph = pm.do(model.build(), {"lower_asymptote": lower}, make_interventions_shared=False)
    (upper,) = set(graph.free_RVs).intersection(ancestors([graph["upper_asymptote"]]))
    value = pt.dvector("upper_value", shape=(count,))
    logp = pm.logp(upper, value)
    evaluate = pytensor.function([value], [logp, pt.grad(logp.sum(), value)])
    actual, gradient = evaluate(point)
    assert_allclose(actual, expected)
    assert_allclose(gradient, derivative)


def test_native_open_truncation_bounds_and_batch_subsets_preserve_parameters():
    prior = pm.TruncatedNormal.dist(
        mu=np.array([0.2, 0.8, 1.1]), sigma=np.array([0.5, 0.9, 1.4]), lower=0
    )
    validate_prior(prior, role="scale", domain="real", shape=(3,))
    assert same_prior(prior, _callback_dist(prior))
    subset = subset_prior(prior, [2, 0], (3,))
    expected = pm.TruncatedNormal.dist(mu=[1.1, 0.2], sigma=[1.4, 0.5], lower=0, shape=2)
    point = np.array([0.6, 0.4])
    actual = pytensor.function([], pm.logp(subset, point), mode="FAST_COMPILE")()
    reference = pytensor.function([], pm.logp(expected, point), mode="FAST_COMPILE")()
    assert_allclose(actual, reference, atol=1e-12)


@pytest.mark.parametrize(
    "updates,valid",
    [
        ({"lower_asymptote": (0.1, 0.95)}, False),
        ({"upper_asymptote": (0.05, 0.15)}, False),
        ({"lower_asymptote": (0.1, 0.85)}, True),
        ({"upper_asymptote": (0.3, 0.4)}, True),
        ({"lower_asymptote": (0.9, 0.95), "upper_asymptote": (0.96, 0.99)}, True),
    ],
)
def test_four_pl_prior_replacement_checks_the_complete_candidate_pair(updates, valid):
    source = fc.load_example(
        "single_binary",
        model=fc.DichotomousIRT(
            kind="4PL",
            priors={
                "lower_asymptote": pm.Uniform.dist(lower=0.05, upper=0.2),
                "upper_asymptote": pm.Uniform.dist(lower=0.4, upper=0.9),
            },
        ),
        n_persons=1,
    ).data
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    previous = model.prepared
    priors = {f"items.{key}": pm.Uniform.dist(lower=a, upper=b) for key, (a, b) in updates.items()}
    if valid:
        model.set_priors(priors)
        for key in updates:
            (layout,) = model.prepared.blocks_by_kind[key]
            assert same_prior(layout.priors[key], priors[f"items.{key}"])
        floor, ceiling = updates.get("upper_asymptote", (0.4, 0.9))
        graph = model.build()
        lower, upper = pm.draw(
            [graph["lower_asymptote"], graph["upper_asymptote"]], draws=16, random_seed=43
        )
        assert np.all((upper > lower) & (upper >= floor) & (upper <= ceiling))
    else:
        with pytest.raises(ValueError, match="empty support"):
            model.set_priors(priors)
        assert model.prepared is previous


def test_four_pl_unsupported_joint_prior_does_not_publish_a_candidate():
    source = fc.load_example("single_binary", model=fc.DichotomousIRT(kind="4PL")).data
    prepared = fc.PreparedInputs.from_inputs(source)
    count = next(
        len(layout.free_flat) for layout in prepared.parameters if layout.kind == "upper_asymptote"
    )
    model = fc.Model(prepared)
    with pytest.raises(ValueError, match=r"support inside \[0, 1\]"):
        model.set_priors({"items.upper_asymptote": pm.Dirichlet.dist(a=np.ones(count))})
    assert model.prepared is prepared and model._built is None


@pytest.mark.parametrize(
    "spec, role, center, intercept_sd, slope_sd, residual_sd",
    [
        *[
            (constructor(), "loading_magnitude", 0, 0.5, 0.25, 0.2)
            for constructor in (
                fc.MGPPM,
                fc.TwoPLRank,
                fc.GGUMRank,
                fc.TIRT,
                fc.GTUM,
                fc.TIRTRank,
                fc.DichotomousIRT,
                fc.GPCM,
                fc.GGUM,
            )
        ],
        (fc.GTUM(), "location_positive", 2, 1, 1, 1),
        (fc.GGUMRank(), "location_negative", -2, 1, 1, 1),
        (fc.GGUM(), "location_positive", 1, 1, 1, 1),
        (fc.GGUM(), "location_negative", -1, 1, 1, 1),
        (fc.GGUM(), "location_free", 0, 1, 1, 1),
        (fc.MGPPM(), "ranking_intercept", 0, 1, 1, 1),
        (fc.MGPPM(), "graded_intercept", 0, 1, 1, 1),
        (fc.GGUMRank(), "threshold", 0, 1, 1, 1),
        (fc.GGUM(), "threshold", 0, 1, 1, 1),
        (fc.TIRT(), "cutpoints", 0, 1, 1, 1),
    ],
)
def test_uniform_regression_defaults_leave_ordinary_laws_and_conditionals_unchanged(
    spec, role, center, intercept_sd, slope_sd, residual_sd
):
    from pyfcirt.models.specification import (
        conditional_defaults,
        default_prior,
        regression_defaults,
    )

    ordinary = default_prior(spec, role)
    conditional = conditional_defaults(spec, role)
    intercept, scale, actual_slope_sd = regression_defaults(spec, role)
    assert same_prior(
        intercept, pm.Normal.dist(mu=0.0 if center == 0 else center, sigma=float(intercept_sd))
    )
    assert same_prior(scale, pm.HalfNormal.dist(sigma=float(residual_sd)))
    assert actual_slope_sd == slope_sd
    assert same_prior(default_prior(spec, role), ordinary)
    if isinstance(spec, (fc.DichotomousIRT, fc.GPCM)) and role == "loading_magnitude":
        assert same_prior(default_prior(spec, role), pm.LogNormal.dist(mu=0.25, sigma=0.5))
    assert conditional_defaults(spec, role) == conditional
    if role.startswith("location_"):
        assert conditional["family"] == ("Normal" if role == "location_free" else "TruncatedNormal")
        assert conditional["bounds"] == (
            [0, None]
            if role == "location_positive"
            else [None, 0]
            if role == "location_negative"
            else None
        )


def test_numpy_anchor_metadata_and_repeated_preparation_keep_independent_records():
    import pandas as pd

    design = pd.DataFrame(
        dict(
            block_id=["b"] * 3,
            position=[1, 2, 3],
            statement_id=[10, 20, 30],
            trait=["A", "B", "C"],
            keying=[1, -1, 1],
            fix_var=[0.5] * 3,
        )
    )
    source = fc.FCInput(
        design,
        fc.ResponseTable(
            pd.DataFrame(
                dict(person_id=[1, 2], block_id=["b", "b"], response=[[1, 2, 3], [3, 2, 1]])
            ),
            format="block_long",
        ),
        model=fc.TIRT(loading_anchors={np.int64(10): np.float64(1.0)}),
    )
    first = fc.PreparedInputs.from_inputs(source)
    (first_loading,) = first.blocks_by_kind["loading"]
    prior = first_loading.priors["loading_magnitude"]
    same_configuration = _callback_dist(prior)
    assert same_prior(prior, same_configuration)
    second = fc.PreparedInputs.from_inputs(source)
    (second_loading,) = second.blocks_by_kind["loading"]
    assert same_prior(second_loading.priors["loading_magnitude"], prior)
    np.testing.assert_array_equal(
        second.constant_data.statement_label, first.constant_data.statement_label
    )
    assert source.model.loading_anchors[np.int64(10)] == 1.0
