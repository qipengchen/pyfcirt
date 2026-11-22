"""Small independent fixed-parameter checks of native measurement distributions."""

from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pytensor.graph.replace import graph_replace
from scipy.special import expit, logsumexp
from scipy.stats import beta, norm

from pyfcirt.models import dist, likert, mgppm, sequential
from pyfcirt.models.thurstonian import gtum, rank, tirt
from tests.reference.likert import subjective_log_probs
from tests.reference.rankings import order_arrays
from tests.reference.sequential import scalar_odds, scalar_probabilities
from tests.reference.thurstonian import gtum_log_probs, tirt_log_probs


def test_public_measurement_distribution_exports_use_native_classes():
    native = (
        likert.DichotomousIRTDistribution,
        likert.GGUMDistribution,
        likert.GPCMDistribution,
        sequential.GGUMRankDistribution,
        sequential.TwoPLRankDistribution,
        gtum.GTUMDistribution,
        mgppm.MGPPMDistribution,
        tirt.TIRTDistribution,
        rank.TIRTRankDistribution,
    )
    assert set(dist.__all__) == {klass.__name__ for klass in native}
    for klass in native:
        assert getattr(dist, klass.__name__) is klass


@pytest.mark.parametrize("family", ["mgppm", "sequential"])
@pytest.mark.parametrize("bad", ["patterns", "membership_type", "membership_count"])
def test_native_ranking_factories_reject_incoherent_event_support(family, bad):
    patterns = np.array([[0, 1], [1, 0]])
    membership = None
    if bad == "patterns":
        patterns = patterns[:1]
    elif bad == "membership_type":
        membership = np.ones((1, 2))
    else:
        membership = np.ones((1, 3), dtype=bool)
    factory = mgppm.MGPPMDistribution if family == "mgppm" else sequential.TwoPLRankDistribution
    support = {"scores" if family == "mgppm" else "orders": patterns}
    with pytest.raises(ValueError, match="patterns|counts|Boolean|pattern count"):
        factory.dist(np.zeros(2), np.ones(2), np.zeros(2), membership=membership, **support)


def test_native_sequential_requires_integer_orders_and_known_tensor_shapes():
    with pytest.raises(TypeError, match="integer"):
        sequential.TwoPLRankDistribution.dist(
            np.zeros(2), np.ones(2), np.zeros(2), orders=np.array([[0.0, 1.0], [1.0, 0.0]])
        )
    with pytest.raises(ValueError, match="shape"):
        sequential.TwoPLRankDistribution.dist(
            pt.matrix("unknown_shape"), np.ones(2), np.zeros(2), orders=np.array([[0, 1], [1, 0]])
        )


@pytest.mark.parametrize("family", ["tirt", "gtum"])
@pytest.mark.parametrize(
    "bad", ["link", "beta_precision", "category_precision", "cutpoints", "beta_effect"]
)
def test_comparison_factories_enforce_link_specific_parameter_contracts(family, bad):
    kwargs = {"cutpoints": np.array([0.0])}
    if family == "tirt":
        factory = tirt.TIRTDistribution
        args = (np.zeros(2), np.ones(2))
        kwargs.update(statement_effect=np.zeros(2), residual_variance=np.ones(2))
    else:
        factory = gtum.GTUMDistribution
        args = (np.zeros(2), np.ones(2), np.array([-1.0, 1.0]))
    if bad == "link":
        kwargs["link"] = "unsupported"
    elif bad in {"beta_precision", "beta_effect"}:
        kwargs["link"] = "beta_probit" if family == "tirt" else "beta_logit"
        if family == "tirt":
            kwargs.pop("statement_effect")
            kwargs.pop("residual_variance")
        if bad == "beta_effect":
            kwargs["precision"] = 2.0
            kwargs["statement_effect" if family == "tirt" else "person_block_effect"] = 1.0
    elif bad == "category_precision":
        kwargs["precision"] = 2.0
    else:
        kwargs["cutpoints"] = np.array([])
    with pytest.raises((ValueError, TypeError)):
        factory.dist(*args, **kwargs)


@pytest.mark.parametrize("family", ["gpcm", "ggum"])
def test_single_item_factories_require_complete_category_axes(family):
    with pytest.raises(ValueError, match="category|threshold"):
        if family == "gpcm":
            likert.GPCMDistribution.dist(np.zeros(2), np.ones(2), np.zeros(1))
        else:
            likert.GGUMDistribution.dist(np.zeros(2), np.ones(2), np.ones(2), np.zeros(1))


@pytest.mark.parametrize("family", ["tirt", "gtum"])
@pytest.mark.parametrize("precision", [0.0, -2.0])
def test_comparison_beta_rejects_invalid_precision_in_direct_evaluation(family, precision):
    if family == "tirt":
        response = tirt.TIRTDistribution.dist(
            [0.0, 0.0], [1.0, 1.0], 0.0, link="beta_probit", precision=precision
        )
    else:
        response = gtum.GTUMDistribution.dist(
            [0.0, 0.0], [1.0, 1.0], [-1.0, 1.0], 0.0, link="beta_logit", precision=precision
        )
    with pytest.raises(ValueError, match="precision > 0"):
        response.eval()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"direction": 0.0}, "direction"),
        ({"cutpoints": [0.2, 0.2]}, "cutpoints must be strictly increasing"),
        ({"residual_variance": [-0.25, 1.0]}, "residual_variance > 0"),
    ],
)
def test_tirt_public_distribution_retains_model_constraints(options, message):
    parameters = {
        "cutpoints": [0.0],
        "statement_effect": [0.0, 0.0],
        "residual_variance": [1.0, 1.0],
    }
    parameters.update(options)
    response = tirt.TIRTDistribution.dist([0.0, 0.0], [1.0, 1.0], **parameters)
    with pytest.raises(ValueError, match=message):
        pm.logp(response, 0).eval()


def test_gtum_beta_rejects_symbolic_block_effect():
    effect = pt.scalar("effect")
    response = gtum.GTUMDistribution.dist(
        [0.0, 0.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        0.0,
        link="beta_logit",
        person_block_effect=effect,
        precision=10.0,
    )
    logp = pytensor.function([effect], pm.logp(response, 0.5))
    assert np.isfinite(logp(0.0))
    with pytest.raises(ValueError, match="person_block_effect"):
        logp(0.1)


@pytest.mark.parametrize("fixed", [False, True])
def test_native_four_pl_builds_ordered_asymptotes_and_fixed_scoring_target(fixed):
    import pandas as pd

    import pyfcirt as fc
    from tests.helpers import labelled_point

    answers = fc.ResponseTable(
        pd.DataFrame({"person_id": [1, 2], "item_id": ["a", "a"], "response": [0, 1]}),
        format="item_long",
    )
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            fc.ItemInput(
                fc.ItemDesignMapping(traits={"A": ["a"]}),
                answers,
                model=fc.DichotomousIRT(kind="4PL"),
            )
        )
    )
    if fixed:
        point = labelled_point(
            model,
            {
                "population_mean": [[0.0]],
                "covariance": [[1.0]],
                "loading": [1.0],
                "item_intercept": [0.0],
                "lower_asymptote": [0.2],
                "upper_asymptote": [0.8],
            },
        )
        model = fc.score_model(model, answers, parameters=point)
    graph = model.build()
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=91)))
    blocks = {block.kind: block for block in model.prepared.blocks}
    upper_rv = graph.named_vars.get(f"_parameter_{blocks['upper_asymptote'].binding_id}_raw")
    assert (upper_rv in graph.free_RVs) is not fixed
    if not fixed:
        logp, gradient = graph.compile_logp(mode="JAX"), graph.compile_dlogp(mode="JAX")
        for lower, upper in ((0.6, 0.9), (0.05, 0.3)):
            point = graph.initial_point(random_seed=91)
            lower_rv = graph[f"_parameter_{blocks['lower_asymptote'].binding_id}_raw"]
            upper_rv = graph[f"_parameter_{blocks['upper_asymptote'].binding_id}_raw"]
            point[graph.rvs_to_values[lower_rv].name] = np.array([np.log(lower / (1 - lower))])
            point[graph.rvs_to_values[upper_rv].name] = np.array(
                [np.log((upper - lower) / (1 - upper))]
            )
            assert np.isfinite(logp(point))
            assert np.isfinite(gradient(point)).all()


CATEGORICAL = (
    "mgppm",
    "two_pl",
    "ggum_rank",
    "tirt_probit",
    "tirt_logit",
    "gtum",
    "normal_rank",
    "1pl",
    "2pl",
    "3pl",
    "4pl",
    "gpcm",
    "ggum",
)
BETA = ("tirt_beta_probit", "tirt_beta_logit", "gtum_beta")
GENERATION_CASES = ("gpcm", "tirt_beta_probit", "normal_rank")


def case(name, *, size=None, batch=False, ability=None, symbolic=None, observed=None):
    """Return a public dist and its independent probability/density reference.

    Batch rows deliberately differ, so transposed or silently repeated rows fail.
    Only scalar equations live here; the production JAX kernels are never references.
    """

    def construct(factory, *args, **kwargs):
        if observed is None:
            return factory.dist(*args, **kwargs)
        return factory("response", *args, observed=observed, dims="person", **kwargs)

    theta = np.array([[0.4, -0.2], [-0.5, 0.8]]) if batch else np.array([0.4, -0.2])
    loading, intercept = np.array([1.2, -0.7]), np.array([0.1, -0.3])
    locations, cuts = np.array([-0.6, 1.1]), np.array([-0.5, 0.8])
    if ability is not None:
        theta = np.array(ability)
    rows = theta if batch else theta[None, :]
    if symbolic is not None:
        theta = symbolic
    expected = []
    kwargs = {"size": size}
    if name == "mgppm":
        scores = np.array([[0, 2], [1, 1], [2, 0]])
        for row in rows:
            logits = [sum((loading * row - intercept) * score) for score in scores]
            expected.append(np.exp(logits - logsumexp(logits)))
        rv = construct(mgppm.MGPPMDistribution, theta, loading, intercept, scores=scores, **kwargs)
    elif name in {"two_pl", "ggum_rank"}:
        orders, _ = order_arrays(2)
        for row in rows:
            utility = (
                loading * row - intercept
                if name == "two_pl"
                else scalar_odds(row, abs(loading), locations, [-0.8, 0.3])
            )
            expected.append(scalar_probabilities(utility))
        if name == "two_pl":
            rv = construct(
                sequential.TwoPLRankDistribution, theta, loading, intercept, orders=orders, **kwargs
            )
        else:
            rv = construct(
                sequential.GGUMRankDistribution,
                theta,
                abs(loading),
                locations,
                [-0.8, 0.3],
                orders=orders,
                **kwargs,
            )
    elif name.startswith("tirt") or name.startswith("gtum"):
        is_beta = "beta" in name
        link = "probit" if "probit" in name else "logit"
        for row in rows:
            if name.startswith("tirt"):
                eta = sum(loading * row * [1, -1])
            else:
                utility = -(loading if is_beta else abs(loading)) * abs(row - locations)
                eta = utility[0] - utility[1]
            if is_beta:
                mean = norm.cdf(-(eta - 0.25)) if link == "probit" else expit(-(eta - 0.25))
                expected.append([mean * 7, (1 - mean) * 7])
            elif name.startswith("tirt"):
                eta = -(eta + 0.3 - (-0.1)) / np.sqrt(1.6)
                expected.append(np.exp(tirt_log_probs(eta, cuts / np.sqrt(1.6), link)))
            else:
                expected.append(np.exp(gtum_log_probs(-(eta + 0.35), cuts)))
        if name.startswith("tirt"):
            extra = (
                {"precision": 7}
                if is_beta
                else {"statement_effect": [0.3, -0.1], "residual_variance": [0.6, 1.0]}
            )
            rv = construct(
                tirt.TIRTDistribution,
                theta,
                loading,
                0.25 if is_beta else cuts,
                link=("beta_" if is_beta else "") + link,
                direction=-1,
                **extra,
                **kwargs,
            )
        else:
            extra = {"precision": 7} if is_beta else {"person_block_effect": 0.35}
            rv = construct(
                gtum.GTUMDistribution,
                theta,
                loading if is_beta else abs(loading),
                locations,
                0.25 if is_beta else cuts,
                link="beta_logit" if is_beta else "logit",
                direction=-1,
                **extra,
                **kwargs,
            )
    elif name == "normal_rank":
        orders, contrasts = order_arrays(2)
        for row in rows:
            mean = loading * row + intercept
            p = norm.cdf((mean[0] - mean[1]) / np.sqrt(1.6))
            expected.append([p, 1 - p])
        rv = construct(
            rank.TIRTRankDistribution,
            theta[..., None],
            loading[:, None],
            intercept,
            [0.6, 1.0],
            orders=orders,
            contrasts=contrasts,
            n_points=128,
            **kwargs,
        )
    elif name in {"1pl", "2pl", "3pl", "4pl"}:
        slopes = np.ones(2) if name == "1pl" else loading
        lower, upper = (0.15 if name in {"3pl", "4pl"} else 0), (0.9 if name == "4pl" else 1)
        for row in rows:
            p = lower + (upper - lower) * expit(sum(slopes * row) - 0.3)
            expected.append([1 - p, p])
        rv = construct(
            likert.DichotomousIRTDistribution,
            theta,
            slopes,
            0.3,
            lower=lower,
            upper=upper,
            **kwargs,
        )
    elif name == "gpcm":
        offsets = np.array([0, 0.7, -0.5])
        for row in rows:
            logits = np.array([k * sum(row * loading) + offset for k, offset in enumerate(offsets)])
            expected.append(np.exp(logits - logsumexp(logits)))
        rv = construct(likert.GPCMDistribution, theta, loading, offsets, **kwargs)
    elif name == "ggum":
        # A unidimensional subjective-state reference gives a distinct derivation.
        for row in rows:
            expected.append(np.exp(subjective_log_probs(row[0], 1.2, -0.6, [0, -1.1, -0.4])))
        rv = construct(
            likert.GGUMDistribution, theta[..., :1], [1.2], [-0.6], [0, -1.1, -0.4], **kwargs
        )
    else:
        raise AssertionError(name)
    expected = np.array(expected)
    return rv, expected if batch else expected[0]


@pytest.mark.parametrize("name", CATEGORICAL)
@pytest.mark.parametrize("batch", [False, True])
def test_native_categorical_density_and_support(name, batch):
    rv, probabilities = case(name, batch=batch)
    category_count = probabilities.shape[-1]
    values = np.arange(category_count)[:, None] if batch else np.arange(category_count)
    actual = pm.logp(rv, values).eval()
    assert_allclose(actual, np.log(probabilities).T, atol=2e-7, rtol=2e-7)
    assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-12)
    invalid = np.array([-1, category_count])
    if batch:
        invalid = invalid[:, None]
    assert np.isneginf(pm.logp(rv, invalid).eval()).all()


@pytest.mark.parametrize("name", BETA)
@pytest.mark.parametrize("batch", [False, True])
def test_native_beta_density_and_support(name, batch):
    rv, shapes = case(name, batch=batch)
    values = np.array([0.13, 0.55, 0.91])
    if batch:
        values = values[:, None]
    expected = beta.logpdf(values, shapes[..., 0], shapes[..., 1])
    assert_allclose(pm.logp(rv, values).eval(), expected, atol=2e-7, rtol=2e-7)
    invalid = np.array([-0.1, 1.1])[:, None] if batch else np.array([-0.1, 1.1])
    assert np.isneginf(pm.logp(rv, invalid).eval()).all()


@pytest.mark.parametrize("name", GENERATION_CASES)
@pytest.mark.parametrize(
    "size,batch,shape", [(None, True, (2,)), (3, False, (3,)), ((2, 2), True, (2, 2))]
)
def test_native_random_size_batch_and_support(name, size, batch, shape):
    rv, expected = case(name, size=size, batch=batch)
    draws = pm.draw(rv, draws=2, random_seed=812, mode="FAST_COMPILE")
    assert draws.shape == (2, *shape)
    if name in BETA:
        assert np.all((draws > 0) & (draws < 1))
    else:
        assert np.issubdtype(draws.dtype, np.integer)
        assert np.all((draws >= 0) & (draws < expected.shape[-1]))


@pytest.mark.parametrize("name", ["mgppm", "normal_rank", "tirt_beta_probit"])
def test_native_mixture_and_observed_registration(name):
    component, reference = case(name)
    other = (
        pm.Beta.dist(2, 3)
        if name in BETA
        else pm.Categorical.dist(p=np.ones(len(reference)) / len(reference))
    )
    values = np.array([0.2, 0.7]) if name in BETA else np.array([0, 1])
    density = beta.pdf(values, *reference) if name in BETA else reference[values]
    other_density = beta.pdf(values, 2, 3) if name in BETA else np.ones(2) / len(reference)
    with pm.Model() as model:
        pm.Mixture("response", w=[0.3, 0.7], comp_dists=[component, other], observed=values)
    assert len(model.observed_RVs) == 1
    assert_allclose(
        model.compile_logp(mode="FAST_COMPILE")({}),
        np.log(0.3 * density + 0.7 * other_density).sum(),
        atol=2e-7,
        rtol=2e-7,
    )
    draws = pm.draw(model["response"], draws=2, random_seed=91, mode="FAST_COMPILE")
    assert draws.shape == (2, 2)


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("event", ["rank", "pick", "mole"])
def test_successive_choice_and_mgppm_event_aggregation(size, event):
    orders, _ = order_arrays(size)
    theta, loading = np.linspace(-0.6, 0.7, size), np.linspace(0.5, 1.3, size)
    intercept = np.linspace(-0.2, 0.4, size)
    utility = loading * theta - intercept
    if event == "rank":
        membership = np.eye(factorial(size), dtype=bool)
    elif event == "pick":
        membership = np.array([orders[:, 0] == item for item in range(size)])
    else:
        membership = np.array(
            [
                (orders[:, 0] == best) & (orders[:, -1] == worst)
                for best in range(size)
                for worst in range(size)
                if best != worst
            ]
        )
    complete = scalar_probabilities(utility)
    expected = np.array(
        [sum(p for p, keep in zip(complete, row, strict=True) if keep) for row in membership]
    )
    logs = sequential.sequential_log_probs(utility, orders, membership)
    assert_allclose(np.exp(logs), expected, atol=1e-12, rtol=1e-12)
    scores = np.argsort(orders, axis=-1)
    logits = np.array([sum(utility * score) for score in scores])
    complete = np.exp(logits - logsumexp(logits))
    expected = np.array(
        [sum(p for p, keep in zip(complete, row, strict=True) if keep) for row in membership]
    )
    assert_allclose(
        np.exp(
            mgppm.MGPPMDistribution.kernel(
                theta, loading, intercept, scores=scores, membership=membership
            )
        ),
        expected,
        atol=1e-12,
    )


@pytest.mark.parametrize("distance", [0.0, 0.003, 0.2, 4.0])
def test_distance_ggum_value_and_derivatives_at_ideal_point(distance):
    def logs(theta):
        return likert.GGUMDistribution.kernel(
            theta, jnp.array([1.2]), jnp.array([0.4]), jnp.array([0, -1.1, -0.4])
        )

    theta = np.array([0.4 + distance])
    assert_allclose(
        logs(theta), subjective_log_probs(theta[0], 1.2, 0.4, [0, -1.1, -0.4]), atol=1e-12
    )
    assert np.isfinite(jax.jacfwd(logs)(theta)).all()
    assert np.isfinite(jax.jacfwd(jax.jacrev(logs))(theta)).all()
    if distance == 0:
        assert_allclose(jax.jacfwd(logs)(theta), 0, atol=1e-12)


@pytest.mark.parametrize("name", CATEGORICAL + BETA)
def test_native_theta_gradient_matches_independent_density_difference(name):
    theta = pt.vector("theta", shape=(2,))
    point = np.array([0.4, -0.2])
    rv, _ = case(name, symbolic=theta)
    value = 0.55 if name in BETA else 1
    target = pm.logp(rv, value)
    evaluate = pytensor.function([theta], pt.grad(target, theta), mode="FAST_COMPILE")

    def reference(x):
        _, expected = case(name, ability=x)
        return beta.logpdf(value, *expected) if name in BETA else np.log(expected[value])

    finite = [
        (reference(point + delta) - reference(point - delta)) / 2e-5 for delta in np.eye(2) * 1e-5
    ]
    assert_allclose(evaluate(point), finite, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("link", ["probit", "logit"])
@pytest.mark.parametrize("eta", [-1000.0, -30.0, 30.0, 1000.0])
def test_tirt_extreme_graded_intervals_and_gradients(link, eta):
    cuts = np.array([-0.8, 0.2, 1.1])

    def logs(theta):
        return tirt.TIRTDistribution.kernel(
            theta, jnp.ones(2), jnp.array(cuts), jnp.zeros(2), jnp.array([0.5, 0.5]), 1.0, link=link
        )

    point = jnp.array([eta, 0.0])
    expected = tirt_log_probs(eta, cuts, link)
    assert_allclose(logs(point), expected, atol=3e-9, rtol=2e-13)
    derivative = jax.jacfwd(logs)(point)
    finite = (
        tirt_log_probs(eta + 1e-4, cuts, link) - tirt_log_probs(eta - 1e-4, cuts, link)
    ) / 2e-4
    assert_allclose(derivative[:, 0], finite, atol=1e-5, rtol=3e-7)
    assert_allclose(derivative[:, 0], -derivative[:, 1], atol=1e-12)


def declaration_input(declaration, kind):
    """Build a high-level declaration and the native responses for its scale."""
    import pandas as pd

    import pyfcirt as fc

    name, _, link = declaration.partition("-")
    options = (
        {"link": link, "priors": {"response_precision": pm.Gamma.dist(alpha=4.0, beta=0.5)}}
        if link
        else {}
    )
    if declaration == "TIRT":
        options["statement_effect_parameterization"] = "centered"
    if declaration == "GTUM":
        options["block_effect"] = True
    spec = getattr(fc, name)(**options)
    if "item" in kind:
        design = pd.DataFrame(
            {"item_id": ["a", "b", "c"], "trait": ["A", "B", "C"], "location_side": [1, 1, 1]}
        )
        answers = pd.DataFrame(
            {
                "person_id": [1] * 3 + [2] * 3,
                "item_id": ["a", "b", "c"] * 2,
                "response": [0, 1, 0, 1, 0, 1] if kind == "binary-item" else [0, 3, 6, 2, 4, 1],
            }
        )
        data = fc.ItemInput(
            design,
            fc.ResponseTable(answers, format="item_long"),
            model=spec,
            categories=(0, 1) if kind == "binary-item" else tuple(range(7)),
        )
    else:
        design = pd.DataFrame(
            {
                "block_id": ["block"] * 3,
                "position": [1, 2, 3],
                "statement_id": ["a", "b", "c"],
                "trait": ["A", "B", "C"],
                "location_side": [1, 1, 1],
                "keying": [1, 1, 1],
            }
        )
        if kind == "rank":
            answers = pd.DataFrame(
                {"person_id": [1, 2], "block_id": ["block"] * 2, "response": [[1, 2, 3], [2, 3, 1]]}
            )
            data = fc.FCInput(design, fc.ResponseTable(answers, format="block_long"), model=spec)
        else:
            answers = pd.DataFrame(
                {
                    "person_id": [1] * 3 + [2] * 3,
                    "block_id": ["block"] * 6,
                    "statement_1": ["a", "a", "b"] * 2,
                    "statement_2": ["b", "c", "c"] * 2,
                    "response": [0.2, 0.8, 0.4, 0.7, 0.3, 0.6]
                    if kind == "beta"
                    else [0, 3, 6, 2, 4, 1],
                }
            )
            data = fc.FCInput(
                design,
                fc.ResponseTable(answers, format="comparison_long"),
                model=spec,
                response_format=None,
                scale=fc.UnitIntervalScale()
                if kind == "beta"
                else fc.CategoryScale(tuple(range(7))),
            )
    return data


@pytest.mark.parametrize(
    "declaration,kind",
    [
        pytest.param("MGPPM", "rank", id="mgppm"),
        pytest.param("TwoPLRank", "rank", id="2pl-rank"),
        pytest.param("GGUMRank", "rank", id="ggum-rank"),
        pytest.param("TIRTRank", "rank", id="gaussian-rank"),
        pytest.param("TIRT", "graded", id="tirt-graded"),
        pytest.param("GTUM", "graded", id="gtum-graded-effect"),
        pytest.param("DichotomousIRT", "binary-item", id="binary-item"),
        pytest.param("GPCM", "ordinal-item", id="gpcm-seven-categories"),
        pytest.param("GGUM", "ordinal-item", id="ggum-seven-categories"),
        pytest.param("TIRT-beta_probit", "beta", id="tirt-beta-probit"),
        pytest.param("TIRT-beta_logit", "beta", id="tirt-beta-logit"),
        pytest.param("GTUM-beta_logit", "beta", id="gtum-beta-logit"),
    ],
)
def test_high_level_declaration_native_target_and_real_prior_generation(
    declaration, kind, monkeypatch
):
    import pyfcirt as fc

    name = declaration.split("-", maxsplit=1)[0]
    if name == "TIRTRank":
        from pyfcirt.models.thurstonian.normal import qmc

        def unexpected_nodes(*args, **kwargs):
            pytest.fail("Rank fitting or response generation allocated marginal Sobol nodes")

        monkeypatch.setattr(qmc, "Sobol", unexpected_nodes)
    data = declaration_input(declaration, kind)
    model = fc.Model(fc.PreparedInputs.from_inputs(data))
    graph = model.build()
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=29)))
    initializations = dict(graph.rvs_to_initial_values)
    generated = fc.prior_predictive(model, draws=2, random_seed=31, store=False)
    assert dict(graph.rvs_to_initial_values) == initializations
    point = generated["prior"].to_dataset().isel(chain=0, draw=0, drop=True)
    assert "theta" in point and not any(name.endswith("_raw") for name in point)
    assert all(np.isfinite(value.values).all() for value in point.data_vars.values())
    for block in model.prepared.blocks:
        values = point[block.kind].values.ravel()[block.entry_index]
        if block.ordered:
            assert (np.diff(values, axis=-1) > 0).all()
        if declaration == "GGUM" and block.key == "threshold":
            assert (values[..., 0] == 0).all()
    for response in model._bindings.responses:
        values = generated["prior_predictive"][response.batch.response_name].values
        if kind == "beta":
            # Native Beta sampling can round extreme-tail draws to an endpoint.
            # The package neither clips those draws nor changes its prior law.
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        else:
            assert np.issubdtype(values.dtype, np.integer)
            category_count = model.prepared.data["constant_data"].sizes[response.batch.category_dim]
            assert ((values >= 0) & (values < category_count)).all()
    if declaration in {"TwoPLRank", "TIRT-beta_logit", "GGUM"}:
        fixed = fc.score_model(model, data.responses, parameters=point)
        assert (
            fixed.build() is not graph
            and fixed._bindings.parameters["theta"] is not model._bindings.parameters["theta"]
        )
        assert "posterior" not in fixed.idata
        for layout in fixed.prepared.parameters:
            if layout.fixed_for_scoring:
                assert fixed.build()[layout.kind] is not graph[layout.kind]
        assert np.isfinite(
            fixed.build().compile_logp(mode="JAX")(fixed.build().initial_point(random_seed=32))
        )


@pytest.mark.parametrize(
    ("declaration", "kind", "targets"),
    [
        ("TIRT", "graded", ("cutpoints", "residual_variance")),
        ("GTUM", "graded", ("location",)),
        ("TIRTRank", "rank", ("intercept", "residual_variance")),
        ("TwoPLRank", "rank", ("intercept",)),
        ("GGUMRank", "rank", ("location", "threshold")),
        ("MGPPM", "rank", ("intercept",)),
        ("DichotomousIRT", "binary-item", ("loading_magnitude", "intercept")),
        ("GTUM-beta_logit", "beta", ("response_precision",)),
    ],
)
def test_dif_targets_feed_each_native_measurement_law(declaration, kind, targets):
    import pandas as pd

    import pyfcirt as fc
    from pyfcirt.data.regressions import prepare_regressions

    pytest.importorskip("formulae")
    source = declaration_input(declaration, kind)
    model = fc.Model(source)
    people = pd.DataFrame({"person_id": [1, 2], "site": ["A", "B"]})
    channel = "items" if isinstance(source, fc.ItemInput) else "fc"
    regression = fc.Regression("site", people, id="person_id", parameter_residual_sd=0.3)
    model._update_parameters(
        prepare_regressions,
        [((channel, target), regression) for target in targets],
        person_effects=True,
    )
    prepared = model.prepared
    for slot in (slot for batch in prepared.batches for slot in batch.slots if slot.key in targets):
        assert slot.effects
        layout = prepared.parameter_layouts[slot.kind]
        affected = slot.entry_index.ravel()[slot.effects[0].position]
        assert np.isnan(layout.fixed.ravel()[affected]).all()
        if slot.kind in {"location", "residual_variance", "response_precision", "loading"}:
            assert slot.effects[0].link == "log"
    native = model.build()
    assert np.isfinite(native.compile_logp(mode="JAX")(native.initial_point(random_seed=29)))
    prior = fc.prior_predictive(model, draws=2, random_seed=31, store=False)
    for response in model._bindings.responses:
        simulated = prior.prior_predictive[response.batch.response_name].values
        assert np.isfinite(simulated).all()

    if "response_precision" in targets or declaration == "DichotomousIRT":
        point = prior.prior.dataset.isel(chain=0, draw=0, drop=True)
        fixed = fc.score_model(model, source.responses, parameters=point, person_data=people)
        assert {rv.name for rv in fixed.build().free_RVs} == {"theta"}
        assert np.isfinite(fixed.build().compile_logp(mode="JAX")(fixed.build().initial_point()))


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("all_fixed", [False, True])
def test_canonical_parameter_scatter_preserves_precision_and_fixed_entries(precision, all_fixed):
    import pandas as pd
    from pytensor.graph.traversal import ancestors

    import pyfcirt as fc

    with pytensor.config.change_flags(floatX=precision):
        common = fc.DichotomousIRT(kind="1PL")
        source = fc.ItemInput(
            pd.DataFrame(
                {
                    "item_id": ["a", "b", "c"],
                    "trait": ["T"] * 3,
                    "fix_loading": [1.25, np.nan, 0.75 if all_fixed else np.nan],
                }
            ),
            fc.ResponseTable(
                pd.DataFrame(
                    {"person_id": [1] * 3, "item_id": ["a", "b", "c"], "response": [0, 1, 0]}
                ),
                format="item_long",
            ),
            items=(fc.ItemOverride("a", model=common), fc.ItemOverride("b", model=common)),
        )
        model = fc.Model(fc.PreparedInputs.from_inputs(source))
        graph = model.build()
        binding = model._bindings.parameters["loading"]
        blocks = [block for block in model.prepared.blocks if block.kind == "loading"]
        assert len(blocks) == 2 and [block.shared for block in blocks] == [True, False]
        assert binding.type.shape == (2,)
        assert all(
            parameter.dtype == precision for parameter in model._bindings.parameters.values()
        )
        # Supply native free draws explicitly; evaluating known entries needs no RNG or JAX mode.
        dependencies = set(ancestors([binding]))
        inputs = [rv for rv in graph.free_RVs if rv in dependencies]
        assert len(inputs) == (0 if all_fixed else 1)
        evaluate = pytensor.function(inputs, binding, mode="FAST_COMPILE")
        actual = evaluate(*[np.full(rv.type.shape, 2.25, dtype=precision) for rv in inputs])
        assert actual.dtype == np.dtype(precision)
        assert_allclose(actual, np.asarray([1.25, 0.75 if all_fixed else 2.25], dtype=precision))
        layout = next(layout for layout in model.prepared.parameters if layout.kind == "loading")
        assert_allclose(actual[layout.fixed_flat], layout.fixed_value.astype(precision))


def test_ordered_prior_native_resize_keeps_normalized_density_and_support():
    from pymc.distributions.shape_utils import change_dist_size
    from scipy.special import gammaln

    from pyfcirt.models.graph import ParameterFactory

    with pm.Model():
        ordered = ParameterFactory.ordered(
            "cuts", pm.StudentT.dist(nu=5, mu=0.2, sigma=0.8), rows=2, count=6
        )
    expanded = change_dist_size(ordered, new_size=(4, 2), expand=False)
    draws = pm.draw(expanded, draws=3, random_seed=45, mode="FAST_COMPILE")
    assert draws.shape == (3, 4, 2, 6) and (np.diff(draws, axis=-1) > 0).all()
    native = pm.logp(expanded, draws[0])
    actual = pytensor.function([], native, mode="JAX")()
    # The prior law belongs to native PyMC; independently verify the ordering
    # normalizer without asserting upstream StudentT constants equal SciPy.
    base = pm.logp(pm.StudentT.dist(nu=5, mu=0.2, sigma=0.8), draws[0])
    expected = pytensor.function([], base, mode="JAX")().sum(axis=-1) + gammaln(7)
    assert_allclose(actual, expected, atol=2e-13, rtol=0)
    invalid = pm.logp(expanded, draws[0, ..., ::-1])
    assert np.isneginf(pytensor.function([], invalid, mode="JAX")()).all()


@pytest.mark.parametrize("count", [1, 3])
def test_ordered_conditional_density_and_gradient_include_iid_normalizer(count):
    from scipy.special import gammaln

    from pyfcirt.models.graph import ParameterFactory

    mean = pt.tensor("mean", dtype="float64", shape=(2,))
    scale = pt.tensor("scale", dtype="float64", shape=(2,))
    value = pt.dmatrix("value")
    with pm.Model():
        cuts = ParameterFactory.ordered(
            "cuts",
            pm.Normal.dist(mu=mean, sigma=scale),
            rows=2,
            count=count,
        )
    logp = pm.logp(cuts, value).reshape((2,))
    evaluate = pytensor.function(
        [value, mean, scale], [logp, *pt.grad(logp.sum(), [value, mean, scale])], mode="JAX"
    )
    means, scales = np.array([-0.5, 1.2]), np.array([0.6, 1.3])
    point = means[:, None] + scales[:, None] * np.linspace(-0.7, 0.9, count)
    residual = point - means[:, None]
    expected = (
        norm.logpdf(point, means[:, None], scales[:, None]).sum(axis=-1) + gammaln(count + 1),
        -residual / scales[:, None] ** 2,
        (residual / scales[:, None] ** 2).sum(axis=1),
        (residual**2 / scales[:, None] ** 3 - 1 / scales[:, None]).sum(axis=1),
    )
    for actual, reference in zip(evaluate(point, means, scales), expected, strict=True):
        assert_allclose(actual, reference, atol=1e-12, rtol=1e-10)
    fixed = graph_replace(
        cuts, replace={mean: pt.as_tensor_variable(means), scale: pt.as_tensor_variable(scales)}
    )
    draws = pm.draw(fixed, draws=3, random_seed=63, mode="JAX")
    assert draws.shape == (3, 2, count)
    assert np.isfinite(draws).all()
    assert (np.diff(draws, axis=-1) > 0).all()
    if count > 1:
        assert np.isneginf(evaluate(point[:, ::-1], means, scales)[0]).all()
        ties = point.copy()
        ties[:, 1] = ties[:, 0]
        assert np.isneginf(evaluate(ties, means, scales)[0]).all()
        with pytest.raises((AssertionError, ValueError), match="[Ss]hape"):
            evaluate(point[:, :-1], means, scales)


@pytest.mark.parametrize("upper_only", [False, True])
def test_ordered_native_bounds_align_with_cutpoint_rows(upper_only):
    from pyfcirt.models.graph import ParameterFactory

    prior = (
        pm.TruncatedNormal.dist(upper=[-1.0, 3.0])
        if upper_only
        else pm.Uniform.dist(lower=[-6.0, 0.0], upper=[-1.0, 3.0])
    )
    point = np.array([[-4.3, -2.2, -1.7], [0.8, 1.9, 2.2]])
    with pm.Model() as model:
        cuts = ParameterFactory.ordered("cuts", prior, rows=2, count=3)
    transform = model.rvs_to_transforms[cuts]
    raw = transform.forward(pt.as_tensor_variable(point), *cuts.owner.inputs)
    recovered = transform.backward(raw, *cuts.owner.inputs)
    jacobian = transform.log_jac_det(raw, *cuts.owner.inputs)
    values, logdet = pytensor.function([], [recovered, jacobian], mode="JAX")()
    assert_allclose(values, point, rtol=0, atol=1e-12)
    assert np.isfinite(logdet).all()


@pytest.mark.parametrize("side", ["positive", "negative", "free"])
def test_location_conditionals_include_truncation_normalizer(side):
    from types import SimpleNamespace

    from scipy.special import log_ndtr

    from pyfcirt.covariates.model import RegressionBinding
    from pyfcirt.models.graph import ParameterFactory

    mean = np.array([-0.8, 0.6])
    scale = np.array([0.7, 1.1])
    bounds = {"positive": [0, None], "negative": [None, 0], "free": None}[side]
    family = "Normal" if side == "free" else "TruncatedNormal"
    binding = RegressionBinding(
        mean=pt.as_tensor_variable(mean),
        scale=pt.as_tensor_variable(scale),
        conditional={"family": family, "bounds": bounds},
        noncentered=False,
    )
    layout = SimpleNamespace(binding_id=0, ordered=False)
    with pm.Model():
        location = ParameterFactory.regression(
            layout, role=f"location_{side}", size=2, binding=binding
        )
    x = pt.dvector("x")
    logp = pm.logp(location, x).sum()
    evaluate = pytensor.function([x], [logp, pt.grad(logp, x)], mode="FAST_COMPILE")
    point = np.array([0.2, 1.0]) * (-1 if side == "negative" else 1)
    expected = norm.logpdf(point, mean, scale)
    if side != "free":
        expected -= log_ndtr((mean if side == "positive" else -mean) / scale)
    actual, gradient = evaluate(point)
    native = (
        pm.Normal.dist(mu=mean, sigma=scale)
        if side == "free"
        else pm.TruncatedNormal.dist(mu=mean, sigma=scale, lower=bounds[0], upper=bounds[1])
    )
    native_density = pytensor.function([], pm.logp(native, point).sum(), mode="FAST_COMPILE")()
    assert_allclose(actual, native_density, rtol=0, atol=1e-12)
    # PyMC's native log-CDF folds sqrt(2.0) as float32; its independent SciPy
    # comparison therefore uses that constant's precision, not a different law.
    assert_allclose(actual, expected.sum(), rtol=0, atol=np.finfo(np.float32).eps)
    assert_allclose(gradient, -(point - mean) / scale**2, atol=1e-12)
    if side != "free":
        assert np.isneginf(evaluate(-point)[0])


@pytest.mark.parametrize("batched", [False, True])
def test_centered_affine_student_t_preserves_density_jacobian_and_gradient(batched):
    from scipy.stats import t

    from pyfcirt.models.graph import ParameterFactory

    nu, mu, sigma = 5.0, 0.3, 0.8
    scale = np.array([0.5, 1.7]) if batched else np.array(1.7)
    offset = np.array([-0.2, 1.0]) if batched else np.array(-0.2)
    with pm.Model():
        rv = ParameterFactory.scaled(
            "affine",
            pm.StudentT.dist(nu=nu, mu=mu, sigma=sigma),
            pt.as_tensor_variable(scale),
            offset=pt.as_tensor_variable(offset),
            shape=(2,),
            noncentered=False,
        )
    x = pt.dvector("x")
    logp = pm.logp(rv, x).sum()
    evaluate = pytensor.function([x], [logp, pt.grad(logp, x)], mode="FAST_COMPILE")
    point = np.array([0.1, 0.6])
    z = (point - offset) / scale
    expected = (t.logpdf(z, nu, loc=mu, scale=sigma) - np.log(scale)).sum()
    expected_gradient = -(nu + 1) * (z - mu) / (nu * sigma**2 + (z - mu) ** 2) / scale
    actual, gradient = evaluate(point)
    assert_allclose(actual, expected, atol=1e-12)
    assert_allclose(gradient, expected_gradient, atol=1e-12)


def test_centered_affine_bounds_follow_intervened_scale_and_offset():
    from pyfcirt.models.graph import ParameterFactory

    with pm.Model() as model:
        scale = pm.LogNormal("scale", shape=2)
        offset = pm.Normal("offset", shape=2)
        ParameterFactory.scaled(
            "x",
            pm.Uniform.dist(lower=[-1.0, 2.0], upper=[3.0, 5.0]),
            scale,
            offset=offset,
            shape=(2,),
            noncentered=False,
        )
    fixed = pm.do(model, {"scale": [0.5, 3.0], "offset": [1.0, -2.0]})
    node = fixed["x"]
    transformed = fixed.rvs_to_transforms[node].backward(pt.zeros(2), *node.owner.inputs)
    assert_allclose(pytensor.function([], transformed, mode="JAX")(), [1.5, 8.5], atol=1e-12)


@pytest.mark.parametrize("name", GENERATION_CASES)
def test_named_observations_preserve_native_registration_and_coordinates(name):
    observed = np.array([0.2, 0.7]) if name in BETA else np.array([0, 1])
    with pm.Model(coords={"person": ["p", "q"]}) as model:
        rv, _ = case(name, batch=True, observed=observed)
        assert model.observed_RVs == [rv]
        assert model.named_vars_to_dims["response"] == ("person",)
        generated = pm.sample_prior_predictive(draws=2, random_seed=92)
    draws = generated["prior_predictive"]["response"]
    assert draws.dims == ("chain", "draw", "person")
    assert draws.shape == (1, 2, 2)
    assert draws.person.values.tolist() == ["p", "q"]


def test_named_distribution_uses_native_shape_and_rejects_conflicting_size():
    with pm.Model():
        rv = likert.GPCMDistribution("rating", [0.2], [1.0], [0, 0.1, -0.2], shape=3)
    assert pm.draw(rv, random_seed=4).shape == (3,)
    with pytest.raises(ValueError, match="only one"):
        likert.GPCMDistribution.dist([0.2], [1.0], [0, 0.1], shape=3, size=2)


@pytest.mark.parametrize("threshold", [[0, 0.8, -0.4], [0, 1.2, 0.3], [0, -0.2, -1.1]])
def test_ggum_native_density_accepts_signed_unordered_thresholds(threshold):
    values = np.arange(len(threshold))[:, None]
    theta = np.array([[-1.3], [-0.3], [0.7]])
    rv = likert.GGUMDistribution.dist(theta, [1.2], [-0.3], threshold)
    expected = np.array(
        [subjective_log_probs(float(person[0]), 1.2, -0.3, threshold) for person in theta]
    ).T
    assert_allclose(pm.logp(rv, values).eval(), expected, atol=2e-13)
    assert_allclose(np.exp(expected).sum(axis=0), 1, atol=1e-14)
    invalid = likert.GGUMDistribution.dist(theta, [1.2], [-0.3], [0.1, *threshold[1:]])
    with pytest.raises(ValueError, match="start at zero"):
        pm.logp(invalid, values).eval()
