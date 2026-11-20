"""Gaussian marginal ranking probabilities and exact augmentation Jacobians."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, quad_vec
from scipy.stats import multivariate_normal, norm

from pyfcirt.models.graph import observe_augmented
from pyfcirt.models.thurstonian.rank import TIRTRankDistribution as Rank
from tests.reference.rankings import order_arrays


def independent_winner_reference(mean, variance):
    """Integrate winner zero directly, including mean and log-variance derivatives."""
    mean, scale = np.asarray(mean), np.sqrt(variance)

    def integrand(z):
        standardized = (mean[0] + scale[0] * z - mean[1:]) / scale[1:]
        cdf = norm.cdf(standardized)
        derivative = np.array(
            [
                norm.pdf(value) * np.prod(np.delete(cdf, index)) / scale[index + 1]
                for index, value in enumerate(standardized)
            ]
        )
        mean_gradient = np.r_[derivative.sum(), -derivative]
        variance_gradient = np.r_[
            scale[0] * z * derivative.sum() / 2,
            -standardized * derivative * scale[1:] / 2,
        ]
        return norm.pdf(z) * np.r_[np.prod(cdf), mean_gradient, variance_gradient]

    result, error = quad_vec(integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)
    assert error < 1e-11
    return result[0], result[1:]


@pytest.mark.parametrize(
    "options", [{"dimension": 0}, {"dimension": 2, "n_points": 127}, {"dimension": 2, "seed": -1}]
)
def test_region_node_configuration_rejects_invalid_integration_requests(options):
    from pyfcirt.models.thurstonian.normal import sobol_nodes

    with pytest.raises(ValueError):
        sobol_nodes(**options)


@pytest.mark.parametrize("problem", ["covariance", "limits", "nodes", "chunk"])
def test_region_integral_rejects_misaligned_dimensions(problem):
    from pyfcirt.models.thurstonian.normal import region_logcdf

    arguments = dict(
        upper=np.zeros(2), mean=np.zeros(2), covariance=np.eye(2), nodes=np.ones((4, 1)) * 0.5
    )
    if problem == "covariance":
        arguments["covariance"] = np.eye(3)
    elif problem == "limits":
        arguments["upper"] = np.zeros(3)
    elif problem == "nodes":
        arguments["nodes"] = np.empty((0, 1))
    else:
        arguments["chunk_size"] = 0
    with pytest.raises(ValueError):
        region_logcdf(**arguments)


def test_comparison_links_match_independent_interval_and_beta_means():
    from scipy.special import expit

    from pyfcirt.models.thurstonian.normal import beta_shapes, interval_log_probs

    cuts = np.array([-2.0, 0.3, 3.0])
    expected = np.diff(np.r_[0.0, expit(cuts), 1.0])
    assert_allclose(np.exp(interval_log_probs(cuts, link="logit")), expected, atol=1e-14)
    alpha, beta = beta_shapes(cuts, 4.0, link="logit")
    assert_allclose(alpha / (alpha + beta), expit(cuts))
    for function in (interval_log_probs, lambda value, **kw: beta_shapes(value, 4.0, **kw)):
        with pytest.raises(ValueError, match="link"):
            function(cuts, link="unsupported")


@pytest.mark.parametrize(
    "change",
    [
        "orders",
        "order_dtype",
        "contrasts",
        "mean",
        "covariance",
        "membership_type",
        "membership_count",
        "points",
        "seed",
        "chunk",
        "residual_scalar",
    ],
)
def test_native_gaussian_rank_factory_requires_coherent_event_geometry(change):
    orders, contrasts = order_arrays(2)
    args = dict(
        theta=np.zeros(2),
        loading=np.ones(2),
        intercept=np.zeros(2),
        residual_variance=np.ones(2),
        orders=orders,
        contrasts=contrasts,
    )
    if change == "orders":
        args["orders"] = orders[0]
    elif change == "order_dtype":
        args["orders"] = orders.astype(float)
    elif change == "contrasts":
        args["contrasts"] = np.zeros((2, 2, 2))
    elif change == "mean":
        args.update(theta=np.zeros(3), loading=np.ones(3), intercept=np.zeros(3))
    elif change == "covariance":
        args["residual_variance"] = np.ones(3)
    elif change.startswith("membership"):
        args["membership"] = (
            np.ones((1, 2)) if change.endswith("type") else np.ones((1, 3), dtype=bool)
        )
    elif change == "points":
        args["n_points"] = 32
    elif change == "seed":
        args["seed"] = -1
    elif change == "chunk":
        args["chunk_size"] = 0
    else:
        args["residual_variance"] = 1.0
    with pytest.raises((ValueError, TypeError)):
        Rank.dist(**args)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_equal_gaussian_utilities_complete_and_partial_event_probabilities(size):
    orders, contrasts = order_arrays(size)
    membership = np.array([orders[:, 0] == i for i in range(size)])
    logs = Rank.kernel(
        jnp.zeros(size),
        jnp.ones(size),
        jnp.zeros(size),
        jnp.ones(size),
        contrasts=contrasts,
        n_points=1024,
    )
    assert_allclose(np.exp(logs), 1 / len(orders), atol=3e-4)
    partial = Rank.kernel(
        jnp.zeros(size),
        jnp.ones(size),
        jnp.zeros(size),
        jnp.ones(size),
        contrasts=contrasts,
        membership=membership,
        n_points=1024,
    )
    assert_allclose(np.exp(partial), 1 / size, atol=5e-4)
    assert_allclose(
        jax.grad(
            lambda x: Rank.kernel(
                x,
                jnp.ones(size),
                jnp.zeros(size),
                jnp.ones(size),
                contrasts=contrasts,
                n_points=1024,
            )[0]
        )(jnp.zeros(size)).sum(),
        0,
        atol=2e-12,
    )


def test_actual_four_statement_pick_tail_and_all_utility_parameter_gradients():
    from pyfcirt.models.thurstonian.normal import sobol_nodes

    # Native fitted state: source 8f0d, TIRTRank_7/b7, chain 1, draw 428, person 1107.
    # Its old 4,096-node complete-order union was 1.0000505547514775.
    mean = np.array(
        [3.729143804740122, -0.6006140062818319, -3.121059820284097, -0.9156306270556585]
    )
    variance = np.array([0.5, 0.568987061408665, 1.452807234687589, 0.6932445566345367])
    orders, contrasts = order_arrays(4)
    membership = orders[:, 0] == np.arange(4)[:, None]
    nodes = sobol_nodes(3, n_points=4096, seed=914)
    reference, reference_gradient = independent_winner_reference(mean, variance)

    def probabilities(value):
        return jnp.exp(
            Rank._log_probs(
                value[:4],
                jnp.diag(jnp.exp(value[4:])),
                contrasts,
                membership,
                nodes=nodes,
            )
        )

    point = jnp.asarray(np.r_[mean, np.log(variance)])
    actual = np.asarray(probabilities(point))
    assert np.all(np.isfinite(actual) & (actual >= 0) & (actual <= 1))
    assert_allclose(actual[0], reference, atol=2e-9, rtol=0)
    assert_allclose(actual.sum(), 1, atol=2e-12, rtol=0)
    assert_allclose(
        jax.grad(lambda value: probabilities(value)[0])(point),
        reference_gradient,
        atol=5e-9,
        rtol=0,
    )


def test_generic_nonconvex_row_batched_gaussian_events_and_gradients():
    from pyfcirt.models.thurstonian.normal import sobol_nodes

    orders, contrasts = order_arrays(4)
    positions = np.argsort(orders, axis=-1)
    first = positions[:, 0] < positions[:, 1]
    second = positions[:, 2] < positions[:, 3]
    exclusive = first != second
    membership = np.array([[exclusive, ~exclusive], [~exclusive, exclusive], [first, ~first]])
    mean = jnp.array([[0.9, 0, -0.9, 0], [0.9, 0, -0.9, 0], [-0.8, 0, 0.4, 0]])
    variance = jnp.array([[0.6, 1, 0.7, 1.2], [0.6, 1, 0.7, 1.2], [0.6, 1, 0.7, 1.2]])
    nodes = sobol_nodes(3, n_points=4096)

    def production(value):
        return jnp.exp(
            Rank._log_probs(
                value,
                jnp.eye(4) * variance[..., None, :],
                contrasts,
                membership,
                nodes=nodes,
            )
        )

    def reference(value):
        # Disjoint utility pairs have independent normal differences; XOR is nonconvex.
        first_p = jax.scipy.special.ndtr(
            (value[:, 0] - value[:, 1]) / jnp.sqrt(variance[:, 0] + variance[:, 1])
        )
        second_p = jax.scipy.special.ndtr(
            (value[:, 2] - value[:, 3]) / jnp.sqrt(variance[:, 2] + variance[:, 3])
        )
        exclusive_p = first_p * (1 - second_p) + (1 - first_p) * second_p
        selected = jnp.array([exclusive_p[0], 1 - exclusive_p[1], first_p[2]])
        return jnp.stack((selected, 1 - selected), axis=-1)

    assert_allclose(production(mean), reference(mean), atol=5e-5, rtol=0)
    assert_allclose(jax.jacrev(production)(mean), jax.jacrev(reference)(mean), atol=5e-5, rtol=0)


def test_full_and_empty_gaussian_events_have_exact_mass_and_zero_adjoint():
    from pyfcirt.models import categorical
    from pyfcirt.models.thurstonian.normal import sobol_nodes

    orders, contrasts = order_arrays(3)
    membership = np.array([np.ones(len(orders), bool), np.zeros(len(orders), bool)])
    nodes = sobol_nodes(2, n_points=128)

    def logs(mean):
        return Rank._log_probs(mean, jnp.eye(3), contrasts, membership, nodes=nodes)

    point = jnp.array([0.4, -0.2, 0.1])
    actual = np.asarray(logs(point))
    assert actual[0] == 0 and np.isneginf(actual[1])
    derivative = np.asarray(jax.jacrev(logs)(point))
    assert np.isfinite(derivative).all()
    assert_allclose(derivative, 0, atol=0)
    component = jnp.log(jnp.array([0.2, 0.3, 0.5]))
    assert_allclose(
        jax.jacrev(lambda value: categorical.event_log_probs(value, membership[:, :3]))(component)[
            1
        ],
        0,
        atol=0,
    )


@pytest.mark.parametrize("difference", [-0.02, 0.02])
def test_near_half_gaussian_event_value_and_each_side_gradient(difference):
    from pyfcirt.models.thurstonian.normal import sobol_nodes

    orders, contrasts = order_arrays(3)
    positions = np.argsort(orders, axis=-1)
    selected = positions[:, 0] < positions[:, 1]
    membership = np.array([selected, ~selected])
    nodes = sobol_nodes(2, n_points=4096)
    variance = jnp.array([0.7, 1.2, 0.9])

    def production(mean):
        return jnp.exp(
            Rank._log_probs(mean, jnp.diag(variance), contrasts, membership, nodes=nodes)
        )[0]

    def reference(mean):
        return jax.scipy.special.ndtr((mean[0] - mean[1]) / jnp.sqrt(variance[:2].sum()))

    # These are local checks on either side of one half. Fixed-QMC estimates can
    # disagree at the representation switch; this is not a global smoothness claim.
    point = jnp.array([difference, 0, 0.3])
    assert_allclose(production(point), reference(point), atol=2e-5, rtol=0)
    assert_allclose(jax.grad(production)(point), jax.grad(reference)(point), atol=5e-5, rtol=0)


def test_gaussian_event_aggregation_preserves_unresolved_partition_error(monkeypatch):
    from pyfcirt.models.thurstonian.normal import sobol_nodes

    orders, contrasts = order_arrays(3)
    component = np.array([0.1, 0.12, 0.14, 0.16, 0.18, 0.32])
    membership = np.repeat(np.eye(3, dtype=bool), 2, axis=1)
    monkeypatch.setattr(
        "pyfcirt.models.thurstonian.rank.region_logcdf",
        lambda *args, **kwargs: jnp.log(component),
    )
    actual = np.exp(
        Rank._log_probs(
            jnp.zeros(3), jnp.eye(3), contrasts, membership, nodes=sobol_nodes(2, n_points=128)
        )
    )
    # All three direct event sums are the smaller side. Their integration error
    # remains visible instead of being hidden by division by the estimated total.
    assert_allclose(actual, [0.22, 0.3, 0.5], atol=2e-15, rtol=0)
    assert_allclose(actual.sum(), 1.02, atol=2e-15, rtol=0)


@pytest.mark.parametrize("chunk_size", [7, 127, 512])
def test_gaussian_event_values_and_gradients_are_chunk_invariant(chunk_size):
    from pyfcirt.models.thurstonian.normal import sobol_nodes

    orders, contrasts = order_arrays(3)
    membership = orders[:, 0] == np.arange(3)[:, None]
    nodes = sobol_nodes(2, n_points=256)

    def evaluate(mean, chunk):
        return Rank._log_probs(
            mean,
            jnp.diag(jnp.array([0.6, 1, 1.4])),
            contrasts,
            membership,
            nodes=nodes,
            chunk_size=chunk,
        )

    point = jnp.array([-0.3, 0.1, 0.7])
    assert_allclose(evaluate(point, chunk_size), evaluate(point, 128), atol=2e-13, rtol=0)
    assert_allclose(
        jax.jacrev(lambda value: evaluate(value, chunk_size))(point),
        jax.jacrev(lambda value: evaluate(value, 128))(point),
        atol=2e-13,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("chunk_size", [7, 128, 511])
def test_near_certain_gaussian_cdf_stays_bounded_without_clipping(dtype, chunk_size):
    from pyfcirt.models.thurstonian.normal import region_logcdf, sobol_nodes

    upper = jnp.full(3, 8, dtype=dtype)
    mean = jnp.zeros(3, dtype=dtype)
    covariance = jnp.asarray(np.eye(3) * 0.65 + 0.35, dtype=dtype)
    actual = region_logcdf(
        upper,
        mean,
        covariance,
        nodes=sobol_nodes(3, n_points=128),
        chunk_size=chunk_size,
    )
    assert actual.dtype == dtype
    assert np.isfinite(actual) and actual <= 0
    assert 0 <= np.exp(actual) <= 1


def test_dyad_augmented_density_integrates_to_native_marginal():
    mean = np.array([0.4, -0.2])
    variance = np.array([0.6, 1.0])
    # The common utility level has disappeared; only its difference is integrated.
    probability = quad(
        lambda difference: norm.pdf(difference, mean[0] - mean[1], np.sqrt(variance.sum())),
        0,
        np.inf,
    )[0]
    _, contrasts = order_arrays(2)
    native = Rank.kernel(
        jnp.array(mean),
        jnp.ones(2),
        jnp.zeros(2),
        jnp.array(variance),
        contrasts=contrasts,
        n_points=128,
    )
    assert_allclose(np.exp(native[0]), probability, atol=1e-12)


def test_neighboring_differences_match_joint_density_and_native_jacobian(monkeypatch):
    from types import SimpleNamespace

    import pymc as pm

    basis = np.array([[[1, -1, 0], [0, 1, -1]]], dtype=float)
    differences = np.array([[0.4, 1.3]])
    mean = np.array([[0.4, -0.2, 0.1]])
    covariance = np.diag([0.6, 1.0, 0.8])
    batch = SimpleNamespace(
        name="b",
        statement=np.zeros((1, 3)),
        obs_dim="obs",
        events=SimpleNamespace(mask=None, basis=basis),
    )

    # Constructing the exact augmented target must never execute a region integral.
    def forbidden(*args, **kwargs):
        raise AssertionError("Marginal CDF entered the augmented sampling target")

    monkeypatch.setattr(Rank, "_log_probs", forbidden)
    with pm.Model(coords={"obs": ["row"], "b_difference": [0, 1]}) as model:
        unused = pm.Normal("unused")
        differences_rv, observed = observe_augmented(
            batch, *Rank.augmentation(batch, mean, covariance)
        )
    assert model.free_RVs == [unused, differences_rv]
    assert model.observed_RVs == [observed]
    assert not model.potentials
    assert model.named_vars_to_dims[differences_rv.name] == ("obs", "b_difference")
    assert model.named_vars_to_dims[observed.name] == ("obs",)
    model = pm.do(model, {}, prune_vars=True)
    assert [rv.name for rv in model.free_RVs] == [differences_rv.name]
    assert [rv.name for rv in model.observed_RVs] == [observed.name]
    initial = model.initial_point()
    assert initial["b_contrast_raw_log__"].shape == (1, 2)
    assert np.isfinite(model.compile_logp(mode="FAST_COMPILE")(initial))

    outside = pm.do(model, {differences_rv.name: np.array([[-0.4, 1.3]])}, prune_vars=True)
    assert np.isneginf(outside.compile_logp(mode="FAST_COMPILE")({}))
    expected = multivariate_normal.logpdf(
        differences[0], mean=[0.6, -0.3], cov=[[1.6, -1.0], [-1.0, 1.8]]
    )
    point = {"b_contrast_raw_log__": np.log(differences)}
    assert_allclose(
        model.compile_logp(jacobian=False, mode="FAST_COMPILE")(point), expected, atol=1e-12
    )
    assert_allclose(
        model.compile_logp(mode="FAST_COMPILE")(point),
        expected + np.log(differences).sum(),
        atol=1e-12,
    )
    expected_gradient = jax.grad(
        lambda log_d: (
            jax.scipy.stats.multivariate_normal.logpdf(
                jnp.exp(log_d), jnp.array([0.6, -0.3]), jnp.array([[1.6, -1.0], [-1.0, 1.8]])
            )
            + log_d.sum()
        )
    )(jnp.log(differences[0]))
    assert_allclose(model.compile_dlogp(mode="FAST_COMPILE")(point), expected_gradient, atol=1e-12)


def test_reduced_target_gradients_include_utility_means_and_free_variances():
    from types import SimpleNamespace

    import pymc as pm
    import pytensor.tensor as pt

    basis = np.array([[1, -1, 0], [0, 1, -1]], dtype=float)
    batch = SimpleNamespace(
        name="b",
        statement=np.zeros((1, 3)),
        obs_dim="obs",
        events=SimpleNamespace(mask=None, basis=basis[None]),
    )
    with pm.Model(coords={"obs": [0], "b_difference": [0, 1]}) as model:
        mean = pm.Flat("mean", shape=3)
        variance = pm.HalfFlat("variance", shape=3)
        observe_augmented(batch, *Rank.augmentation(batch, mean[None], pt.diag(variance)))
    mean, log_variance, log_difference = (
        np.array([0.4, -0.2, 0.1]),
        np.log([0.6, 1, 0.8]),
        np.log([0.4, 1.3]),
    )

    def reference(vector):
        covariance = basis @ jnp.diag(jnp.exp(vector[3:6])) @ basis.T
        return (
            jax.scipy.stats.multivariate_normal.logpdf(
                jnp.exp(vector[6:]), basis @ vector[:3], covariance
            )
            + vector[3:].sum()
        )

    point = dict(mean=mean, variance_log__=log_variance, b_contrast_raw_log__=log_difference[None])
    expected = jax.grad(reference)(jnp.asarray(np.r_[mean, log_variance, log_difference]))
    assert_allclose(model.compile_dlogp(mode="FAST_COMPILE")(point), expected, atol=1e-12)


def test_comparison_constraints_reject_cycles_instead_of_filling_unknown_orders():
    with pytest.raises(ValueError, match="Cyclic"):
        Rank.observed_region(3, ((0, 1, 1), (1, 2, 1), (0, 2, 0)))
    mask, order = Rank.observed_region(4, ((0, 1, 1), (2, 3, 0)))
    assert mask.sum() == 2
    assert order.tolist() == [1, 2, 0, 3]


@pytest.mark.parametrize(
    "edges",
    [
        ((0, 1, 1), (0, 2, 1)),  # PICK: two losers have no relative order.
        ((0, 3, 1), (1, 3, 1), (2, 3, 1), (0, 1, 1), (0, 2, 1)),  # MOLE
        ((0, 1, 1), (2, 3, 0)),  # Two disconnected comparisons.
    ],
)
def test_partial_difference_density_integrates_common_level_without_ordering_middle(edges):
    from types import SimpleNamespace

    import pymc as pm

    size = max(max(a, b) for a, b, _ in edges) + 1
    mask, order = Rank.observed_region(size, edges)
    inverse = np.argsort(order)
    basis = np.eye(size)[order[1:]] - np.eye(size)[order[:1]]
    raw = jnp.linspace(-0.3, 0.5, size - 1)
    relative = Rank._project_differences(raw, mask)
    derivative = jax.jacfwd(lambda value: Rank._project_differences(value, mask))(raw)
    jacobian = np.linalg.slogdet(derivative)[1]
    mean, covariance = np.linspace(-0.4, 0.6, size), np.diag(np.linspace(0.6, 1.4, size))
    difference_mean, difference_covariance = Rank._contrast_moments(mean, covariance, basis)
    relative_utility = np.r_[0, relative][inverse]
    for first, second, bit in edges:
        assert (relative_utility[first] > relative_utility[second]) == bool(bit)
    old_density = quad(
        lambda level: np.exp(
            multivariate_normal.logpdf(relative_utility + level, mean, covariance)
        ),
        -np.inf,
        np.inf,
        epsabs=1e-12,
    )[0]
    new_density = np.exp(
        multivariate_normal.logpdf(relative, difference_mean, difference_covariance)
    )
    assert_allclose(new_density, old_density, rtol=1e-10)
    batch = SimpleNamespace(
        name="b",
        statement=np.zeros((1, size)),
        obs_dim="obs",
        events=SimpleNamespace(mask=mask[None], basis=basis[None]),
    )
    with pm.Model(coords={"obs": [0], "b_difference": range(size - 1)}) as model:
        differences_rv, observed = observe_augmented(
            batch, *Rank.augmentation(batch, mean[None], covariance)
        )
    assert model.observed_RVs == [observed]
    assert model.free_RVs == [differences_rv]
    logp = model.compile_logp(mode="FAST_COMPILE")
    assert np.isfinite(logp(model.initial_point()))
    expected = np.log(old_density) + float(jacobian)
    assert_allclose(
        logp({"b_contrast_raw_partial_order__": np.asarray(raw)[None]}),
        expected,
        atol=1e-10,
    )
