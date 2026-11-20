"""Independent single-item equations, tails, gradients and ideal-point curvature."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, log_expit, logsumexp

from pyfcirt.models.likert import DichotomousIRTDistribution as Binary
from pyfcirt.models.likert import GGUMDistribution as GGUM
from pyfcirt.models.likert import GPCMDistribution as GPCM
from tests.reference.likert import subjective_log_probs


@pytest.mark.parametrize("lower,upper", [(0.0, 1.0), (0.2, 1.0), (0.2, 0.9)])
def test_binary_tail_values_and_ability_gradients(lower, upper):
    point = np.array([[-1000.0, 0.7], [0.3, -0.2], [1000.0, -0.4]])
    loading = jnp.array([1.2, -0.7])

    def logs(theta):
        return Binary.kernel(theta, loading, 0.3, lower, upper)

    eta = point @ loading - 0.3
    if lower == 0 and upper == 1:
        reference = np.stack((log_expit(-eta), log_expit(eta)), -1)
    else:
        lower + (upper - lower) * expit(eta)
        reference = np.stack(
            (
                np.logaddexp(
                    np.log1p(-upper) if upper < 1 else -np.inf,
                    np.log(upper - lower) + log_expit(-eta),
                ),
                np.logaddexp(
                    np.log(lower) if lower else -np.inf, np.log(upper - lower) + log_expit(eta)
                ),
            ),
            -1,
        )
    assert_allclose(logs(point), reference, atol=3e-13, rtol=0)

    def selected(theta):
        return logs(theta)[jnp.arange(3), jnp.array([0, 1, 0])].sum()

    finite = np.zeros_like(point)
    for index in np.ndindex(point.shape):
        delta = np.zeros_like(point)
        delta[index] = 1e-4

        def ref(x):
            z = x @ loading - 0.3
            if lower == 0 and upper == 1:
                return np.array([log_expit(-z[0]), log_expit(z[1]), log_expit(-z[2])]).sum()
            yes = np.logaddexp(
                np.log(lower) if lower else -np.inf, np.log(upper - lower) + log_expit(z)
            )
            no = np.logaddexp(
                np.log1p(-upper) if upper < 1 else -np.inf, np.log(upper - lower) + log_expit(-z)
            )
            return no[0] + yes[1] + no[2]

        finite[index] = (ref(point + delta) - ref(point - delta)) / 2e-4
    assert_allclose(jax.grad(selected)(jnp.array(point)), finite, atol=3e-9)


@pytest.mark.parametrize("count", [2, 3, 5])
def test_partial_credit_adjacent_odds_and_gradients(count):
    point = jnp.array([-0.7, 0.8])
    loading = jnp.array([1.2, -0.6])
    cuts = jnp.r_[0.0, jnp.linspace(-0.7, 0.8, count - 1)]

    def logs(x):
        return GPCM.kernel(x, loading, cuts)

    expected = np.arange(count) * float(point @ loading) + np.asarray(cuts)
    assert_allclose(logs(point), expected - logsumexp(expected), atol=2e-14)
    assert_allclose(np.diff(logs(point)), float(point @ loading) + np.diff(cuts), atol=2e-14)
    probs = np.exp(logs(point))
    score = count - 1 - np.arange(count) @ probs
    assert_allclose(jax.grad(lambda x: logs(x)[-1])(point), score * loading, atol=2e-14)


@pytest.mark.parametrize("count", [2, 3, 5])
@pytest.mark.parametrize("loading", [1.2, -1.2])
@pytest.mark.parametrize("steps", ["negative", "positive", "mixed"])
def test_unfolding_subjective_state_values_and_gradients(count, loading, steps):
    free = np.linspace(2.0, 0.4, count - 1)
    if steps == "negative":
        free = -free
    elif steps == "mixed":
        free[1::2] *= -1
    cuts = np.r_[0.0, free]
    for x in [-1.3, -0.3, 0.7]:

        def logs(v):
            return GGUM.kernel(v, jnp.array([loading]), jnp.array([-0.3]), jnp.array(cuts))

        def ref(v):
            return subjective_log_probs(v, loading, -0.3, cuts)

        assert_allclose(logs(jnp.array([x])), ref(x), atol=2e-13)
        finite = (ref(x + 1e-5) - ref(x - 1e-5)) / 2e-5
        assert_allclose(jax.jacfwd(logs)(jnp.array([x]))[:, 0], finite, atol=2e-9)


def test_multidimensional_ideal_point_hessian_is_exact_and_finite():
    loading = jnp.array([1.2, 0.7])
    location = jnp.array([0.4, -0.6])
    cuts = jnp.array([0.0, -2.0, -1.2, -0.75])

    def logs(x):
        return GGUM.kernel(x, loading, location, cuts)

    center = -float(loading.sum()) * np.cumsum(cuts)
    probability = np.exp(center - logsumexp(center))
    coefficient = (2 * len(cuts) - 1) / 2 - np.arange(len(cuts))
    curvature = coefficient[-1] ** 2 - probability @ coefficient**2
    assert_allclose(logs(location), center - logsumexp(center), atol=2e-14)
    assert_allclose(jax.grad(lambda x: logs(x)[-1])(location), 0, atol=1e-14)
    assert_allclose(
        jax.hessian(lambda x: logs(x)[-1])(location),
        np.diag(curvature * np.array(loading) ** 2),
        atol=2e-13,
    )
    assert np.isfinite(jax.hessian(lambda x: logs(x)[-1])(location + 1e-8)).all()


CASES = json.loads(
    (Path(__file__).parents[1] / "reference/forcechoice_single_item.json").read_text()
)["cases"]


@pytest.mark.parametrize("fixture", CASES, ids=lambda f: f["parameters"]["id"])
def test_executed_forcechoice_cpp_fixed_likelihood(fixture):
    case = fixture["parameters"]
    theta = jnp.array(case["theta"])
    if case["family"] == "mirt":
        logs = Binary.kernel(theta, jnp.array(case["a"]), case["b"], case["c"], case["upper"])
    elif case["family"] == "mgpcm":
        logs = GPCM.kernel(theta, jnp.array(case["a"]), jnp.array(case["d"]))
    else:
        par = case["par"]
        logs = GGUM.kernel(theta, jnp.array(par[:3]), jnp.array(par[3:6]), jnp.array(par[6:]))
    selected = np.asarray(logs)[np.arange(len(theta)), case["response"]]
    reference = np.array(fixture["cpp"]["individual"])
    if case["id"] == "mirt_extreme":
        eta = np.array(theta) @ case["a"] - case["b"]
        stable = np.where(np.array(case["response"]) == 1, log_expit(eta), log_expit(-eta))
        assert_allclose(selected, stable, atol=3e-13)
        clipped = np.clip(1 / (1 + np.exp(-np.clip(eta, -700, 700))), 1e-10, 1 - 1e-10)
        original = np.where(np.array(case["response"]) == 1, np.log(clipped), np.log1p(-clipped))
        assert_allclose(reference, original, atol=5e-14, rtol=0)
        assert np.max(np.abs(selected - reference)) > 900
    elif case["family"] == "mggum":
        par = np.array(case["par"])

        def distance_law(epsilon):
            radius = np.sqrt(
                np.sum((par[:3] * (np.array(theta) - par[3:6])) ** 2, axis=-1) + epsilon
            )
            rows = [
                subjective_log_probs(value, 1.0, 0.0, par[:3].sum() * par[6:]) for value in radius
            ]
            return np.array(rows)[np.arange(len(theta)), case["response"]]

        assert_allclose(reference, distance_law(1e-10), atol=5e-14, rtol=0)
        assert_allclose(selected, distance_law(0.0), atol=5e-14, rtol=0)
        assert np.max(np.abs(reference - selected)) > 1e-12
    else:
        assert_allclose(selected, reference, atol=3e-13, rtol=0)
