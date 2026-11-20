"""Executed external reference probabilities and expected Fisher matrices."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyfcirt.analysis.information import beta_fisher, categorical_fisher
from pyfcirt.models.likert import DichotomousIRTDistribution as Binary
from pyfcirt.models.likert import GGUMDistribution as GGUM
from pyfcirt.models.likert import GPCMDistribution as GPCM
from pyfcirt.models.mgppm import MGPPMDistribution as MGPPM
from pyfcirt.models.sequential import GGUMRankDistribution, TwoPLRankDistribution
from pyfcirt.models.thurstonian.rank import TIRTRankDistribution
from pyfcirt.models.thurstonian.tirt import TIRTDistribution
from tests.reference.rankings import order_arrays


def cases(name):
    return json.loads((Path(__file__).parent / "reference" / name).read_text())["cases"]


@pytest.mark.parametrize("case", cases("mirt_information.json"), ids=lambda c: c["id"])
def test_mirt_external_full_fisher(case):
    if case["id"] == "mgpcm":

        def kernel(x):
            return GPCM.kernel(x, jnp.array(case["loading"]), jnp.array(case["category_intercept"]))
    else:

        def kernel(x):
            return Binary.kernel(
                x,
                jnp.array(case["loading"]),
                case["native_intercept"],
                case["lower"],
                case["upper"],
            )

    for row, p, info in zip(
        case["theta"], case["R_probabilities"], case["R_information"], strict=True
    ):
        actual, _, fisher = categorical_fisher(kernel, jnp.array(row))
        assert_allclose(actual, p, atol=1e-12)
        assert_allclose(fisher, info, atol=1e-10)


@pytest.mark.parametrize("case", cases("forcechoice_mggum_information.json"), ids=lambda c: c["id"])
def test_ggum_external_probability_derivative_fisher(case):
    par = case["par"]

    def kernel(x):
        return GGUM.kernel(x, jnp.array(par[:3]), jnp.array(par[3:6]), jnp.array(par[6:]))

    for i, row in enumerate(case["theta"]):
        p, _, info = categorical_fisher(kernel, jnp.array(row))
        assert_allclose(p, case["probabilities"][i], atol=1e-7)
        assert_allclose(info, case["fisher_d1"][i], atol=1e-7)
        assert_allclose(info, case["fisher_d2"][i], atol=1e-7)


@pytest.mark.parametrize("case", cases("mgppm_author_information.json"), ids=lambda c: c["id"])
def test_mgppm_author_probabilities_and_information(case):
    par = np.array(case["par"])
    loading = par[::2]
    intercept = -par[1::2]
    orders, _ = order_arrays(len(loading))
    scores = len(loading) - 1 - np.argsort(orders, axis=-1)

    def kernel(x):
        return MGPPM.kernel(x, jnp.array(loading), jnp.array(intercept), scores=scores)

    for i, row in enumerate(case["theta"]):
        p, _, info = categorical_fisher(kernel, jnp.array(row, dtype=float))
        assert_allclose(p, case["probabilities"][i], atol=2e-10)
        assert_allclose(info, case["information"][i], atol=2e-8)


@pytest.mark.parametrize("case", cases("forcechoice_rank_information.json"), ids=lambda c: c["id"])
def test_rank_external_probabilities_and_fisher(case):
    orders, _ = order_arrays(case["size"])
    response_format = case["response_protocol"]

    def matches(order, event):
        if response_format == "rank":
            return tuple(order) == tuple(np.array(event) - 1)
        if response_format == "pick":
            return order[0] == event[0] - 1
        return order[0] == event[0] - 1 and order[-1] == event[-1] - 1

    membership = np.array(
        [[matches(order, event) for order in orders] for event in case["patterns"]]
    )
    loading = np.array(case["magnitude"])
    if case["family"] == "two_pl_rank":
        cls = TwoPLRankDistribution
        loading = loading * np.array(case["keying"])
        parameters = (jnp.array(loading), jnp.array(case["ranking_intercept"]))
    else:
        cls = GGUMRankDistribution
        parameters = (jnp.array(loading), jnp.array(case["location"]), jnp.array(case["threshold"]))
    index = np.array(case["trait_index"])

    def kernel(x):
        return cls.kernel(x[index], *parameters, orders=orders, membership=membership)

    for i, row in enumerate(case["theta"]):
        p, _, info = categorical_fisher(kernel, jnp.array(row))
        assert_allclose(p, case["R_probability"][i], atol=1e-7)
        assert_allclose(info, case["R_fisher_d1"][i], atol=1e-7)
        assert_allclose(info, case["R_fisher_d2"][i], atol=1e-7)


@pytest.mark.parametrize("case", cases("betareg_information.json"), ids=lambda c: c["id"])
def test_beta_original_expected_hessian(case):
    for x, offset, expected in zip(
        case["X"], case["offset"], case["expected_observation_information"], strict=True
    ):
        # A two-statement contrast maps the reference design vector exactly.
        x = jnp.array(x)

        def shapes(theta, x=x, offset=offset):
            return TIRTDistribution._beta_parameters(
                theta,
                x * jnp.array([1.0, -1.0]),
                -offset,
                case["precision"],
                1.0,
                link=case["link"],
            )

        _, _, info = beta_fisher(shapes, jnp.array(case["theta"]))
        assert_allclose(info, expected, atol=1e-10)


@pytest.mark.parametrize("case", cases("mfcblockinfo_information.json"), ids=lambda c: c["id"])
def test_tirt_pair_information_against_executed_mfcblockinfo(case):
    load = jnp.array(case["loading_matrix"])
    variance = np.array(case["uniqueness_variances"])
    for person, theta in enumerate(case["theta"]):
        for i, (a, b) in enumerate(case["pair_indices_zero_based"]):
            # Project global ability into the two statement utilities explicitly.
            def kernel(x, a=a, b=b, i=i):
                return TIRTDistribution.kernel(
                    jnp.stack((load[a] @ x, load[b] @ x)),
                    jnp.ones(2),
                    jnp.array([case["native_gamma"][i]]),
                    jnp.zeros(2),
                    jnp.array(variance[[a, b]]),
                    1.0,
                    link="probit",
                )

            _, _, info = categorical_fisher(kernel, jnp.array(theta, dtype=float))
            assert_allclose(info, case["R_pair_expected"][person][i], atol=1e-9)


@pytest.mark.parametrize(
    "case", cases("mfcblockinfo_joint_information.json"), ids=lambda c: c["id"]
)
def test_gaussian_joint_rank_information_against_executed_mfcblockinfo(case):
    load = jnp.array(case["loading_matrix"])
    size = len(load)
    orders, contrasts = order_arrays(size)
    # Frozen rows already use one-based preference order.
    frozen_orders = np.array(case["ranking"]) - 1
    selection = [list(map(tuple, orders)).index(tuple(row)) for row in frozen_orders]

    def kernel(x):
        return TIRTRankDistribution.kernel(
            load @ x,
            jnp.ones(size),
            jnp.array(case["intercept"]),
            jnp.array(case["residual_sd"]) ** 2,
            contrasts=contrasts,
            n_points=16384,
        )

    reference = json.loads(
        (Path(__file__).parent / "reference/mfcblockinfo_joint_information.json").read_text()
    )
    fisher_atol = reference["provenance"]["information_absolute_tolerance"][case["id"]]
    # Exact dyad CDF retains the frozen probability gate. Higher-dimensional QMC
    # uses an explicit approximation-specific regression tolerance, not a bound
    # or a claim to pass the frozen reference's original 1e-10 probability gate.
    probability_atol = 1e-10 if size == 2 else 1e-6
    for i, theta in enumerate(case["theta"]):
        p, _, info = categorical_fisher(kernel, jnp.array(theta, dtype=float))
        assert_allclose(
            p[np.array(selection)], case["probability"][i], atol=probability_atol, rtol=0
        )
        assert_allclose(info, case["block_information"][i], atol=fisher_atol, rtol=0)


@pytest.mark.parametrize("kind", ["dyad", "triplet"])
def test_rank2pl_author_item_information(kind):
    fixture = json.loads(
        (Path(__file__).parent / "reference/rank2pl_author_information.json").read_text()
    )
    case = fixture["cases"][kind]
    par = case["parameters"]
    count = 2 if kind == "dyad" else 3
    loading = np.array(par[:2] if count == 2 else par[:6:2])
    if count == 2:
        loading[1] *= -1
    intercept = -np.array([par[5], 0.0] if count == 2 else par[1:6:2])
    orders, _ = order_arrays(count)
    events = fixture["provenance"][f"{kind}_event_order"]
    frozen_orders = np.array(events) - 1
    selection = [list(map(tuple, orders)).index(tuple(row)) for row in frozen_orders]

    def kernel(x):
        return TwoPLRankDistribution.kernel(
            x[:count], jnp.array(loading), jnp.array(intercept), orders=orders
        )

    for i, theta in enumerate(fixture["theta"]):
        p, _, info = categorical_fisher(kernel, jnp.array(theta, dtype=float))
        assert_allclose(p[np.array(selection)], case["probabilities"][i], atol=1e-8)
        assert_allclose(info, case["information"][i], atol=1e-6)
