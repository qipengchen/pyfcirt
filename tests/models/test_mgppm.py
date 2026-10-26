"""Independent fixed-parameter probability, gradient and generator checks."""

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from pyfcirt import FCData, FCDesign
from pyfcirt.models.mgppm._data import prepare_data
from pyfcirt.models.mgppm._likelihood import (
    event_log_probs,
    numpy_event_log_probs,
    statement_utilities,
)
from pyfcirt.simulation import simulate_mgppm
from tests._reference_mgppm import category_logps, partial_logp

CASES = [(task, size) for task in ("rank", "pick", "mole") for size in (2, 3, 4)] + [("graded", 2)]


def make_design(task, size):
    kwargs = {}
    if task == "graded":
        kwargs["graded_scale"] = {
            "categories": ["second++", "second", "equal", "first", "first++"],
            "higher_prefers": "position_1",
        }
    return FCDesign.from_mapping(
        traits={f"T{i}": [f"S{i}"] for i in range(size)},
        blocks={"B": [f"S{i}" for i in range(size)]},
        keying={f"S{i}": (-1 if i % 2 else 1) for i in range(size)},
        response_format=task,
        **kwargs,
    )


def group_for(design):
    task = design.response_formats["B"]
    size = design.block_size
    order = list(range(1, size + 1))
    if task == "pick":
        order[1:] = [None] * (size - 1)
    elif task == "mole":
        order[1:-1] = [None] * (size - 2)
    elif task == "graded":
        order = design.graded_scale["categories"][0]
    data = FCData.from_frames(
        pd.DataFrame({"person_id": ["P"], "block_id": ["B"], "response": [order]}), design
    )
    return prepare_data(data).groups[0]


def reference_events(utilities, group):
    if group.task == "graded":
        return np.array([category_logps(row, group.scores) for row in utilities])
    return np.array([[partial_logp(row, event) for event in group.events] for row in utilities])


@pytest.fixture(scope="module", params=CASES, ids=lambda c: f"{c[0]}-k{c[1]}")
def kernel(request):
    task, size = request.param
    group = group_for(make_design(task, size))
    utility = pt.dmatrix("utility")
    response = pt.lvector("response")
    logp = event_log_probs(utility, group.scores, group.membership)
    selected = logp[pt.arange(utility.shape[0]), response].sum()
    # Compile each graph once and reuse it across its mathematical checks.
    # Use the supported Numba linker; the legacy C path is not a test dependency.
    function = pytensor.function(
        [utility, response], [logp, pt.grad(selected, utility)], mode="NUMBA"
    )
    return group, function


def test_fixed_logp_and_gradient_match_scalar_reference(kernel):
    group, function = kernel
    size = group.scores.shape[1]
    utility = np.array([np.linspace(-0.6, 0.9, size), np.linspace(0.7, -0.3, size)])
    response = np.array([0, len(group.events) - 1], dtype="int64")
    actual, gradient = function(utility, response)
    expected = reference_events(utility, group)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(
        numpy_event_log_probs(utility, group.scores, group.membership),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(np.exp(actual).sum(axis=-1), 1, atol=1e-14)
    numerical = np.empty_like(utility)
    step = 1e-5
    for index in np.ndindex(utility.shape):
        plus, minus = utility.copy(), utility.copy()
        plus[index] += step
        minus[index] -= step
        hi = reference_events(plus, group)[np.arange(2), response].sum()
        lo = reference_events(minus, group)[np.arange(2), response].sum()
        numerical[index] = (hi - lo) / (2 * step)
    assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)


def test_zero_utilities_and_extreme_finite_logits(kernel):
    group, function = kernel
    size = group.scores.shape[1]
    response = np.array([0, len(group.events) - 1], dtype="int64")
    uniform, _ = function(np.zeros((2, size)), response)
    assert_allclose(np.exp(uniform), 1 / len(group.events), atol=1e-14)
    extreme = np.array([np.linspace(-1000, 1000, size), np.linspace(800, -900, size)])
    actual, gradient = function(extreme, response)
    assert np.isfinite(actual).all()
    assert np.isfinite(gradient).all()
    assert_allclose(actual, reference_events(extreme, group), atol=2e-10, rtol=1e-12)


def test_ranking_shift_invariance_is_not_applied_to_graded(kernel):
    group, function = kernel
    values = np.array([np.linspace(-0.4, 0.7, group.scores.shape[1])])
    response = np.array([0], dtype="int64")
    before, _ = function(values, response)
    after, _ = function(values + 2.5, response)
    if group.task == "graded":
        assert not np.allclose(before, after)
    else:
        assert_allclose(before, after, atol=1e-13)


def test_parameter_gather_and_gradients_match_labelled_reference(design):
    frame = pd.DataFrame(
        {"person_id": ["P1", "P2"], "block_id": ["B1", "B2"], "response": [[2, 3, 1], [3, 1, 2]]}
    )
    inputs = prepare_data(FCData.from_frames(frame, design))
    group = inputs.groups[0]
    theta, magnitude, beta = pt.dmatrix("theta"), pt.dvector("magnitude"), pt.dvector("beta")
    utility = statement_utilities(theta, magnitude, beta, inputs, group)
    logp = event_log_probs(utility, group.scores, group.membership)
    total = logp[pt.arange(2), group.response].sum()
    function = pytensor.function(
        [theta, magnitude, beta],
        [utility, *pt.grad(total, [theta, magnitude, beta])],
        mode="NUMBA",
    )
    values = [
        np.array([[0.2, -0.4, 0.8], [-0.5, 0.6, 0.1]]),
        np.array([0.7, 1.2, 0.8, 1.3, 0.9, 1.1]),
        np.array([0.1, -0.2, 0.1, -0.3, 0.1, 0.2]),
    ]
    members = design.to_frame().set_index(["block_id", "statement_id"])

    def reference(parameters, return_utility=False):
        traits = {label: i for i, label in enumerate(inputs.trait_ids)}
        statements = {label: i for i, label in enumerate(inputs.statement_ids)}
        rows = []
        for person, block in zip(frame.person_id, frame.block_id, strict=True):
            individual = inputs.person_ids.index(person)
            rows.append(
                [
                    parameters[1][statements[s]]
                    * members.loc[(block, s), "keying"]
                    * parameters[0][individual, traits[members.loc[(block, s), "trait"]]]
                    - parameters[2][statements[s]]
                    for s in design.to_frame().loc[
                        design.to_frame().block_id == block, "statement_id"
                    ]
                ]
            )
        if return_utility:
            return np.asarray(rows)
        return sum(
            partial_logp(row, order) for row, order in zip(rows, frame.response, strict=True)
        )

    actual, *gradients = function(*values)
    assert_allclose(actual, reference(values, return_utility=True), atol=1e-14)
    for parameter, expected_gradient in enumerate(gradients):
        numerical = np.empty_like(values[parameter])
        for index in np.ndindex(numerical.shape):
            plus, minus = [v.copy() for v in values], [v.copy() for v in values]
            plus[parameter][index] += 1e-5
            minus[parameter][index] -= 1e-5
            numerical[index] = (reference(plus) - reference(minus)) / 2e-5
        assert_allclose(expected_gradient, numerical, rtol=2e-6, atol=2e-7)


def test_leading_batch_axes_are_preserved():
    group = group_for(make_design("pick", 3))
    values = np.arange(18, dtype=float).reshape(2, 3, 3) / 10
    symbolic = pt.dtensor3("utilities")
    function = pytensor.function(
        [symbolic], event_log_probs(symbolic, group.scores, group.membership), mode="NUMBA"
    )
    expected = reference_events(values.reshape(-1, 3), group).reshape(2, 3, 3)
    assert_allclose(function(values), expected, atol=1e-12)
    assert_allclose(
        numpy_event_log_probs(values, group.scores, group.membership), expected, atol=1e-12
    )


def test_default_backend_matches_reference():
    group = group_for(make_design("pick", 3))
    utility = pt.dmatrix("utility")
    function = pytensor.function(
        [utility], event_log_probs(utility, group.scores, group.membership)
    )
    values = np.array([[0.2, -0.6, 0.8]])
    assert_allclose(function(values), reference_events(values, group), atol=1e-12)


@pytest.mark.parametrize("task,size", CASES)
def test_generator_cdf_matches_independent_probabilities(monkeypatch, task, size):
    design = make_design(task, size)
    group = group_for(design)
    n = len(group.events)
    trait = np.linspace(-0.3, 0.5, size)
    loading = np.linspace(0.7, 1.2, size)
    beta = np.linspace(-0.3, 0.4, size)
    if task != "graded":
        beta -= beta.mean()
    keying = np.array([1 if i % 2 == 0 else -1 for i in range(size)])
    utility = loading * keying * trait - beta
    probs = np.exp(reference_events(utility[None, :], group)[0])
    uniforms = probs.cumsum() - probs / 2

    class ControlledUniforms:
        def random(self, shape):
            assert shape == (n, 1)
            return uniforms[:, None]

    monkeypatch.setattr(
        "pyfcirt.simulation.np.random.default_rng", lambda seed: ControlledUniforms()
    )
    result = simulate_mgppm(
        design,
        pd.DataFrame(
            np.tile(trait, (n, 1)),
            index=[f"P{i}" for i in range(n)],
            columns=[f"T{i}" for i in range(size)],
        ),
        pd.Series(loading, index=[f"S{i}" for i in range(size)]),
        pd.Series(beta, index=[f"S{i}" for i in range(size)]),
        random_seed=31,
    )
    expected = list(group.events) if task == "graded" else [list(event) for event in group.events]
    assert result.to_frame().response.tolist() == expected


def test_generator_aligns_parameter_labels_and_respects_local_seed():
    design = make_design("rank", 3)
    theta = pd.DataFrame(
        [[0.4, -0.3, 0.2], [-0.1, 0.6, 0.5]], index=["A", "B"], columns=["T0", "T1", "T2"]
    )
    magnitude = pd.Series([0.8, 1.2, 0.7], index=["S0", "S1", "S2"])
    beta = pd.Series([0.1, -0.2, 0.1], index=magnitude.index)
    original = theta.copy()
    first = simulate_mgppm(design, theta, magnitude, beta, random_seed=29)
    second = simulate_mgppm(
        design, theta.iloc[:, ::-1], magnitude.iloc[::-1], beta.iloc[::-1], random_seed=29
    )
    assert_frame_equal(first.to_frame(), second.to_frame())
    assert_frame_equal(theta, original)


def test_generator_keeps_mixed_size_and_task_support():
    design = FCDesign.from_mapping(
        traits={f"T{i}": [f"S{i}"] for i in range(9)},
        blocks={
            "rank": ["S0", "S1", "S2"],
            "grade": ["S3", "S4"],
            "mole": ["S5", "S6", "S7", "S8"],
        },
        allow_variable_block_size=True,
        block_response_formats={"grade": "graded", "mole": "mole"},
        graded_scale={"categories": [0, 1, 2, 3, 4], "higher_prefers": "position_1"},
    )
    theta = pd.DataFrame(np.zeros((2, 9)), columns=[f"T{i}" for i in range(9)], index=["P", "Q"])
    magnitude = pd.Series(np.ones(9), index=[f"S{i}" for i in range(9)])
    result = simulate_mgppm(design, theta, magnitude, magnitude * 0, random_seed=17)
    assert result.summary()["n_observations"] == 6
    assert result.to_rank_frame().shape[0] == 2 * (3 + 4)
    assert len(result.to_graded_frame()) == 2
    assert all(
        result.to_frame()
        .query("block_id == 'mole'")
        .response.map(lambda response: response[1:-1] == [None, None])
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("loading", 0),
        ("loading", -1),
        ("loading", np.nan),
        ("theta", np.inf),
        ("intercept", np.nan),
    ],
)
def test_generator_rejects_invalid_fixed_parameters(field, value):
    design = make_design("rank", 2)
    theta = pd.DataFrame([[0.1, 0.2]], index=["P"], columns=["T0", "T1"])
    magnitude = pd.Series([1.0, 1.0], index=["S0", "S1"])
    beta = pd.Series([0.0, 0.0], index=magnitude.index)
    if field == "theta":
        theta.iloc[0, 0] = value
    elif field == "loading":
        magnitude.iloc[0] = value
    else:
        beta.iloc[0] = value
    with pytest.raises(ValueError):
        simulate_mgppm(design, theta, magnitude, beta)


def test_ranking_generator_rejects_unidentified_intercept_location():
    design = make_design("rank", 2)
    theta = pd.DataFrame([[0.1, 0.2]], index=["P"], columns=["T0", "T1"])
    magnitude = pd.Series([1.0, 1.0], index=["S0", "S1"])
    with pytest.raises(ValueError, match="sum to zero"):
        simulate_mgppm(design, theta, magnitude, magnitude * 0.1)


def test_nonfinite_fixed_logits_are_not_silently_clipped():
    group = group_for(make_design("rank", 2))
    with pytest.raises(ValueError, match="finite logits"):
        numpy_event_log_probs(np.array([[np.nan, 1]]), group.scores, group.membership)
