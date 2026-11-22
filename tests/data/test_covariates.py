"""Frozen covariate encodings, parameter pools and native conditional densities."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from pytensor.graph.replace import graph_replace
from scipy import stats

import pyfcirt as fc
from pyfcirt.covariates.design import apply_design, fit_design
from pyfcirt.covariates.model import RegressionBinding, build_regressions
from pyfcirt.data.coordinates import ParameterBlock
from pyfcirt.data.regressions import _resolve_coefficients, prepare_regressions
from pyfcirt.models.graph import ParameterFactory
from pyfcirt.models.priors import same_prior
from tests.helpers import replay_design, runtime_snapshot, supplied_posterior


def _normal_arguments(prior):
    node = prior.owner
    assert isinstance(node.op, pm.Normal.rv_type)
    return node.op.dist_params(node)


@pytest.mark.parametrize(
    ("model", "target"),
    [
        (fc.GPCM(), "category_intercept"),
        (fc.GGUM(), "threshold"),
        (fc.DichotomousIRT(kind="3PL"), "lower_asymptote"),
        (fc.DichotomousIRT(kind="1PL"), "loading_magnitude"),
    ],
)
def test_model_declarations_keep_unavailable_regression_targets_rejected(model, target):
    source = item_input(
        pd.DataFrame(dict(item_id=list("abc"), trait=["A"] * 3, location_side=[1] * 3)),
        model=model,
        categories=(0, 1) if isinstance(model, fc.DichotomousIRT) else (0, 1, 2),
    )
    analysis = fc.Model(fc.PreparedInputs.from_inputs(source))
    prepared = analysis.prepared
    with pytest.raises(ValueError):
        analysis.add_regressions(
            f"items.{target} ~ 1", item_data=pd.DataFrame({"item_id": list("abc")})
        )
    assert analysis.prepared is prepared and analysis._built is None


def test_scaled_statement_effect_keeps_its_original_regression_exclusion():
    from tests.helpers import mixed_inputs

    source = replace(mixed_inputs()[0], model=fc.TIRT())
    analysis = fc.Model(fc.PreparedInputs.from_inputs(source))
    prepared = analysis.prepared
    features = pd.DataFrame({"statement_id": prepared.constant_data.statement_label.values})
    with pytest.raises(ValueError, match="Unsupported parameter target"):
        analysis.add_regressions("fc.statement_effect ~ 1", statement_data=features)
    assert analysis.prepared is prepared and analysis._built is None


@pytest.fixture
def formulae():
    pytest.importorskip("formulae")


@pytest.fixture
def interaction_design(formulae):
    people = pd.DataFrame(
        {"pid": [4, 3, 2, 1], "age": [40.0, 30.0, 20.0, 10.0], "site": ["A", "B", "A", "B"]}
    )
    keys = pd.DataFrame({"pid": [1, 2, 3, 4]})
    spec = fc.Regression("age * site", people, id="pid", standardize_predictors=True)
    return fit_design(spec, keys), people, keys


def test_typed_id_alignment_and_frozen_interaction_replay(interaction_design):
    training, people, keys = interaction_design
    z = (np.array([10.0, 20.0, 30.0, 40.0]) - 25) / np.sqrt(125)
    expected = {"Intercept": np.ones(4), "age": z, "site[A]": [0, 1, 0, 1]}
    expected["age:site[A]"] = z * expected["site[A]"]
    assert set(training.term.values) == expected.keys()
    for term, values in expected.items():
        assert_allclose(training.X.sel(term=term), values)
    assert training.key_0.values.tolist() == [1, 2, 3, 4]
    assert [type(value) for value in training.key_0.values] == [int, int, int, int]
    assert_allclose(replay_design(training, people, keys), training.X)

    people.loc[:, "age"] = 999.0
    new = pd.DataFrame({"pid": [9], "age": [50.0], "site": ["A"]})
    replay = replay_design(training, new, new[["pid"]])
    assert replay.term.identical(training.term)
    assert_allclose(replay.sel(term="age"), [25 / np.sqrt(125)])
    assert_allclose(replay.sel(term="site[A]"), [1])
    assert_allclose(replay.sel(term="age:site[A]"), [25 / np.sqrt(125)])


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"site": "unknown"}, "unknown"),
        ({"age": np.nan}, "missing"),
        ({"age": np.inf}, "nonfinite"),
    ],
)
def test_single_person_replay_rejects_unknown_or_invalid_features(
    interaction_design, changes, message
):
    training, _, _ = interaction_design
    new = pd.DataFrame({"pid": [9], "age": [50.0], "site": ["B"]}).assign(**changes)
    with pytest.raises(ValueError, match=message):
        replay_design(training, new, new[["pid"]])


def test_composite_covariate_keys_preserve_large_integer_ids(formulae):
    ids = [2**80, 2**80 + 1]
    data = pd.DataFrame(
        {"block": ["b1", "b2", "b1"], "statement": [ids[0], ids[0], ids[1]], "x": [1.0, 2.0, 4.0]}
    )
    keys = data[["block", "statement"]].iloc[[2, 0, 1]]
    spec = fc.Regression("0 + x", data, id=("block", "statement"), center_predictors=False)
    training = fit_design(spec, keys)
    assert training.key_0.values.tolist() == ["b1", "b1", "b2"]
    assert training.key_1.values.tolist() == [ids[1], ids[0], ids[0]]
    assert_allclose(training.X.sel(term="x"), [4.0, 1.0, 2.0])
    xr.testing.assert_identical(replay_design(training, data, keys), training.X)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "boolean"])
def test_required_covariate_ids_are_not_coerced_or_repaired(formulae, failure):
    keys = pd.DataFrame({"pid": [1, 2]})
    people = pd.DataFrame({"pid": [1, 2], "age": [10.0, 30.0]}).astype({"pid": object})
    if failure == "missing":
        people = people.iloc[:1]
    elif failure == "duplicate":
        people = pd.concat([people, people.iloc[:1]], ignore_index=True)
    else:
        people.at[0, "pid"] = True
    with pytest.raises(ValueError, match="missing required IDs|duplicate IDs|nonempty string"):
        fit_design(fc.Regression("age", people, id="pid"), keys)


@pytest.mark.parametrize("formula", ["log(age)", "age ** 2", "age - site", "age / site"])
def test_formula_syntax_requires_explicit_feature_columns(formulae, formula):
    people = pd.DataFrame({"pid": [1, 2], "age": [10.0, 30.0], "site": ["A", "B"]})
    with pytest.raises(ValueError, match="Regression formulas"):
        fit_design(fc.Regression(formula, people, id="pid"), people[["pid"]])


def test_reference_rows_control_numeric_transform_without_recoding_categories(formulae):
    people = pd.DataFrame(
        {"pid": [1, 2, 3, 4], "x": [10.0, 20.0, 100.0, 200.0], "site": ["A", "A", "B", "B"]}
    )
    design = fit_design(
        fc.Regression("x + site", people, id="pid", standardize_predictors=True),
        people[["pid"]],
        reference_rows=[True, True, False, False],
        include_intercept=False,
    )
    assert "Intercept" not in design.term
    assert_allclose(design.X.sel(term="x"), [-1, 1, 17, 37])
    assert_allclose(design.X.sel(term="site[B]"), [0, 0, 1, 1])
    assert design.attrs["feature_records"][0]["center"] == 15
    assert design.attrs["feature_records"][0]["scale"] == 5
    assert design.attrs["prior_scale_divisors"] == {"x": 1.0, "site[B]": 1.0}
    raw = fit_design(
        fc.Regression("x + site", people, id="pid"),
        people[["pid"]],
        reference_rows=[True, True, False, False],
        include_intercept=False,
    )
    assert raw.attrs["prior_scale_divisors"] == {"x": 5.0, "site[B]": 1.0}


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("center_predictors", [False, True])
@pytest.mark.parametrize("include_intercept", [False, True])
def test_prior_scale_factors_follow_actual_formulae_columns(
    formulae, standardize, center_predictors, include_intercept
):
    frame = pd.DataFrame(
        {
            "id": range(8),
            "x": [-3.0, -1.0, 1.0, 3.0] * 2,
            "z": [-2, -2, 2, 2] * 2,
            "group": ["C", "C", "C", "A:B", "C", "C", "A:B", "A:B"],
            "site": ["M", "N"] * 4,
        }
    )
    spec = fc.Regression(
        "x * z + x * group + group * site",
        frame,
        id="id",
        standardize_predictors=standardize,
        center_predictors=center_predictors,
    )
    design = fit_design(spec, frame[["id"]], include_intercept=include_intercept)
    sx, sz = (1.0, 1.0) if standardize else (np.sqrt(5), 2.0)
    expected = {
        "x": sx,
        "z": sz,
        "x:z": sx * sz,
        "group[A:B]": 1.0,
        "x:group[A:B]": sx,
        "site[N]": 1.0,
        "group[A:B]:site[N]": 1.0,
    }
    if include_intercept:
        expected["Intercept"] = 1.0
    factors = design.attrs["prior_scale_divisors"]
    assert set(factors) == set(design.term.values) == expected.keys()
    assert_allclose([factors[term] for term in expected], list(expected.values()))
    no_intercept = fit_design(replace(spec, formula="0 + group + x:group"), frame[["id"]])
    expected = {
        "group[C]": 1.0,
        "group[A:B]": 1.0,
        "x:group[C]": sx,
        "x:group[A:B]": sx,
    }
    factors = no_intercept.attrs["prior_scale_divisors"]
    assert factors.keys() == expected.keys()
    assert_allclose([factors[term] for term in expected], list(expected.values()))


@pytest.mark.parametrize("center_predictors", [False, True])
def test_default_prior_unit_changes_preserve_predictions_and_jacobian_adjusted_density(
    formulae, center_predictors
):
    frame = pd.DataFrame(
        {
            "id": range(8),
            "x": np.arange(8) + 10.0,
            "z": [-2.0, -1.0, 2.0, 4.0] * 2,
            "group": ["A"] * 6 + ["B"] * 2,
        }
    )
    spec = fc.Regression("x * z + x * group", frame, id="id", center_predictors=center_predictors)
    raw = fit_design(spec, frame[["id"]])
    changed = fit_design(
        replace(spec, data=frame.assign(x=12 * frame.x, z=3 * frame.z)), frame[["id"]]
    )
    standardized = fit_design(replace(spec, standardize_predictors=True), frame[["id"]])
    base = [(fc.MGPPM(), "loading_magnitude")]
    normalized_beta = np.linspace(-0.4, 0.4, raw.sizes["term"])
    log_densities, predictions = [], []
    for design in (raw, changed, standardized):
        factors = np.array([design.attrs["prior_scale_divisors"][t] for t in design.term.values])
        records, roles, _, _ = _resolve_coefficients(spec, design, base)
        scales = np.array(
            [
                _normal_arguments(records[int(role.removeprefix("prior_"))])[1].eval()
                for role in roles[0]
            ]
        )
        beta = normalized_beta / factors
        predictions.append(design.X.values @ beta)
        log_densities.append(stats.norm.logpdf(beta, scale=scales).sum() - np.log(factors).sum())
    assert_allclose(predictions[0], predictions[1], atol=1e-12)
    assert_allclose(predictions[0], predictions[2], atol=1e-12)
    assert_allclose(log_densities, log_densities[0], atol=1e-12)
    factors = raw.attrs["prior_scale_divisors"]
    changed_factors = changed.attrs["prior_scale_divisors"]
    assert_allclose(changed_factors["x"], 12 * factors["x"])
    assert_allclose(changed_factors["x:z"], 36 * factors["x:z"])
    assert_allclose(changed_factors["x:group[B]"], 12 * factors["x:group[B]"])
    assert changed_factors["group[B]"] == factors["group[B]"] == 1.0
    balanced = fit_design(
        replace(spec, data=frame.assign(group=["A"] * 4 + ["B"] * 4)), frame[["id"]]
    )
    assert balanced.attrs["prior_scale_divisors"] == factors
    before = dict(raw.attrs["prior_scale_divisors"])
    new = frame.iloc[[0]].assign(id=99, x=100.0)
    replay = replay_design(raw, new, new[["id"]])
    assert replay.shape == (1, raw.sizes["term"])
    assert raw.attrs["prior_scale_divisors"] == before


@pytest.mark.parametrize("magnitude", [1e110, 1e-110])
def test_nonfinite_or_zero_interaction_prior_scale_requires_new_units(formulae, magnitude):
    frame = pd.DataFrame(
        {
            "id": range(6),
            "x": [-magnitude, magnitude, 0, 0, 0, 0],
            "y": [0, 0, -magnitude, magnitude, 0, 0],
            "z": [0, 0, 0, 0, -magnitude, magnitude],
        }
    )
    with pytest.raises(ValueError, match="prior scale.*adjust feature units"):
        fit_design(
            fc.Regression("0 + x:y:z", frame, id="id", center_predictors=False), frame[["id"]]
        )


@pytest.mark.parametrize("override", ["common", "x"])
def test_only_automatic_priors_convert_the_coefficient_sd(formulae, override):
    frame = pd.DataFrame({"id": [1, 2], "x": [-1.0, 1.0]})
    spec = fc.Regression("x", frame, id="id")
    design = fit_design(spec, frame[["id"]])
    design.attrs["prior_scale_divisors"]["x"] = 1e-310
    base = [(fc.MGPPM(), "loading_magnitude")]
    with pytest.raises(ValueError, match="automatic prior SD"):
        _resolve_coefficients(spec, design, base)
    prior = pm.Normal.dist(mu=0.2, sigma=0.7)
    records, roles, _, _ = _resolve_coefficients(
        replace(spec, priors={override: prior}), design, base
    )
    slope = list(design.term.values).index("x")
    assert same_prior(records[int(roles[0, slope].removeprefix("prior_"))], prior)


def item_input(design, *, model=None, categories=(0, 1)):
    labels = design.item_id.drop_duplicates().tolist()
    return fc.ItemInput(
        design,
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": ["p"] * len(labels), "item_id": labels, "response": [0] * len(labels)}
            ),
            format="item_long",
        ),
        model=fc.DichotomousIRT() if model is None else model,
        categories=categories,
    )


def test_item_trait_pools_exclude_fixed_and_inactive_slots(formulae):
    source = item_input(
        pd.DataFrame(
            {
                "item_id": ["pair", "pair", "7", "third"],
                "trait": ["A", "B", "A", "B"],
                "fix_loading": [1.0, np.nan, np.nan, np.nan],
            }
        )
    )
    spec = fc.Regression("1", pd.DataFrame({"item": ["third", "7", "pair"]}), id="item")
    prepared = prepare_regressions(
        fc.PreparedInputs.from_inputs(source), {("items", "loading_magnitude"): spec}
    )
    plan = prepared.regressions[0]
    target = plan.targets[0]
    assert plan.design.key_0.values.tolist() == ["pair", "7", "third"]
    assert_array_equal(target.local_position, [1, 2, 5])
    assert_array_equal(target.row_index, [0, 1, 2])
    assert_array_equal(target.pool_index, [0, 1, 0])
    assert plan.coefficient_positions.shape == (2, 1)
    assert target.noncentered is True
    assert not target.pool_index.flags.writeable
    (layout,) = prepared.blocks_by_kind["loading"]
    assert_array_equal(np.flatnonzero(layout.entry_index.ravel() < 0), [3, 4])
    canonical = next(value for value in prepared.parameters if value.kind == "loading")
    assert canonical.shape == (4,)
    assert_array_equal(canonical.fixed_flat, [0])
    assert_array_equal(canonical.fixed_value, [1])
    assert_array_equal(layout.entry_index, [[0, 1], [2, -1], [-1, 3]])
    residual = next(layout for layout in plan.blocks if layout.key == "residual_sd")
    assert same_prior(plan.priors[residual.prior_role], pm.HalfNormal.dist(sigma=0.2))
    assert residual.prior_role_index is None

    bindings, _ = build_regressions(
        (plan,),
        {
            "regression_coefficient": pt.as_tensor_variable([0.25, 0.75]),
            "regression_sd": pt.as_tensor_variable([0.5, 1.5]),
        },
    )
    binding = bindings[target.binding_id, target.prior_role]
    mean, scale = pytensor.function([], [binding.mean, binding.scale], mode="JAX")()
    assert_allclose(mean, [0.25, 0.75, 0.25])
    assert_allclose(scale, [0.5, 1.5, 0.5])


@pytest.mark.parametrize("channel", ["fc", "items"])
@pytest.mark.parametrize("family", ["Normal", "LogNormal", "TruncatedNormal"])
@pytest.mark.parametrize("standardize", [False, True])
def test_loading_slope_defaults_follow_target_role_and_explicit_priors_take_precedence(
    formulae, channel, family, standardize
):
    from tests.helpers import mixed_inputs

    base = getattr(pm, family).dist(
        mu=0.4, sigma=0.8, **{"lower": 0} if family == "TruncatedNormal" else {}
    )
    if channel == "fc":
        source = replace(mixed_inputs()[0], model=fc.MGPPM(priors={"loading_magnitude": base}))
        labels = list("abcdefghi")
    else:
        labels = list("abcd")
        source = item_input(
            pd.DataFrame({"item_id": labels, "trait": ["A"] * len(labels)}),
            model=fc.DichotomousIRT(priors={"loading_magnitude": base}),
        )
    with pytest.warns(fc.IdentificationWarning) if family == "Normal" else nullcontext():
        state = fc.PreparedInputs.from_inputs(source)
    features = pd.DataFrame(
        {"id": labels, "x": np.arange(len(labels)), "z": np.arange(len(labels)) ** 2}
    )
    spec = fc.Regression("x + z", features, id="id", standardize_predictors=standardize)
    for overrides in (
        {},
        {"common": pm.Normal.dist(mu=0.3, sigma=0.7)},
        {
            "common": pm.Normal.dist(mu=0.3, sigma=0.7),
            "x": pm.Normal.dist(mu=-0.2, sigma=0.8),
            "Intercept": pm.StudentT.dist(nu=5, mu=0.4, sigma=0.9),
        },
    ):
        plan = prepare_regressions(
            state, {(channel, "loading_magnitude"): replace(spec, priors=overrides)}
        ).regressions[0]
        expected = {
            "Intercept": overrides.get("Intercept", pm.Normal.dist(mu=0.0, sigma=0.5)),
            "x": overrides.get(
                "x",
                overrides.get(
                    "common",
                    pm.Normal.dist(
                        mu=0, sigma=0.25 / plan.design.attrs["prior_scale_divisors"]["x"]
                    ),
                ),
            ),
            "z": overrides.get(
                "common",
                pm.Normal.dist(mu=0, sigma=0.25 / plan.design.attrs["prior_scale_divisors"]["z"]),
            ),
        }
        roles = plan.blocks[0].prior_role_index.reshape(plan.coefficient_positions.shape)
        for column, term in enumerate(plan.design.term.values):
            assert all(same_prior(plan.priors[role], expected[term]) for role in roles[:, column])
        residual = next(layout for layout in plan.blocks if layout.key == "residual_sd")
        assert same_prior(plan.priors[residual.prior_role], pm.HalfNormal.dist(sigma=0.2))
        assert residual.prior_role_index is None


def _regression_pool_uses(state, plan):
    coefficient = next(
        layout for layout in state.parameters if layout.kind == "regression_coefficient"
    )
    uses = state.parameter_uses
    mask = (uses.kind.values == coefficient.kind) & np.isin(
        uses.entry_index, plan.coefficient_entries
    )
    pool = coefficient.entry_data.pool_index.values[uses.entry_index.values[mask]]
    return np.unique(
        np.column_stack((pool, uses.unit_index.values[mask], uses.trait_index.values[mask])), axis=0
    )


def test_shared_statement_training_sd_uses_distinct_free_entities(formulae):
    source = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a"], "B": ["b", "c"]},
            blocks={"one": ["a", "b"], "two": ["a", "c"]},
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": ["p", "p"], "block_id": ["one", "two"], "response": [[1, 2], [2, 1]]}
            ),
            format="block_long",
        ),
    )
    features = pd.DataFrame({"id": list("abc"), "x": [10.0, 30.0, 50.0]})
    spec = fc.Regression("x", features, id="id")
    state = prepare_regressions(
        fc.PreparedInputs.from_inputs(source), {("fc", "loading_magnitude"): spec}
    )
    plan = state.regressions[0]
    assert plan.design.sizes["row"] == 3
    assert_allclose(plan.design.attrs["prior_scale_divisors"]["x"], np.std([10, 30, 50], ddof=0))
    target = plan.targets[0]
    assert_array_equal(target.local_position, [0, 1, 2])
    assert target.unit_indices == (0, 1)
    assert_array_equal(_regression_pool_uses(state, plan), [(0, 0, -1), (0, 1, -1)])


@pytest.mark.parametrize("noncentered", [None, False, True])
def test_mixed_location_sides_resolve_noncentering_per_role(formulae, noncentered):
    source, features = location_input()
    spec = fc.Regression("x", features, id="item_id", deviation_noncentered=noncentered)
    state = fc.PreparedInputs.from_inputs(source)
    if noncentered is True:
        with pytest.raises(NotImplementedError, match="centered"):
            prepare_regressions(state, {("items", "location"): spec})
        return
    plan = prepare_regressions(state, {("items", "location"): spec}).regressions[0]
    assert {target.prior_role for target in plan.targets} == {
        "location_positive",
        "location_negative",
        "location_free",
    }
    for target in plan.targets:
        assert target.noncentered is (noncentered is None and target.prior_role == "location_free")
        assert target.conditional["family"] == (
            "Normal" if target.prior_role == "location_free" else "TruncatedNormal"
        )
    roles = plan.blocks[0].prior_role_index.reshape(plan.coefficient_positions.shape)
    assert_allclose(
        [_normal_arguments(plan.priors[role])[0].eval() for role in roles[:, 0]], [1, -1, 0]
    )
    assert all(
        same_prior(plan.priors[role], pm.Normal.dist(mu=0.0 if center == 0 else center, sigma=1.0))
        for role, center in zip(roles[:, 0], [1, -1, 0], strict=True)
    )
    assert all(
        same_prior(plan.priors[role], pm.Normal.dist(mu=0, sigma=1 / np.std(features.x)))
        for role in roles[:, 1]
    )


@pytest.mark.parametrize("residual_sd", [None, pm.HalfStudentT.dist(nu=4, sigma=0.4), 0.6])
@pytest.mark.parametrize("key", ["loading_magnitude", "intercept"])
def test_residual_priors_and_fixed_values_apply_to_all_pools(formulae, residual_sd, key):
    source = item_input(
        pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}),
        model=fc.DichotomousIRT(priors={key: pm.Normal.dist(mu=0, sigma=1)}),
    )
    source = replace(
        source,
        items=(
            fc.ItemOverride(
                "b", model=fc.DichotomousIRT(priors={key: pm.LogNormal.dist(sigma=0.7)})
            ),
        ),
    )
    spec = fc.Regression(
        "1", pd.DataFrame({"id": list("abc")}), id="id", parameter_residual_sd=residual_sd
    )
    with pytest.warns(fc.IdentificationWarning) if key == "loading_magnitude" else nullcontext():
        state = fc.PreparedInputs.from_inputs(source)
    prepared = prepare_regressions(state, {("items", key): spec})
    plan = prepared.regressions[0]
    residual = next(layout for layout in plan.blocks if layout.key == "residual_sd")
    assert residual.shape == (2,)
    canonical = prepared.parameter_layouts["regression_sd"]
    assert_array_equal(canonical.entry_data.domain, ["positive", "positive"])
    if isinstance(residual_sd, float):
        assert residual.free_flat.size == 0
        assert_allclose(canonical.fixed_value, [0.6, 0.6])
        assert all(target.conditional["residual_sd_source"] == "fixed" for target in plan.targets)
    else:
        assert_array_equal(residual.free_flat, [0, 1])
        expected = (
            pm.HalfNormal.dist(sigma=0.2 if key == "loading_magnitude" else 1.0)
            if residual_sd is None
            else residual_sd
        )
        assert same_prior(plan.priors[residual.prior_role], expected)
        from pyfcirt.data.prepare import _validate_parameters

        point = xr.Dataset(
            {"regression_sd": (canonical.dims, [0.2, 0.3])},
            coords={dim: prepared.coords[dim] for dim in canonical.dims},
        )
        assert_allclose(_validate_parameters(prepared, point).regression_sd, [0.2, 0.3])
        for invalid in (0.0, -0.2):
            point.regression_sd.values[0] = invalid
            with pytest.raises(ValueError, match="regression_sd: values must be positive"):
                _validate_parameters(prepared, point)
    assert {target.conditional["family"] for target in plan.targets} == (
        {"LogNormal"} if key == "loading_magnitude" else {"Normal"}
    )


def cutpoint_input():
    rows = [
        (block, position, statement, trait, 1, 0)
        for block, statements in zip(
            ("one", "two", "three", "other"), ("ab", "cd", "ef", "gh"), strict=True
        )
        for position, (statement, trait) in enumerate(zip(statements, ("A", "B"), strict=True), 1)
    ]
    source = fc.FCInput(
        pd.DataFrame(
            rows,
            columns=["block_id", "position", "statement_id", "trait", "keying", "location_side"],
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p"] * 4,
                    "block_id": ["one", "two", "three", "other"],
                    "response": [0] * 4,
                }
            ),
            format="block_long",
        ),
        model=fc.GTUM(),
        response_format=None,
        scale=fc.CategoryScale((0, 1)),
        blocks=(
            fc.BlockSpec("three", scale=fc.CategoryScale((0, 1, 2))),
            fc.BlockSpec(
                "other",
                scale=fc.CategoryScale((0, 1, 2)),
                model=fc.GTUM(priors={"cutpoints": pm.Normal.dist(mu=0.7, sigma=0.8)}),
            ),
        ),
    )
    features = pd.DataFrame(
        {
            "block": ["one", "two", "three", "other"],
            "first": list("bceg"),
            "second": list("adfh"),
            "x": [1.0, 3.0, 4.0, 6.0],
        }
    )
    return source, features


@pytest.mark.parametrize("baseline_first", [True, False])
def test_person_cutpoint_effect_uses_tirt_pair_rows_and_preserves_order(formulae, baseline_first):
    source, pair_info = cutpoint_input()
    responses = pd.concat(
        [source.responses.frame.assign(person_id=person) for person in ("p1", "p2", "p3", "p4")],
        ignore_index=True,
    )
    source = replace(
        source,
        responses=fc.ResponseTable(responses, format="block_long"),
        model=fc.TIRT(loading_anchors={"a": 0.9, "b": 1.1}),
        blocks=tuple(replace(block, model=None) for block in source.blocks),
    )
    model = fc.Model(source)
    people = pd.DataFrame({"person_id": ["p4", "p3", "p2", "p1"], "site": ["B", "B", "A", "A"]})
    target = ("fc", "cutpoints")
    effect = fc.Regression("site", people, id="person_id", parameter_residual_sd=0.5)

    def add_effect(spec):
        model._update_parameters(prepare_regressions, [(target, spec)], person_effects=True)

    def add_baseline():
        model.add_regressions(
            "fc.cutpoints ~ 1",
            pair_data=pair_info,
            id_columns={"pair": ("block", "first", "second")},
        )

    if baseline_first:
        add_baseline()
        add_effect(effect)
    else:
        add_effect(effect)
        add_baseline()
    prepared = model.prepared
    effect_plan = next(plan for plan in prepared.regressions if plan.person_effects)
    native = model.build()
    model_graph = model._bindings
    for binding in model_graph.responses:
        batch = binding.batch
        cuts = next(slot for slot in batch.slots if slot.key == "cutpoints")
        (offset,) = cuts.effects
        ((_, lookup, _),) = prepared.person_effect_pools[cuts.kind]
        assert not any(
            array.flags.writeable for array in (lookup, offset.person, offset.pool, offset.position)
        )
        assert_array_equal(offset.person, np.repeat(batch.person, cuts.entry_index.shape[-1]))
        assert effect_plan.design.key_0.values.tolist() == ["p1", "p2", "p3", "p4"]
        baseline = model_graph.parameters[cuts.kind]
        beta = model_graph.parameters["regression_coefficient"]
        baseline_values = np.arange(
            prepared.parameter_layouts[cuts.kind].fixed.size, dtype=baseline.dtype
        )
        for coefficient in (0, 2):
            fixed = {
                baseline: pt.as_tensor_variable(baseline_values),
                beta: pt.ones_like(beta) * coefficient,
            }
            actual = pm.draw(
                graph_replace(binding.parameters[1], fixed, strict=False), random_seed=7
            )
            deviation = effect_plan.design.X.values[batch.person, :] * (0.5 * coefficient)
            assert_allclose(actual, cuts.entry_index + deviation)
            assert np.all(np.diff(actual, axis=-1) > 0)
    assert np.isfinite(native.compile_logp(mode="JAX")(native.initial_point(random_seed=7)))

    with pytest.raises(ValueError, match="unknown|missing|absent|unmatched"):
        add_effect(replace(effect, data=people.iloc[:-1]))
    assert model.prepared is prepared and model.build() is native
    with pytest.raises(ValueError, match="linearly dependent"):
        add_effect(replace(effect, formula="1 + site"))
    assert model.prepared is prepared and model.build() is native
    add_effect(replace(effect, parameter_residual_sd=0.25))
    assert len(model.prepared.regressions) == 2

    from pyfcirt.data.prepare import _prepare_generation, _prepare_scoring

    new = pd.DataFrame({"person_id": ["new2", "new1"], "site": ["A", "B"]})
    schedule = fc.generation_schedule(model, ["new1", "new2"], person_data=new)
    generated = _prepare_generation(model.prepared, schedule)
    answers = fc.ResponseTable(
        pd.concat(
            [responses.iloc[:4].assign(person_id=person) for person in ("new1", "new2")],
            ignore_index=True,
        ),
        format="block_long",
    )
    scored = _prepare_scoring(model.prepared, answers, person_data=new)
    assert (
        generated.calibration_identity
        == scored.calibration_identity
        == model.prepared.calibration_identity
    )
    assert_array_equal(
        generated.person_effect_regressions[0].design.X,
        scored.person_effect_regressions[0].design.X,
    )
    assert_array_equal(scored.person_effect_regressions[0].design.X[:, 0], [1, 0])
    assert {kind for kind in scored.point_layouts if kind.startswith("regression_")} == {
        "regression_coefficient",
        "regression_sd",
    }
    changed = _prepare_scoring(model.prepared, answers, person_data=new.assign(site=["B", "A"]))
    assert changed.scoring_batch_identity != scored.scoring_batch_identity
    with pytest.raises(ValueError, match="unknown training categories"):
        _prepare_scoring(model.prepared, answers, person_data=new.assign(site=["C", "A"]))


@pytest.mark.parametrize("cutoff", [None, 0.5])
def test_pairwise_dif_keeps_joint_pools_and_replays_category_labels(formulae, cutoff):
    from scipy.linalg import helmert

    from pyfcirt.data.prepare import _prepare_generation

    source, _ = cutpoint_input()
    categories = pd.Categorical(["north", "west", "east"], categories=["east", "north", "west"])
    people = pd.DataFrame({"person_id": ["p0", "p1", "p2"], "site": categories})
    responses = pd.concat(
        [source.responses.frame.assign(person_id=person) for person in people.person_id],
        ignore_index=True,
    )
    source = replace(
        source,
        responses=fc.ResponseTable(responses, format="block_long"),
        blocks=(
            fc.BlockSpec("one", model=fc.TIRT(loading_anchors={"a": 0.9, "b": 1.1})),
            *source.blocks,
        ),
    )
    model = fc.Model(source)
    target = ("fc", "cutpoints")
    term = fc.Regression("site", people, id="person_id")
    options = {"strength": 1.5, "cutoff": cutoff}
    model._update_parameters(
        prepare_regressions,
        [(target, term)],
        person_effects=True,
        dif_options={(target, "site"): {"pairwise": options}},
    )
    (plan,) = model.prepared.person_effect_regressions
    (block,) = plan.blocks
    assert block.prior_shapes["prior_0"] == [4, 2]
    basis = helmert(3).T
    assert_allclose(plan.design.X.values, basis[[1, 2, 0]])
    native = model.build()
    assert np.isfinite(native.compile_logp(mode="JAX")(native.initial_point(random_seed=17)))
    assert pm.draw(native["regression_coefficient"], random_seed=41).shape == (8,)
    new_people = pd.DataFrame({"person_id": ["pX", "pY"], "site": ["east", "west"]})
    schedule = fc.generation_schedule(model, ["pY", "pX"], person_data=new_people)
    replay = _prepare_generation(model.prepared, schedule)
    assert_allclose(replay.person_effect_regressions[0].design.X.values, basis[[2, 0]])
    assert replay.calibration_identity == model.prepared.calibration_identity
    reordered = people.copy()
    reordered["site"] = reordered["site"].cat.reorder_categories(["west", "north", "east"])
    incompatible = fc.Model(source)
    incompatible._update_parameters(
        prepare_regressions,
        [(target, replace(term, data=reordered))],
        person_effects=True,
        dif_options={(target, "site"): {"pairwise": options}},
    )
    assert_array_equal(
        incompatible.prepared.person_effect_regressions[0].design.term,
        plan.design.term,
    )
    with pytest.raises(ValueError, match="different frozen design"):
        _prepare_generation(incompatible.prepared, schedule)

    # A more specific replacement keeps complete events in the original pools.
    tirt = ("fc", "TIRT", "cutpoints")
    model._update_parameters(prepare_regressions, [(tirt, term)], person_effects=True)
    retained, replaced = model.prepared.person_effect_regressions
    (retained_block,) = retained.blocks
    assert retained_block.prior_shapes["prior_0"] == [3, 2]
    assert replaced.design.term.values.tolist() == ["site[north]", "site[west]"]
    assert_allclose(retained.design.X, plan.design.X)


def test_dif_summary_uses_common_baseline_and_held_predictor_on_parameter_scale(formulae):
    from pyfcirt.analysis.dif import summarize_dif
    from tests.helpers import supplied_posterior

    source = item_input(pd.DataFrame({"item_id": ["a", "b"], "trait": ["A", "A"]}))
    people = pd.DataFrame(
        {"person_id": ["p1", "p2", "p3", "p4"], "site": ["A", "A", "B", "B"], "x": [0, 1, 0, 1]}
    )
    responses = pd.concat(
        [source.responses.frame.assign(person_id=person) for person in people.person_id],
        ignore_index=True,
    )
    model = fc.Model(replace(source, responses=replace(source.responses, frame=responses)))
    target = ("items", "loading_magnitude")
    terms = [
        (target, fc.Regression(name, people, id="person_id", priors={"common": pm.Normal.dist()}))
        for name in ("site", "x")
    ]
    model._update_parameters(prepare_regressions, terms, person_effects=True)
    assert "regression_sd" not in model.prepared.parameter_layouts
    original = fc.prior_predictive(model, draws=1, random_seed=49, store=False)
    point = original.prior.dataset.isel(chain=0, draw=0, drop=True)
    site, x = model.prepared.person_effect_regressions
    coefficients = np.zeros_like(point.regression_coefficient.values)
    coefficients[site.coefficient_entries.ravel()] = [np.log(2), 0.0]
    coefficients[x.coefficient_entries.ravel()] = np.log(3)
    point = point.assign(
        regression_coefficient=point.regression_coefficient.copy(data=coefficients),
        loading=point.loading.copy(data=np.asarray([1.5, 2.0])),
    )
    supplied_posterior(model, point, inference_method="mcmc")
    summary = summarize_dif(model, contrasts={"site": ("A", "B")}, at={"x": 1.5})
    assert summary.predictor.tolist() == ["site", "site"]
    assert summary.units.tolist() == [("a",), ("b",)]
    assert summary.detected.tolist() == [True, False]
    assert_allclose(summary[["mean", "lower", "upper"]], [[4.5] * 3, [0.0] * 3])
    assert summarize_dif(model, contrasts={"site": ("A", "B")}, significant_only=True).shape[0] == 1


def test_dif_public_api_handles_crossed_groups_multiple_targets_and_overrides(formulae):
    source = item_input(pd.DataFrame({"item_id": ["a", "b"], "trait": ["A", "A"]}))
    people = pd.DataFrame(
        {
            "person_id": [f"p{i}" for i in range(8)],
            "site": ["north"] * 4 + ["west"] * 4,
            "language": ["en", "en", "zh", "zh"] * 2,
            "age": [25, 44, 21, 35, 40, 30, 19, 56],
        }
    )
    responses = pd.concat(
        [source.responses.frame.assign(person_id=person) for person in people.person_id],
        ignore_index=True,
    )
    model = fc.Model(replace(source, responses=replace(source.responses, frame=responses)))
    with pytest.raises(ValueError, match="base model provides the intercept"):
        model.add_dif_terms(
            "items.intercept",
            person_data=people,
            groups="site",
            prior_options={"priors": {"Intercept": pm.Normal.dist()}},
        )
    model.add_dif_terms(
        ["items.loading_magnitude", "items.intercept"],
        person_data=people,
        groups=[("site", "language")],
        formula="age",
        overrides={"items.intercept": {"prior": "truncated_l1", "pooling": "shared"}},
    )
    designs = [plan.design.X.copy() for plan in model.prepared.person_effect_regressions]
    model.set_priors({"items.intercept": pm.Normal.dist(sigma=0.3)})
    plans = model.prepared.person_effect_regressions
    for plan, original in zip(plans, designs, strict=True):
        xr.testing.assert_identical(plan.design.X, original)
    assert len(plans) == 4
    joint = next(plan for plan in plans if "group_basis" in plan.design.attrs)
    assert joint.coefficient_entries.shape == (1, 3)
    assert "DIF: item_intercept ~ site:language" in model.summary()
    assert "conditional Normal" not in model.summary()
    assert joint.design.attrs["group_levels"] == (
        ("north", "en"),
        ("north", "zh"),
        ("west", "en"),
        ("west", "zh"),
    )
    native = model.build()
    assert np.isfinite(native.compile_logp(mode="JAX")(native.initial_point()))
    predictions = fc.prior_predictive(model, draws=3, random_seed=22, store=False)
    posterior = predictions.prior.dataset
    model.attach_samples(xr.DataTree.from_dict({"posterior": posterior}))
    result = model.dif_summary(
        {"site:language": (("north", "en"), ("west", "zh"))},
        at={"age": 35},
    )
    assert set(result.parameter) == {"loading", "item_intercept"}
    assert result.reference.tolist() == [("north", "en")] * 4
    assert result.compared.tolist() == [("west", "zh")] * 4
    assert set(result.units) == {("a",), ("b",)}
    assert np.isfinite(result["mean"]).all()

    new_features = people.iloc[[2, 4]].assign(person_id=["new-zh", "new-en"])
    new_responses = responses.loc[responses.person_id.isin(["p2", "p4"])].assign(
        person_id=lambda frame: frame.person_id.map({"p2": "new-zh", "p4": "new-en"})
    )
    scoring = fc.score_model(
        model,
        fc.ResponseTable(new_responses, format=source.responses.format),
        person_data=new_features,
    )
    assert {rv.name for rv in scoring.build().free_RVs} == {"theta"}
    for training, replayed in zip(plans, scoring.prepared.person_effect_regressions, strict=True):
        assert_allclose(replayed.design.X, training.design.X.values[[2, 4]])


def test_dif_summary_encodes_only_profiles_used_by_each_target(formulae):
    source = item_input(pd.DataFrame({"item_id": ["a", "b"], "trait": ["A", "A"]}))
    people = pd.DataFrame(
        {
            "person_id": list("abcdef"),
            "site": ["A", "A", "A", "A", "B", "B"],
            "language": ["en", "en", "zh", "zh", "zh", "zh"],
        }
    )
    responses = pd.concat(
        [source.responses.frame.assign(person_id=person) for person in people.person_id],
        ignore_index=True,
    )
    model = fc.Model(replace(source, responses=replace(source.responses, frame=responses)))
    model.add_dif_terms(
        "items.loading_magnitude", person_data=people, groups=[("site", "language")]
    )
    model.add_dif_terms("items.intercept", person_data=people, groups="site")
    prior = fc.prior_predictive(model, draws=3, random_seed=19, store=False)
    model.attach_samples(xr.DataTree.from_dict({"posterior": prior.prior.dataset}))

    result = model.dif_summary()
    assert result.groupby("predictor").parameter.unique().map(list).to_dict() == {
        "site": ["item_intercept"],
        "site:language": ["loading"],
    }
    assert len(result) == 8
    with pytest.raises(ValueError, match="Unknown training combinations"):
        model.dif_summary({"site:language": (("A", "en"), ("B", "en"))})


def test_dif_default_contrasts_respect_each_targets_fitted_levels(formulae):
    source = item_input(pd.DataFrame({"item_id": ["a", "b"], "trait": ["A", "A"]}))
    people = pd.DataFrame({"person_id": list("abcdef"), "site": list("AABBCC")})
    responses = pd.concat(
        [source.responses.frame.assign(person_id=person) for person in people.person_id],
        ignore_index=True,
    )
    model = fc.Model(replace(source, responses=replace(source.responses, frame=responses)))
    model.add_dif_terms("items.loading_magnitude", person_data=people, groups="site")
    model.add_dif_terms("items.intercept", person_data=people.replace({"C": "B"}), groups="site")
    prior = fc.prior_predictive(model, draws=3, random_seed=19, store=False)
    model.attach_samples(xr.DataTree.from_dict({"posterior": prior.prior.dataset}))

    result = model.dif_summary()
    assert len(result) == 8
    assert set(
        result[["parameter", "reference", "compared"]].itertuples(index=False, name=None)
    ) == {
        ("loading", "A", "B"),
        ("loading", "A", "C"),
        ("loading", "B", "C"),
        ("item_intercept", "A", "B"),
    }
    explicit = model.dif_summary({"site": ("A", "B")})
    common = result.loc[(result.reference == "A") & (result.compared == "B")]
    pd.testing.assert_frame_equal(explicit, common.reset_index(drop=True))


def test_mixed_measurement_dif_can_share_covariates_with_theta(formulae):
    from tests.helpers import mixed_inputs

    people = pd.DataFrame(
        {
            "person_id": np.arange(1, 9),
            "site": ["A", "A", "B", "B"] * 2,
            "language": ["en", "zh"] * 4,
        }
    )
    channels = []
    for source in mixed_inputs():
        original = source.responses.frame
        responses = pd.concat(
            [original[original.person_id == 1].assign(person_id=p) for p in people.person_id],
            ignore_index=True,
        )
        channels.append(replace(source, responses=replace(source.responses, frame=responses)))
    model = fc.Model().prepare(
        tuple(channels),
        regressions={"theta": fc.Regression("site", people, id="person_id")},
    )
    model.add_dif_terms(
        ["fc.intercept", "items.intercept"],
        person_data=people,
        groups=["site", "language"],
        overrides={"fc.MGPPM.intercept": {"prior": "pairwise_l1"}},
    )
    prepared = model.prepared
    assert prepared.theta_regression.design.attrs["formula"] == "site"
    assert len(prepared.person_effect_regressions) == 4
    assert {
        target.kind for plan in prepared.person_effect_regressions for target in plan.targets
    } == {"statement_intercept", "item_intercept"}
    native = model.build()
    assert np.isfinite(native.compile_logp(mode="JAX")(native.initial_point()))
    with pytest.raises(ValueError, match="linearly dependent"):
        model.add_dif_terms("items.intercept", person_data=people, formula="1 + site")
    assert model.prepared is prepared and model.build() is native


@pytest.mark.parametrize("change_units", [1.0, 1000.0])
def test_person_effect_slopes_scale_with_training_units_and_keep_explicit_priors(
    formulae, change_units
):
    people = pd.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "x": np.array([-2.0, 0.0, 1.0, 3.0]) * change_units,
            "site": ["A", "A", "B", "B"],
        }
    )
    spec = fc.Regression("x * site", people, id="pid", priors={"site[B]": pm.Normal.dist(sigma=2)})
    matrix = fit_design(spec, people[["pid"]], include_intercept=False)
    priors, roles, _, standardized = _resolve_coefficients(
        spec, matrix, [(fc.TIRT(), "cutpoints")], person_effects=True
    )
    terms = matrix.term.values.tolist()
    assert terms == ["x", "site[B]", "x:site[B]"]
    assert_array_equal(standardized[0], [True, False, True])
    effects = np.column_stack(
        [
            matrix.X[:, i] * _normal_arguments(priors[int(role.split("_")[1])])[1].eval()
            for i, role in enumerate(roles[0])
        ]
    )
    x = np.array([-2.0, 0.0, 1.0, 3.0])
    z = (x - x.mean()) / x.std()
    b = np.array([0.0, 0.0, 1.0, 1.0])
    assert_allclose(effects, np.column_stack((z, 2 * b, z * b)))


def test_available_covariate_targets_follow_mixed_design_and_fixed_roles(formulae):
    source, _ = cutpoint_input()
    source = replace(
        source,
        model=fc.TIRT(residuals="fixed", loading_anchors={"a": 0.9, "b": 1.1}),
        blocks=(fc.BlockSpec("three", model=fc.GTUM()), source.blocks[1]),
    )
    item = item_input(
        pd.DataFrame({"item_id": ["a", "b"], "trait": ["A", "A"]}),
        model=fc.DichotomousIRT(kind="1PL"),
    )
    model = fc.Model().prepare((source, item))
    before = model.prepared
    targets = model.available_covariate_targets()
    assert model.prepared is before
    assert set(model.available_covariate_targets(level="person").target) == {"theta"}
    assert set(model.available_covariate_targets(level="item").target) == {"items.intercept"}
    assert "fc.residual_variance" not in set(targets.target)
    assert model.available_covariate_targets(level="person", kind="dif").empty
    with pytest.raises(ValueError, match="level must be"):
        model.available_covariate_targets(level="cutpoint")
    with pytest.raises(ValueError, match="kind must be"):
        model.available_covariate_targets(kind="unknown")

    statement = model.available_covariate_targets(level="statement", kind="regression")
    target = ("fc", "GTUM", "loading_magnitude")
    gtum = statement.loc[statement.target.map(lambda value: value == target)].iloc[0]
    assert gtum.predictor_level == "statement"
    features = pd.DataFrame({"statement_id": list("abcdefgh"), "length": np.arange(8)})
    model.add_regressions({gtum.target: fc.Regression("length", features, id="statement_id")})
    (plan,) = model.prepared.regressions
    assert {
        model.prepared.constant_data.unit_model.values[unit]
        for bound in plan.targets
        for unit in bound.unit_indices
    } == {"GTUM"}

    pairs = model.available_covariate_targets(level="pair", kind="dif")
    assert set(pairs.target) == {"fc.TIRT.cutpoints", "fc.GTUM.cutpoints"}
    assert set(pairs.predictor_level) == {"person"}


def test_model_qualified_priors_share_one_formula_and_preserve_other_models(formulae):
    source, _ = cutpoint_input()
    tirt = fc.TIRT(loading_anchors={"a": 0.9, "b": 1.1})
    source = replace(
        source,
        model=tirt,
        blocks=(fc.BlockSpec("three", model=fc.GTUM()), source.blocks[1]),
    )
    model = fc.Model(source)
    features = pd.DataFrame({"statement_id": list("abcdefgh"), "length": np.arange(8)})
    common = pm.Normal.dist(mu=0.5, sigma=0.3)
    tirt_slope = pm.Normal.dist(mu=0.2, sigma=0.1)
    gtum_slope = pm.Normal.dist(mu=-0.2, sigma=0.4)
    formula = "fc.loading_magnitude ~ length"
    options = {
        "fc.loading_magnitude": {"priors": {"Intercept": common, "length": common}},
        "fc.TIRT.loading_magnitude": {"priors": {"length": tirt_slope}},
        "fc.GTUM.loading_magnitude": {"priors": {"common": gtum_slope}},
    }
    model.add_regressions(formula, statement_data=features, options=options)
    before = model.prepared
    initial = fc.PreparedInputs.from_inputs(
        source, regressions=formula, statement_data=features, regression_options=options
    )
    (initial_plan,) = initial.regressions
    (plan,) = before.regressions
    for role, prior in plan.priors.items():
        assert same_prior(initial_plan.priors[role], prior)
    assert plan.design.term.values.tolist() == ["Intercept", "length"]
    for target in plan.targets:
        (family,) = set(before.constant_data.unit_model.values[list(target.unit_indices)])
        expected = tirt_slope if family == "TIRT" else gtum_slope
        for pool in np.unique(target.pool_index):
            roles = plan.blocks[0].prior_role_index[plan.coefficient_positions[pool]]
            assert same_prior(plan.priors[roles[0]], common)
            assert same_prior(plan.priors[roles[1]], expected)
    loading = before.parameter_layouts["loading"]
    assert_allclose(loading.fixed_value, [0.9, 1.1])
    native = model.build()
    assert np.isfinite(native.compile_logp(mode="JAX")(native.initial_point(random_seed=45)))
    with pytest.raises(ValueError, match="No applicable free parameter targets"):
        model.add_regressions(
            "fc.loading_magnitude ~ length",
            statement_data=features,
            options={"fc.TIRTRank.loading_magnitude": {"priors": {"length": tirt_slope}}},
        )
    assert model.prepared is before and model.build() is native

    ordinary = pm.LogNormal.dist(mu=0.4, sigma=0.25)
    model.set_priors({"fc.TIRT.loading_magnitude": ordinary})
    (after,) = model.prepared.regressions
    assert after.design.attrs["encoder"] is plan.design.attrs["encoder"]
    assert {unit for target in after.targets for unit in target.unit_indices} == {2, 3}
    tirt_binding = next(block for block in model.prepared.blocks_by_kind["loading"] if block.priors)
    assert same_prior(tirt_binding.priors["loading_magnitude"], ordinary)
    np.testing.assert_array_equal(model.prepared.coords["loading_id"], before.coords["loading_id"])


def test_mixed_model_prior_replacement_compacts_middle_pool_and_preserves_labels(formulae):
    source, features = cutpoint_input()
    source = replace(
        source,
        blocks=(
            fc.BlockSpec("two", model=fc.TIRT(loading_anchors={"c": 0.9, "d": 1.1})),
            source.blocks[0],
        ),
    )
    model = fc.Model(source)
    means = np.array([0.1, 0.4, 0.7])
    scales = np.array([0.2, 0.3, 0.5])
    model.add_regressions(
        "fc.cutpoints ~ 1",
        pair_data=features,
        id_columns={"pair": ("block", "first", "second")},
        options={
            "fc.cutpoints": {
                "priors": {"Intercept": pm.Normal.dist(mu=means, sigma=scales)},
                "parameter_residual_sd": pm.HalfNormal.dist(sigma=scales),
            }
        },
    )
    original = model.prepared
    (before,) = original.regressions
    assert [target.pool_index.item() for target in before.targets] == [0, 1, 2, 0]

    replacement = pm.Normal.dist(mu=-0.3, sigma=0.8)
    model.set_priors({"fc.TIRT.cutpoints": replacement})
    prepared = model.prepared
    (after,) = prepared.regressions
    assert after.name == before.name
    assert after.design.attrs["encoder"] is before.design.attrs["encoder"]
    xr.testing.assert_identical(after.design.X, before.design.X)
    assert [target.pool_index.item() for target in after.targets] == [0, 1, 0]
    assert [target.pool_index.item() for target in before.targets] == [0, 1, 2, 0]
    assert_array_equal(_regression_pool_uses(prepared, after), [(0, 0, -1), (0, 3, -1), (1, 2, -1)])
    remaining = [0, 2]
    for previous, current in zip(before.blocks, after.blocks, strict=True):
        dim = f"{current.kind}_id"
        assert current.binding_id == previous.binding_id
        assert_array_equal(
            prepared.coords[dim][current.entry_index],
            original.coords[dim][previous.entry_index[remaining]],
        )
    coefficient, residual = after.blocks
    values = np.array([0.3, 0.6])
    intercept_prior = after.priors[coefficient.prior_role_index[0]]
    scale_prior = after.priors[residual.prior_role]
    assert_allclose(
        pm.logp(intercept_prior, values).eval(),
        stats.norm.logpdf(values, loc=means[remaining], scale=scales[remaining]),
    )
    assert_allclose(
        pm.logp(scale_prior, values).eval(), stats.halfnorm.logpdf(values, scale=scales[remaining])
    )
    (removed_target,) = (target for target in before.targets if target.unit_indices == (1,))
    restored = next(
        block
        for block in prepared.blocks_by_kind["cutpoint"]
        if block.binding_id == removed_target.binding_id
    )
    assert same_prior(restored.priors["cutpoints"], replacement)

    graph = model.build()
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=45)))
    anchors = model.coef(blocks=["two"], var_names=["loading"], kind="stats")
    assert anchors.attrs["model_response_format"] == "TIRT"
    assert anchors.index.get_level_values("statement").tolist() == ["c", "d"]
    assert_allclose(anchors["mean"], [0.9, 1.1])
    assert anchors[["sd", "hdi95_lb", "hdi95_ub"]].isna().all().all()
    dim = "regression_coefficient_id"
    ids = prepared.coords[dim]
    posterior_values = xr.Dataset({"regression_coefficient": (dim, values)}, coords={dim: ids})
    supplied_posterior(model, posterior_values, inference_method="mcmc", draws=range(8))
    for entry, expected in zip(ids, values, strict=True):
        summary = model.coef(
            component="regression",
            var_names=["regression_coefficient"],
            coords={dim: [entry]},
            kind="stats",
        )
        assert summary.attrs["model"] == "GTUM"
        assert summary.attrs["target"] == "cutpoint"
        assert summary.index.tolist() == ["Intercept"]
        assert_allclose(summary["mean"], [expected])


def test_equal_model_qualified_array_priors_bind_their_own_pools(formulae):
    source, features = cutpoint_input()
    source = replace(
        source,
        model=fc.TIRT(loading_anchors={"a": 1, "b": 1}),
        blocks=(
            fc.BlockSpec("two", scale=fc.CategoryScale((0, 1, 2))),
            fc.BlockSpec("three", model=fc.GTUM(), scale=fc.CategoryScale((0, 1, 2))),
            fc.BlockSpec("other", model=fc.GTUM()),
        ),
    )
    model = fc.Model(source)
    options = {
        f"fc.{family}.cutpoints": {
            "priors": {"Intercept": pm.Normal.dist(mu=[0.1, 0.2], sigma=0.4)}
        }
        for family in ("TIRT", "GTUM")
    }
    pairing = {"pair": ("block", "first", "second")}
    model.add_regressions(
        "fc.cutpoints ~ 1", pair_data=features, id_columns=pairing, options=options
    )
    (plan,) = model.prepared.regressions
    roles = plan.blocks[0].prior_role_index.reshape(plan.coefficient_positions.shape)
    assert roles.shape == (4, 1)
    assert roles[0, 0] == roles[1, 0] and roles[2, 0] == roles[3, 0]
    assert roles[0, 0] != roles[2, 0]
    for role in (roles[0, 0], roles[2, 0]):
        assert_array_equal(_normal_arguments(plan.priors[role])[0].eval(), [0.1, 0.2])

    before = model.prepared
    with pytest.raises(TypeError, match="unnamed native"):
        model.add_regressions(
            "fc.cutpoints ~ 1",
            pair_data=features,
            id_columns=pairing,
            options={"fc.TIRT.cutpoints": {"priors": {"Intercept": pt.dscalar()}}},
        )
    assert model.prepared is before


def test_cutpoints_pool_across_blocks_but_keep_count_and_prior_distinct(formulae):
    source, features = cutpoint_input()
    spec = fc.Regression("1", features, id=("block", "first", "second"))
    state = prepare_regressions(fc.PreparedInputs.from_inputs(source), {("fc", "cutpoints"): spec})
    plan = state.regressions[0]
    assert [target.pool_index.item() for target in plan.targets] == [0, 0, 1, 2]
    assert [target.row_index.shape for target in plan.targets] == [(1,)] * 4
    assert [target.noncentered for target in plan.targets] == [True, True, False, False]
    for index, target in enumerate(plan.targets):
        assert_array_equal(target.local_position, [0])
        assert target.unit_indices == (index,)
        assert not target.local_position.flags.writeable
    assert_array_equal(
        _regression_pool_uses(state, plan), [(0, 0, -1), (0, 1, -1), (1, 2, -1), (2, 3, -1)]
    )
    coefficients = plan.blocks[0]
    priors = [plan.priors[role] for role in coefficients.prior_role_index]
    assert [_normal_arguments(prior)[0].eval() for prior in priors] == [0, 0, 0]
    assert [_normal_arguments(prior)[1].eval() for prior in priors] == [1, 1, 1]
    residual = next(layout for layout in plan.blocks if layout.key == "residual_sd")
    assert same_prior(plan.priors[residual.prior_role], pm.HalfNormal.dist(sigma=1.0))
    assert residual.prior_role_index is None
    slopes = prepare_regressions(
        fc.PreparedInputs.from_inputs(source), {("fc", "cutpoints"): replace(spec, formula="0 + x")}
    ).regressions[0]
    for role in slopes.blocks[0].prior_role_index:
        prior = slopes.priors[role]
        assert same_prior(prior, pm.Normal.dist(mu=0, sigma=1 / np.std(features.x, ddof=0)))


@pytest.mark.parametrize("row_index", [range(4), [10, 20, 30, 40], [3, 1, 0, 2], [1, 1, 2, 2]])
def test_reversed_pair_ids_preserve_features_and_duplicate_pairs_fail(formulae, row_index):
    source, features = cutpoint_input()
    features.index = row_index
    spec = fc.Regression("0 + x", features, id=("block", "first", "second"))
    plan = prepare_regressions(
        fc.PreparedInputs.from_inputs(source), {("fc", "cutpoints"): spec}
    ).regressions[0]
    assert plan.design.key_1.values.tolist() == list("aceg")
    assert_allclose(plan.design.X.values.ravel(), [-2.5, -0.5, 0.5, 2.5])
    duplicate = features.iloc[[0]].rename(columns={"first": "second", "second": "first"})
    with pytest.raises(ValueError, match="duplicate or reversed-pair"):
        prepare_regressions(
            fc.PreparedInputs.from_inputs(source),
            {("fc", "cutpoints"): replace(spec, data=pd.concat([features, duplicate]))},
        )


def test_rank_deficient_design_and_unknown_coefficient_prior_fail_before_graph(formulae):
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    features = pd.DataFrame({"item": list("abc"), "x": [-1.0, 0.0, 1.0], "twice": [-2.0, 0.0, 2.0]})
    state = fc.PreparedInputs.from_inputs(source)
    with pytest.raises(ValueError, match="linearly dependent"):
        prepare_regressions(
            state, {("items", "intercept"): fc.Regression("x + twice", features, id="item")}
        )
    location, covariates = location_input()
    covariates.loc[:5, "x"] = 0.0
    with pytest.raises(
        ValueError, match="pool 0: regression design columns are linearly dependent"
    ):
        fc.Model(location).add_regressions("items.location ~ x", item_data=covariates)
    with pytest.raises(ValueError, match="unknown design columns"):
        prepare_regressions(
            state,
            {
                ("items", "intercept"): fc.Regression(
                    "x", features, id="item", priors={"typo": pm.Normal.dist(mu=0, sigma=1)}
                )
            },
        )


def conditional_layout():
    return ParameterBlock(
        binding_id=0,
        key="intercept",
        kind="item_intercept",
        component="measurement",
        entry_index=np.arange(3),
        priors={},
        dims=("entity",),
        shape=(3,),
        prior_role="base",
        prior_groups=(("base", np.arange(2)),),
        free_flat=np.array([0, 2]),
    )


@pytest.mark.parametrize("family", ["Normal", "LogNormal", "TruncatedNormal"])
def test_native_conditional_density_gradients_and_fixed_slots(family):
    mean, scale, value = pt.dvectors("mean", "scale", "value")
    bounds = {"lower": -1.0, "upper": 2.0} if family == "TruncatedNormal" else {}
    prior = getattr(pm, family).dist(mu=19, sigma=5, **bounds)
    layout = replace(conditional_layout(), priors={"base": prior})
    prepared = SimpleNamespace(
        parameter_layouts={
            layout.kind: SimpleNamespace(
                kind=layout.kind,
                dims=layout.dims,
                shape=layout.shape,
                fixed=np.array([np.nan, 7.0, np.nan]),
            )
        },
        blocks_by_kind={layout.kind: [layout]},
    )
    with pm.Model(coords={"entity": range(3)}) as graph:
        bindings = ParameterFactory(prepared).build(
            [layout.kind],
            regressions={
                (0, "base"): RegressionBinding(
                    conditional={
                        "family": family,
                        "link": "identity",
                        "bounds": (-1, 2) if bounds else None,
                    },
                    mean=mean,
                    scale=scale,
                    noncentered=False,
                )
            },
        )
    final = bindings[layout.kind]
    raw = graph.free_RVs[0]
    logp = pm.logp(raw, value).sum()
    evaluate = pytensor.function(
        [value, mean, scale], [logp, *pt.grad(logp, [value, mean, scale])], mode="FAST_COMPILE"
    )
    mu, sd, q = np.array([0.2, -0.4]), np.array([0.7, 1.3]), np.array([0.5, 1.2])
    residual = (np.log(q) if family == "LogNormal" else q) - mu
    dmean, dscale = residual / sd**2, residual**2 / sd**3 - 1 / sd
    dvalue = -dmean
    if family == "LogNormal":
        expected = stats.lognorm.logpdf(q, s=sd, scale=np.exp(mu))
        dvalue = -(dmean + 1) / q
    elif family == "TruncatedNormal":
        a, b = (-1 - mu) / sd, (2 - mu) / sd
        expected = stats.truncnorm.logpdf(q, a, b, loc=mu, scale=sd)
        normalizer = stats.norm.cdf(b) - stats.norm.cdf(a)
        dmean -= (stats.norm.pdf(a) - stats.norm.pdf(b)) / (sd * normalizer)
        dscale -= (a * stats.norm.pdf(a) - b * stats.norm.pdf(b)) / (sd * normalizer)
    else:
        expected = stats.norm.logpdf(q, loc=mu, scale=sd)
    actual = evaluate(q, mu, sd)
    # The native truncated-normal CDF approximation differs slightly from SciPy.
    tolerance = 2e-8 if family == "TruncatedNormal" else 1e-10
    for result, reference in zip(actual, [expected.sum(), dvalue, dmean, dscale], strict=True):
        assert_allclose(result, reference, atol=tolerance, rtol=tolerance)
    scattered = graph_replace(final, replace={raw: value})
    scatter = pytensor.function([value], scattered, mode="FAST_COMPILE")
    assert_allclose(scatter(q), [q[0], 7, q[1]])
    if family != "Normal":
        invalid = [-0.1, 1.2] if family == "LogNormal" else [-1.1, 2.1]
        assert np.isneginf(evaluate(invalid, mu, sd)[0])


@pytest.mark.parametrize("theta", [False, True])
@pytest.mark.parametrize("noncentered", [False, True])
def test_beta_and_B_use_centered_priors_unless_explicitly_noncentered(formulae, theta, noncentered):
    source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["a", "b"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": [1, 2, 3, 4],
                    "item_id": ["a", "b", "a", "b"],
                    "response": [0, 1, 1, 0],
                }
            ),
            format="item_long",
        ),
    )
    features = pd.DataFrame(
        {
            "id": [1, 2, 3, 4] if theta else ["a", "b"],
            "x": [-3.0, -1.0, 1.0, 3.0] if theta else [-1.0, 1.0],
        }
    )
    # A single theta term can occupy the entire term-by-trait shape.
    slope_shape = (1, 1) if theta and noncentered else ()
    priors = {
        "common": pm.Normal.dist(mu=np.full(slope_shape, 0.4), sigma=np.full(slope_shape, 0.7))
    }
    if not theta:
        priors["Intercept"] = pm.Normal.dist(mu=-0.2, sigma=0.8)
    spec = fc.Regression(
        "x",
        features,
        id="id",
        priors=priors,
        parameter_residual_sd=None if theta else 0.6,
        **({"coefficient_noncentered": True} if noncentered else {}),
    )
    target = "theta" if theta else ("items", "intercept")
    defaults = prepare_regressions(
        fc.PreparedInputs.from_inputs(source), {target: replace(spec, priors=None)}
    ).regressions[0]
    for column, term in enumerate(defaults.design.term.values):
        expected_mu = 0.0 if term == "Intercept" else 0
        expected_sd = 1.0 if term == "Intercept" else float(1 / np.std(features.x, ddof=0))
        roles = defaults.blocks[0].prior_role_index[defaults.coefficient_positions[:, column]]
        assert all(
            same_prior(defaults.priors[role], pm.Normal.dist(mu=expected_mu, sigma=expected_sd))
            for role in roles
        )
    prepared = prepare_regressions(fc.PreparedInputs.from_inputs(source), {target: spec})
    plan = prepared.regressions[0]
    with pm.Model(coords=prepared.coords) as graph:
        factory = ParameterFactory(prepared)
        globals_ = factory.build(
            kind for kind in ("regression_coefficient", "regression_sd") if kind in factory.layouts
        )
        targets, _ = build_regressions(prepared.regressions, globals_)
    key = "regression_coefficient"
    assert globals_[key].name == "regression_coefficient"
    assert any("noncentered" in rv.name for rv in graph.free_RVs) is noncentered
    values = [np.full(rv.type.shape, 0.25) for rv in graph.free_RVs]
    logp = sum(pm.logp(rv, value).sum() for rv, value in zip(graph.free_RVs, values, strict=True))
    outputs = graph.replace_rvs_by_values(
        [
            globals_[key],
            targets[(plan.targets[0].binding_id, plan.targets[0].prior_role)].mean,
            logp,
        ]
    )
    evaluate = pytensor.function(
        [graph.rvs_to_values[rv] for rv in graph.free_RVs], outputs, mode="JAX"
    )
    coefficients, means, density = evaluate(*values)
    expected = np.full(plan.coefficient_positions.shape, 0.25)
    expected[:, list(plan.design.term.values).index("x")] = (
        0.4 + 0.7 * 0.25 if noncentered else 0.25
    )
    assert_allclose(coefficients.ravel()[plan.coefficient_positions], expected)
    assert_allclose(means.ravel(), (plan.design.X.values @ expected.T).ravel())
    expected_density = stats.norm.logpdf(
        0.25, loc=0 if noncentered else 0.4, scale=1 if noncentered else 0.7
    )
    if not theta:
        expected_density += stats.norm.logpdf(0.25, loc=-0.2, scale=0.8)
    assert_allclose(density, expected_density)
    if not theta and noncentered:
        assert any("centered" in rv.name and "noncentered" not in rv.name for rv in graph.free_RVs)


@pytest.mark.parametrize(
    "options, error",
    [
        ({"formula": 1}, TypeError),
        ({"formula": " "}, ValueError),
        ({"id": ["id"]}, TypeError),
        ({"id": ()}, ValueError),
        ({"id": ("id", "id")}, ValueError),
        ({"center_predictors": 1}, TypeError),
        ({"deviation_noncentered": 1}, TypeError),
        ({"priors": []}, TypeError),
        ({"priors": {"": pm.Normal.dist()}}, ValueError),
        ({"priors": {"x": "Normal"}}, TypeError),
        ({"parameter_residual_sd": "1"}, TypeError),
        ({"parameter_residual_sd": 0}, ValueError),
    ],
)
def test_regression_declaration_rejects_invalid_contracts(options, error):
    arguments = dict(formula="x", data=pd.DataFrame({"id": [1, 2], "x": [0, 1]}), id="id")
    with pytest.raises(error):
        fc.Regression(**(arguments | options))


@pytest.mark.parametrize(
    "formula, options, error",
    [
        ("True", {}, ValueError),
        ("x - 1", {"include_intercept": 1}, TypeError),
        ("x", {"reference_rows": [1, 0]}, TypeError),
        ("x", {"reference_rows": [False, False]}, ValueError),
        ("x", {"reference_rows": [True]}, ValueError),
        ("0", {}, ValueError),
    ],
)
def test_design_contracts_are_checked_before_freezing(formulae, formula, options, error):
    frame = pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0]})
    with pytest.raises(error):
        fit_design(fc.Regression(formula, frame, id="id"), frame[["id"]], **options)


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "columns",
        "missing_id",
        "missing_feature",
        "constant",
        "mixed",
        "float_category",
        "complex",
    ],
)
def test_feature_training_rejects_ambiguous_or_degenerate_values(formulae, problem):
    frame = pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0]})
    keys = frame[["id"]]
    if problem == "empty":
        keys = keys.iloc[:0]
    elif problem == "columns":
        frame.columns = ["id", "id"]
    elif problem == "missing_id":
        frame.loc[0, "id"] = np.nan
    elif problem == "missing_feature":
        frame.loc[0, "x"] = np.nan
    elif problem == "constant":
        frame["x"] = 1.0
    elif problem == "mixed":
        frame["x"] = pd.Series(["A", 1], dtype=object)
    elif problem == "float_category":
        frame["x"] = pd.Categorical([0.1, 0.2])
    else:
        frame["x"] = [1j, 2j]
    with pytest.raises(ValueError):
        fit_design(fc.Regression("x", frame, id="id"), keys)


@pytest.mark.parametrize("kind", ["bool", "int", "str"])
def test_categorical_recipes_keep_declared_order_and_exclude_unobserved_levels(formulae, kind):
    levels = {"bool": [True, False], "int": [2, 1, 3], "str": ["B", "A", "unused"]}[kind]
    frame = pd.DataFrame({"id": [1, 2], "x": pd.Categorical(levels[:2], categories=levels)})
    design = fit_design(fc.Regression("x", frame, id="id"), frame[["id"]])
    assert design.attrs["feature_records"][0]["levels"] == levels[:2]
    assert_allclose(replay_design(design, frame, frame[["id"]]), design.X)


def test_design_requires_training_dataset_and_valid_intercept_spec(interaction_design):
    training, people, keys = interaction_design
    with pytest.raises(TypeError):
        fit_design("age", keys)
    with pytest.raises(TypeError):
        apply_design(training.X, people, np.arange(len(keys)), pd.Index(keys.pid))
    spec = fc.Regression("age - 1", people, id="pid", priors={"Intercept": pm.Normal.dist()})
    with pytest.raises(ValueError, match="Intercept prior"):
        fit_design(spec, keys, include_intercept=False)


@pytest.mark.parametrize(
    "target, options, error",
    [
        ("invalid", {}, ValueError),
        (("items", "location"), {}, ValueError),
        (("items", "intercept"), {"id": ("item", "other")}, ValueError),
        ("theta", {"id": ("item", "other")}, ValueError),
        ("theta", {"parameter_residual_sd": 1.0}, ValueError),
        ("theta", {"deviation_noncentered": False}, ValueError),
        (
            ("items", "intercept"),
            {"coefficient_noncentered": True, "priors": {"common": pm.StudentT.dist(nu=5)}},
            NotImplementedError,
        ),
    ],
)
def test_target_specific_regression_contracts_fail_before_build(formulae, target, options, error):
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    features = pd.DataFrame({"item": list("abc"), "other": list("abc"), "x": [-1.0, 0.0, 1.0]})
    spec = fc.Regression("x", features, **({"id": "item"} | options))
    with pytest.raises(error):
        prepare_regressions(fc.PreparedInputs.from_inputs(source), {target: spec})


def test_regression_preparation_appends_without_refitting_existing_records(formulae, monkeypatch):
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    state = fc.PreparedInputs.from_inputs(source)
    features = pd.DataFrame({"item": list("abc"), "x": [-1.0, 0.0, 1.0]})
    spec = fc.Regression("x", features, id="item")
    for options in (
        {"marginal_correlation": 1},
        {"regressions": {"theta": "x"}},
    ):
        with pytest.raises(TypeError):
            prepare_regressions(state, **options)
    frozen = prepare_regressions(state, {("items", "intercept"): spec})
    snapshot = runtime_snapshot(frozen)
    before = frozen.regressions[0]
    assert prepare_regressions(frozen) is frozen
    calls = []

    def record_fit(declaration, *args, **kwargs):
        calls.append(declaration)
        return fit_design(declaration, *args, **kwargs)

    monkeypatch.setattr("pyfcirt.data.regressions.fit_design", record_fit)
    features.loc[:, "x"] = [-100.0, 0.0, 100.0]
    appended = prepare_regressions(frozen, {("items", "loading_magnitude"): spec})
    assert calls == [spec]
    assert frozen.data.identical(snapshot)
    for original, updated in zip(frozen.batches, appended.batches, strict=True):
        for first, second in zip(original.slots, updated.slots, strict=True):
            assert (first.key, first.kind) == (second.key, second.kind)
            assert np.shares_memory(first.entry_index, second.entry_index)
    after = appended.regressions
    assert [plan.name for plan in after] == ["r0", "r1"]
    assert after[0].design.identical(before.design)
    assert after[0].design.attrs["resolved_priors"] == before.design.attrs["resolved_priors"]
    assert_array_equal(after[0].coefficient_positions, before.coefficient_positions)
    for old, new in zip(before.targets, after[0].targets, strict=True):
        assert old.binding_id == new.binding_id
        assert_array_equal(old.row_index, new.row_index)
        assert_array_equal(old.pool_index, new.pool_index)
    assert (
        appended.data["constant_data"].attrs["regression_scopes"][0]
        == snapshot["constant_data"].attrs["regression_scopes"][0]
    )
    changed = prepare_regressions(appended, marginal_correlation=True)
    assert changed.data["constant_data"].attrs["marginal_correlation"] is True
    assert changed.data["covariates/r0"].identical(snapshot["covariates/r0"])
    replaced = prepare_regressions(frozen, {("items", "intercept"): spec})
    assert [plan.name for plan in replaced.regressions] == ["r1"]
    assert "covariates/r0" not in replaced.data
    unobserved_data = state.data.copy(deep=False)
    unobserved_data.attrs["purpose"] = "generate"
    unobserved = fc.PreparedInputs(unobserved_data)
    updated = prepare_regressions(unobserved, {("items", "intercept"): spec})
    assert updated.data.attrs["purpose"] == "generate"
    assert len(updated.regressions) == 1


@pytest.mark.parametrize("result_group", ["posterior", "estimates", "fixed_parameters"])
def test_regression_binder_rejects_fitted_or_fixed_states(formulae, result_group):
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    state = fc.PreparedInputs.from_inputs(source)
    state.data[result_group] = state.data["constant_data"].copy(deep=True)
    snapshot = state.data.copy(deep=True)
    spec = fc.Regression("1", pd.DataFrame({"id": list("abc")}), id="id")
    with pytest.raises(ValueError, match="before estimation or fixing"):
        prepare_regressions(state, {("items", "intercept"): spec})
    with pytest.raises(ValueError, match="fixed after estimation"):
        prepare_regressions(state, marginal_correlation=True)
    assert state.data.identical(snapshot)


def test_regression_binder_failed_batch_leaves_all_prior_records_unchanged(formulae):
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    spec = fc.Regression("1", pd.DataFrame({"id": list("abc")}), id="id")
    frozen = prepare_regressions(
        fc.PreparedInputs.from_inputs(source), {("items", "intercept"): spec}
    )
    snapshot = runtime_snapshot(frozen)
    invalid = fc.Regression("x", pd.DataFrame({"id": ["p"], "x": [None]}), id="id")
    with pytest.raises(ValueError, match="missing"):
        prepare_regressions(frozen, {("items", "loading_magnitude"): spec, "theta": invalid})
    assert frozen.data.identical(snapshot)
    assert "covariates/r1" not in frozen.data


@pytest.mark.parametrize("prior", [pm.StudentT.dist(nu=5), pm.Normal.dist(tau=1)])
def test_item_regression_replaces_arbitrary_ordinary_native_law(formulae, prior):
    source = item_input(
        pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}),
        model=fc.DichotomousIRT(priors={"intercept": prior}),
    )
    spec = fc.Regression("1", pd.DataFrame({"item": list("abc")}), id="item")
    plan = prepare_regressions(
        fc.PreparedInputs.from_inputs(source), {("items", "intercept"): spec}
    ).regressions[0]
    assert plan.targets[0].conditional["family"] == "Normal"
    assert same_prior(plan.priors[plan.blocks[0].prior_role], pm.Normal.dist(mu=0.0, sigma=1.0))


def test_absent_optional_formulae_reports_the_install_extra(monkeypatch):
    import sys

    frame = pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0]})
    monkeypatch.setitem(sys.modules, "formulae", None)
    with pytest.raises(ImportError, match=r"pyfcirt\[covariates\]"):
        fit_design(fc.Regression("x", frame, id="id"), frame[["id"]])


def test_finite_raw_features_cannot_produce_an_overflowing_interaction(formulae):
    frame = pd.DataFrame(
        {"id": [1, 2], "x": [-1e110, 1e110], "y": [-1e110, 1e110], "z": [-1e110, 1e110]}
    )
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite design"):
        fit_design(fc.Regression("x:y:z", frame, id="id"), frame[["id"]])


def test_cutpoint_regressions_require_pair_ids_and_normal_ordered_priors(formulae):
    source, features = cutpoint_input()
    with pytest.raises(ValueError, match="id=.*block"):
        prepare_regressions(
            fc.PreparedInputs.from_inputs(source),
            {("fc", "cutpoints"): fc.Regression("1", features, id=("block", "first"))},
        )
    source = replace(
        source,
        model=fc.GTUM(priors={"cutpoints": pm.LogNormal.dist(mu=0, sigma=1)}),
    )
    prepared = prepare_regressions(
        fc.PreparedInputs.from_inputs(source),
        {("fc", "cutpoints"): fc.Regression("1", features, id=("block", "first", "second"))},
    )
    assert all(
        target.conditional["family"] == "Normal" for target in prepared.regressions[0].targets
    )


def test_native_residual_prior_and_shared_item_entities_are_retained(formulae):
    source = item_input(pd.DataFrame({"item_id": ["a", "a", "b"], "trait": ["A", "B", "A"]}))
    prior = pm.Exponential.dist(lam=2)
    spec = fc.Regression(
        "0 + x",
        pd.DataFrame({"id": ["a", "b"], "x": [10.0, 30.0]}),
        id="id",
        parameter_residual_sd=prior,
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    model.add_regressions({("items", "loading_magnitude"): spec})
    plan = model.prepared.regressions[0]
    assert plan.design.sizes["row"] == 2 and plan.coefficient_positions.shape == (2, 1)
    assert plan.design.attrs["prior_scale_divisors"] == {"x": 10.0}
    assert plan.design.attrs["residual_sd"] is None
    layout = next(layout for layout in plan.blocks if layout.key == "residual_sd")
    assert same_prior(plan.priors[layout.prior_role], prior)


def test_intercept_pools_keep_rank_and_graded_prior_roles_separate(formulae):
    from tests.helpers import mixed_inputs

    source = mixed_inputs()[0]
    answers = source.responses.frame.copy(deep=True)
    pair = answers.block_id.eq("pair")
    answers.loc[pair, "response"] = [1, 0]
    source = replace(
        source,
        responses=fc.ResponseTable(answers, format="block_long"),
        blocks=(
            *source.blocks,
            fc.BlockSpec("pair", response_format=None, scale=fc.CategoryScale((0, 1))),
        ),
    )
    spec = fc.Regression("1", pd.DataFrame({"id": list("abcdefghi")}), id="id")
    prepared = prepare_regressions(
        fc.PreparedInputs.from_inputs(source), {("fc", "intercept"): spec}
    )
    assert prepared.regressions[0].coefficient_positions.shape == (2, 1)
    (intercept,) = prepared.blocks_by_kind["statement_intercept"]
    assert intercept.prior_role_index is not None
    plan = prepared.regressions[0]
    roles = plan.blocks[0].prior_role_index[plan.coefficient_positions[:, 0]]
    assert all(same_prior(plan.priors[role], pm.Normal.dist(mu=0.0, sigma=1.0)) for role in roles)


@pytest.mark.parametrize("custom_ids", [False, True])
def test_public_regression_strings_and_mapping_share_frozen_designs(formulae, custom_ids):
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    source = replace(
        source,
        responses=fc.ResponseTable(
            pd.concat(
                [source.responses.frame.assign(person_id=person) for person in ["p", "q", "r"]],
                ignore_index=True,
            ),
            format="item_long",
        ),
    )
    person_id, item_id = ("pid", "iid") if custom_ids else ("person_id", "item_id")
    people = pd.DataFrame({person_id: ["r", "p", "q"], "age": [30.0, 10.0, 20.0]})
    items = pd.DataFrame({item_id: ["c", "a", "b"], "x": [3.0, 1.0, 2.0]})
    settings = dict(
        standardize_predictors=True, parameter_residual_sd=0.4, coefficient_noncentered=True
    )
    state = fc.PreparedInputs.from_inputs(source)
    string_model = fc.Model(state)
    assert (
        string_model.add_regressions(
            "theta ~ age\nitems.loading_magnitude ~ x",
            person_data=people,
            item_data=items,
            id_columns={"person": person_id, "item": item_id},
            options={"items.loading_magnitude": settings},
        )
        is None
    )
    mapping_model = fc.Model(state)
    mapping_model.add_regressions(
        {
            "theta": fc.Regression("age", people, id=person_id),
            ("items", "loading_magnitude"): fc.Regression("x", items, id=item_id, **settings),
        }
    )
    assert string_model.idata.identical(mapping_model.idata)
    prepared = fc.PreparedInputs.from_inputs(
        source,
        regressions="theta ~ age\nitems.loading_magnitude ~ x",
        person_data=people,
        item_data=items,
        id_columns={"person": person_id, "item": item_id},
        regression_options={"items.loading_magnitude": settings},
    )
    from tests.helpers import assert_same_preparation

    assert_same_preparation(prepared, string_model.prepared)
    for actual, expected in zip(
        prepared.regressions, string_model.prepared.regressions, strict=True
    ):
        xr.testing.assert_equal(actual.design, expected.design)
        assert actual.design.attrs["noncentered"] == expected.design.attrs["noncentered"]
        for name in actual.priors:
            assert same_prior(actual.priors[name], expected.priors[name])
    direct = fc.Model(prepared).build()
    incremental = mapping_model.build()
    assert direct.named_vars_to_dims == incremental.named_vars_to_dims
    assert_allclose(
        direct.compile_logp()(direct.initial_point()),
        incremental.compile_logp()(incremental.initial_point()),
    )


def test_public_cutpoint_regression_uses_default_ordered_pair_columns(formulae):
    source, features = cutpoint_input()
    features = features.rename(
        columns={"block": "block_id", "first": "statement_1", "second": "statement_2"}
    )
    state = fc.PreparedInputs.from_inputs(source)
    string_model = fc.Model(state)
    string_model.add_regressions("fc.cutpoints ~ 1", pair_data=features)
    mapping_model = fc.Model(state)
    mapping_model.add_regressions(
        {
            ("fc", "cutpoints"): fc.Regression(
                "1", features, id=("block_id", "statement_1", "statement_2")
            )
        }
    )
    assert string_model.idata.identical(mapping_model.idata)


@pytest.mark.parametrize(
    "problem, error",
    [
        ("mapping_table", ValueError),
        ("mapping_ids", ValueError),
        ("mapping_options", ValueError),
        ("unknown_entity", ValueError),
        ("unknown_target", ValueError),
        ("unknown_option", ValueError),
        ("invalid_option", TypeError),
        ("wrong_id", ValueError),
        ("missing_table", ValueError),
    ],
)
def test_public_regression_declaration_rejects_mixed_or_invalid_inputs(formulae, problem, error):
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    items = pd.DataFrame({"item_id": list("abc"), "x": [-1.0, 0.0, 1.0]})
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    formula = "items.intercept ~ x"
    kwargs = {"item_data": items}
    if problem.startswith("mapping_"):
        formula = {("items", "intercept"): fc.Regression("x", items, id="item_id")}
        kwargs = {
            "mapping_table": {"item_data": items},
            "mapping_ids": {"id_columns": {}},
            "mapping_options": {"options": {}},
        }[problem]
    elif problem == "unknown_entity":
        kwargs["id_columns"] = {"items": "item_id"}
    elif problem == "unknown_target":
        kwargs["options"] = {"items.location": {}}
    elif problem == "unknown_option":
        kwargs["options"] = {"items.intercept": {"standardized": True}}
    elif problem == "invalid_option":
        kwargs["options"] = {"items.intercept": {"standardize_predictors": 1}}
    elif problem == "wrong_id":
        kwargs["id_columns"] = {"item": "absent"}
    elif problem == "missing_table":
        kwargs = {}
    else:
        formula += "\nitems.intercept ~ 1"
    with pytest.raises(error):
        model.add_regressions(formula, **kwargs)


def location_input():
    labels = list("abcdefghijklmnopqr")
    sides = np.repeat([1, -1, 0], 6)
    source = item_input(
        pd.DataFrame({"item_id": labels, "trait": ["A"] * len(labels), "location_side": sides}),
        model=fc.GGUM(),
        categories=(0, 1, 2),
    )
    features = pd.DataFrame(
        {
            "item_id": labels,
            "x": [-2.0, 0.0, 2.0, -1.0, 1.0, 3.0] * 3,
            "group": pd.Categorical(["a", "a", "a", "b", "b", "b"] * 3),
        }
    )
    return source, features


@pytest.mark.parametrize("ordinary_last", [False, True])
@pytest.mark.parametrize("fixed_scale", [False, True])
def test_native_prior_and_regression_latest_call_wins_for_one_location_side(
    formulae, ordinary_last, fixed_scale
):
    source, features = location_input()
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    native = pm.StudentT.dist(nu=5, mu=3, sigma=0.7)
    if not ordinary_last:
        with pytest.warns(fc.IdentificationWarning):
            model.set_priors({"items.location_positive": native})
    means = np.array([0.1, 0.2, 0.3])
    settings = {"priors": {"x": pm.Normal.dist(mu=means, sigma=0.4)}}
    if fixed_scale:
        settings["parameter_residual_sd"] = 0.4
    model.add_regressions(
        "items.location ~ x * group", item_data=features, options={"items.location": settings}
    )
    original = model.prepared
    frozen = original.regressions[0]
    encoder = frozen.design.copy(deep=False)
    if ordinary_last:
        with pytest.warns(fc.IdentificationWarning):
            model.set_priors({"items.location_positive": native})
    plans = model.prepared.regressions
    roles = {target.prior_role for plan in plans for target in plan.targets}
    assert roles == (
        {"location_negative", "location_free"}
        if ordinary_last
        else {"location_positive", "location_negative", "location_free"}
    )
    active = plans[0]
    canonical = model.prepared.parameter_layouts["regression_sd"]
    earlier = original.parameter_layouts["regression_sd"]
    assert_array_equal(canonical.entry_data.domain, ["positive"] * len(active.targets))
    assert_array_equal(earlier.entry_data.domain, ["positive"] * 3)
    assert canonical.fixed_source.tolist() == (
        ["fixed"] * len(active.targets) if fixed_scale else []
    )
    if fixed_scale:
        assert_array_equal(canonical.fixed_value, np.full(len(active.targets), 0.4))
        assert earlier.fixed_source.tolist() == ["fixed"] * 3
    xr.testing.assert_identical(model.prepared.observed_data, original.observed_data)
    xr.testing.assert_identical(
        model.prepared.parameter_uses.isel(
            parameter_use=~model.prepared.parameter_uses.kind.isin(
                ["regression_coefficient", "regression_sd"]
            )
        ),
        original.parameter_uses.isel(
            parameter_use=~original.parameter_uses.kind.isin(
                ["regression_coefficient", "regression_sd"]
            )
        ),
    )
    assert active.name == "r0"
    assert active.design.X.identical(encoder.X)
    assert active.design.attrs["encoder"] is encoder.attrs["encoder"]
    assert active.design.attrs["feature_records"] == encoder.attrs["feature_records"]
    assert active.design.sizes["row"] == encoder.sizes["row"]
    assert active.coefficient_positions.shape[0] == (2 if ordinary_last else 3)
    retained = slice(1, None) if ordinary_last else slice(None)
    for previous, current in zip(frozen.blocks, active.blocks, strict=True):
        key = f"{previous.kind}_id"
        assert current.binding_id == previous.binding_id
        assert_array_equal(
            model.prepared.coords[key][current.entry_index],
            original.coords[key][previous.entry_index[retained]],
        )
    x_position = list(active.design.term.values).index("x")
    prior = active.priors[active.blocks[0].prior_role_index[x_position]]
    selected_means = means[retained]
    expected = -0.5 * (selected_means / 0.4) ** 2 - np.log(0.4 * np.sqrt(2 * np.pi))
    assert_allclose(pm.logp(prior, np.zeros(len(expected))).eval(), expected)
    start = 6 if ordinary_last else 0
    expected_uses = []
    for pool, target in enumerate(active.targets):
        units = np.arange(start + 6 * pool, start + 6 * (pool + 1))
        assert_array_equal(target.local_position, units)
        assert target.unit_indices == tuple(units)
        expected_uses.extend((pool, unit, 0) for unit in units)
    assert_array_equal(_regression_pool_uses(model.prepared, active), expected_uses)
    # A successful partial update does not remap any older immutable prepared view.
    assert_array_equal(_regression_pool_uses(original, frozen)[-6:, 0], np.full(6, 2))
    if ordinary_last:
        (location,) = model.prepared.blocks_by_kind["location"]
        assert same_prior(location.priors["location_positive"], native)
        assert set(active.design.attrs["resolved_priors"]) == set(active.priors)
    graph = model.build()
    assert len([name for name in graph.named_vars if name == "location"]) == 1
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=51)))


@pytest.mark.parametrize(("index", "side"), [(0, "positive"), (1, "negative")])
def test_partial_prior_replacement_keeps_equal_coefficient_and_scale_templates_distinct(
    formulae, index, side
):
    source, features = location_input()
    sigma = np.array([0.3, 0.4, 0.5])
    prior = pm.HalfNormal.dist(sigma=sigma)
    model = fc.Model(source)
    model.add_regressions(
        "items.location ~ residual_sd",
        item_data=features.rename(columns={"x": "residual_sd"}),
        options={
            "items.location": {"priors": {"residual_sd": prior}, "parameter_residual_sd": prior}
        },
    )
    previous = model.prepared.regressions[0]
    model.set_priors({f"items.location_{side}": pm.HalfNormal.dist(sigma=0.9)})
    active = model.prepared.regressions[0]
    assert previous.coefficient_positions.shape[0] == 3
    assert active.coefficient_positions.shape[0] == 2
    assert not any(name.endswith("_entry_ids") for name in model.prepared.constant_data)
    coefficient, scale = active.blocks
    term = list(active.design.term.values).index("residual_sd")
    coefficient_role = coefficient.prior_role_index[term]
    assert coefficient_role != scale.prior_role
    values = np.full(2, 0.4)
    selected = np.delete(sigma, index)
    expected = np.log(np.sqrt(2 / np.pi) / selected) - (values / selected) ** 2 / 2
    for role in (coefficient_role, scale.prior_role):
        assert_allclose(pm.logp(active.priors[role], values).eval(), expected)
    assert previous.design.attrs["encoder"] is active.design.attrs["encoder"]


@pytest.mark.parametrize("generic_last", [False, True])
def test_location_multiline_order_and_partial_replacement_keep_sparse_monotonic_ids(
    formulae, generic_last
):
    source, features = location_input()
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    lines = ["items.location ~ x * group", "items.location_positive ~ 1"]
    if generic_last:
        lines.reverse()
    model.add_regressions("\n".join(lines), item_data=features)
    plans = model.prepared.regressions
    assert [plan.name for plan in plans] == (["r0"] if generic_last else ["r0", "r1"])
    if not generic_last:
        assert {t.prior_role for t in plans[0].targets} == {"location_negative", "location_free"}
        assert {t.prior_role for t in plans[1].targets} == {"location_positive"}
    general = pm.Normal.dist(mu=0, sigma=2)
    positive = pm.HalfNormal.dist(sigma=1)
    updates = [("items.location", general), ("items.location_positive", positive)]
    model.set_priors(dict(reversed(updates) if generic_last else updates))
    (location,) = model.prepared.blocks_by_kind["location"]
    priors = location.priors
    assert same_prior(priors["location_positive"], general if generic_last else positive)
    assert same_prior(priors["location_negative"], general)
    assert not model.prepared.regressions
    assert "covariates" not in model.idata
    constant = model.prepared.data["constant_data"]
    assert not any(name.startswith("r") and name[1:2].isdigit() for name in constant.variables)
    expected_id = 1 if generic_last else 2
    assert constant.attrs["next_regression_id"] == expected_id
    model.add_regressions("items.location_free ~ x", item_data=features)
    assert [p.name for p in model.prepared.regressions] == [f"r{expected_id}"]
    fc.Model(model.prepared)  # Sparse IDs are valid external records.


def test_model_update_transaction_preserves_cache_and_outputs_on_failure(formulae):
    source, features = location_input()
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    model.add_regressions("items.location ~ x * group", item_data=features)
    graph = model.build()
    before = model.idata.copy(deep=True)
    with pytest.raises(ValueError, match="linearly dependent"):
        model.add_regressions(
            "items.location_positive ~ x + twice", item_data=features.assign(twice=2 * features.x)
        )
    assert model.idata.identical(before)
    assert model.build() is graph
    with pytest.raises((ValueError, TypeError)):
        model.set_priors(
            {"items.location_positive": pm.Normal.dist(), "items.typo": pm.Normal.dist()}
        )
    assert model.idata.identical(before)
    assert model.build() is graph
    model.add_regressions("items.location_positive ~ 1", item_data=features)
    assert model.build() is not graph


def test_replacing_regressions_keeps_other_owned_axes_in_a_mixed_grouped_model(formulae):
    from tests.helpers import mixed_inputs

    inputs = tuple(
        replace(
            source,
            responses=replace(
                source.responses,
                frame=pd.concat(
                    [
                        source.responses.frame,
                        source.responses.frame.assign(
                            person_id=source.responses.frame.person_id + 2
                        ),
                    ],
                    ignore_index=True,
                ),
            ),
        )
        for source in mixed_inputs()
    )
    people = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "group": ["A", "A", "B", "B"],
            "x": [-1.0, 1.0, -2.0, 2.0],
            "z": [0.0, 1.0, 2.0, 4.0],
        }
    )
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            inputs,
            population=fc.PopulationSpec(group="group", reference_group="A"),
            person_data=people,
        )
    )
    model.add_regressions(
        "theta ~ x\nitems.loading_magnitude ~ 1",
        person_data=people,
        item_data=pd.DataFrame({"item_id": ["pair", "ordinal"]}),
    )

    def coefficient_identity(prepared, plan):
        axis = "regression_coefficient_id"
        entries = prepared.coords[axis][plan.coefficient_entries.ravel()]
        names = [
            "regression_coefficient_parameter_set_index",
            "regression_coefficient_regression_index",
        ]
        return prepared.output_coords[names].sel({axis: entries})

    earlier = model.prepared
    first_theta, item = earlier.regressions
    model.add_regressions("theta ~ z", person_data=people)
    replacement = model.prepared
    current_item, current_theta = replacement.regressions
    assert current_item.design.attrs["encoder"] is item.design.attrs["encoder"]
    xr.testing.assert_identical(
        coefficient_identity(replacement, current_item), coefficient_identity(earlier, item)
    )
    assert replacement.coords["theta_predictor"].tolist() == ["z"]
    xr.testing.assert_identical(earlier.observed_data, replacement.observed_data)
    for kind, layout in earlier.parameter_layouts.items():
        if kind in ("regression_coefficient", "regression_sd"):
            continue
        retained = replacement.parameter_layouts[kind]
        xr.testing.assert_identical(layout.entry_data, retained.entry_data)
        if layout.fixed.size:
            assert np.shares_memory(layout.fixed, retained.fixed)
    assert "theta_predictor" in earlier.constant_data.dims
    assert not any(name.startswith(f"{first_theta.name}_") for name in replacement.constant_data)

    model.set_priors({"items.loading_magnitude": pm.LogNormal.dist(mu=0.1, sigma=0.4)})
    after = model.prepared
    (retained,) = after.regressions
    assert retained.design.attrs["encoder"] is current_theta.design.attrs["encoder"]
    xr.testing.assert_identical(
        coefficient_identity(after, retained), coefficient_identity(replacement, current_theta)
    )
    assert after.coords["theta_predictor"].tolist() == ["z"]
    assert not any(name.startswith(f"{item.name}_") for name in after.constant_data)
    uses = earlier.parameter_uses
    measurement = ~uses.kind.isin(["regression_coefficient", "regression_sd"])
    xr.testing.assert_identical(uses.isel(parameter_use=measurement), after.parameter_uses)
    graph = model.build()
    assert np.isfinite(graph.compile_logp(mode="JAX")(graph.initial_point(random_seed=4)))


def test_exact_repeated_target_uses_last_line_and_only_validates_active_design(formulae):
    source, features = location_input()
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    with pytest.raises(ValueError, match="Invalid regression formula syntax"):
        model.add_regressions(
            "items.location_free ~ x +\nitems.location_free ~ x", item_data=features
        )
    model.add_regressions(
        "items.location_free ~ absent\nitems.location_free ~ x", item_data=features
    )
    plans = model.prepared.regressions
    assert len(plans) == 1 and plans[0].name == "r0"
    assert plans[0].design.attrs["formula"] == "x"


def test_pair_keys_keep_typed_ids_and_reversed_interaction_features(formulae):
    blocks = [1, 2, 3, 4, 5, 6]
    left = [10, 12, 20, 22, 30, 32]
    right = [11, 13, 21, 23, 31, 33]
    design = pd.DataFrame(
        [
            (block, position, statement, trait, 1, 0)
            for block, a, b in zip(blocks, left, right, strict=True)
            for position, statement, trait in [(1, a, "A"), (2, b, "B")]
        ],
        columns=["block_id", "position", "statement_id", "trait", "keying", "location_side"],
    )
    source = fc.FCInput(
        design,
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p"] * 6, "block_id": blocks, "response": [0] * 6}),
            format="block_long",
        ),
        model=fc.GTUM(),
        response_format=None,
        scale=fc.CategoryScale((0, 1)),
    )
    features = pd.DataFrame(
        {
            "block": blocks,
            "left": right,
            "right": left,
            "x": [-2.0, 0.0, 2.0, -1.0, 1.0, 3.0],
            "group": pd.Categorical(["a"] * 3 + ["b"] * 3),
        }
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    model.add_regressions(
        "fc.cutpoints ~ x * group",
        pair_data=features,
        id_columns={"pair": ("block", "left", "right")},
    )
    plan = model.prepared.regressions[0]
    assert plan.design.key_0.values.tolist() == blocks
    assert [type(v) for v in plan.design.key_0.values] == [int, int] * 3
    assert plan.design.key_1.values.tolist() == left
    assert plan.design.key_2.values.tolist() == right
    assert_allclose(plan.design.X.sel(term="x"), features.x - features.x.mean())
    assert_allclose(
        plan.design.X.sel(term="x:group[b]"),
        (features.x - features.x.mean()) * (features.group == "b"),
    )


@pytest.mark.parametrize(
    "option", ["statement_data", "item_data", "pair_data", "id_columns", "regression_options"]
)
def test_factory_rejects_unused_regression_settings(option):
    with pytest.raises(ValueError, match="require regressions"):
        fc.PreparedInputs.from_inputs(
            item_input(pd.DataFrame({"item_id": ["a"], "trait": ["A"]})), **{option: {}}
        )


@pytest.mark.parametrize("old_name", ["standardize", "noncentered", "residual_sd"])
def test_removed_ambiguous_regression_names_are_rejected(old_name):
    with pytest.raises(TypeError, match=old_name):
        fc.Regression("1", pd.DataFrame({"id": [1]}), id="id", **{old_name: True})


def test_first_regression_preparation_publishes_once_without_pruning(formulae, monkeypatch):
    from pyfcirt.data import regressions
    from tests.helpers import mixed_inputs

    constructed = []
    post_init = fc.PreparedInputs.__post_init__

    def observe(prepared):
        constructed.append(prepared)
        post_init(prepared)

    def reject_pruning(*args):
        pytest.fail("First preparation has no regression pools to prune")

    monkeypatch.setattr(fc.PreparedInputs, "__post_init__", observe)
    monkeypatch.setattr(regressions, "_prune_regressions", reject_pruning)
    person_data = pd.DataFrame({"person_id": [2, 1], "age": [30.0, 10.0]})
    prepared = fc.PreparedInputs.from_inputs(
        mixed_inputs(), regressions="theta ~ age", person_data=person_data
    )

    assert len(constructed) == 1 and constructed[0] is prepared
    assert prepared.theta_regression.name == "r0"
    assert_array_equal(prepared.theta_regression.design.X.values[:, 0], [-10.0, 10.0])


def test_factory_mapping_retains_population_groups_and_final_fixed_coefficients(formulae):
    from tests.helpers import mixed_inputs, workflow_point

    people = pd.DataFrame({"person_id": [1, 2], "age": [10.0, 30.0]})
    regressions = {"theta": fc.Regression("age", people, id="person_id")}
    inputs = mixed_inputs()
    reference = fc.Model(fc.PreparedInputs.from_inputs(inputs, regressions=regressions))
    point = workflow_point(reference)
    coefficient = next(
        layout
        for layout in reference.prepared.parameters
        if layout.kind == "regression_coefficient"
    )
    point["regression_coefficient"] = xr.DataArray(
        [0.1, -0.2],
        dims=coefficient.dims,
        coords={dim: reference.prepared.coords[dim] for dim in coefficient.dims},
    )
    prepared = fc.PreparedInputs.from_inputs(
        inputs, regressions=regressions, fixed_parameters=point
    )
    assert "regression_coefficient" in prepared.data["fixed_parameters"]
    assert {rv.name for rv in fc.Model(prepared).build().free_RVs} == {"theta"}
    grouped_inputs = tuple(
        replace(
            source,
            responses=replace(
                source.responses,
                frame=pd.concat(
                    [
                        source.responses.frame,
                        source.responses.frame.assign(
                            person_id=source.responses.frame.person_id + 2
                        ),
                    ],
                    ignore_index=True,
                ),
            ),
        )
        for source in inputs
    )
    groups = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "age": [10.0, 30.0, 20.0, 40.0],
            "group": ["reference", "reference", "other", "other"],
        }
    )
    grouped = fc.PreparedInputs.from_inputs(
        grouped_inputs,
        population=fc.PopulationSpec(group="group", reference_group="reference"),
        person_data=groups,
        regressions={"theta": fc.Regression("age", groups, id="person_id")},
    )
    assert grouped.population.recipe["groups"] == ["reference", "other"]
    assert_array_equal(grouped.regressions[0].design.X.values[:, 0], [-10.0, 10.0, 0.0, 20.0])


@pytest.mark.parametrize("by_group", [False, True])
def test_theta_pool_order_replays_distinct_terms_traits_and_an_omitted_training_group(
    formulae, by_group
):
    from pyfcirt.data.prepare import _prepare_generation

    source = item_input(pd.DataFrame({"item_id": list("ab"), "trait": ["A", "B"]}))
    source = replace(
        source,
        responses=replace(
            source.responses,
            frame=pd.concat([source.responses.frame.assign(person_id=p) for p in range(6)]),
        ),
    )
    people = pd.DataFrame(
        {
            "person_id": range(6),
            "group": ["reference"] * 3 + ["other"] * 3,
            "x": [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0],
            "z": [1.0, 0.0, 1.0, 4.0, 0.0, 4.0],
        }
    )
    prepared = fc.PreparedInputs.from_inputs(
        source,
        person_data=people,
        population=fc.PopulationSpec(group="group", regression_by_group=by_group),
        regressions={"theta": fc.Regression("x + z", people.iloc[::-1], id="person_id")},
    )
    # Native B has (group, predictor, trait) axes; values distinguish every axis.
    B = np.array([[[0.2, -0.3], [0.7, 0.4]], [[-0.5, 0.6], [0.9, -0.8]]])
    coefficients = B if by_group else B[0]
    new_people = pd.DataFrame(
        {
            "person_id": ["first", "last"],
            "group": ["other", "other"],
            "x": [3.0, -1.0],
            "z": [9.0, 1.0],
        }
    )
    schedule = fc.generation_schedule(fc.Model(prepared), ["last", "first"], person_data=new_people)
    replay = _prepare_generation(prepared, schedule)
    assert replay.calibration_identity == prepared.calibration_identity
    for current, expected_rows, expected_pools, groups, features in (
        (
            prepared,
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
            [[0, 1]] * 3 + ([[2, 3]] * 3 if by_group else [[0, 1]] * 3),
            [0, 0, 0, 1, 1, 1],
            people,
        ),
        (
            replay,
            [[0, 0], [1, 1]],
            [[2, 3] if by_group else [0, 1]] * 2,
            [1, 1],
            new_people.iloc[::-1],
        ),
    ):
        plan = current.regressions[0]
        target = plan.targets[0]
        assert_array_equal(target.row_index, expected_rows)
        assert_array_equal(target.pool_index, expected_pools)
        assert plan.coefficient_positions.shape == ((4, 2) if by_group else (2, 2))
        assert current.population.recipe["groups"] == ["reference", "other"]
        layout = next(p for p in current.parameters if p.kind == "regression_coefficient")
        entry_count = coefficients.size
        assert_array_equal(layout.entry_data.target_kind, ["theta"] * entry_count)
        assert_array_equal(layout.entry_data.scale, ["trait"] * entry_count)
        assert_array_equal(layout.entry_data.term, ["x", "x", "z", "z"] * (2 if by_group else 1))
        assert_array_equal(layout.entry_data.trait_index, [0, 1] * (entry_count // 2))
        assert_array_equal(
            layout.entry_data.population_group_index,
            np.repeat([0, 1], 4) if by_group else [-1] * entry_count,
        )
        assert_array_equal(
            current.coords["regression_coefficient_id"],
            prepared.coords["regression_coefficient_id"],
        )
        bindings, _ = build_regressions(
            current.regressions,
            {layout.kind: pt.as_tensor_variable(coefficients.ravel())},
        )
        actual = pytensor.function([], bindings[(-1, "theta")].mean, mode="JAX")()
        x = features[["x", "z"]].to_numpy() - [0.0, 2 / 3]
        expected = np.stack(
            [row @ B[group if by_group else 0] for row, group in zip(x, groups, strict=True)]
        )
        assert_allclose(actual, expected)


def test_removed_preparation_aliases_are_not_exported():
    assert not hasattr(fc, "prepare_data")
    assert not hasattr(fc, "PreparedData")


@pytest.mark.parametrize("person_id", ["pid", ("pid",)])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mapping", [False, True])
def test_factory_person_id_forms_retain_group_and_formula_alignment(
    formulae, person_id, grouped, mapping, monkeypatch
):
    from pyfcirt.covariates import design as covariate_design
    from pyfcirt.data import grouping, regressions

    calls = []
    align = grouping.align_rows

    def count_alignment(data, keys, columns, context):
        calls.append(columns)
        return align(data, keys, columns, context)

    monkeypatch.setattr(grouping, "align_rows", count_alignment)
    monkeypatch.setattr(covariate_design, "align_rows", count_alignment)
    monkeypatch.setattr(regressions, "align_rows", count_alignment)
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    source = replace(
        source,
        responses=replace(
            source.responses,
            frame=pd.concat([source.responses.frame.assign(person_id=p) for p in [1, 2, 3, 4]]),
        ),
    )
    people = pd.DataFrame(
        {
            "pid": [4, 1, 3, 2],
            "age": [40.0, 10.0, 30.0, 20.0],
            "group": ["other", "reference", "other", "reference"],
        }
    )
    population = fc.PopulationSpec(group="group", reference_group="reference") if grouped else None
    if mapping:
        arguments = dict(
            regressions={"theta": fc.Regression("age", people, id=person_id)},
            person_data=people.rename(columns={"pid": "person_id"}),
        )
    else:
        arguments = dict(
            regressions="theta ~ age", person_data=people, id_columns={"person": person_id}
        )
    state = fc.PreparedInputs.from_inputs(source, population=population, **arguments)
    expected = [("person_id",), ("pid",)] if mapping else [("person_id",)]
    assert calls == expected
    assert_array_equal(state.regressions[0].design.key_0, [1, 2, 3, 4])
    center = 15.0 if grouped else 25.0
    assert_allclose(
        state.regressions[0].design.X[:, 0], [10 - center, 20 - center, 30 - center, 40 - center]
    )
    assert_array_equal(state.population.person_group, [0, 0, 1, 1] if grouped else [0] * 4)
    if not mapping:
        with pytest.raises(ValueError, match="one column"):
            fc.PreparedInputs.from_inputs(
                source,
                population=population,
                regressions="theta ~ age",
                person_data=people,
                id_columns={"person": ("pid", "age")},
            )


def test_generation_factory_binds_regressions_simulates_and_validates_final_point(formulae):
    source = item_input(pd.DataFrame({"item_id": list("abc"), "trait": ["A"] * 3}))
    source = replace(
        source,
        responses=fc.ResponseTable(
            pd.DataFrame(
                [(p, item) for p in [1, 2, 3, 4] for item in "abc"],
                columns=["person_id", "item_id"],
            ),
            format="item_schedule",
        ),
    )
    options = dict(
        regressions="theta ~ age\nitems.loading_magnitude ~ length",
        person_data=pd.DataFrame({"person_id": [1, 2, 3, 4], "age": [10.0, 20.0, 30.0, 40.0]}),
        item_data=pd.DataFrame({"item_id": list("abc"), "length": [2.0, 3.0, 4.0]}),
        purpose="generate",
    )
    prepared = fc.PreparedInputs.from_inputs(source, **options)
    model = fc.Model(prepared)
    generated = fc.simulate(model, random_seed=21)
    assert generated.prepared.data.attrs["purpose"] == "observed"
    assert len(generated.prepared.regressions) == 2
    assert len(fc.response_data(generated)) == 12
    point = generated.idata["parameters/generation"].to_dataset()
    assert "regression_coefficient" in point
    fixed = fc.Model(fc.PreparedInputs.from_inputs(source, fixed_parameters=point, **options))
    assert {rv.name for rv in fixed.build().free_RVs} == {"theta"}
    with pytest.raises(ValueError, match="before fitting or attaching"):
        fixed.add_regressions("theta ~ age", person_data=options["person_data"])
    with pytest.raises(ValueError, match="Generate real responses"):
        model.fit()
