"""Read-only summaries of frozen specifications and supplied draws; never inference."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pyfcirt as fc
from tests.helpers import labelled_point, mixed_model, supplied_posterior


def coefficient_model():
    model = mixed_model()
    point = labelled_point(
        model,
        {
            "loading": np.ones(11),
            "statement_intercept": np.zeros(9),
            "item_intercept": [0.0],
            "category_intercept": [0.0, 0.2, -0.3],
        },
    )
    point = point.assign_coords(
        {
            name: value
            for name, value in model.prepared.output_coords.items()
            if set(value.dims) <= set(point.dims)
        }
    )
    return supplied_posterior(
        model, point, inference_method="mcmc", chains=("a", "b"), draws=range(12)
    )


def test_data_summary_without_posterior_or_graph(monkeypatch):
    model = mixed_model()
    before = model.idata.copy(deep=True)
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("summary assembled a graph"))
    summary = fc.data_coverage(model)
    pd.testing.assert_frame_equal(summary, fc.data_coverage(model))
    assert summary.index.name == "model_response_format"
    assert summary.loc["MGPPM / pick", "n_blocks"] == 1
    blocks = fc.data_coverage(model, by="block", details=True)
    items = fc.data_coverage(model, by="item", details=True)
    assert blocks.index.name == "block" and items.index.name == "item"
    assert blocks.loc["7", "model_response_format"] == "MGPPM / pick"
    assert blocks.loc["7", "n_answered"] == 1
    assert blocks.loc["7", "n_response_categories"] == 3
    assert blocks.loc["quad", "n_response_categories"] == 12
    assert blocks.loc["pair", "n_obs"] == 2
    assert items.loc["ordinal", "n_categories"] == 3
    assert items.loc["pair", "model_response_format"] == "DichotomousIRT"
    assert items.loc["pair", "link"] == "logit"
    assert pd.isna(items.loc["ordinal", "link"])
    text = model.summary()
    assert "Population:" in text and "fixed=" in text
    assert "X @" not in text
    assert "DichotomousIRT" in text and "MGPPM / pick" in text
    assert "r_hat" not in text and "hdi95" not in text
    xr.testing.assert_identical(model.idata, before)


def test_data_summary_separates_declared_answers_and_fitted_factor_support():
    import pymc as pm

    units = {"pairwise": 3, "unanswered": 3, "joint": 3, "continuous": 2}
    design = pd.DataFrame(
        [
            (block, f"{block}{position}", position, 1.0 if position == 1 else np.nan)
            for block, size in units.items()
            for position in range(1, size + 1)
        ],
        columns=["block_id", "statement_id", "position", "fix_loading"],
    ).assign(trait="A", keying=1)
    responses = pd.DataFrame(
        [
            (person, block, f"{block}{a}", f"{block}{b}", answer)
            for person, block, pairs, answer in (
                (1, "pairwise", [(1, 2), (1, 3), (2, 3)], 1),
                (1, "joint", [(1, 2)], 1),
                (2, "joint", [(1, 2), (1, 3), (2, 3)], 1),
                (1, "continuous", [(1, 2)], 0.75),
            )
            for a, b in pairs
        ],
        columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
    )
    inputs = fc.FCInput(
        design,
        fc.ResponseTable(responses, format="comparison_long"),
        model="TIRT",
        blocks=(
            fc.BlockSpec("joint", model="MGPPM"),
            fc.BlockSpec(
                "continuous",
                model=fc.TIRT(
                    link="beta_probit", priors={"response_precision": pm.HalfNormal.dist()}
                ),
                response_format=None,
                scale=fc.UnitIntervalScale(),
            ),
        ),
    )
    with pytest.warns(UserWarning, match="Statements without observed comparisons"):
        prepared = fc.PreparedInputs.from_inputs(inputs)
    summary = fc.data_coverage(prepared, by="block", details=True)
    assert summary.n_obs.to_dict() == {"pairwise": 3, "unanswered": 0, "joint": 2, "continuous": 1}
    assert summary.n_categories.iloc[:3].tolist() == [2, 2, (2, 6)]
    assert summary.n_response_categories.iloc[:3].tolist() == [6, 6, 6]
    assert pd.isna(summary.loc["continuous", "n_response_categories"])
    assert pd.isna(summary.loc["continuous", "n_categories"])


def test_model_summary_reads_class_metadata_without_restoring_prior_graphs(monkeypatch):
    model = mixed_model()

    def unexpected_reconstruction(*args, **kwargs):
        pytest.fail("Metadata-only summary reconstructed native priors or the model")

    monkeypatch.setattr(fc.Model, "build", unexpected_reconstruction)
    with monkeypatch.context() as context:
        context.setattr(fc.PreparedInputs, "observed_data", property(unexpected_reconstruction))
        assert "MGPPM" in model.summary()
    assert fc.data_coverage(model, by="item", details=True).loc["pair", "link"] == "logit"


def test_tirt_overview_is_compact_and_distinguishes_blocks_from_comparisons(monkeypatch):
    model = fc.Model(
        fc.PreparedInputs.from_inputs(fc.load_example("tirt_study", model="TIRT").data)
    )
    before = model.idata.copy(deep=True)
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("summary built a graph"))
    text = model.summary()
    assert len(text.splitlines()) <= 28
    assert text.count("cutpoint [72] ~") == 1 and "entries" not in text
    assert "loading [72] ~" in text and "residual_variance [48] ~" in text
    assert "7,200 pair comparisons" in text and "72 statements" in text
    assert "residual_variance: fixed=0.5 [24 fixed]" in text
    assert "sigma=sqrt(residual_variance)" in text
    assert "theta ~ MvNormal(mu=0, cov=R)" in text and "R [4 x 4] ~ LKJCorr(n=4, eta=1)" in text
    for internal in (
        "scope=",
        "np.float",
        "native coordinates",
        "transform=",
        "response_format=None",
        "B24",
    ):
        assert internal not in text
    overview = fc.data_coverage(model)
    assert overview.columns.tolist() == ["n_blocks", "size", "n_persons", "n_answered", "n_missing"]
    assert overview.loc["TIRT"].tolist() == [24, 3, 100, 2400, 0]
    assert fc.data_coverage(model, details=True).loc["TIRT", "n_obs"] == 7200
    detailed = model.summary(details=True)
    assert "B24" in detailed and "Blocks/items:" in detailed and "fixed=0.5" in detailed
    xr.testing.assert_identical(model.idata, before)


def test_joint_fc_summary_has_one_population_and_family_owned_priors():
    data = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a", "d"], "B": ["b", "e"], "C": ["c", "f"]},
            blocks={"dominance": ["a", "b", "c"], "preference": ["d", "e", "f"]},
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": [1, 1, 2, 2],
                    "block_id": ["dominance", "preference"] * 2,
                    "response": [[1, 2, 3], [2, 1, 3], [3, 2, 1], [1, 3, 2]],
                }
            ),
            format="block_long",
        ),
        blocks=(fc.BlockSpec("dominance", model="TIRT"),),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(data))
    text = model.summary()
    assert "TIRT / rank:" in text and "MGPPM / rank:" in text
    assert text.count("theta ~ MvNormal") == 1 and text.count("~ LKJCorr") == 1
    assert "6 pair comparisons; 2 block events" in text
    priors = text.split("Priors:", 1)[1].split("Constraints:", 1)[0]
    assert "  TIRT:" in priors and "  MGPPM:" in priors
    overview = fc.data_coverage(model)
    assert overview.n_persons.tolist() == [2, 2]
    assert overview.n_answered.tolist() == [2, 2]
    assert overview.n_blocks.sum() == 2


@pytest.mark.parametrize("difference", ["link", "prior", "categories"])
def test_distinct_measurement_laws_are_not_hidden_by_compact_grouping(difference):
    from dataclasses import replace

    import pymc as pm

    example = fc.load_example("tirt_study", model="TIRT", n_persons=2)
    model = fc.TIRT(
        link="logit" if difference == "link" else "probit",
        priors={"loading_magnitude": pm.LogNormal.dist(mu=0.25, sigma=0.3)}
        if difference == "prior"
        else None,
    )
    blocks = tuple(
        replace(
            block,
            model=model,
            scale=fc.CategoryScale((0, 1, 2)) if difference == "categories" else block.scale,
        )
        if block.block_id == "B2"
        else block
        for block in example.data.blocks
    )
    analysis = fc.Model(fc.PreparedInputs.from_inputs(replace(example.data, blocks=blocks)))
    text = analysis.summary()
    overview = fc.data_coverage(analysis)
    if difference == "prior":
        assert "mu=0.25, sigma=0.3" in text and "scope=('B2',)" in text
        assert len(overview) == 1  # data counts are independent of parameter priors
    elif difference == "link":
        assert len(overview) == 2 and set(overview.link) == {"probit", "logit"}
        assert "link multiplier=1.702" in text
    else:
        assert len(overview) == 2 and set(overview.n_categories) == {2, 3}


def test_grouped_counts_deduplicate_people_and_preserve_recorded_statuses():
    data = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["a", "b"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": [1, 1, 2], "item_id": ["a", "b", "a"], "response": [0, 1, 1]}
            ),
            format="item_long",
        ),
        administration=pd.DataFrame(
            {
                "person_id": [1, 1, 2, 2, 3, 4],
                "item_id": ["a", "b", "a", "b", "b", "b"],
                "status": [
                    "answered",
                    "answered",
                    "answered",
                    "missing",
                    "refused",
                    "not_administered",
                ],
            }
        ),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(data))
    row = fc.data_coverage(model, details=True).loc["DichotomousIRT"]
    assert row.n_items == 2 and row.n_persons == 4
    assert row.n_answered == 3 and row.n_missing == 1
    assert row.n_refused == 1 and row.n_not_administered == 1 and row.n_obs == 3
    assert fc.data_coverage(model, by="block").empty
    assert fc.data_coverage(model, by="block").index.name == "block"
    with pytest.raises(ValueError, match="by must"):
        fc.data_coverage(model, by="unknown")


def test_generation_summary_retains_unknown_answers_and_absent_channel():
    data = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["a"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1], "item_id": ["a"], "response": [None]}),
            format="item_schedule",
        ),
        administration=pd.DataFrame({"person_id": [1], "item_id": ["a"], "status": ["scheduled"]}),
    )
    state = fc.PreparedInputs.from_inputs(data, purpose="generate")
    result = fc.data_coverage(state, details=True)
    assert result.n_persons.tolist() == [1] and result.n_obs.tolist() == [0]
    assert result[["n_answered", "n_missing", "n_refused", "n_not_administered"]].isna().all().all()
    assert fc.data_coverage(state, by="block", details=True).empty


def test_coef_native_statistics_fixed_rows_and_colliding_ids(monkeypatch):
    model = coefficient_model()
    before = model.idata.copy(deep=True)
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("coef assembled a graph"))
    table = fc.coef(model, kind="stats")
    pd.testing.assert_frame_equal(table, fc.coef(model, kind="stats"))
    assert {"channel", "unit", "parameter"} <= set(table.index.names)
    assert ("fc", "pair") in table.index.to_frame(index=False).set_index(["channel", "unit"]).index
    assert ("items", "pair") in table.index.to_frame(index=False).set_index(
        ["channel", "unit"]
    ).index
    uncertainty = ["sd", "hdi95_lb", "hdi95_ub"]
    assert {"fixed", "fixed_value"}.isdisjoint(table.columns)
    fixed = table.loc[table.sd.isna()]
    assert len(fixed) and fixed[uncertainty].isna().all().all()
    assert fixed["mean"].notna().all()
    assert all(np.isclose(v, [0.0, 0.2, -0.3, 1.0]).any() for v in table["mean"].dropna())
    import arviz_stats as azs

    native = azs.summary(
        model.idata["posterior"].to_dataset()[["item_intercept"]],
        kind="stats",
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    )
    selected = fc.coef(model, items=["pair"], var_names=["item_intercept"], kind="stats")
    np.testing.assert_allclose(selected[["mean", *uncertainty]].values, native.values)
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize(
    "selection",
    [
        dict(blocks=["unknown"]),
        dict(items=["pair", "pair"]),
        dict(blocks="7"),
        dict(coords={"pair": [0]}),
        dict(fmt="long"),
        dict(group="prior"),
        dict(sample_dims=["draw"]),
    ],
)
def test_coef_rejects_unknown_selection_and_wrong_summary_modes(selection):
    with pytest.raises((ValueError, TypeError)):
        fc.coef(coefficient_model(), **selection)


def test_coef_requires_final_samples_and_explicit_mcmc_source():
    model = coefficient_model()
    model.idata.attrs["inference_method"] = "advi"
    with pytest.raises(NotImplementedError, match="MCMC"):
        fc.coef(model)
    model.idata.attrs["inference_method"] = "mcmc"
    model.idata["posterior"] = model.idata["posterior"].to_dataset().drop_vars("loading")
    with pytest.raises(ValueError, match="final"):
        fc.coef(model)


@pytest.mark.parametrize("channel", ["fc", "items"])
def test_single_channel_coef_uses_real_parameter_axes(channel, monkeypatch):
    from tests.helpers import mixed_inputs

    source = mixed_inputs()[0 if channel == "fc" else 1]
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    values = (
        {"loading": np.ones(9), "statement_intercept": np.zeros(9)}
        if channel == "fc"
        else {
            "loading": np.ones(2),
            "item_intercept": [0.25],
            "category_intercept": [0.0, -0.3, 0.2],
        }
    )
    supplied_posterior(
        model, labelled_point(model, values), inference_method="mcmc", draws=range(8)
    )
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("summary assembled a graph"))
    result = fc.coef(model, kind="stats")
    assert "channel" not in result.index.names and result.attrs["channel"] == channel
    assert result["mean"].notna().any()
    if channel == "items":
        assert "category_intercept" in result.index.get_level_values("parameter")
        assert "trait" in result.index.names and "B" in result.index.get_level_values("trait")


def test_coefficient_empty_selection_and_all_fixed_statistics_require_no_posterior(monkeypatch):
    import arviz_stats as azs

    def forbidden(*args, **kwargs):
        pytest.fail("Fixed/empty summaries must not request native statistics")

    monkeypatch.setattr(azs, "summary", forbidden)
    from tests.helpers import mixed_inputs, workflow_point

    model = mixed_model()
    fixed = fc.Model(
        fc.PreparedInputs.from_inputs(mixed_inputs(), fixed_parameters=workflow_point(model))
    )
    table = fc.coef(fixed, kind="diagnostics_median")
    statistics = ["ess_median", "ess_tail", "r_hat", "mcse_median"]
    assert len(table) and table[statistics].isna().all().all()
    assert table["mean"].notna().all()
    assert table.columns.tolist() == ["mean", *statistics, "model_response_format"]
    assert fc.coef(fixed, blocks=[], items=[]).empty
    assert fc.coef(fixed, var_names=[]).empty
    for options in [dict(ci_kind="invalid"), dict(ci_prob=1), dict(kind="invalid")]:
        with pytest.raises(ValueError, match="Invalid"):
            fc.coef(fixed, **options)


def test_coef_selects_actual_parameter_labels_before_native_statistics():
    model = coefficient_model()
    layout = next(p for p in model.prepared.parameters if p.kind == "loading")
    dimension = layout.dims[0]
    labels = model.prepared.coords[dimension][:1].tolist()
    selected = fc.coef(
        model, blocks=["pair"], var_names=[layout.kind], coords={dimension: labels}, kind="stats"
    )
    assert len(selected) == 1 and selected.index.get_level_values("parameter").tolist() == [
        "loading"
    ]


@pytest.mark.parametrize("kind", ["stats", "stats_median"])
@pytest.mark.parametrize("round_to", ["none", 2, "auto"])
def test_coef_intervals_and_explicit_rounding_follow_native_arviz(kind, round_to):
    import arviz_stats as azs
    from arviz_base import rcParams

    model = coefficient_model()
    name = "item_intercept"
    values = np.linspace(-0.41, 1.17, 24).reshape(2, 12, 1) ** 3
    model.idata["posterior"][name] = model.idata["posterior"][name].copy(data=values)
    before = dict(rcParams)
    native = azs.summary(
        model.idata["posterior"].to_dataset()[[name]],
        kind=kind,
        ci_prob=0.95,
        ci_kind="hdi",
        round_to=round_to,
    )
    result = fc.coef(model, items=["pair"], var_names=[name], kind=kind, round_to=round_to)
    pd.testing.assert_frame_equal(
        pd.DataFrame(result[native.columns]).reset_index(drop=True),
        pd.DataFrame(native).reset_index(drop=True),
    )
    assert ("eti95_lb" if kind == "stats_median" else "hdi95_lb") in result
    if round_to == "auto":
        for column, formatting in native._fmt_map.items():
            assert result._fmt_map[column][result.index[0]] == formatting[native.index[0]]
    assert dict(rcParams) == before


def test_coef_preserves_requested_typed_unit_order():
    result = fc.coef(
        coefficient_model(), blocks=["7", "pair"], items=["ordinal", "pair"], kind="stats"
    )
    selected_units = result.index.to_frame(index=False)[["channel", "unit"]].drop_duplicates()
    assert list(selected_units.itertuples(index=False, name=None)) == [
        ("fc", "7"),
        ("fc", "pair"),
        ("items", "ordinal"),
        ("items", "pair"),
    ]
    assert result.loc[("fc", "7"), "model_response_format"].eq("MGPPM / pick").all()


@pytest.mark.parametrize("items", [("a",), ("a", "b")])
def test_model_summary_reports_sharing_across_distinct_items(items):
    model = (
        fc.Model()
        .add_items(
            "[traits]\nA ~ " + " ".join(items) + "\nB ~ " + " ".join(items),
            pd.DataFrame(
                {
                    "person_id": [1, 2] * len(items),
                    "item_id": np.repeat(items, 2),
                    "response": [1, 0] * len(items),
                }
            ),
            model=fc.DichotomousIRT(kind="1PL"),
            format="item_long",
        )
        .prepare()
    )
    slots = model.prepared.parameter_uses
    assert slots.slot_position.values[slots.kind.values == "loading"].tolist() == [0, 1] * len(
        items
    )
    text = model.summary(details=True)
    assert ("Parameters shared across blocks/items" in text) == (len(items) > 1)


def _stored_draws(model, layouts, values):
    """Publish supplied canonical arrays and prepared constants, without a graph."""
    prepared = model.prepared
    model.idata["posterior"] = xr.Dataset(
        {
            layout.kind: xr.DataArray(
                np.where(np.isfinite(layout.fixed), layout.fixed, values[layout.kind]),
                dims=("chain", "draw", *layout.dims),
                coords={dim: prepared.coords[dim] for dim in layout.dims},
            )
            for layout in layouts
            if layout.kind in values
        },
        coords={"chain": ["left", "right"], "draw": np.arange(12)},
    )
    posterior = model.idata["posterior"].to_dataset()
    model.idata["posterior"] = posterior.assign_coords(
        {
            name: value
            for name, value in prepared.output_coords.items()
            if set(value.dims) <= set(posterior.dims)
        }
    )
    model.idata.attrs["inference_method"] = "mcmc"


def test_coef_handles_many_parameter_axes_and_preserves_native_formatting(monkeypatch):
    item_ids = [f"I{index:02d}" for index in range(70)]
    responses = pd.DataFrame(
        [
            (person, item, response)
            for person, response in (("P1", 0), ("P2", 1))
            for item in item_ids
        ],
        columns=["person_id", "item_id", "response"],
    )
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            fc.ItemInput(
                fc.ItemDesignMapping(traits={"trait": item_ids}),
                fc.ResponseTable(responses, format="item_long"),
                model=fc.GPCM(),
                categories=(0, 1, 2),
            )
        )
    )
    layouts = [
        p for p in model.prepared.parameters if (p.entry_data.component == "measurement").any()
    ]
    assert len(layouts) == 2 and sum(p.fixed.size for p in layouts) == 4 * len(item_ids)
    values = {
        p.kind: np.broadcast_to(
            np.linspace(0.3, 1.4, 24).reshape(2, 12, 1), (2, 12, *p.shape)
        ).copy()
        for p in layouts
    }
    _stored_draws(model, layouts, values)
    before = model.idata.copy(deep=True)
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("coef assembled a graph"))
    result = fc.coef(model, kind="stats", round_to="auto")
    assert len(result) == 4 * len(item_ids) and result.sd.isna().sum() == len(item_ids)
    fixed = result.loc[result.sd.isna()]
    assert fixed["mean"].eq(0).all()
    assert fixed[["hdi95_lb", "hdi95_ub"]].isna().all().all()
    for item in (item_ids[0], item_ids[-1]):
        selected = fc.coef(model, items=[item], kind="stats", round_to="auto")
        actual = result.loc[selected.index]
        pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(selected))
        for column, formatting in selected._fmt_map.items():
            for index, value in formatting.items():
                assert result._fmt_map[column][index] == value
    xr.testing.assert_identical(model.idata, before)


def test_shared_statement_statistics_are_computed_once_then_associated_with_blocks(monkeypatch):
    import arviz_stats as azs

    data = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a", "c"], "B": ["b"]}, blocks={"pair": ["a", "b"], "7": ["a", "c"]}
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {"person_id": [1, 1], "block_id": ["pair", "7"], "response": [[1, 2], [2, 1]]}
            ),
            format="block_long",
        ),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(data))
    layouts = [
        p for p in model.prepared.parameters if (p.entry_data.component == "measurement").any()
    ]
    values = {
        layout.kind: np.broadcast_to(
            np.linspace(0.3, 1.4, 24).reshape(2, 12, 1), (2, 12, *layout.shape)
        ).copy()
        + np.arange(np.prod(layout.shape)).reshape(layout.shape) / 10
        + offset
        for offset, layout in enumerate(layouts)
    }
    _stored_draws(model, layouts, values)
    calls = []
    native = azs.summary

    def capture(dataset, **kwargs):
        calls.append(sum(value.isel(chain=0, draw=0).size for value in dataset.data_vars.values()))
        result = native(dataset, **kwargs)
        return result.isel({dim: slice(None, None, -1) for dim in result.dims if dim != "summary"})

    monkeypatch.setattr(azs, "summary", capture)
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("summary assembled a graph"))
    assert "loading [3] ~" in model.summary()
    result = fc.coef(model, kind="stats")
    free = result.loc[result.sd.notna()]
    assert calls == [sum(len(layout.free_flat) for layout in layouts)]
    shared = free.xs("a", level="statement").xs("loading", level="parameter")
    assert len(shared) == 2
    loading = next(layout for layout in layouts if layout.kind == "loading")
    position = np.flatnonzero(loading.entry_data.statement_index.values == 0)
    from pyfcirt.analysis.summary import _select_posterior_draws

    selected = _select_posterior_draws(
        model, ["loading"], {"block": ["7", "pair"], "statement": ["a"]}
    )
    xr.testing.assert_identical(
        selected.loading,
        model.idata["posterior"].loading.isel(loading_id=position),
    )
    native_shared = native(
        model.idata["posterior"].to_dataset()[[loading.kind]].isel({loading.dims[0]: position}),
        kind="stats",
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    )
    np.testing.assert_allclose(
        shared[native_shared.columns], np.repeat(native_shared.values, 2, axis=0)
    )
    assert result.index.is_unique


@pytest.mark.parametrize("count", [2, 3, 5])
@pytest.mark.parametrize(
    "names", [("a", "b", "c"), (11, 7, 23), ("S1 vs S2", "S3", "S4"), ("S1 > S2", "S3", "S4")]
)
@pytest.mark.parametrize("higher_prefers", ["first", "second"])
@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_comparison_coefficients_use_the_frozen_pair_dimension(count, names, higher_prefers):
    data = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": [names[0], names[2]], "B": [names[1]]}, blocks={"triple": list(names)}
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": [1],
                    "block_id": ["triple"],
                    "statement_1": [names[0]],
                    "statement_2": [names[1]],
                    "response": [1],
                }
            ),
            format="comparison_long",
        ),
        model=fc.TIRT(),
        response_format=None,
        scale=fc.CategoryScale(tuple(range(count)), higher_prefers=higher_prefers),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(data))
    constant = model.prepared.data["constant_data"]
    layout = next(p for p in model.prepared.parameters if p.kind == "cutpoint")
    values = np.broadcast_to(
        layout.entry_data.cutpoint_position.values, (2, 12, *layout.shape)
    ).copy()
    _stored_draws(model, [layout], {"cutpoint": values})
    result = fc.coef(model, var_names=[layout.kind], kind="stats")
    assert result.index.names == (
        ["unit", "pair", "position", "parameter"] if count > 2 else ["unit", "pair", "parameter"]
    )
    assert result.attrs["channel"] == "fc"
    separator = " > " if higher_prefers == "first" else " < "
    labels = ("'S1 > S2'", "'S3'", "'S4'") if names[0] == "S1 > S2" else tuple(map(str, names))
    assert result.index.get_level_values("pair").drop_duplicates().tolist() == [
        separator.join((labels[0], labels[1])),
        separator.join((labels[0], labels[2])),
        separator.join(map(str, names[1:])),
    ]
    positions = constant.pair_positions.values
    np.testing.assert_array_equal(positions, [[0, 1], [0, 2], [1, 2]])
    np.testing.assert_array_equal(
        np.column_stack((constant.pair_statement_1.values, constant.pair_statement_2.values)),
        constant.fc_statement.values[0, positions],
    )
    observed = fc.response_data(model)
    assert observed.pair_label.unique().tolist() == [separator.join(labels[:2])]
    assert observed.pair.iloc[0] == names[:2]
    assert result.index.get_level_values("parameter").unique().tolist() == ["cutpoint"]
    assert layout.entry_data.pair_index.values.tolist() == np.repeat([0, 1, 2], count - 1).tolist()
    assert len(model.prepared.coords[layout.dims[0]]) == 3 * (count - 1)
    assert result.index.is_unique
    selected = fc.coef(
        model,
        var_names=[layout.kind],
        coords={"cutpoint_pair_index": [1]},
        kind="stats",
        idata=model.idata,
    )
    pd.testing.assert_frame_equal(
        selected,
        result.xs(separator.join((labels[0], labels[2])), level="pair"),
    )
    assert selected.attrs["pair"] == separator.join((labels[0], labels[2]))
    np.testing.assert_allclose(result["mean"], np.tile(np.arange(count - 1), 3))
    assert "model_response_format" not in result
    assert result.attrs["model_response_format"] == "TIRT"
    summary = fc.data_coverage(model, by="block", details=True)
    assert summary.loc["triple", "link"] == ("ordered_probit" if count > 2 else "probit")
    assert summary.loc["triple", "n_categories"] == count
    assert summary.loc["triple", "n_persons"] == 1


@pytest.mark.parametrize("members", [("d", "e", "f"), ("c", "d", "a")])
@pytest.mark.parametrize("response_format", ["rank", "pick", "mole", None])
@pytest.mark.filterwarnings("ignore:^Statements without observed comparisons:UserWarning")
def test_pair_labels_resolve_each_block_membership(members, response_format):
    blocks = {"B1": ["a", "b", "c"], "B2": list(members)}
    statements = list(dict.fromkeys([*blocks["B1"], *blocks["B2"]]))
    responses = {"rank": [1, 2, 3], "pick": [1, None, None], "mole": [1, None, 3]}
    rows = (
        pd.DataFrame(
            [(1, block, values[0], values[1], 1) for block, values in blocks.items()],
            columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
        )
        if response_format is None
        else pd.DataFrame(
            {
                "person_id": [1, 1],
                "block_id": list(blocks),
                "response": [responses[response_format]] * 2,
            }
        )
    )
    source = fc.FCInput(
        fc.FCDesignMapping(traits={"A": statements}, blocks=blocks),
        fc.ResponseTable(
            rows, format="comparison_long" if response_format is None else "block_long"
        ),
        model="TIRT",
        response_format=response_format,
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    layouts = [p for p in model.prepared.parameters if p.kind == "cutpoint"]
    _stored_draws(model, layouts, {"cutpoint": np.zeros((2, 12, *layouts[0].shape))})
    result = fc.coef(model, var_names="cutpoint", kind="stats")
    for block, (first, second, third) in blocks.items():
        assert result.xs(block, level="unit").index.get_level_values("pair").tolist() == [
            f"{first} > {second}",
            f"{first} > {third}",
            f"{second} > {third}",
        ]


def _population_model(*, regression=True, grouped=True, group_labels=("ref", "other")):
    people = pd.DataFrame(
        {
            "person_id": ["p0", "p1", "p2", "p3", "p4"],
            "group": [group_labels[0]] * 3 + [group_labels[1]] * 2,
            "x": [10.0, 15.0, 20.0, 40.0, 50.0],
        }
    )
    data = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["a"], "B": ["b"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": np.repeat(people.person_id, 2),
                    "item_id": ["a", "b"] * len(people),
                    "response": [0, 1] * len(people),
                }
            ),
            format="item_long",
        ),
    )
    population = (
        fc.PopulationSpec(
            group="group",
            reference_group=group_labels[0],
            theta_residual_scale="group",
            regression_by_group=regression,
        )
        if grouped
        else fc.PopulationSpec()
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(data, person_data=people, population=population))
    if regression:
        model.add_regressions("theta ~ x", person_data=people)
    u = np.linspace(-0.8, 1.1, 24).reshape(2, 12)
    # Unequal spacing gives a unique shortest interval despite backend roundoff.
    t = u + 0.05 * u**2
    means = np.zeros((2, 12, 2 if grouped else 1, 2))
    if grouped:
        means[:, :, 1] = np.stack([0.4 + t, -0.3 + t**2 / 2], axis=-1)
    covariance = np.broadcast_to(np.eye(2), (2, 12, 2, 2)).copy()
    covariance[:, :, 0, 1] = covariance[:, :, 1, 0] = 0.15 + 0.1 * t
    if grouped:
        other = covariance.copy()
        other[:, :, 0, 0], other[:, :, 1, 1] = 1.4 + 0.1 * t, 1.8 + 0.2 * t
        covariance = np.stack([covariance, other], axis=2)
    values = {"population_mean": means, "covariance": covariance}
    if regression:
        slopes = np.stack([0.04 + 0.02 * t, -0.03 + 0.01 * t], axis=-1)[:, :, None, :]
        if grouped:
            other = np.stack([-0.02 + 0.03 * t, 0.05 - 0.01 * t], axis=-1)[:, :, None, :]
            slopes = np.stack([slopes, other], axis=2)
        values["B"] = slopes
        expected_x = [-5, 0, 5, 25, 35] if grouped else [-17, -12, -7, 13, 23]
        np.testing.assert_allclose(model.prepared.data["covariates/r0"].X.values[:, 0], expected_x)
    published = {key: value for key, value in values.items() if key != "B"}
    if regression:
        published["regression_coefficient"] = values["B"].reshape(2, 12, -1)
    _stored_draws(model, model.prepared.parameters, published)
    return model, values


@pytest.mark.parametrize("grouped", [False, True])
def test_regression_groups_are_coordinates_and_reference_uncertainty_is_na(grouped, monkeypatch):
    model, _ = _population_model(grouped=grouped, group_labels=("Group A", "Group B"))
    before = model.idata.copy(deep=True)
    monkeypatch.setattr(fc.Model, "build", lambda *args: pytest.fail("summary built a graph"))
    result = fc.coef(model, component="regression")
    assert result.index.is_unique
    if grouped:
        assert result.index.get_level_values("population_group").unique().tolist() == [
            "Group A",
            "Group B",
        ]
    else:
        assert "population_group" not in result.index.names
    reference = fc.coef(
        model,
        component="regression",
        var_names=["population_mean"],
        coords={"population_group": ["Group A"]} if grouped else {},
        kind="stats",
    )
    assert len(reference) == 2 and reference["mean"].eq(0).all()
    assert reference[["sd", "hdi95_lb", "hdi95_ub"]].isna().all().all()
    measurement = fc.coef(model, var_names=["lower_asymptote"], kind="stats")
    assert "population_group" not in measurement.index.names and len(measurement) == 2
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize("groups", [None, ["Group A"], ["Group B"]])
def test_regression_group_filter_preserves_native_statistics(groups):
    import arviz_stats as azs

    model, _ = _population_model(group_labels=("Group A", "Group B"))
    layout = next(p for p in model.prepared.parameters if p.kind == "regression_coefficient")
    metadata = layout.entry_data
    group_indices = [
        model.prepared.coords["population_group"].tolist().index(group)
        for group in (groups or ["Group A", "Group B"])
    ]
    positions = np.flatnonzero(
        (metadata.trait_index.values == 1)
        & np.isin(metadata.population_group_index.values, group_indices)
    )
    coords = {
        "regression_coefficient_trait_index": [1],
        "regression_coefficient_population_group_index": group_indices,
    }
    native = azs.summary(
        model.idata["posterior"].to_dataset()[[layout.kind]].isel({layout.dims[0]: positions}),
        kind="stats",
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    )
    result = fc.coef(
        model, component="regression", var_names=[layout.kind], coords=coords, kind="stats"
    )
    actual_groups = (
        result.index.get_level_values("population_group").unique().tolist()
        if "population_group" in result.index.names
        else [result.attrs["population_group"]]
    )
    assert actual_groups == (groups or ["Group A", "Group B"])
    assert result.index.get_level_values("parameter").tolist() == ["x"] * len(result)
    np.testing.assert_allclose(result[native.columns], native)


def _reference_trait_draws(values, *, grouped, regression, by_group, x=None, groups=None):
    """Independent NumPy law of total covariance, with equal weights per current person."""
    groups = (
        np.array([0, 0, 0, 1, 1] if grouped else [0] * 5) if groups is None else np.array(groups)
    )
    x = (
        np.array([-5, 0, 5, 25, 35] if grouped else [-17, -12, -7, 13, 23])
        if x is None
        else np.array(x)
    )
    selections = (
        [np.flatnonzero(groups == group) for group in range(2)]
        if by_group
        else [np.arange(len(groups))]
    )
    output = {"trait_mean": [], "trait_sd": [], "trait_correlation": []}
    for selection in selections:
        means, sds, correlations = [], [], []
        for chain in range(2):
            for draw in range(12):
                person_means, person_covariances = [], []
                for person in selection:
                    group = groups[person]
                    mean = values["population_mean"][chain, draw, group].copy()
                    if regression:
                        slopes = (
                            values["B"][chain, draw, group, 0]
                            if grouped
                            else values["B"][chain, draw, 0]
                        )
                        mean += x[person] * slopes
                    person_means.append(mean)
                    cov = values["covariance"][chain, draw]
                    person_covariances.append(cov[group] if grouped else cov)
                marginal = np.mean(person_means, axis=0)
                total = sum(
                    cov + np.outer(mean - marginal, mean - marginal)
                    for mean, cov in zip(person_means, person_covariances, strict=True)
                ) / len(selection)
                sd = np.sqrt(np.diag(total))
                means.append(marginal)
                sds.append(sd)
                correlations.append(total / np.outer(sd, sd))
        for name, value in zip(output, [means, sds, correlations], strict=True):
            output[name].append(np.array(value).reshape(2, 12, *np.shape(value)[1:]))
    axes = {
        "trait_mean": ("trait",),
        "trait_sd": ("trait",),
        "trait_correlation": ("trait", "trait_aux"),
    }
    coordinates = {
        "chain": ["left", "right"],
        "draw": np.arange(12),
        "trait": ["A", "B"],
        "trait_aux": ["A", "B"],
    }
    if by_group:
        coordinates["population_group"] = ["ref", "other"]
    return xr.Dataset(
        {
            name: (
                ("chain", "draw", *(("population_group",) if by_group else ()), *axes[name]),
                np.stack(value, axis=2) if by_group else value[0],
            )
            for name, value in output.items()
        },
        coords=coordinates,
    )


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("by_group", [False, True])
def test_trait_summary_transforms_each_draw_before_native_statistics(
    regression, by_group, monkeypatch
):
    import arviz_stats as azs

    model, values = _population_model(regression=regression)
    before = model.idata.copy(deep=True)
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("summary assembled a graph"))
    monkeypatch.setattr(
        fc, "parameter_point", lambda *a, **k: pytest.fail("summary used point means")
    )
    import pyfcirt.analysis.output as output

    monkeypatch.setattr(output, "read_parameter_point", lambda *a, **k: pytest.fail("point means"))
    draws = _reference_trait_draws(values, grouped=True, regression=regression, by_group=by_group)
    expected = azs.summary(
        draws, fmt="xarray", kind="stats", ci_prob=0.95, ci_kind="hdi", round_to="none"
    )
    for trait in ["A", "B"]:
        expected.trait_correlation.loc[dict(trait=trait, trait_aux=trait)] = np.nan
        expected.trait_correlation.loc[dict(trait=trait, trait_aux=trait, summary="mean")] = 1
    if by_group:
        expected.trait_mean.loc[dict(population_group="ref")] = np.nan
        expected.trait_mean.loc[dict(population_group="ref", summary="mean")] = 0
        if not regression:
            expected.trait_sd.loc[dict(population_group="ref")] = np.nan
            expected.trait_sd.loc[dict(population_group="ref", summary="mean")] = 1
    native_summary = azs.summary
    calls = []

    def capture(samples, **options):
        calls.append(set(samples.data_vars))
        return native_summary(samples, **options)

    monkeypatch.setattr(azs, "summary", capture)
    actual = fc.trait_summary(model, by_group=by_group, kind="stats")
    assert calls == [{"trait_mean", "trait_sd", "trait_correlation"}]
    expected = xr.Dataset(
        {name: expected[name].transpose(*value.dims) for name, value in actual.data_vars.items()}
    )
    xr.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    xr.testing.assert_identical(actual, fc.trait_summary(model, by_group=by_group, kind="stats"))
    assert actual.attrs == {
        "target": "marginal_population",
        "reference_distribution": "current_persons",
    }
    xr.testing.assert_identical(model.idata, before)


def test_trait_summary_native_filters_median_intervals_and_no_group_equivalence(monkeypatch):
    import arviz_stats as azs
    import jax

    model, values = _population_model(grouped=False)
    compiled_shapes = []
    jit = jax.jit

    def compile(function):
        compiled = jit(function)

        def evaluate(*values):
            result = compiled(*values)
            compiled_shapes.append(tuple(value.shape for value in result))
            return result

        return evaluate

    monkeypatch.setattr(jax, "jit", compile)
    options = dict(
        kind="stats_median", ci_prob=0.8, var_names=["trait_sd"], coords={"trait": ["B"]}
    )
    result = fc.trait_summary(model, **options)
    draws = _reference_trait_draws(values, grouped=False, regression=True, by_group=False)
    expected = azs.summary(
        draws[["trait_sd"]].sel(trait=["B"]),
        fmt="xarray",
        kind="stats_median",
        ci_prob=0.8,
        ci_kind="hdi",
        round_to="none",
    )
    expected = expected.transpose("trait", "summary")
    xr.testing.assert_allclose(result, expected)
    assert result.summary.values.tolist() == ["median", "mad", "eti80_lb", "eti80_ub"]
    xr.testing.assert_identical(result, fc.trait_summary(model, by_group=True, **options))
    assert compiled_shapes and all(shapes == ((12, 1),) for shapes in compiled_shapes)


@pytest.mark.parametrize("by_group", [False, True])
@pytest.mark.parametrize(
    "options, names, coords",
    [
        (dict(var_names=[]), [], {}),
        (dict(filter_vars="like", var_names=["mean"]), ["trait_mean"], {}),
        (
            dict(filter_vars="regex", var_names=["mean|sd"]),
            ["trait_mean", "trait_sd"],
            {"trait": ["B", "A"]},
        ),
        (
            dict(var_names=["trait_correlation"]),
            ["trait_correlation"],
            {"trait": "B", "trait_aux": "A"},
        ),
        (dict(var_names=["~trait_correlation"]), ["trait_mean", "trait_sd"], {"trait": []}),
    ],
)
def test_trait_summary_selects_output_entries_before_retaining_draws(
    by_group, options, names, coords
):
    model, _ = _population_model()
    complete = fc.trait_summary(model, by_group=by_group, kind="stats")
    expected = xr.Dataset({name: complete[name] for name in names}, attrs=complete.attrs)
    expected = expected.sel(coords).reset_coords(drop=True)
    actual = fc.trait_summary(model, by_group=by_group, kind="stats", coords=coords, **options)
    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("integer_labels", [False, True])
@pytest.mark.parametrize("scalar_chain", [False, True])
def test_trait_summary_preserves_draw_selection_before_native_statistics(
    integer_labels, scalar_chain
):
    import arviz_stats as azs

    model, values = _population_model()
    draws = _reference_trait_draws(values, grouped=True, regression=True, by_group=True)
    labels = {
        "chain": [1, 0] if integer_labels else ["left", "right"],
        "draw": [29, 4, 10, 7, 3, 81, 12, 2, 8, 1, 0, 19],
    }
    model.idata["posterior"] = model.idata["posterior"].to_dataset().assign_coords(labels)
    draws = draws.assign_coords(labels)
    selected = {"chain": [labels["chain"][1]], "draw": [19, 4, 2, 3, 0, 29]}
    expected = azs.summary(
        draws[["trait_mean"]].sel(population_group=["other"], trait=["B"], **selected),
        fmt="xarray",
        kind="stats",
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    ).transpose("population_group", "trait", "summary")
    actual = fc.trait_summary(
        model,
        by_group=True,
        kind="stats",
        var_names=["trait_mean"],
        coords={
            "population_group": ["other"],
            "trait": ["B"],
            **selected,
            "chain": selected["chain"][0] if scalar_chain else selected["chain"],
        },
    )
    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("axis,index", [("chain", 1), ("draw", 3)])
@pytest.mark.parametrize("container", ["numpy", "coordinate"])
def test_trait_summary_accepts_zero_dimensional_sample_labels(axis, index, container):
    import arviz_stats as azs

    model, values = _population_model()
    before = model.idata.copy(deep=True)
    posterior = model.idata["posterior"].to_dataset()
    label = posterior[axis].isel({axis: index})
    if container == "numpy":
        label = label.values
    draws = _reference_trait_draws(values, grouped=True, regression=True, by_group=True)
    expected = azs.summary(
        draws[["trait_mean"]].sel(
            population_group=["other"], trait=["B"], **{axis: [label.item()]}
        ),
        fmt="xarray",
        kind="stats",
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    ).transpose("population_group", "trait", "summary")
    actual = fc.trait_summary(
        model,
        by_group=True,
        kind="stats",
        var_names=["trait_mean"],
        coords={"population_group": ["other"], "trait": ["B"], axis: label},
    )
    xr.testing.assert_allclose(actual, expected)
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize(
    "coords,error",
    [
        ({"chain": [1]}, KeyError),
        ({"draw": [99]}, KeyError),
        ({"chain": np.array("missing")}, KeyError),
        ({"draw": xr.DataArray(99)}, KeyError),
        ({"chain": []}, ValueError),
        ({"draw": []}, ValueError),
    ],
)
def test_trait_summary_rejects_unknown_or_empty_sample_labels(coords, error):
    model, _ = _population_model()
    with pytest.raises(error):
        fc.trait_summary(model, kind="stats", coords=coords)


def test_trait_summary_no_regression_baseline_and_one_trait_fixed_diagonal():
    model, _ = _population_model(regression=False, grouped=False)
    result = fc.trait_summary(model, kind="stats")
    np.testing.assert_array_equal(result.trait_mean.sel(summary="mean"), [0, 0])
    np.testing.assert_array_equal(result.trait_sd.sel(summary="mean"), [1, 1])
    assert result.trait_mean.sel(summary="sd").isnull().all()
    assert result.trait_sd.sel(summary="sd").isnull().all()
    unfitted = fc.Model(model.prepared)
    fixed_only = unfitted.trait_summary(kind="stats", var_names=["trait_mean", "trait_sd"])
    xr.testing.assert_identical(fixed_only, result[["trait_mean", "trait_sd"]])
    diagonal = unfitted.trait_summary(
        kind="stats", var_names="trait_correlation", coords={"trait": "A", "trait_aux": "A"}
    )
    assert diagonal.trait_correlation.sel(summary="mean").item() == 1
    assert diagonal.trait_correlation.drop_sel(summary="mean").isnull().all()
    with pytest.raises(ValueError, match="final.*covariance"):
        unfitted.trait_summary(kind="stats")
    data = fc.ItemInput(
        fc.ItemDesignMapping(traits={"T": ["a"]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": [1], "item_id": ["a"], "response": [1]}), format="item_long"
        ),
    )
    fixed_model = fc.Model(fc.PreparedInputs.from_inputs(data))
    fixed = fc.trait_summary(fixed_model, kind="all")
    assert fixed.trait_correlation.shape == (1, 1, len(fixed.summary))
    assert fixed.trait_correlation.sel(summary="mean").item() == 1
    assert fixed.trait_correlation.drop_sel(summary="mean").isnull().all()
    with pytest.raises(ValueError, match="labelled posterior"):
        fc.trait_summary(fixed_model, coords={"chain": ["right"]})
    _stored_draws(fixed_model, [], {})
    posterior = fixed_model.idata["posterior"].to_dataset()
    for coords in (
        {"chain": "right", "draw": 3},
        {"chain": np.array("right"), "draw": np.array(3)},
        {"chain": posterior.chain.isel(chain=1), "draw": posterior.draw.isel(draw=3)},
    ):
        selected = fc.trait_summary(fixed_model, kind="all", coords=coords)
        xr.testing.assert_identical(selected, fixed)
    with pytest.raises(KeyError):
        fc.trait_summary(fixed_model, coords={"chain": ["missing"]})


@pytest.mark.parametrize("inference_method", ["advi", "map", "laplace"])
def test_trait_summary_rejects_non_mcmc_and_missing_final_nodes(inference_method):
    model, _ = _population_model()
    model.idata.attrs["inference_method"] = inference_method
    with pytest.raises(NotImplementedError, match="MCMC"):
        fc.trait_summary(model)
    model.idata.attrs["inference_method"] = "mcmc"
    means = model.trait_summary(var_names="trait_mean")
    full = model.idata["posterior"].to_dataset()
    model.idata["posterior"] = full.drop_vars("covariance")
    xr.testing.assert_identical(model.trait_summary(var_names="trait_mean"), means)
    model.idata["posterior"] = full.drop_vars("regression_coefficient")
    with pytest.raises(ValueError, match="final.*regression_coefficient"):
        fc.trait_summary(model)


def test_regression_coefficients_use_final_native_nodes_and_resolved_scales(monkeypatch):
    import arviz_stats as azs

    model = _item_regression_draws("mixed")
    prepared, graph = model.prepared, model.build()
    bindings = dict(model._bindings.parameters)
    before = model.idata.copy(deep=True)
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("summary assembled a graph"))
    layout = next(p for p in prepared.parameters if p.kind == "regression_coefficient")
    assert bindings[layout.kind] is graph[layout.kind]
    assert not any("::" in name for name in graph.named_vars)
    entries = layout.entry_data
    assert set(entries.target_kind.values) == {"theta", "loading"}
    result = fc.coef(model, component="regression", kind="stats")
    assert result.index.is_unique and set(result.index.get_level_values("target")) == {
        "theta",
        "loading",
    }
    assert result.loc[result.sd.isna(), "mean"].notna().all()
    positions = np.flatnonzero(entries.target_kind.values == "theta")
    coords = {layout.dims[0]: prepared.coords[layout.dims[0]][positions]}
    selected = fc.coef(
        model, component="regression", var_names=[layout.kind], coords=coords, kind="stats"
    )
    native = azs.summary(
        model.idata["posterior"].to_dataset()[[layout.kind]].isel({layout.dims[0]: positions}),
        kind="stats",
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    )
    np.testing.assert_allclose(selected[native.columns], native)
    assert selected.attrs["scale"] == "trait"
    assert set(result.xs("loading", level="target")["scale"]) == {"log_loading"}
    for key, binding in bindings.items():
        assert model._bindings.parameters[key] is binding and binding is graph[binding.name]
    for selection in [dict(blocks=[]), dict(items=["pair"])]:
        with pytest.raises(ValueError, match="blocks/items"):
            fc.coef(model, component="regression", **selection)
    assert fc.coef(model, component="regression", var_names=[], kind="stats").empty
    xr.testing.assert_identical(model.idata, before)


def test_fixed_scoring_traits_use_current_people_and_frozen_training_center(monkeypatch):
    reference, values = _population_model()
    entries = {}
    for layout in reference.prepared.parameters:
        if not layout.fixed_for_scoring:
            continue
        name = layout.kind
        if name in reference.idata["posterior"]:
            value = reference.idata["posterior"][name].isel(chain=0, draw=0, drop=True)
        else:
            initial = 1.0 if name in {"loading", "upper_asymptote"} else 0.0
            value = xr.DataArray(
                np.where(np.isnan(layout.fixed), initial, layout.fixed),
                dims=layout.dims,
                coords={dim: reference.prepared.coords[dim] for dim in layout.dims},
            )
        entries[name] = value
    point = xr.Dataset(entries)
    people = pd.DataFrame(
        {"person_id": ["new0", "new1"], "group": ["other", "other"], "x": [50.0, 60.0]}
    )
    responses = fc.ResponseTable(
        pd.DataFrame({"person_id": people.person_id, "item_id": ["a", "a"], "response": [0, 1]}),
        format="item_long",
    )
    monkeypatch.setattr(
        fc.Model, "build", lambda self: pytest.fail("scoring summary built a graph")
    )
    scored = fc.score_model(reference, responses, parameters=point, person_data=people)
    np.testing.assert_allclose(scored.prepared.data["covariates/r0"].X.values[:, 0], [35, 45])
    result = fc.trait_summary(scored, kind="all")
    slope = values["B"][0, 0, 1, 0]
    mean = values["population_mean"][0, 0, 1] + 40 * slope
    total = values["covariance"][0, 0, 1] + 25 * np.outer(slope, slope)
    sd = np.sqrt(np.diag(total))
    np.testing.assert_allclose(result.trait_mean.sel(summary="mean"), mean)
    np.testing.assert_allclose(result.trait_sd.sel(summary="mean"), sd)
    np.testing.assert_allclose(
        result.trait_correlation.sel(summary="mean"), total / np.outer(sd, sd)
    )
    for value in result.data_vars.values():
        assert value.drop_sel(summary="mean").isnull().all()
    with pytest.raises(ValueError, match="ref.*no people"):
        fc.trait_summary(scored, by_group=True)
    regression = fc.coef(scored, component="regression", kind="stats")
    assert regression["mean"].notna().all()
    assert regression[["sd", "hdi95_lb", "hdi95_ub"]].isna().all().all()


@pytest.mark.parametrize(
    "options",
    [
        dict(by_group=1),
        dict(fmt="wide"),
        dict(group="prior"),
        dict(sample_dims=["draw"]),
        dict(ci_prob=1),
    ],
)
def test_trait_summary_rejects_incompatible_native_modes(options):
    model, _ = _population_model()
    with pytest.raises((TypeError, ValueError)):
        fc.trait_summary(model, **options)


def _item_regression_draws(mode="single", *, formula="1", options=None, population_groups=False):
    """Three typed items and explicit draws, with no native model construction."""
    traits = (
        {"A": ["pair", "7"], "B": ["third"]} if mode == "groups" else {"A": ["pair", "7", "third"]}
    )
    items = {"third": {"model": fc.GPCM(), "categories": (0, 1, 2)}} if mode == "mixed" else {}
    responses = pd.DataFrame(
        {"person_id": [1, 2, 1], "item_id": ["pair", "7", "third"], "response": [0, 1, 0]}
    )
    if population_groups:
        responses.loc[len(responses)] = [3, "pair", 0]
        responses.loc[len(responses)] = [4, "third", 1]
    source = fc.ItemInput(
        fc.ItemDesignMapping(traits=traits),
        fc.ResponseTable(responses, format="item_long"),
        items=items,
    )
    people = pd.DataFrame({"person_id": [1, 2], "x": [-1.0, 1.0]})
    population = None
    if population_groups:
        people.loc[2] = [3, 0.0]
        people.loc[3] = [4, 2.0]
        people["group"] = ["Group A", "Group B", "Group A", "Group B"]
        population = fc.PopulationSpec(
            group="group", reference_group="Group A", regression_by_group=True
        )
    model = fc.Model(
        fc.PreparedInputs.from_inputs(source, person_data=people, population=population)
    )
    model.add_regressions(
        f"theta ~ x\nitems.loading_magnitude ~ {formula}",
        person_data=people,
        item_data=pd.DataFrame(
            {"item_id": ["pair", "7", "third"], "residual_sd": [-1.0, 0.0, 1.0]}
        ),
        options={"items.loading_magnitude": options} if options else None,
    )
    layouts = [
        p
        for p in model.prepared.parameters
        if p.kind in {"population_mean", "regression_coefficient", "regression_sd"}
    ]
    values = {}
    for layout in layouts:
        t = np.linspace(0.2, 0.9, 24).reshape(2, 12, *([1] * len(layout.shape))) ** 3
        if layout.kind == "regression_coefficient":
            offset = np.where(layout.entry_data.target_kind.values == "theta", 0.1, 0.4)
        else:
            offset = 0.8 if layout.kind == "regression_sd" else 0
        positions = layout.entry_data.local_position.values.reshape(layout.shape)
        values[layout.kind] = t + offset + positions / 10
    _stored_draws(model, layouts, values)
    return model


def test_grouped_population_and_shared_measurement_regressions_use_distinct_index_levels():
    model = _item_regression_draws("mixed", population_groups=True)
    result = fc.coef(model, component="regression", kind="stats")
    assert result.index.is_unique and "population_group" in result.index.names
    assert "parameter_set_index" not in result.index.names
    theta = result.xs("theta", level="target")
    assert set(theta.index.get_level_values("population_group")) == {"Group A", "Group B"}
    shared = result.xs("loading", level="target")
    assert shared.index.get_level_values("population_group").isna().all()
    layout = next(p for p in model.prepared.parameters if p.kind == "regression_coefficient")
    entries = layout.entry_data
    positions = np.flatnonzero(entries.target_kind.values == "loading")
    selected = fc.coef(
        model,
        component="regression",
        var_names=[layout.kind],
        coords={layout.dims[0]: model.prepared.coords[layout.dims[0]][positions]},
        kind="stats",
    )
    assert "population_group" not in selected.index.names
    assert len(selected) == len(positions)
    np.testing.assert_allclose(
        selected["mean"],
        shared.loc[shared.index.get_level_values("parameter") == "Intercept", "mean"],
    )
    assert fc.coef(model, component="regression", var_names=[], kind="stats").empty


@pytest.mark.parametrize("mode", ["single", "groups", "mixed"])
def test_regression_index_and_text_follow_actual_independent_groups(mode, monkeypatch):
    import arviz_stats as azs

    model = _item_regression_draws(mode)
    before = model.idata.copy(deep=True)
    calls, native = [], azs.summary

    def capture(dataset, **kwargs):
        calls.append(tuple(dataset.data_vars))
        result = native(dataset, **kwargs)
        return result.isel({dim: slice(None, None, -1) for dim in result.dims if dim != "summary"})

    monkeypatch.setattr(azs, "summary", capture)
    monkeypatch.setattr(fc.Model, "build", lambda *args: pytest.fail("summary built a graph"))
    result = fc.coef(model, component="regression", kind="stats")
    assert result.index.is_unique and len(calls) == 1
    measurement = result.xs("loading", level="target")
    labels = measurement.index.get_level_values("parameter")
    mean = np.mean(np.linspace(0.2, 0.9, 24) ** 3)
    np.testing.assert_allclose(measurement.loc[labels == "Intercept", "mean"].iloc[0], mean + 0.4)
    np.testing.assert_allclose(
        measurement.loc[labels == "regression_sd", "mean"].iloc[0], mean + 0.8
    )
    if mode == "groups":
        assert "trait" in result.index.names and set(
            measurement.index.get_level_values("trait")
        ) == {"A", "B"}
    elif mode == "mixed":
        assert set(result.index.get_level_values("model")) == {
            "Population",
            "DichotomousIRT",
            "GPCM",
        }
    text = model.summary()
    assert "log-loading" in text and "conditional LogNormal" in text
    assert "independent parameter pool(s)" in text
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize("kind", ["stats", "stats_median"])
def test_filtered_regression_statistics_keep_native_intervals_and_compact_context(kind):
    import arviz_stats as azs

    model = _item_regression_draws("mixed")
    layout = next(p for p in model.prepared.parameters if p.kind == "regression_coefficient")
    entries = layout.entry_data
    position = np.flatnonzero(
        (entries.target_kind.values == "loading") & (entries.term.values == "Intercept")
    )[0]
    coords = {layout.dims[0]: [model.prepared.coords[layout.dims[0]][position]]}
    native = azs.summary(
        model.idata["posterior"].to_dataset()[[layout.kind]].isel({layout.dims[0]: [position]}),
        kind=kind,
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    )
    result = fc.coef(
        model, component="regression", var_names=[layout.kind], coords=coords, kind=kind
    )
    assert len(result) == 1 and result.index.is_unique
    assert result.attrs["model"] == "DichotomousIRT"
    np.testing.assert_allclose(result[native.columns], native)
    assert all(pd.api.types.is_numeric_dtype(result[column]) for column in native.columns)
    assert ("eti95_lb" if kind == "stats_median" else "hdi95_lb") in result


def test_residual_sd_predictor_and_remaining_sd_have_distinct_semantic_labels():
    import pymc as pm

    model = _item_regression_draws(
        formula="residual_sd",
        options={
            "priors": {"common": pm.HalfNormal.dist(sigma=0.4)},
            "parameter_residual_sd": pm.HalfNormal.dist(sigma=0.4),
        },
    )
    result = fc.coef(model, component="regression", kind="stats")
    measurement = result.xs("loading", level="target")
    labels = measurement.index.get_level_values("parameter")
    assert "residual_sd" in labels and "regression_sd" in labels and result.index.is_unique
    layout = next(p for p in model.prepared.parameters if p.kind == "regression_coefficient")
    assert "residual_sd" in layout.entry_data.term.values
    text = model.summary(details=True)
    assert "regression_coefficient" in text and "regression_sd" in text and "residual_sd" in text


def test_fixed_scoring_omits_inactive_item_regressions_before_context(monkeypatch):
    import arviz_stats as azs

    from tests.helpers import mixed_inputs, workflow_point

    model = mixed_model()
    model.add_regressions(
        "theta ~ x\nitems.loading_magnitude ~ 1",
        person_data=pd.DataFrame({"person_id": [1, 2], "x": [-1.0, 1.0]}),
        item_data=pd.DataFrame({"item_id": ["pair", "ordinal"]}),
    )
    parameters = workflow_point(model)
    parameters["regression_coefficient"] = xr.zeros_like(parameters.regression_coefficient)
    fixed = fc.score_model(
        model,
        {"fc": mixed_inputs()[0].responses, "items": mixed_inputs()[1].responses},
        parameters=parameters,
        person_data=pd.DataFrame({"person_id": [1, 2], "x": [-1.0, 1.0]}),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Fixed or empty regression output requested native statistics")

    monkeypatch.setattr(azs, "summary", forbidden)
    result = fc.coef(fixed, component="regression", kind="stats")
    assert result.index.is_unique and "population_group" not in result.index.names
    assert result.attrs["target"] == "theta"
    assert "regression_sd" not in fixed.build().named_vars
    assert result["mean"].notna().all()
    assert result[["sd", "hdi95_lb", "hdi95_ub"]].isna().all().all()
    empty = fc.coef(fixed, component="regression", var_names=[], kind="stats")
    assert empty.empty and "parameter" in empty.index.names


@pytest.mark.parametrize("details", [False, True])
def test_model_summary_is_independent_of_coefficient_expansion(monkeypatch, details):
    import importlib

    import arviz_stats as azs

    module = importlib.import_module("pyfcirt.analysis.summary")
    model = _item_regression_draws("mixed")
    before = model.idata.copy(deep=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Model summary expanded coefficient tables, statistics or a graph")

    monkeypatch.setattr(module, "_entry_tables", forbidden)
    monkeypatch.setattr(module, "_coefficient_result", forbidden)
    monkeypatch.setattr(azs, "summary", forbidden)
    monkeypatch.setattr(fc.Model, "build", forbidden)
    text = model.summary(details=details)
    assert "theta ~ MvNormal(mu=X @ regression_coefficient, cov=1)" in text
    assert "regression_coefficient" in text and "regression_sd" in text
    assert "conditional LogNormal" in text
    xr.testing.assert_identical(model.idata, before)


def test_population_summary_does_not_substitute_posterior_estimates_for_free_parameters():
    model, _ = _population_model(regression=False)
    text = model.summary()
    assert "mu=population_mean[group]" in text
    assert "cov=diag(population_sd[group]) @ R @ diag(population_sd[group])" in text
    assert "population_mean [2] ~ Normal" in text and "population_sd [2] ~ HalfNormal" in text


def test_scoring_summary_displays_supplied_population_values_without_unused_priors():
    from tests.helpers import mixed_inputs, workflow_point

    model = mixed_model()
    point = workflow_point(model)
    covariance = "covariance"
    point[covariance].values[:] = [[1.0, 0.25], [0.25, 1.0]]
    scored = fc.score_model(
        model,
        {"fc": mixed_inputs()[0].responses, "items": mixed_inputs()[1].responses},
        parameters=point,
    )
    before = scored.idata.copy(deep=True)
    text = scored.summary()
    assert "theta ~ MvNormal(mu=0, cov=[[1.0, 0.25], [0.25, 1.0]])" in text
    assert "LKJCorr" not in text and "population_sd" not in text
    assert "supplied parameter values" in text
    xr.testing.assert_identical(scored.idata, before)


@pytest.mark.parametrize("first_id", [1, 2**53])
def test_response_tables_preserve_integer_ids_across_fc_and_items(first_id):
    ids = [first_id, first_id + 1]
    fc_input = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a", "b", "c", "d"]},
            blocks={ids[0]: ["a", "b"], ids[1]: ["c", "d"]},
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p1", "p2", "p3"] * 2,
                    "block_id": np.repeat(ids, 3),
                    "response": [[1, 2], [1, 2], [2, 1], [1, 2], [2, 1], [2, 1]],
                }
            ),
            format="block_long",
        ),
        model="TIRT",
    )
    item_input = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ids}),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": ["p1", "p2", "p3"] * 2,
                    "item_id": np.repeat(ids, 3),
                    "response": [1, 1, 0, 1, 0, 0],
                }
            ),
            format="item_long",
        ),
    )
    state = fc.PreparedInputs.from_inputs((fc_input, item_input))
    before = state.data.copy(deep=True)
    long = fc.response_data(state)
    summary = fc.response_counts(long)
    for axis in ("block", "item"):
        selected = long.loc[long[axis].notna()]
        assert selected[axis].drop_duplicates().tolist() == ids
        assert selected[f"{axis}_label"].drop_duplicates().tolist() == list(map(str, ids))
        counts = summary.loc[summary[axis].notna()].set_index([axis, "category"])
        assert len(counts) == 4
        for index, unit_id in enumerate(ids):
            assert counts.loc[(unit_id, 1), "count"] == 2 - index
            assert counts.loc[(unit_id, 1), "n_valid"] == 3
            assert counts.loc[(unit_id, 1), "proportion"] == pytest.approx((2 - index) / 3)
    xr.testing.assert_identical(state.data, before)


@pytest.mark.parametrize("categories", [(1, "1", "'1'"), (1, "1", "'1'", '"1"')])
def test_response_display_labels_do_not_merge_cascading_collisions(categories):
    responses = [value for count, value in enumerate(categories, 1) for _ in range(count)]
    source = fc.FCInput(
        fc.FCDesignMapping(traits={"A": ["a"], "B": ["b"]}, blocks={"pair": ["a", "b"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": range(len(responses)),
                    "block_id": "pair",
                    "response": responses,
                }
            ),
            format="block_long",
        ),
        model=fc.TIRT(loading_anchors={"a": 1.0, "b": 1.0}),
        response_format=None,
        scale=fc.CategoryScale(categories),
    )
    state = fc.PreparedInputs.from_inputs(source)
    long = fc.response_data(state)
    summary = fc.response_counts(long)
    assert summary.category.tolist() == list(categories)
    assert summary.category_label.is_unique
    assert summary["count"].tolist() == list(range(1, len(categories) + 1))
    np.testing.assert_allclose(summary.proportion, summary["count"] / len(responses))
    assert long.category_label.nunique() == len(categories)


def test_graded_dyad_display_projects_whole_block_to_its_single_pair():
    source = fc.FCInput(
        fc.FCDesignMapping(traits={"A": [11], "B": [7]}, blocks={"B": [11, 7]}),
        fc.ResponseTable(
            pd.DataFrame({"person_id": ["p"], "block_id": ["B"], "response": [4]}),
            format="block_long",
        ),
        model="MGPPM",
        response_format=None,
        scale=fc.CategoryScale((0, 1, 2, 3, 4)),
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    assert prepared.constant_data.obs_pair.values.tolist() == [-1]
    display = fc.response_data(prepared, representation="pairwise")
    assert display.pair.tolist() == [(11, 7)]
    assert display.category.tolist() == [4]


@pytest.mark.parametrize("model", ["TIRT", "TIRTRank", "MGPPM"])
def test_rank_categories_recover_from_object_typed_comparisons(model):
    frame = pd.DataFrame(
        [
            (person, "B", first, second, bit)
            for person, bits in [("p1", [1, 1, 1]), ("p2", [0, 0, 0]), ("p3", [1, None, 1])]
            for (first, second), bit in zip([("a", "b"), ("a", "c"), ("b", "c")], bits, strict=True)
        ],
        columns=["person_id", "block_id", "statement_1", "statement_2", "response"],
        dtype=object,
    )
    source = fc.FCInput(
        fc.FCDesignMapping(traits={"A": ["a", "c"], "B": ["b"]}, blocks={"B": ["a", "b", "c"]}),
        fc.ResponseTable(frame, format="comparison_long"),
        model=model,
        response_format="rank",
    )
    prepared = fc.PreparedInputs.from_inputs(source)
    categories = fc.response_data(prepared, representation="categories").set_index("person")
    assert categories.loc["p1", "category"] == ("a", "b", "c")
    assert categories.loc["p2", "category"] == ("c", "b", "a")
    assert categories.loc["p3", "category"] == ("a", "b", "c")
    assert "pair" not in categories
    pd.testing.assert_frame_equal(source.responses.frame, frame)


@pytest.mark.parametrize(
    "selection,channel,model_response_format",
    [
        ({}, None, None),
        ({"blocks": ["pair"]}, "fc", "MGPPM / rank"),
        ({"blocks": ["pair", "7"]}, "fc", None),
        ({"items": ["pair"]}, "items", "DichotomousIRT"),
        ({"items": ["pair", "ordinal"]}, "items", None),
        ({"var_names": ["item_intercept"]}, "items", "DichotomousIRT"),
        ({"var_names": ["item_intercept"], "filter_vars": "like"}, "items", "DichotomousIRT"),
    ],
)
def test_measurement_coef_context_uses_final_selection(selection, channel, model_response_format):
    model = coefficient_model()
    before = model.idata.copy(deep=True)
    result = fc.coef(model, kind="stats", **selection)
    assert result.attrs.get("channel") == channel
    assert ("channel" in result.index.names) == (channel is None)
    assert result.attrs.get("model_response_format") == model_response_format
    assert ("model_response_format" in result) == (model_response_format is None)
    assert result.index.is_unique
    assert all("::" not in label for label in result.index.get_level_values("parameter"))
    assert not any(
        label.startswith("theta") for label in result.index.get_level_values("parameter")
    )
    xr.testing.assert_identical(model.idata, before)


def test_measurement_coef_empty_context_is_not_inferred():
    result = fc.coef(coefficient_model(), var_names=[])
    assert result.empty and "parameter" in result.index.names
    assert "channel" not in result.attrs and "model_response_format" not in result.attrs


def test_measurement_coef_short_labels_preserve_native_variable_and_coord_filters():
    model = fc.Model(fc.PreparedInputs.from_inputs(fc.load_example("joint_fc", model="TIRT").data))
    constant = model.prepared.data["constant_data"]
    layout = next(p for p in model.prepared.parameters if p.kind == "residual_variance")
    values = np.broadcast_to(np.linspace(0.5, 1.5, 24).reshape(2, 12, 1), (2, 12, *layout.shape))
    _stored_draws(model, [layout], {"residual_variance": values})
    dimension = layout.dims[0]
    position = layout.free_flat[0]
    statement = constant[dimension].values[position]
    before = model.idata.copy(deep=True)
    result = fc.coef(model, var_names=[layout.kind], coords={dimension: [statement]}, kind="stats")
    short = fc.coef(
        model, var_names=["residual_variance"], coords={dimension: [statement]}, kind="stats"
    )
    pd.testing.assert_frame_equal(short, result)
    assert set(result.index.get_level_values("parameter")) == {"residual_variance"}
    np.testing.assert_allclose(result["mean"], 1.0)
    xr.testing.assert_identical(model.idata, before)


def test_measurement_coef_mixed_fc_families_keep_model_response_format_and_owner_rows():
    source = fc.FCInput(
        fc.FCDesignMapping(
            traits={"A": ["a"], "B": ["b"]}, blocks={"pair": ["a", "b"], "7": ["a", "b"]}
        ),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": [1, 1],
                    "block_id": ["pair", "7"],
                    "response": [[1, 2], [2, 1]],
                }
            ),
            format="block_long",
        ),
        blocks=(fc.BlockSpec("pair", model="TwoPLRank"),),
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    layouts = [
        p for p in model.prepared.parameters if (p.entry_data.component == "measurement").any()
    ]
    values = {
        layout.kind: np.broadcast_to(
            np.linspace(0.3, 1.4, 24).reshape(2, 12, *([1] * len(layout.shape))),
            (2, 12, *layout.shape),
        )
        for layout in layouts
    }
    _stored_draws(model, layouts, values)
    loadings = [layout.kind for layout in layouts if layout.kind == "loading"]
    loading = next(p for p in layouts if p.kind == "loading")
    sets = loading.entry_data.parameter_set_index.values
    draws = model.idata["posterior"]["loading"].values.copy()
    draws[..., sets == sets.max()] += 1
    model.idata["posterior"]["loading"] = model.idata["posterior"]["loading"].copy(data=draws)
    result = fc.coef(model, var_names="loading", kind="stats")
    native = fc.coef(model, var_names=loadings, kind="stats")
    pd.testing.assert_frame_equal(result, native)
    assert result.attrs["channel"] == "fc" and "channel" not in result.index.names
    assert "model_response_format" not in result.attrs
    assert set(result.model_response_format) == {"TwoPLRank / rank", "MGPPM / rank"}
    assert result.index.is_unique
    loading_rows = result.loc[result.index.get_level_values("parameter").isin(["loading"])]
    assert loading_rows.groupby(level="unit", sort=False)["mean"].mean().nunique() == 2
    assert result.index.get_level_values("unit").unique().tolist() == ["pair", "7"]
    selected = fc.coef(model, blocks=["pair"], kind="stats")
    assert (
        "model_response_format" not in selected
        and selected.attrs["model_response_format"] == "TwoPLRank / rank"
    )


@pytest.mark.parametrize("categories", [(0, 1, 2), (1, 2, 3)])
def test_compact_measurement_labels_keep_declared_integer_categories_distinct(categories):
    source = fc.ItemInput(
        fc.ItemDesignMapping(traits={"A": ["item"]}),
        fc.ResponseTable(
            pd.DataFrame(
                {
                    "person_id": range(3),
                    "item_id": "item",
                    "response": list(categories),
                }
            ),
            format="item_long",
        ),
        model="GPCM",
        categories=categories,
    )
    model = fc.Model(fc.PreparedInputs.from_inputs(source))
    layout = next(p for p in model.prepared.parameters if p.kind == "category_intercept")
    values = np.broadcast_to(np.linspace(0.3, 1.4, 24).reshape(2, 12, 1), (2, 12, *layout.shape))
    _stored_draws(model, [layout], {"category_intercept": values})
    result = fc.coef(model, var_names=[layout.kind], kind="stats")
    assert len(result) == 3 and result.index.is_unique
    assert result.index.get_level_values("parameter").tolist() == ["category_intercept"] * 3
    assert result.index.get_level_values("category").tolist() == list(categories)
    assert result.attrs["channel"] == "items" and result.attrs["model_response_format"] == "GPCM"
    assert layout.kind in model.idata["posterior"]


@pytest.mark.parametrize("selection", [{}, {"blocks": ["pair"]}, {"items": ["pair", "ordinal"]}])
@pytest.mark.parametrize("requested", ["loading", ["loading"], ["loading", "item_intercept"]])
def test_coef_canonical_names_select_matching_usage_entries(requested, selection):
    model = coefficient_model()
    before = model.idata.copy(deep=True)
    names = [requested] if isinstance(requested, str) else requested
    actual = fc.coef(model, var_names=requested, kind="stats", **selection)
    expected = fc.coef(model, var_names=names, kind="stats", **selection)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.index.is_unique
    assert set(actual.index.get_level_values("parameter")) <= set(names)
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize(
    "unknown", ["loading_magnitude", "loading_raw", "o0::loading_magnitude", "not_a_parameter"]
)
def test_coef_canonical_matching_rejects_aliases_and_unknown_names(unknown):
    with pytest.raises(KeyError):
        fc.coef(coefficient_model(), var_names=[unknown], kind="stats")


def test_coef_exclusions_and_empty_selections_keep_native_semantics():
    model = coefficient_model()
    actual = fc.coef(model, var_names=["~loading"], kind="stats")
    names = [p.kind for p in model.prepared.parameters if p.kind != "loading"]
    expected = fc.coef(model, var_names=names, kind="stats")
    pd.testing.assert_frame_equal(actual, expected)
    assert fc.coef(model, var_names=[], kind="stats").empty
    assert fc.coef(model, blocks=[], items=[], var_names=["loading"], kind="stats").empty
    assert fc.coef(model, blocks=["pair"], var_names=["category_intercept"], kind="stats").empty


@pytest.mark.parametrize("selection", ["exact", "like", "regex", "exclusion"])
def test_tirt_coef_omits_statement_effects_by_default_but_keeps_explicit_selection(
    monkeypatch, selection
):
    model = fc.Model(
        fc.PreparedInputs.from_inputs(
            fc.FCInput(
                fc.FCDesignMapping(
                    traits={"A": ["a", "c"], "B": ["b"]},
                    blocks={"triple": ["a", "b", "c"]},
                ),
                fc.ResponseTable(
                    pd.DataFrame(
                        {"person_id": [1], "block_id": ["triple"], "response": [[1, 2, 3]]}
                    ),
                    format="block_long",
                ),
                model=fc.TIRT(),
            )
        )
    )
    layouts = [
        p for p in model.prepared.parameters if (p.entry_data.component == "measurement").any()
    ]
    values = {
        layout.kind: np.where(
            np.isnan(layout.fixed),
            np.broadcast_to(
                np.linspace(0.3, 1.4, 24).reshape(2, 12, *([1] * len(layout.shape))),
                (2, 12, *layout.shape),
            ),
            layout.fixed,
        )
        for layout in layouts
    }
    _stored_draws(model, layouts, values)
    before = model.idata.copy(deep=True)
    monkeypatch.setattr(fc.Model, "build", lambda self: pytest.fail("coef built a graph"))
    default = fc.coef(model, kind="stats")
    assert not default.index.get_level_values("parameter").isin(["statement_effect"]).any()
    pd.testing.assert_frame_equal(default, fc.coef(model, idata=model.idata, kind="stats"))
    pd.testing.assert_frame_equal(
        default, fc.coef(model, var_names=["~statement_effect"], kind="stats")
    )
    name = next(layout.kind for layout in layouts if layout.kind == "statement_effect")
    options = {
        "exact": {"var_names": [name]},
        "like": {"var_names": ["statement_effect"], "filter_vars": "like"},
        "regex": {"var_names": [r"^statement_effect$"], "filter_vars": "regex"},
        "exclusion": {"var_names": ["~loading"]},
    }[selection]
    selected = fc.coef(model, kind="stats", **options)
    effects = selected.loc[selected.index.get_level_values("parameter").isin(["statement_effect"])]
    assert len(effects) == 3
    np.testing.assert_allclose(effects["mean"], 0.85)
    xr.testing.assert_identical(model.idata, before)


@pytest.mark.parametrize(
    "filter_vars,queries",
    [
        ("like", ["loading"]),
        ("like", ["~loading"]),
        ("regex", [r"^(loading|item_intercept)$"]),
        ("regex", [r"^loading$"]),
    ],
)
def test_coef_pattern_filtering_matches_native_canonical_names(filter_vars, queries):
    from arviz_base import extract

    model = coefficient_model()
    names = [p.kind for p in model.prepared.parameters]
    selected = extract(
        xr.Dataset({name: 0 for name in names}),
        var_names=queries,
        filter_vars=filter_vars,
        sample_dims=[],
        combined=False,
        keep_dataset=True,
    )
    actual = fc.coef(model, var_names=queries, filter_vars=filter_vars, kind="stats")
    expected = fc.coef(model, var_names=list(selected.data_vars), kind="stats")
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("kind", ["population_mean", "regression_coefficient", "regression_sd"])
def test_regression_coef_canonical_kinds_follow_component_metadata(kind):
    model = _item_regression_draws(mode="mixed")
    layout = next(p for p in model.prepared.parameters if p.kind == kind)
    assert (layout.entry_data.component == "regression").all()
    actual = fc.coef(model, component="regression", var_names=[kind], kind="stats")
    assert len(actual) and actual.index.is_unique
    if kind == "regression_coefficient":
        assert set(actual.index.get_level_values("target")) == {"theta", "loading"}


def test_r_posterior_label_export_roundtrips_large_integers_and_literal_strings():
    from types import SimpleNamespace

    from pyfcirt.analysis.summary import _draw_labels, _select_posterior_draws

    prepared = mixed_model().prepared
    wide = 2**53 + 1
    for people in ([wide - 1, wide], ["integer:9007199254740992", "integer:9007199254740993"]):
        samples = xr.Dataset(
            {"theta": (("chain", "draw", "person"), [[[1, 2], [3, 4]]])},
            coords={"chain": [0], "draw": [wide - 1, wide], "person": people},
        )
        model = SimpleNamespace(
            prepared=prepared, idata=xr.DataTree.from_dict({"posterior": samples})
        )
        selected = _select_posterior_draws(
            model,
            ["theta"],
            {"draw": [_draw_labels(samples.draw)[1]], "person": [_draw_labels(samples.person)[1]]},
        )
        assert selected.theta.item() == 4
        assert selected.person.item() == people[1]
        assert selected.draw.item() == wide
