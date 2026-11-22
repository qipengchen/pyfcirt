"""Independent simulated TIRT trials. Keep tuning and validation seeds disjoint."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import arviz_stats
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from scipy.stats import binomtest

import pyfcirt as fc
from pyfcirt.covariates.model import indexed_mean

CANDIDATES = {
    "hierarchical": (0.25, 0.50, 0.75),
    "pairwise": (0.75, 1.50, 3.00),
    "truncated": ((0.75, 0.25), (1.50, 0.50), (2.00, 0.50)),
}
CONFIRMATION = {
    "hierarchical": (0.50,),
    "pairwise": (3.00,),
    "truncated": ((1.50, 0.50, 0.55), (1.50, 0.50, 0.75), (1.50, 0.50, 1.00)),
}
CASES = (
    "null",
    "partial",
    "unbalanced",
    "three_groups",
    "mean_only",
    "mean_dif",
    "theta_only",
    "theta_dif",
    "sd_only",
    "corr_only",
    "same_direction",
    "small_effect",
)


def trial(scenario, seed, size):
    rng = np.random.default_rng(seed)
    labels = list("ABC" if scenario == "three_groups" else "AB")
    proportion = (
        [0.5, 0.3, 0.2]
        if len(labels) == 3
        else ([0.75, 0.25] if scenario == "unbalanced" else [0.50, 0.50])
    )
    counts = np.floor(np.array(proportion) * size).astype(int)
    counts[0] += size - counts.sum()
    site = rng.permutation(np.repeat(np.arange(len(labels)), counts))
    sites = np.asarray(labels)[site]
    people = pd.DataFrame(
        {
            "person_id": np.arange(size),
            "site": pd.Categorical(sites, categories=labels),
        }
    )
    shifts = np.zeros((len(labels), 2))
    if scenario in {
        "mean_only",
        "mean_dif",
        "theta_only",
        "theta_dif",
        "same_direction",
        "three_groups",
    }:
        shifts[1] = 0.45, -0.30
        if len(labels) == 3:
            shifts[2] = -0.30, 0.35
    residual_sd = np.ones((len(labels), 2))
    correlations = np.full(len(labels), 0.35)
    if scenario == "sd_only":
        residual_sd[1] = 1.30, 0.75
    if scenario == "corr_only":
        correlations[1] = -0.25
    theta = rng.normal(size=(size, 2))
    rho = correlations[site]
    theta[:, 1] = rho * theta[:, 0] + np.sqrt(1 - rho**2) * theta[:, 1]
    theta = residual_sd[site] * theta + shifts[site]
    pairs = [(0, 0)] * 4 + [(1, 1)] * 4 + [(0, 1)] * 4
    block_ids = [f"b{i + 1:02d}" for i in range(len(pairs))]
    loading = np.tile([0.9, 1.1], (len(pairs), 1))
    keying = np.tile([1, -1], (len(pairs), 1))
    keying[10:, 1] = 1
    rows = [
        (block, position + 1, f"{block}_{position + 1}", ("T1", "T2")[index], keying[j, position])
        for j, (block, pair) in enumerate(zip(block_ids, pairs, strict=True))
        for position, index in enumerate(pair)
    ]
    design = pd.DataFrame(rows, columns=["block_id", "position", "statement_id", "trait", "keying"])
    dif = np.zeros((len(labels), len(pairs)))
    if scenario in {
        "partial",
        "unbalanced",
        "mean_dif",
        "theta_dif",
        "three_groups",
        "small_effect",
    }:
        magnitude = 0.65 if scenario == "small_effect" else 1.5
        dif[1, [1, 5, 9]] = magnitude * np.asarray([1, -1, 1])
        if len(labels) == 3:
            dif[2, [2, 5, 10]] = magnitude * np.asarray([-1, -1, 1])
    if scenario == "same_direction":
        dif[1, [0, 1, 2, 4, 5, 6, 8, 9]] = 1.25
    selected = theta[:, np.asarray(pairs)]
    utility_difference = (selected * loading * keying) @ np.array([1, -1])
    cuts = np.linspace(-0.5, 0.5, len(pairs)) + dif[site]
    observed = (utility_difference + rng.normal(size=cuts.shape) >= cuts).astype(np.int64)
    answers = pd.DataFrame(
        {
            "person_id": np.repeat(np.arange(size), len(pairs)),
            "block_id": np.tile(block_ids, size),
            "response": observed.ravel(),
        }
    )
    spec = fc.TIRT(
        loading_anchors=dict(zip(design.statement_id, loading.ravel(), strict=True)),
        residuals="fixed",
    )
    inputs = fc.FCInput(
        design,
        fc.ResponseTable(answers, format="block_long"),
        model=spec,
        response_format=None,
        scale=fc.CategoryScale((0, 1)),
    )
    if scenario.startswith("theta_"):
        analysis = fc.Model().prepare(
            inputs, regressions={"theta": fc.Regression("site", people, id="person_id")}
        )
    else:
        grouped = scenario in {
            "mean_only",
            "mean_dif",
            "same_direction",
            "three_groups",
            "sd_only",
            "corr_only",
        }
        population = fc.PopulationSpec(
            group="site" if grouped else None,
            reference_group="A" if grouped else None,
            theta_residual_scale="group" if scenario == "sd_only" else "unit",
            theta_correlation="group" if scenario == "corr_only" else "shared",
        )
        analysis = fc.Model().prepare(
            inputs, population=population, person_data=people if grouped else None
        )
    return analysis, people, block_ids, labels, dif, shifts, residual_sd, correlations


def evaluate(case, seed, n, family, setting, draws, tune, chains, sampler):
    analysis, people, blocks, labels, dif, shifts, sds, correlations = trial(case, seed, n)
    if family == "hierarchical":
        prior, options = family, {"parameter_residual_sd": pm.HalfNormal.dist(sigma=setting)}
    elif family == "pairwise":
        prior, options = "pairwise_l1", {"strength": setting}
    else:
        strength, cutoff, *base = setting
        prior = "truncated_l1"
        options = {"strength": strength, "cutoff": cutoff, "base_sd": base[0] if base else 1.0}
    analysis.add_dif_terms(
        "fc.cutpoints", person_data=people, groups="site", prior=prior, prior_options=options
    )
    start = perf_counter()
    idata = analysis.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=1,
        random_seed=seed + 1000,
        nuts_sampler=sampler,
        target_accept=0.90,
        progressbar=False,
    )
    summary = analysis.dif_summary()
    true = np.asarray(
        [
            dif[labels.index(row.compared), blocks.index(row.units[0])]
            - dif[labels.index(row.reference), blocks.index(row.units[0])]
            for row in summary.itertuples()
        ]
    )
    posterior = idata.posterior.dataset
    estimated = [
        name
        for name in ("cutpoint", "regression_coefficient", "population_mean", "covariance")
        if name in posterior
    ]
    diagnostic_draws = posterior[estimated]
    rp = arviz_stats.rhat(diagnostic_draws)
    bulk = arviz_stats.ess(diagnostic_draws, method="bulk")
    worst_rhat = max(float(rp[name].max()) for name in rp.data_vars)
    min_ess = min(float(bulk[name].min()) for name in bulk.data_vars)

    prepared = analysis.prepared

    def mean(name):
        if name in posterior:
            return posterior[name].mean(("chain", "draw"))
        layout = prepared.parameter_layouts[name]
        return xr.DataArray(
            layout.fixed,
            dims=layout.dims,
            coords={dim: prepared.constant_data[dim] for dim in layout.dims},
        )

    def by_site(value):
        if "population_group" not in value.dims:
            return value.expand_dims(site=labels).values
        if prepared.population.groups:
            return value.sel(population_group=labels).values
        return value.isel(population_group=0, drop=True).expand_dims(site=labels).values

    means = by_site(mean("population_mean"))
    theta_plan = prepared.theta_regression
    if theta_plan is not None:
        (target,) = theta_plan.targets
        coefficients = mean("regression_coefficient").values.ravel()[theta_plan.coefficient_entries]
        sites = people.set_index("person_id").loc[prepared.constant_data.person.values, "site"]
        rows = [int(np.flatnonzero(sites == label)[0]) for label in labels]
        means = means + np.asarray(
            indexed_mean(
                theta_plan.design.X.values,
                coefficients,
                target.row_index[rows],
                target.pool_index[rows],
            )
        )
    estimates_sd = by_site(mean("population_sd"))
    estimates_rho = by_site(mean("trait_correlation").sel(trait="T1", trait_aux="T2"))
    info = dict(
        case=case,
        seed=seed,
        n=n,
        family=family,
        setting=setting,
        rhat=worst_rhat,
        bulk_ess=min_ess,
        seconds=round(perf_counter() - start, 2),
        divergence=int(idata.sample_stats.dataset.diverging.sum()),
        truth=true.tolist(),
        estimates=summary["mean"].to_list(),
        detected=summary.detected.to_list(),
        positive_direction=(summary["mean"] > 0).to_list(),
        intervals=summary[["lower", "upper"]].values.tolist(),
        true_means=shifts.tolist(),
        true_sds=sds.tolist(),
        true_rho=correlations.tolist(),
        group_labels=labels,
        group_counts=people.site.value_counts(sort=False).to_dict(),
        means=means.tolist(),
        sds=estimates_sd.tolist(),
        rho=estimates_rho.tolist(),
    )
    return info


def _rates(counts):
    """Compute contrast rates from totals over independent datasets."""
    null, alternative, false_positive, true_positive, correct_direction, affected = counts.T
    with np.errstate(invalid="ignore", divide="ignore"):
        false_rate = false_positive / null
        sensitivity = true_positive / alternative
        return np.column_stack(
            (
                false_rate,
                sensitivity,
                (true_positive + null - false_positive) / (null + alternative),
                (sensitivity + 1 - false_rate) / 2,
                correct_direction / true_positive,
                affected,
            )
        )


def summarize(path):
    """Print estimates and bootstrap intervals that resample whole datasets."""
    rng = np.random.default_rng(73_002)
    trials = pd.DataFrame(map(json.loads, path.read_text(encoding="utf-8").splitlines()))
    if trials.empty:
        raise ValueError("There are no completed DIF trials")
    trials["setting"] = trials.setting.astype(str)
    replicate_keys = ["case", "family", "setting", "seed"]
    if trials.duplicated(replicate_keys).any():
        raise ValueError("A DIF replication occurs more than once in the trial records")
    fields = (
        "false_positive_rate",
        "sensitivity",
        "accuracy",
        "balanced_accuracy",
        "direction_accuracy",
        "any_false_positive",
    )
    for configuration, group in trials.groupby(["case", "family", "setting"], sort=False):
        counts = []
        for row in group.itertuples():
            truth = np.asarray(row.truth)
            detected = np.asarray(row.detected)
            null = truth == 0
            fp = int(np.sum(detected & null))
            tp = int(np.sum(detected & ~null))
            correct = int(np.sum(detected & ~null & ((truth > 0) == row.positive_direction)))
            counts.append([np.sum(null), np.sum(~null), fp, tp, correct, int(fp > 0)])
        counts = np.asarray(counts)
        replicates = len(counts)
        total = counts.sum(axis=0, keepdims=True)
        estimate = _rates(total)[0]
        estimate[-1] /= replicates
        idx = rng.integers(0, replicates, size=(4_000, replicates))
        bootstrap = _rates(counts[idx].sum(axis=1))
        bootstrap[:, -1] /= replicates
        rates = {}
        for index, field in enumerate(fields):
            if np.isfinite(estimate[index]):
                if field == "any_false_positive":
                    lower, upper = binomtest(int(total[0, 5]), replicates).proportion_ci()
                    interval = [round(lower, 4), round(upper, 4)]
                else:
                    sample = bootstrap[np.isfinite(bootstrap[:, index]), index]
                    interval = (
                        [None, None]
                        if replicates == 1 or np.ptp(sample) == 0
                        else np.quantile(sample, [0.025, 0.975]).round(4).tolist()
                    )
                rates[field] = [round(float(estimate[index]), 4), *interval]
        if total[0, 1]:
            all_correct = int(np.sum(counts[:, 4] == counts[:, 1]))
            lower, upper = binomtest(all_correct, replicates).proportion_ci()
            rates["all_planted_detected_in_correct_direction"] = [
                round(all_correct / replicates, 4),
                round(lower, 4),
                round(upper, 4),
            ]
        contrast_error = np.abs(np.asarray(group.estimates.tolist()) - group.truth.tolist())
        null = np.asarray(group.truth.tolist()) == 0
        recovery = {
            name: round(float(contrast_error[mask].mean()), 4)
            for name, mask in (("true_zero", null), ("planted", ~null))
            if mask.any()
        }
        population_recovery = {}
        for name, estimated, truth in (
            ("trait_mean", group.means, group.true_means),
            ("trait_sd", group.sds, group.true_sds),
            ("trait_correlation", group.rho, group.true_rho),
        ):
            error = np.abs(np.asarray(estimated.tolist()) - truth.tolist()).mean(axis=0)
            population_recovery[name] = {
                site: round(float(np.mean(site_error)), 4)
                for site, site_error in zip(group.group_labels.iloc[0], error, strict=True)
            }
        print(
            json.dumps(
                {
                    "case": configuration[0],
                    "family": configuration[1],
                    "setting": configuration[2],
                    "replicates": replicates,
                    "null_contrasts": int(total[0, 0]),
                    "nonzero_contrasts": int(total[0, 1]),
                    "false_positives": int(total[0, 2]),
                    "true_positives": int(total[0, 3]),
                    "correct_directions": int(total[0, 4]),
                    "datasets_with_false_positive": int(total[0, 5]),
                    "rates_with_95_percent_intervals": rates,
                    "dif_mean_absolute_error": recovery,
                    "population_mean_absolute_error_by_site": population_recovery,
                    "min_bulk_ess": float(group.bulk_ess.min()),
                    "max_rhat": float(group.rhat.max()),
                    "fits_above_rhat_1_05": int((group.rhat > 1.05).sum()),
                    "divergences": int(group.divergence.sum()),
                }
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "tune", "confirm", "validation", "summary"), required=True
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--reps", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--family", choices=tuple(CANDIDATES), action="append")
    parser.add_argument("--sampler", choices=("numpyro", "blackjax"), default="numpyro")
    args = parser.parse_args()
    if args.mode == "summary":
        summarize(args.output)
        return
    fc.set_platform(args.platform)
    if args.mode == "smoke":
        cases = args.case or ("mean_dif",)
        configs = {family: values[1:2] for family, values in CANDIDATES.items()}
        draws, tune, chains, n, reps, origin = 160, 260, 2, 180, 1, 609100
    elif args.mode == "tune":
        cases = args.case or ("null", "mean_dif", "unbalanced", "three_groups")
        configs = CANDIDATES
        draws, tune, chains, n, reps, origin = 350, 500, 2, 320, 2, 610100
    elif args.mode == "confirm":
        cases = args.case or ("null", "mean_dif", "three_groups")
        configs = CONFIRMATION
        draws, tune, chains, n, reps, origin = 700, 1100, 4, 400, 1, 612100
    else:
        if args.selected is None:
            parser.error("--mode validation requires --selected")
        cases = args.case or CASES
        configs = {name: (value,) for name, value in json.loads(args.selected.read_text()).items()}
        if configs.keys() != CANDIDATES.keys():
            parser.error("--selected requires all three prior families")
        draws, tune, chains, n, reps, origin = 600, 1000, 4, 400, 12, 620100
    if args.family:
        configs = {name: configs[name] for name in args.family}
    repeats = reps if args.reps is None else args.reps
    if args.start < 0 or repeats < 1 or args.start + repeats > 100:
        parser.error("replication indices must fit in the scenario's 100-seed range")
    seen = set()
    if args.output.exists():
        for row in map(json.loads, args.output.read_text().splitlines()):
            seen.add((row["case"], row["seed"], row["family"], str(row["setting"])))
    with args.output.open("a", encoding="utf8", buffering=1) as handle:
        for case in cases:
            for repeat in range(args.start, args.start + repeats):
                seed = origin + 100 * CASES.index(case) + repeat
                for family, values in configs.items():
                    for setting in values:
                        if (
                            case,
                            seed,
                            family,
                            str(list(setting) if isinstance(setting, tuple) else setting),
                        ) in seen:
                            continue
                        result = evaluate(
                            case, seed, n, family, setting, draws, tune, chains, args.sampler
                        )
                        handle.write(json.dumps(result, allow_nan=False) + "\n")
                        detected = np.asarray(result["detected"])
                        truth = np.asarray(result["truth"]) != 0
                        print(
                            case,
                            seed,
                            family,
                            setting,
                            int((detected & truth).sum()),
                            int((detected & ~truth).sum()),
                            "rhat",
                            round(result["rhat"], 3),
                            "div",
                            result["divergence"],
                            flush=True,
                        )


if __name__ == "__main__":
    main()
