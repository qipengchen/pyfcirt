"""Render gallery previews from saved summaries; this script does not fit models."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BLUE, PURPLE, ORANGE = "#196d8c", "#8560a0", "#c67d36"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titlepad": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#373a3c",
        "text.color": "#373a3c",
    }
)


def canvas():
    return plt.subplots(figsize=(7, 4.5), layout="constrained")


def finish(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=170)
    plt.close(fig)


def render(data, output):
    output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((data / "provenance.json").read_text())

    frame = pd.read_csv(data / "tirt_rank_variances.csv")
    fig, ax = canvas()
    for i, row in frame.iterrows():
        color = "#90969a" if row.fixed else BLUE
        if not row.fixed:
            ax.plot([row.low, row.high], [i, i], color=color, lw=2.5)
        ax.plot(row["mean"], i, "s" if row.fixed else "o", color=color, ms=6)
    ax.set(
        yticks=range(len(frame)),
        yticklabels=frame.label,
        xlabel="Statement residual variance",
        title="Residual estimates and 95% HDIs",
    )
    ax.invert_yaxis()
    ax.plot([], [], "o", color=BLUE, label="Estimated")
    ax.plot([], [], "s", color="#90969a", label="Fixed reference")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    finish(fig, output, "tirt_rank_analysis")

    frame = pd.read_csv(data / "two_pl_loadings.csv")
    density = pd.read_csv(data / "two_pl_loading_densities.csv")
    fig, ax = canvas()
    for i, row in frame.iterrows():
        curve = density.loc[density.statement == row.statement]
        width = 0.34 * curve.density / curve.density.max()
        ax.fill_between(curve.x, i - width, i + width, color=BLUE, alpha=0.28, linewidth=0)
        ax.plot([row.low, row.high], [i, i], color=BLUE, lw=2)
        ax.plot(row["mean"], i, "o", color=BLUE, ms=4)
    ax.set(
        yticks=range(len(frame)),
        yticklabels=frame.label,
        xlabel="Loading magnitude",
        title="Posterior loading distributions",
    )
    ax.invert_yaxis()
    finish(fig, output, "two_pl_rank_analysis")

    frame = pd.read_csv(data / "mgppm_intercepts.csv")
    fig, ax = canvas()
    ax.hlines(range(len(frame)), frame.low, frame.high, color=BLUE, linewidth=2.5)
    ax.plot(frame["mean"], range(len(frame)), "o", color=BLUE)
    ax.axvline(0, color="0.6", linestyle="--")
    ax.set(
        yticks=range(len(frame)),
        yticklabels=frame.label,
        xlabel="Utility-reducing intercept",
        title="Free intercept estimates and 95% HDIs",
    )
    ax.invert_yaxis()
    finish(fig, output, "mgppm_analysis")

    frame = pd.read_csv(data / "ggum_rank_location_densities.csv")
    intervals = pd.read_csv(data / "ggum_rank_locations.csv").set_index("position")
    fig, ax = canvas()
    for (position, curve), color in zip(
        frame.groupby("position"), [BLUE, PURPLE, ORANGE], strict=True
    ):
        interval = intervals.loc[position]
        ax.fill_between(
            curve.location,
            0,
            curve.density,
            color=color,
            alpha=0.18,
            where=curve.location.between(interval.low, interval.high),
            linewidth=0,
        )
        ax.plot(curve.location, curve.density, color=color, lw=2, label=f"Position {position}")
    ax.axvline(0, color="#a0a5aa", lw=1, ls="--")
    ax.set(
        xlabel="Location (latent SD units)",
        ylabel="Posterior density",
        title="Ideal-point location posteriors",
    )
    ax.legend(frameon=False, fontsize=10)
    finish(fig, output, "ggum_rank_analysis")

    frame = pd.read_csv(data / "gtum_locations.csv")
    fig, ax = canvas()
    for i, row in frame.iterrows():
        ax.plot([row.low, row.high], [i, i], color=ORANGE, lw=3)
        ax.plot(row["mean"], i, "o", color=ORANGE, ms=8)
    ax.axvline(0, color="#a0a5aa", lw=1, ls="--")
    ax.set(
        yticks=range(len(frame)),
        yticklabels=frame.statement,
        ylim=(-0.65, len(frame) - 0.35),
        xlabel="Location (latent SD units)",
        title="Ideal-point estimates and 95% HDIs",
    )
    ax.invert_yaxis()
    finish(fig, output, "gtum_locations")

    frame = pd.read_csv(data / "loading_covariate_density.csv")
    summary = provenance["covariates"]["summary"]
    fig, ax = canvas()
    ax.plot(frame.coefficient, frame.density, color=PURPLE, lw=2)
    ax.fill_between(
        frame.coefficient,
        0,
        frame.density,
        where=frame.coefficient.between(summary["low"], summary["high"]),
        color=PURPLE,
        alpha=0.25,
        label="95% HDI",
    )
    ax.axvline(summary["mean"], color=PURPLE, ls="--", lw=1.5, label="Posterior mean")
    ax.set(
        xlabel="Feature coefficient (log-loading scale)",
        ylabel="Posterior density",
        title="Statement-feature association",
    )
    ax.legend(frameon=False, fontsize=10)
    finish(fig, output, "tirt_covariates_analysis")

    effect = pd.read_csv(data / "covariate_loading_effect.csv")
    fig, ax = canvas()
    ax.fill_between(effect.predictor_change, effect.low, effect.high, color=BLUE, alpha=0.2)
    ax.plot(effect.predictor_change, effect["median"], color=BLUE)
    ax.axhline(1, color="0.5", linestyle="--")
    ax.axvline(0, color="0.7", linewidth=1)
    predictor = pd.read_csv(data / "covariate_loading_predictor.csv")
    sns.rugplot(x=predictor.predictor, color="0.4", ax=ax)
    ax.set(xlabel="Standardized log-loading predictor", ylabel="Loading multiplier")
    finish(fig, output, "covariate_loading_effect")

    # These are functions of the saved estimates, with the actual fitted-score support.
    curves = pd.read_csv(data / "statement_curves_data.csv")
    scores = pd.read_csv(data / "statement_curves_scores.csv")
    support = scores.loc[scores.trait == "agency", "score_mean"]
    low, high = support.quantile([0.05, 0.95])
    for model, slug in [
        ("TIRTRank", "tirt_rank"),
        ("TwoPLRank", "two_pl_rank"),
        ("MGPPM", "mgppm"),
        ("GGUMRank", "ggum_rank"),
    ]:
        selected = curves.loc[(curves.model == model) & (curves.trait == "agency")].copy()
        selected["Statement"] = selected.statement.str.replace(f"{model}_", "", regex=False)
        fig, ax = canvas()
        ax.axvspan(low, high, color="0.93")
        sns.lineplot(data=selected, x="theta", y="value", hue="Statement", palette="deep", ax=ax)
        sns.rugplot(x=support, color="0.4", alpha=0.25, ax=ax)
        label = (
            "Reference-scale endorsement"
            if model in {"TwoPLRank", "GGUMRank"}
            else "Expected utility"
        )
        ax.set(xlabel="Agency, θ", ylabel=label)
        finish(fig, output, f"{slug}_statement_functions")

    # Reuse prepared response labels; do not reproduce rank/category encoding here.
    import pyfcirt as fc

    for model, slug in [
        ("TIRTRank", "tirt_rank"),
        ("TwoPLRank", "two_pl_rank"),
        ("MGPPM", "mgppm"),
        ("GGUMRank", "ggum_rank"),
    ]:
        example = fc.load_example(f"simulated_{slug}", model=model, n_persons=50)
        prepared = fc.PreparedInputs.from_inputs(example.data)
        counts = fc.response_counts(fc.response_data(prepared, blocks=["B1"]))
        fig, ax = canvas()
        sns.barplot(
            data=counts, x="proportion", y="category_label", color="C0", errorbar=None, ax=ax
        )
        ax.set(xlabel="Observed proportion", ylabel="B1 preference order", xlim=(0, 1))
        finish(fig, output, f"{slug}_observed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, default=Path("gallery_figures"))
    args = parser.parse_args()
    render(args.data, args.output)
