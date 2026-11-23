---
html_theme.sidebar_secondary.remove: true
---

# Examples

Choose a measurement model and follow its data, estimation and model-checking workflow.

## Forced-choice models

::::{grid} 1 2 2 3
:gutter: 2
:class-container: model-gallery fc-model-gallery

:::{grid-item-card}
:shadow: none
:link: tutorials/tirt_workflow
:link-type: doc
:header: **Thurstonian item response theory (TIRT)** Dominance · pairwise comparisons
:footer: HEXACO data: information for Honesty–Humility and Openness.

```{image} _static/thumbnails/tirt_analysis.png
:alt: Diagonal test information plotted against posterior mean Honesty–Humility and Openness scores for 120 HEXACO respondents.
```
:::

:::{grid-item-card}
:shadow: none
:link: tutorials/tirt_rank_workflow
:link-type: doc
:header: **Thurstonian item response theory for joint rankings (TIRTRank)** Dominance · joint Gaussian rankings
:footer: HEXACO data: observed and predicted frequencies for six triplet orders.

```{image} _static/thumbnails/tirt_rank_analysis.png
:alt: Observed proportions and 95% posterior predictive intervals for all six triplet orders from 120 HEXACO respondents.
```
:::

:::{grid-item-card}
:shadow: none
:link: tutorials/two_pl_rank_workflow
:link-type: doc
:header: **Two-parameter logistic model for ranking (2PLM-RANK)** Dominance · successive choices
:footer: HEXACO data: fitted statement endorsement curves.

```{image} _static/thumbnails/two_pl_rank_workflow.png
:alt: Endorsement curves for three HEXACO statements evaluated at their fitted discriminations and intercepts.
```
:::

:::{grid-item-card}
:shadow: none
:link: tutorials/mgppm_workflow
:link-type: doc
:header: **Multidimensional generalized partial preference model (MGPPM)** Dominance · ranking-pattern probabilities
:footer: HEXACO data: observed and predicted best–worst responses.

```{image} _static/thumbnails/mgppm_workflow.png
:alt: Observed proportions and 95% posterior predictive intervals for the six best–worst answers in HEXACO block B1.
```
:::

:::{grid-item-card}
:shadow: none
:link: tutorials/ggum_rank_workflow
:link-type: doc
:header: **Generalized graded unfolding model for ranking (GGUM-RANK)** Ideal point · successive choices
:footer: Fitted ideal-point endorsement curves and the observed score range.

```{image} _static/thumbnails/ggum_rank_workflow.png
:alt: Fitted nonmonotonic GGUM-RANK statement endorsement curves with respondents' fitted-score rug and a 5th to 95th percentile score-support band.
```
:::


:::{grid-item-card}
:shadow: none
:link: tutorials/gtum_workflow
:link-type: doc
:header: **Generalized Thurstonian unfolding model (GTUM)** Ideal point · pairwise comparisons
:footer: GTUM tutorial: statement locations and 95% HDIs.

```{image} _static/thumbnails/gtum_locations.png
:alt: Posterior mean locations and 95% HDIs for S1, S13 and S25, for 100 simulated respondents.
```
:::


::::

## Binary and ordinal responses

One notebook covers Dichotomous IRT (1PL–4PL), GPCM and GGUM, with a shared
preparation, fitting, diagnostics and model-checking workflow.

::::{grid} 1 2 2 3
:gutter: 2
:class-container: model-gallery

:::{grid-item-card}
:link: tutorials/binary_ordinal_workflow
:link-type: doc
:shadow: none
:header: **Binary and ordinal responses** Dichotomous item response theory (1PL–4PL) · generalized partial credit model (GPCM) · generalized graded unfolding model (GGUM)
:footer: Worked GPCM analysis: information across fitted trait scores.

```{image} _static/thumbnails/binary_ordinal_responses.png
:alt: Diagonal GPCM test information against posterior-mean trait scores for 50 simulated respondents.
```
:::

::::

## Model combinations and extensions

::::{grid} 1 2 2 3
:gutter: 2
:class-container: model-gallery fc-model-gallery

:::{grid-item-card}
:link: tutorials/joint_workflow
:link-type: doc
:shadow: none
:header: **Combine forced-choice blocks and individual items**
:footer: Compare score uncertainty under shared and separate trait assignments.

```{image} _static/thumbnails/joint_models_analysis.png
:alt: Raincloud plot of agency posterior SDs for the same 80 respondents under two trait assignments.
```
:::

:::{grid-item-card}
:link: tutorials/mgppm_groups
:link-type: doc
:shadow: none
:header: **Multiple groups with the generalized partial preference model**
:footer: Estimate group differences on a shared trait scale.

```{image} _static/thumbnails/mgppm_groups.png
:alt: Posterior density and forest plot for the estimated difference in group means.
```
:::

:::{grid-item-card}
:link: tutorials/tirt_dif
:link-type: doc
:shadow: none
:header: **Differential item functioning with TIRT** (experimental)
:footer: Compare three DIF priors against simulated group differences.
:::

:::{grid-item-card}
:link: tutorials/jrt_tirt_native
:link-type: doc
:shadow: none
:header: **Joint response-time and Thurstonian item response theory (JRT–TIRT)** Connect response times with choice uncertainty
:footer: Observed times and posterior-predictive response distributions.

```{image} _static/thumbnails/jrt_tirt.png
:alt: Observed and posterior-predictive log response-time distributions from the actual 100-person synthetic JRT–TIRT fit.
```
:::


:::{grid-item-card}
:link: tutorials/m_tcir_native
:link-type: doc
:shadow: none
:header: **Normal and random responding in Thurstonian IRT (M-TCIR)** A three-state dyad adaptation
:footer: Compare predictive fit for the same binary choices.

```{image} _static/thumbnails/m_tcir.png
:alt: Leave-one-comparison-out predictive accuracy for TIRT and the three-state mixture.
```
:::

::::

## Estimation and scoring

::::{grid} 1 2 2 3
:gutter: 2
:class-container: model-gallery

:::{grid-item-card}
:link: examples/estimation_backends
:link-type: doc
:shadow: none
:header: **Choose a backend and estimation method** Four NUTS implementations · MAP/Laplace · VI · SMC
:footer: Compare estimator outputs and find the sampler and compilation settings.

```{image} _static/thumbnails/estimation_and_scoring_illustration.png
:alt: NUTS posterior draws, a MAP mode with Laplace SDs, variational draws and SMC particles.
```
:::

:::{grid-item-card}
:link: tutorials/prior_comparison
:link-type: doc
:shadow: none
:header: **Choose a prior for statement preferences**
:footer: Compare the rankings generated by two statement-intercept priors.

```{image} _static/thumbnails/prior_comparison.png
:alt: Prior predictive cumulative distributions of the share giving the most frequent ranking, for two MGPPM intercept priors.
```
:::

:::{grid-item-card}
:link: tutorials/recursive_scoring
:link-type: doc
:shadow: none
:header: **Score new respondents and update the calibration**
:footer: Compare a joint fit with experimental recursive calibration.

```{image} _static/thumbnails/recursive_scoring.png
:alt: Parameter recovery over five joint recursive updates, compared with one fit to all 300 respondents.
```
:::

::::

## R examples

::::{grid} 1 2 2 3
:gutter: 2
:class-container: model-gallery fc-model-gallery

:::{grid-item-card}
:shadow: none
:link: tutorials/tirt_r
:link-type: doc
:header: **Thurstonian item response theory (TIRT) in R** Dominance · pairwise comparisons
:footer: Fit 24 teaching triplets, score 10 held-out respondents and restore the fit.

```{image} _static/thumbnails/tirt_r_scores.png
:alt: Posterior medians and 95% credible intervals for four trait scores of one held-out respondent in the R tutorial.
```
:::

:::{grid-item-card}
:shadow: none
:link: tutorials/joint_r
:link-type: doc
:header: **Combine forced-choice and Likert responses in R**
:footer: Fit two response types on a shared trait and compare their loadings.

```{image} _static/thumbnails/joint_r_loadings.png
:alt: Posterior mean loadings and 95% highest density intervals for three forced-choice statements and two individual items.
```
:::

:::{grid-item-card}
:shadow: none
:link: tutorials/tirt_dif_r
:link-type: doc
:header: **Differential item functioning with TIRT in R** (experimental)
:footer: Compare statement loadings and pair cutpoints across two groups.

```{image} _static/thumbnails/tirt_dif_r_contrasts.png
:alt: Posterior means and 95% credible intervals for between-group cutpoint and loading differences in the R DIF tutorial.
```
:::

::::

See [Datasets](datasets.md) for response formats, sources and generating values.

```{toctree}
:hidden:

TIRT <tutorials/tirt_workflow>
TIRTRank <tutorials/tirt_rank_workflow>
2PLM-RANK <tutorials/two_pl_rank_workflow>
MGPPM <tutorials/mgppm_workflow>
GGUM-RANK <tutorials/ggum_rank_workflow>
GTUM <tutorials/gtum_workflow>
Binary and ordinal responses <tutorials/binary_ordinal_workflow>
Models by block and item <tutorials/joint_workflow>
Multiple groups with MGPPM <tutorials/mgppm_groups>
Differential item functioning with TIRT <tutorials/tirt_dif>
JRT–TIRT <tutorials/jrt_tirt_native>
M-TCIR <tutorials/m_tcir_native>
Statement response functions <tutorials/statement_curves>
Backends and estimation <examples/estimation_backends>
Choose a prior <tutorials/prior_comparison>
Score and update the calibration <tutorials/recursive_scoring>
TIRT in R <tutorials/tirt_r>
TIRT DIF in R <tutorials/tirt_dif_r>
Mixed response types in R <tutorials/joint_r>
Datasets <datasets>
```
