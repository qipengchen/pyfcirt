---
orphan: true
---

# Thurstonian IRT (TIRT)

These supplementary notes explain TIRT's comparison likelihood, parameter
coordinates and residual constraints. Start with the
[TIRT example](../tutorials/tirt_workflow.ipynb) for its statistical
introduction, default priors and an analysis of the prepared study data.

## Response model

For respondent $p$ and statement $i$, define the systematic utility

$$
m_{pi}=\lambda_i\theta_{p,t(i)}+\nu_i,
\qquad \lambda_i=q_i a_i,
$$

where $q_i$ is the design's keying sign and $a_i$ is the loading magnitude.
For a comparison of statements $i$ and $j$, let

$$
\eta_{pij}=m_{pi}-m_{pj},\qquad
s_{ij}=\sqrt{\psi_i^2+\psi_j^2}.
$$

When response 1 means the first statement is preferred, the binary probability is

$$
P(Y_{pij}=1\mid\theta_p,a,\nu,\psi,\gamma)
=\Phi\!\left(\frac{\eta_{pij}-\gamma_{bij}}{s_{ij}}\right).
$$

Here $b$ identifies the presented block and $\Phi$ is the standard normal CDF.
For ordered response categories, probabilities are differences between adjacent
CDF values at the ordered comparison thresholds. The declared response direction
controls which statement higher categories favor; it is separate from keying.

The conditional comparison formula follows Bürkner, Schulte and Holling (2019,
equation 4) and the authors' Stan implementation. It multiplies comparison
probabilities conditional on the model parameters. Brown and Maydeu-Olivares's
joint latent-response formulation also includes within-respondent residual
covariances for comparisons sharing a statement. A global statement effect does
not reproduce those person-specific residuals. See
[comparison and ranking targets](comparisons.md#comparison-and-ranking-targets)
for the implications when comparisons come from one ranking.

## Loadings

`loading_magnitude` is the prior role for $a_i$; saved `loading` and `fc.coef`
report that parameter before keying. It is positive under the default prior;
compatible signed priors can relax that orientation. The response law uses the
effective coefficient $\lambda_i=q_i a_i$, which is also the `loading` input
expected by the low-level `TIRTDistribution.dist`. With positive $a_i$, positive
keying increases utility with the trait and negative keying decreases it.
Keying does not reverse the recorded response codes.

## Residual scales

`residual_variance` is $\psi_i^2$, with statement SD $\psi_i$. The comparison
scale is $s_{ij}$, not either statement SD alone. Fixed reference SDs are described under
[identification](#identification).

A reused statement has one global SD across its blocks. Reference units make
parameters easier to compare across block sizes, but do not guarantee that all
parameters have identical empirical scales or precision.

## Statement effects

`statement_effect` is $\nu_i$, with the conditional prior

$$
\nu_i\mid\psi_i\sim\mathcal N(0,\psi_i^2).
$$

It is a global statement coefficient shared by respondents. It is not a fresh
respondent-by-statement error. Because the likelihood uses the effective threshold
$\gamma_{bij}-\nu_i+\nu_j$, free thresholds can absorb these statement effects.
Their separate estimates therefore depend on the hierarchical priors.

By default, `TIRT()` samples standardized
statement coordinates $z_i\sim\mathcal N(0,1)$ and computes $\nu_i=\psi_i z_i$.
`TIRT(statement_effect_parameterization="centered")` samples $\nu_i$ directly.
The choice preserves the priors on model-scale parameters,
likelihood and residual references; it changes the sampling coordinates.
It does not remove the threshold/effect confounding. Calibration and scoring
continue to use the model-scale `statement_effect` values.

## Comparison thresholds

`cutpoints` configures the prior on $\gamma_{bij}$, or on the ordered
threshold vector for graded categories. Multiple thresholds are ordered within
each comparison. Each threshold belongs to its original block and statement pair.
With the default first-statement preference coding, increasing a cutpoint
shifts probability toward lower response categories. In a binary comparison
this favors the second statement. Reversed response coding changes the
interpretation and must be declared explicitly.

These are **unstandardized utility-scale thresholds**. Both the utility contrast and the
thresholds are divided by $s_{ij}$ in the response probability. For a binary comparison
with the default $\gamma_{bij}\sim\mathcal N(0,1)$ prior, the standardized threshold
$\gamma_{bij}/s_{ij}$ has conditional prior SD $1/s_{ij}$. Graded comparisons use ordered
cutpoints, whose marginal prior SDs need not equal one before scaling. Custom priors can
also change those SDs. Saved cutpoint arrays contain the raw thresholds.

## Trait correlations

`theta` contains the respondent latent traits. Their population mean is zero and
each marginal SD is one. `trait_correlation` is their correlation matrix $R$.
Correlations describe associations
between traits, not residual dependence between comparisons.

## Identification

Automatic residual references use SD $\sqrt{0.5}$ (variance $0.5$).
Dyads fix both statement variances; larger connected statement components need
one reference unless a user constraint already provides it. The design's `fix_var`
column overrides reference values. Each shared statement has one residual parameter.
`TIRT(residuals="fixed")` fixes every otherwise unspecified variance at $0.5$.
Both settings explicitly sample statement effects; noncentered is the default.

`TIRT(link="probit")` is the default. `link="logit"` keeps the $1.702$ probit-scale
approximation. `link="beta_probit"` and `link="beta_logit"` describe continuous
open-interval responses with their respective mean links. Beta comparisons require
an explicit positive `response_precision` prior; their logit link is unscaled.
Exact zero/one responses are rejected at data preparation and are not adjusted.

For isolated same-dimension components, data preparation supplies an automatic
loading reference. Explicit `loading_anchors` or the design's `fix_loading` values
override that choice. The prepared model retains these constraints; keep its declarations when saving samples.

## References

- Brown, A., & Maydeu-Olivares, A. (2011). Item response modeling of forced-choice
  questionnaires. *Educational and Psychological Measurement, 71*(3), 460–502.
  [https://doi.org/10.1177/0013164410375112](https://doi.org/10.1177/0013164410375112).
  Defines structured loadings and within-block residual covariance; comparison
  thresholds are not constrained to be differences of statement intercepts.
- Brown, A., & Maydeu-Olivares, A. (2012). Fitting a Thurstonian IRT model to
  forced-choice data using Mplus. *Behavior Research Methods, 44*, 1135–1147.
  [https://doi.org/10.3758/s13428-012-0217-x](https://doi.org/10.3758/s13428-012-0217-x).
  Explains comparison coding and identification conventions.
- Bürkner, P.-C., Schulte, N., & Holling, H. (2019). On the statistical and
  practical limitations of Thurstonian IRT models. *Educational and Psychological
  Measurement, 79*(5), 827–854.
  [https://doi.org/10.1177/0013164419832063](https://doi.org/10.1177/0013164419832063).
  Equation 4 and the authors' [Stan definition](https://github.com/paul-buerkner/thurstonianIRT/blob/a6edd59449fa5003324a933d63d1facaa2881259/src/stan_files/thurstonian_irt_model.stan)
  provide the conditional comparison formulation. The priors and reference
  constraints stated above are PyFCIRT's specification.


For explicit anchors, see the concrete [statement-to-magnitude mapping and design
column recipe](../reference/generated/pyfcirt.Model.set_priors.md#tirt-loading-anchors).
