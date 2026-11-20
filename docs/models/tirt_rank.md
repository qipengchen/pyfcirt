---
orphan: true
---

# Joint-ranking Thurstonian IRT (TIRTRank)

`TIRTRank` fits the joint probability of a ranking, best choice, or
best–worst choice within a block. See the [TIRTRank example](../tutorials/tirt_rank_workflow.ipynb)
for a worked analysis. For graded pairwise responses, use [TIRT](tirt.md).

The Gaussian utility and joint-ranking construction follows the framework
discussed by Frick (2023) and Yousfi (2025). The priors and identification
conventions below specify PyFCIRT's implementation.

## Model and identification

For person p and statement i measuring trait t(i),

$$
U_{pi}=u_i+a_i s_i\theta_{p,t(i)}+\epsilon_{pi},\qquad
\boldsymbol\theta_p\sim N(0,R),\quad\operatorname{diag}R=1.
$$

In dyads, both statement residual variances are fixed at 0.5. Larger unanchored statement components fix
one reference variance to .5 and estimate the others; design `fix_var` values
override the defaults. Repeated statements share one intercept,
loading and variance. Dyad variance constraints carry into larger blocks; a
shared component without a dyad fixes one reference variance to .5. Separate
presentations share parameters, not realized person-by-statement residuals.
All-same-trait components require an explicit loading anchor. A two-trait design
consisting entirely of cross-trait pairs requires both loading magnitudes in one
reference pair to be explicitly anchored.
These are structural scale constraints, not a proof of empirical identification
for every design, parameter point or partial-ranking instruction.

## Gaussian event augmentation

For a complete ranking, the fitting graph samples only the K-1 neighboring
utility differences, following the reduced implementation of Heister, Doebler
and Frick (2025). For example, A > B > C requires U_A-U_B > 0 and U_B-U_C > 0.
If D = A U, its Gaussian mean and covariance are A m and A Omega A.T.
The fitting graph assigns this normal distribution to the differences and
observes that they lie in the recorded order region. PyMC samples the
differences on a log scale for complete rankings, including the Jacobian.

PICK, MOLE and incomplete comparisons instead use K-1 utilities relative to one
reference, constrained only by the recorded partial order. The omitted common
utility level is analytically integrated out, not fixed as an observed value.
Preparation supplies the contrast basis and any required masks. The sampling
coordinates represent the positive gap above each required predecessor, while
unconstrained differences remain free. Calibration and new-person scoring use
the same augmentation. Prediction and evaluation integrate over the order
region to obtain the marginal probability.

## Information and precision

Use `fc.information(model)` for [joint Gaussian event information](../reference/information.md).
RANK,
PICK and MOLE retain their own event categories. Statement residuals are
integrated out and event-augmentation variables do not enter the calculation;
the fitted residual variances define the Gaussian utility covariance.

Pairs use an analytic normal CDF. Larger events use approximate Genz–Sobol
integration with refinement checks and JAX's native `ndtri_exp` for the inverse
normal CDF on the log-probability scale.
The default information calculation compares 4,096 and 16,384 fixed integration
nodes. This is an approximation: nonbinary event probabilities do not retain the
1e-10 precision of an analytic or specialized quadrature reference. Increase
`n_points` in the low-level distribution when checking sensitive probability
calculations. The returned metadata reports numerical diagnostics, not guaranteed
error bounds. Extreme tails or unresolved integrations can raise an error. This
expected information excludes prior precision and calibration uncertainty.

## References

- Heister, H., Doebler, P., & Frick, S. (2025). Bayesian Thurstonian IRT modeling:
  Logical dependencies as an accurate reflection of Thurstone's law of comparative
  judgment. *Educational and Psychological Measurement, 85*(5), 935–955.
  [Article](https://doi.org/10.1177/00131644251335586);
  [author Stan code](https://osf.io/8fndw/).
- Frick, S. (2023). Estimating and using block information in the Thurstonian
  IRT model. *Psychometrika, 88*, 1556–1589.
  [Article](https://www.cambridge.org/core/journals/psychometrika/article/estimating-and-using-block-information-in-the-thurstonian-irt-model/5BED906AE9E06A1D5FA9CED6515F53B7).
  Distinguishes the joint ranking likelihood and independence approximations.
- Yousfi, S. (2025). Statistical foundations of person parameter estimation in
  the Thurstonian IRT model for forced-choice and pairwise comparison designs.
  *British Journal of Mathematical and Statistical Psychology, 78*, 555–593.
  [https://doi.org/10.1111/bmsp.12364](https://doi.org/10.1111/bmsp.12364).
  Discusses the genuine ranking likelihood and person-parameter estimation.


For explicit anchors, see the concrete [statement-to-magnitude mapping and design
column recipe](../reference/generated/pyfcirt.Model.set_priors.md#tirt-loading-anchors).
