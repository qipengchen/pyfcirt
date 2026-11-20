---
orphan: true
---

# Binary and graded preference comparisons

TIRT and GTUM can analyze explicitly recorded comparisons, or comparisons
obtained from complete and partial block rankings. The data preparation is
shared, but a comparison likelihood and a joint ranking likelihood make
different assumptions.

## Input

An explicit comparison row identifies `person_id`, `block_id`, `statement_1`,
`statement_2` and `response`. In the usual binary coding, 1 favors the first
statement and 0 the second. Missing responses use NA. The [data guide](../reference/generated/pyfcirt.ResponseTable.md)
shows the input tables and how to declare another category order or direction.

Statements retain their original block memberships. Reused statements share
item parameters within the same measurement family; comparison thresholds stay
attached to their original block and statement pair. Declaring an item does not create
responses to comparisons that were not observed.

For ranking-list input, the adapter retains only known comparisons:

- RANK supplies all comparisons implied by the recorded order.
- PICK supplies comparisons between the winner and the other statements.
- MOLE supplies comparisons involving the best or worst statement; comparisons
  among unordered middle statements are missing.

No middle ranking is imputed. Fully missing block responses add no observed
comparisons. A cycle such as A > B > C > A may remain in explicitly recorded
comparison data for TIRT or GTUM; a model requiring a coherent ranking rejects
that cycle before fitting.

## Model probabilities

[TIRT](tirt.md#response-model) uses the difference of signed loading–trait
products and global statement effects. Both that contrast and the raw
comparison thresholds are divided by
$\sqrt{\psi_i^2+\psi_j^2}$. Probit is the default link; the explicitly selected
1.702-scaled logit is an approximation.

[GTUM](gtum.md#response-model) uses the difference of negative absolute-distance
utilities and an unscaled logistic link. Its optional person-by-block normal
effect is shared across that block's comparisons. There is no TIRT residual-SD
denominator. For either family, ordered response probabilities are adjacent CDF
intervals in the declared category order.

These probabilities are normalized over each comparison's categories. They are
not normalized over all possible block rankings.

## Comparison and ranking targets

For separately recorded comparisons, the likelihood assumes independence
conditional on the traits, item parameters and any declared random effects.
This is the conditional comparison model implemented by the current TIRT/GTUM
paths.

A single ranking supplies related comparisons. Their product is a **pairwise
composite likelihood**, not a normalized probability for the observed ranking.
Brown and Maydeu-Olivares's joint TIRT formulation includes structured
within-respondent residual covariance for comparisons sharing statements. The
global TIRT statement effects in the conditional model are not fresh
respondent-by-statement errors and do not reproduce that covariance.

A joint utility-ranking probability instead integrates the relevant correlated
utility differences. It also needs coherent utility parameters. Free comparison
thresholds need not be differences of statement intercepts, and a common GTUM
block factor does not by itself define a transitive ranking distribution.
[Joint Gaussian ranking information](../reference/information.md)
is therefore a separate known-parameter analysis, not an automatic conversion
of an arbitrary fitted comparison model.

### Partial rankings and missingness

Representing unobserved comparisons as NA is the correct data representation.
It does not establish that the product of the retained marginal probabilities
is the joint likelihood of the observed PICK or MOLE event. Which comparisons
are retained depends on the observed selection.

Brown and Maydeu-Olivares (2012) discuss missing-at-random outcomes in a joint
model. That missingness argument does not make an arbitrary independent-pair
approximation equivalent to that joint model. Keep the response protocol and
likelihood target explicit when interpreting a partial-ranking analysis.

## Prediction and model checks

Comparison predictions return labelled comparison responses, including their
category labels. They may contain cycles; no ranking is invented from them.
The [posterior predictive check](../reference/generated/pyfcirt.chi_ppp.md) for this model compares
comparison-category discrepancies. It is not a full-ranking-category check.

`log_likelihood()` retains one log probability per observed person/comparison,
with `obs`, person, block and pair coordinates. [LOO, WAIC and DIC](../comparison.md)
use these native factors. With GTUM block effects, those probabilities condition
on each saved effect; this is not analytic marginalization over new effects.

Match data, event definitions and nuisance conditioning before comparing models.
Equal-shaped likelihood arrays alone do not make a comparison target and a
ranking target comparable. Inspect ArviZ's Pareto diagnostics for unstable LOO
importance sampling, especially when observations have their own latent effects.

## References

- Brown, A., & Maydeu-Olivares, A. (2011). Item response modeling of forced-choice
  questionnaires. *Educational and Psychological Measurement, 71*(3), 460–502.
  [https://doi.org/10.1177/0013164410375112](https://doi.org/10.1177/0013164410375112).
  Structured factor loadings, residual covariance and free comparison thresholds.
- Brown, A., & Maydeu-Olivares, A. (2012). Fitting a Thurstonian IRT model to
  forced-choice data using Mplus. *Behavior Research Methods, 44*, 1135–1147.
  [https://doi.org/10.3758/s13428-012-0217-x](https://doi.org/10.3758/s13428-012-0217-x).
  Comparison coding and the joint model's partial-ranking treatment.
- Bürkner, P.-C., Schulte, N., & Holling, H. (2019). On the statistical and
  practical limitations of Thurstonian IRT models. *Educational and Psychological
  Measurement, 79*(5), 827–854.
  [https://doi.org/10.1177/0013164419832063](https://doi.org/10.1177/0013164419832063).
  Conditional probit comparison formulation and practical design limitations.
- Zhang, B., Tu, N., Angrave, L., Zhang, S., Sun, T., Tay, L., & Li, J. (2024).
  The generalized Thurstonian unfolding model (GTUM):
  Advancing the modeling of forced-choice data. *Organizational Research Methods,
  27*(4), 713–747.
  [https://doi.org/10.1177/10944281231210481](https://doi.org/10.1177/10944281231210481).
  Unfolding comparison probabilities and the random block factor.
