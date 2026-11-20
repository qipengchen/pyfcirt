# Comparison and joint-ranking likelihoods

PyFCIRT distinguishes comparison-based TIRT and GTUM from the joint Gaussian
ranking likelihood in TIRTRank. Choose the response event that matches the data
and the assumptions of the analysis.

## Comparison models

[TIRT](tirt.md) supplies binary or graded comparison probabilities. Its
statement effects, comparison thresholds and residual scales have the meanings
specified in that guide. The statement effects are not respondent-specific
Gaussian utility errors.

[GTUM](gtum.md) uses distance-based statement utilities and logistic comparison
probabilities, with an optional respondent-by-block effect. The default has no
block effect, including for dyads. Explicitly enabling it defines a different
model and is retained in simulation and new-person estimation.

Both models can use known comparisons from larger blocks. A product of
comparisons derived from a single ranking is a composite likelihood. It does
not become a joint distribution over rankings by normalizing after fitting.

## Joint-ranking TIRT

[TIRTRank](tirt_rank.md) assumes Gaussian statement utilities. Adjacent utility
differences for a full ranking have a joint normal distribution; their joint
positive-orthant probability gives the ranking probability. PICK and MOLE use
the corresponding partial-order events without filling unknown ranks.

Fitting uses Gaussian event augmentation. The auxiliary coordinates are
integrated out for categorical log likelihood and Pearson posterior predictive
checks. Prediction generates Gaussian utilities and records their order.
New-person MCMC holds item parameters and the population correlation fixed.

TIRTRank has separate intercept and residual constraints from comparison-based
TIRT. It currently supports disjoint blocks with distinct traits within a block,
subject to its additional loading-anchor requirements. It does not supply a
joint graded-ranking model or arbitrary shared-statement identification.

## Analysis units and other implementations

Compare LOO, WAIC and DIC only for matching data and observation events.
A common person-by-block index does not make different likelihoods comparable.
Information from independent comparisons is also distinct from joint-ranking
information; see [information](../reference/information.md).

not equivalent to the logistic GTUM comparison model.

## References

- Bürkner, Schulte and Holling (2019),
  [conditional TIRT comparison formulation](https://doi.org/10.1177/0013164419832063).
- Frick (2023),
  [genuine and independence likelihoods](https://www.cambridge.org/core/journals/psychometrika/article/estimating-and-using-block-information-in-the-thurstonian-irt-model/5BED906AE9E06A1D5FA9CED6515F53B7).
- Yousfi (2025),
  [person estimation under ranking and pairwise designs](https://doi.org/10.1111/bmsp.12364).
- Zhang et al. (2024),
  [GTUM](https://doi.org/10.1177/10944281231210481).
