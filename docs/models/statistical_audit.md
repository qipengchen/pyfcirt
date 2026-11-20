# Statistical model correspondence

The model guides give each family's probabilities, parameter meanings, priors
and identification constraints:

- [TIRT](tirt.md): conditional binary or graded comparison probabilities.
- [TIRTRank](tirt_rank.md): joint Gaussian ranking-event probabilities.
- [GTUM](gtum.md): distance-based logistic comparison probabilities, with optional
  person-by-block effects.
- [2PLM-RANK](two_pl_rank.md): successive-choice ranking probabilities.
- [GGUM-RANK](ggum_rank.md): unfolding ranking probabilities.
- [MGPPM](mgppm.md): generalized partial-preference category probabilities.
- [Single-item models](single_items.md): MIRT, MGPCM and MGGUM.

A ranking model assigns probabilities to observed block events. PICK and MOLE
retain unknown orders by summing compatible full rankings. Comparison-based
TIRT and GTUM instead use the observed comparisons; a product of comparisons
derived from one ranking is a composite likelihood, not a joint ranking law.

Equal likelihood values at a fixed parameter point do not imply equal priors,
posteriors or identification. Compare LOO and WAIC only when the data and
observation events match. Fixed-calibration trait estimates condition on item
parameters and trait correlations; they do not propagate calibration uncertainty.

See [model comparison](../comparison.md), [information](../reference/information.md) and
[model checks](../reference/generated/pyfcirt.chi_ppp.md) for analysis-specific assumptions.
