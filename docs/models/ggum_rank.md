---
orphan: true
---

# GGUM-RANK

These supplementary notes explain GGUM-RANK's unfolding odds, threshold
convention and comparison projections. Start with the
[GGUM-RANK example](../tutorials/ggum_rank_workflow.ipynb) for the parameter meanings,
default priors, format support and a simulated ranking analysis.

Location sides are substantive assumptions, independent of dominance keying.
Declare them with `side + ~`, `side - ~` or `side 0 ~` in a
[design string](../reference/generated/pyfcirt.ResponseTable.md), or in the
statement table. They cannot be inferred from observed response codes.

## Response probabilities

For loading magnitude a, location delta and statement threshold tau, define

$$
x_{pj}=a_j(\theta_{p,t(j)}-\delta_j),
\qquad
\ell_{pj}=\log(e^{x_{pj}}+e^{2x_{pj}})
-a_j\tau_j-\log(1+e^{3x_{pj}}).
$$

These are log endorsement odds from the binary unfolding response model.
The complete-ranking probability uses the same successive-choice normalization
as [2PLM-RANK](two_pl_rank.md#response-probabilities), substituting these odds.
PICK and MOLE sum the compatible complete-ranking probabilities; unknown ranks
are not imputed as independent observations.

## Reference

Lee, P., Joo, S.-H., Stark, S., & Chernyshenko, O. S. (2019). GGUM-RANK statement
and person parameter estimation with multidimensional forced choice triplets.
*Applied Psychological Measurement, 43*(3), 226–240.
[doi:10.1177/0146621618768294](https://doi.org/10.1177/0146621618768294).
Equations 1–2 describe pair and triplet choice probabilities. The log-odds
expression above states the binary unfolding function used by the implementation;
the supported block sizes and prior conventions are specified above.
