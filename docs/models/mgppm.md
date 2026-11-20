---
orphan: true
---

# Multidimensional generalized partial preference model (MGPPM)

These supplementary notes explain MGPPM's score vectors, response probabilities
and shared-statement constraints. The [MGPPM example](../tutorials/mgppm_workflow.ipynb)
fits the model to observed best–worst responses from HEXACO triplets.
The model follows [Furr and Fu (2025)](https://doi.org/10.1017/psy.2025.10054),
particularly Equations 22–25 for probabilities and Section 4.1.2 for graded pairs.

## Response probabilities

For respondent $p$ and statement $j$, let

$$
u_{pj}=\lambda_j\theta_{p,t(j)}-\beta_j,
\qquad \lambda_j=q_j a_j,
$$

where $q_j$ is the keying sign and $a_j$ the loading magnitude. Increasing
$\beta_j$ lowers statement utility. Keying changes loading direction; it does
not reverse the recorded responses.

For category $r$ in block $b$, with statement score vector $\mathbf s_{br}$,

$$
P(Y_{pb}=r\mid\theta_p,a,\beta)
=\frac{\exp(\mathbf s_{br}^{\mathsf T}\mathbf u_{pb})}
       {\sum_{r'}\exp(\mathbf s_{br'}^{\mathsf T}\mathbf u_{pb})}.
$$

Unlike TIRT's comparison CDF, this is one normalized distribution over the
block's categories. It does not multiply independent pairwise probabilities.

## Rankings, PICK and MOLE

Blocks may contain at least two statements. For a complete preference
order, the statement at rank $r$ receives score $K-r$. Thus `[2, 3, 1]`
assigns scores `[0, 2, 1]` to statements in design order. The support contains
all $K!$ orders.

PICK records the most preferred statement; MOLE records the most and least
preferred statements. Their event probabilities sum over compatible complete
orders. Unobserved positions remain unknown; they do not become independent
comparison observations.

Complete-ranking score vectors have the same sum. A common intercept shift
therefore leaves their probabilities unchanged. The default fixes a reference intercept to zero within each connected group;
remaining intercepts have Normal priors.

## Graded pairs

MGPPM supports ordered categories for two-statement blocks. For five categories,
the score rows, from preference for the second position to preference for the first,
are

$$
(0,2),\quad(0,1),\quad(0,0),\quad(1,0),\quad(2,0).
$$

## Shared statements and observation units

Compatible ranking blocks may reuse a statement, including across block sizes.
It then has one loading and intercept. Under the default prior, the first
reference block in each connected component fixes the intercept shift;
overlapping blocks are not separately centered. See
[shared statements](../reference/generated/pyfcirt.Model.set_priors.md#shared-statements-in-linear-ranking-models).
Automatic shared-statement graded intercepts are not supported. A statement
cannot belong to different effective measurement families through the
high-level `Model` interface.

One respondent/block response is one likelihood event. Missing or unadministered
blocks contribute no response likelihood. Pointwise likelihoods include the
category normalization and condition on the respondent traits and item
parameters; priors and transformation Jacobians are excluded. Use matching
observation units when comparing models. These constraints do not establish
identification for every possible loading design.

## Reference

Furr, D. C., & Fu, J. (2025). Multidimensional generalized partial preference model for forced-choice items. *Psychometrika, 90*(5), 1907–1936. <https://doi.org/10.1017/psy.2025.10054>
