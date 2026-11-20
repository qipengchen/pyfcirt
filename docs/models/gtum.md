---
orphan: true
---

# Generalized Thurstonian unfolding model (GTUM)

These supplementary notes explain GTUM's distance utility, category coding and
optional block effects. The [GTUM example](../tutorials/gtum_workflow.ipynb) introduces the
default priors and analyzes simulated binary comparisons. `GTUM()` has no block
effect; `GTUM(block_effect=True)` adds a person-by-block effect. Both use
comparison probabilities, rather than a joint distribution over rankings.

## Response model

For respondent $p$ and statement $i$, systematic utility is

$$
m_{pi}=-a_i\left|\theta_{p,t(i)}-\delta_i\right|,
\qquad a_i>0.
$$

Let $\eta_{pij}=m_{pi}-m_{pj}$. When binary response 1 favors the first statement,

$$
P(Y_{pij}=1\mid\theta_p,a,\delta,\gamma)
=\operatorname{logistic}(\eta_{pij}-\gamma_{bij}).
$$

With a block effect, replace $\eta_{pij}$ by $\eta_{pij}+\zeta_{pb}$. Graded
probabilities are adjacent differences of the ordered logistic CDF. The logistic
link has scale one; it does not use TIRT's 1.702 multiplier or a Gaussian
statement-residual denominator.

## Locations

`location` is the ideal point $\delta_i$. Statement utility is highest at that
location and decreases on both sides. `location_side` in the design expresses
substantive information about its side of zero:

- `+1`: a positive location magnitude.
- `-1`: the negative of that magnitude.
- `0`: an unrestricted location, rather than a location fixed at zero.

The [GTUM API reference](../reference/generated/pyfcirt.GTUM.md) lists their default distributions.

The default positive and negative location priors have no finite outer bound.
Their prior roles are `location_positive` and `location_negative`; unrestricted
locations use `location_free`. Each role can have its own prior. The design's `keying`
field is not substituted for this location-side declaration. A statement reused
in several blocks retains one loading and one location.

## Comparison thresholds

`cutpoints` supplies the prior for unstandardized logistic cutpoints. By
default, graded cutpoints are the ordered values of independent, identically distributed
Normal draws; the ordered cutpoints are therefore dependent. Each threshold belongs to its
original block and statement pair, including when statements are shared across blocks.

### Mapping the paper's coding

Zhang et al.'s binary equation 12 uses
$\operatorname{logistic}(\eta+\tau)$, so its $\tau$ equals the negative of the
cutpoint used above. Their graded example places preference for the first
statement at the **low** end of the category scale. For those recorded codes,
use `CategoryScale((1, 2, 3, 4, 5), higher_prefers="second")` with `response_format=None`;
the ordered package cutpoints then equal the paper's graded thresholds.
The same direction values apply to comparison and block response tables.

PyFCIRT's usual higher-category-prefers-first convention is also valid. Keep
that coding consistent when interpreting thresholds or comparing a fixed
parameter point with an external implementation. Changing numeric category
labels alone does not define this direction.

## Block effects

With `block_effect=True`, `person_block_effect` is $\zeta_{pb}$, where

$$
\zeta_{pb}\mid\sigma_\zeta\sim\mathcal N(0,\sigma_\zeta^2),
\qquad\sigma_\zeta\sim\operatorname{HalfNormal}(1).
$$

`block_effect_scale` is $\sigma_\zeta$. Each observed person/block has one effect
shared by its comparisons. Conditional on the common SD, effects are independent
across person/block units and independent of focal traits. This is the random block factor in the paper's
equation 14, used to describe local dependence. It has no statement-specific
factor loading and does not enforce transitive rankings.

## Trait correlations

With the baseline population, `theta` has mean zero, unit marginal variances and correlation matrix
`trait_correlation`.

## Identification

Positive loadings and substantively justified location sides determine the usual
orientation. The measurement design still needs sufficient trait coverage and
information. Proper priors and known signs do not establish identification or
recovery for every design.

Current prior defaults are PyFCIRT choices; they differ from the paper's
simulation priors. [User prior overrides](../reference/generated/pyfcirt.Model.set_priors.md#priors-and-identification)
may change their distributions. Allowing signed loadings or location magnitudes
to cross zero changes the default unfolding interpretation and emits an
identification warning.

## Conditional-mode updates (experimental)

`fc.sample_gtum(model)` adds a conditional-mode independent Metropolis–Hastings
update after a retained full-parameter NUTS step. It leaves the declared prior
and likelihood unchanged. NUTS-only warmup adapts the sampler; item parameters
continue to be updated by NUTS. The specialized sampler is explicitly requested.

The proposal uses all relevant comparisons, the correlated trait prior and any
current block effects. Forward and reverse proposal densities enter the MH
correction. The specialized transition uses BlackJAX on the configured JAX CPU or GPU,
with positive warmup and the shared task runner. Ordinary PyMC estimators
remain available through `Model.fit()`.

Inspect R-hat, ESS, divergences and the recorded IMH acceptance/invalid-proposal
statistics. `nuts_energy` describes the NUTS substep, not the full compound-chain
energy for BFMI. Kernel correctness does not guarantee convergence in an
unfolding model. This experimental update is a PyFCIRT method, not a method
established by the GTUM article.

## Reference

Zhang, B., Tu, N., Angrave, L., Zhang, S., Sun, T., Tay, L., & Li, J. (2024).
The generalized Thurstonian unfolding model (GTUM): Advancing the modeling of
forced-choice data. *Organizational Research Methods, 27*(4), 713–747.
[https://doi.org/10.1177/10944281231210481](https://doi.org/10.1177/10944281231210481).
Equations 10–14 define the distance utility, binary/graded probabilities and
random block factor. The paper also discusses the use of separate Likert
calibration information for statement-parameter priors.
