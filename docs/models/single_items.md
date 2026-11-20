---
orphan: true
---

# Binary and ordinal measurement models

These supplementary notes define the single-item response laws and parameter
coordinates. The [binary and ordinal response notebook](../tutorials/binary_ordinal_workflow.ipynb)
uses a simulated dataset to demonstrate a shared fitting and analysis workflow. The [input guide](../likert.md) explains preparation and analysis
calls. Literature also uses MIRT, MGPCM and MGGUM for multidimensional models;
the current Python names are `DichotomousIRT`, `GPCM` and `GGUM`.

With the default population, all three families use standardized correlated traits,
$\theta_p\sim\mathcal N(0,R)$.
The design selects active item-by-trait loadings. Inactive loadings are zero;
positive active loadings and trait labels alone do not guarantee identification.
Declare the full ordered category set even if some categories are unobserved.

(dichotomous-mirt)=

## Dichotomous IRT

The Python declaration is `DichotomousIRT`. For a binary item, write

$$
P(Y_{pj}=1)=c_j+(d_j-c_j)
\operatorname{logistic}(a_j^\top\theta_p-b_j).
$$

`loading_magnitude` contains the active coefficients $a_j$, and `intercept` is
$b_j$: increasing it lowers the probability of the higher response category.
It is an intercept, not a difficulty multiplied by the loading.
`lower_asymptote` and `upper_asymptote` are $c_j$ and $d_j$.
Final fixed entries remain in the prepared constraints and coefficient display,
without being repeated across posterior draws.

- `DichotomousIRT(kind="1PL")` estimates one common active loading, positive under
  the default prior, and fixes $c_j=0$, $d_j=1$,
  including multidimensional designs.
  Inactive loadings are zero; with the default population, trait means and variances
  are zero and one. The common estimated discrimination carries the response scale.
  The default common-discrimination
  prior is LogNormal(0.25, 0.5), on the log scale. It differs from mirt's `Rasch`
  option, which fixes the slope and estimates trait variances.
  The fixed activity design must distinguish the named dimensions.
- `DichotomousIRT(kind="2PL")` estimates loadings, with $c_j=0$ and $d_j=1$.
- `DichotomousIRT(kind="3PL")` also estimates $c_j$, retaining $d_j=1$.
- `DichotomousIRT(kind="4PL")` estimates both asymptotes, with $0\le c_j<d_j\le1$.

Default priors are listed in the [Dichotomous IRT API reference](../reference/generated/pyfcirt.DichotomousIRT.md).
The default templates are $c_j\sim\mathrm{Beta}(1,4)$ and
$d_j\sim\mathrm{Beta}(4,1)$, each covering the full unit interval. The upper prior is conditioned on
exceeding the lower asymptote; it is not an independent prior permitting crossed
asymptotes.

DichotomousIRT requires exactly two declared categories per item. See Chalmers (2012) and
the current mirt model reference below for the compensatory response law.
The mirt additive-intercept convention uses $-b_j$.

(mgpcm)=

## GPCM

For ordered categories indexed $k=0,\ldots,C_j-1$,

$$
P(Y_{pj}=k)=
\frac{\exp\{k\,a_j^\top\theta_p+v_{jk}\}}
{\sum_{h=0}^{C_j-1}\exp\{h\,a_j^\top\theta_p+v_{jh}\}},
\qquad v_{j0}=0.
$$

`loading_magnitude` contains active loadings $a_j$. `category_intercept` is the prior role
for $v_{jk}$ values. Posterior arrays retain the prepared item/category coordinates; use
`coef()` or inspect `model.idata["posterior"]` for the actual node names. An item with
$C_j$ categories has $C_j-1$ free intercepts. These are **not ordered thresholds**: the
adjacent step parameter is $v_{j,k-1}-v_{jk}$. Fixing the baseline intercept removes a
common additive shift in the category logits.

The [GPCM API reference](../reference/generated/pyfcirt.GPCM.md) lists the default priors.
Items can have different declared category counts. Muraki's
cumulative-step representation and these category intercepts express the same
single-item partial-credit probabilities; the active loading vector supplies
the compensatory multidimensional predictor.

(mggum)=

## GGUM

`GGUM()` uses a **weighted-distance unfolding** formulation. For item $j$, define

$$
r_{pj}^2=\sum_t a_{jt}^2(\theta_{pt}-\delta_{jt})^2,
\qquad A_j=\sum_t a_{jt}.
$$

For $C_j$ ordered response categories, its category logits are

$$
\eta_{pjk}=
\log\cosh\!\left[\left(C_j-\tfrac12-k\right)r_{pj}\right]
-A_j\sum_{h=0}^k\tau_{jh},
\qquad P(Y_{pj}=k)=\operatorname{softmax}_k(\eta_{pjk}).
$$

`loading_magnitude` contains positive active $a_{jt}$ values; `location` contains the
item-by-trait ideal points $\delta_{jt}$. Explicit `location_side` values are +1 or -1 for
sign-constrained locations, or 0 for unrestricted locations. Their underlying Normal distributions have means +1, -1
and 0 respectively and SD 1; signed sides are truncated to their declared half-lines.
Custom priors can change these values. A
trait with no signed location anchor raises an identification warning. Inactive loading
and location cells are zero. Each item's threshold vector starts at $\tau_{j0}=0$;
remaining step parameters
have independent Normal(0, 1) priors. They may be positive or negative and are not
ordered. Override them with `GGUM(priors={"threshold": pm.Normal.dist(mu=0, sigma=1)})`.

Default distributions are listed in the
[GGUM API reference](../reference/generated/pyfcirt.GGUM.md). The design's location
directions are substantive assumptions; they are not estimated from response
codes.

This distance model differs from a compensatory unfolding model based on a
scalar projection $a_j^\top\theta_p-\delta_j$. Do not treat their multidimensional
parameters or information functions as interchangeable. The current mirt model
reference gives the distance formulation; a one-active-trait restriction does
not establish equivalence to every multidimensional GGUM variant.

## References

- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
  for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
  [doi:10.18637/jss.v048.i06](https://doi.org/10.18637/jss.v048.i06).
- Muraki, E. (1993). *Information Functions of the Generalized Partial Credit
  Model*. ETS Research Report RR-93-27.
  [Technical report](https://files.eric.ed.gov/fulltext/ED390896.pdf).
- [mirt model reference](https://philchalmers.github.io/mirt/docs/reference/mirt.html).
  Defines current single-item response laws, including the distance unfolding
  model; software notation and prior choices need not match PyFCIRT's.
