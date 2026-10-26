# MGPPM statistical specification

The likelihood, simulation, native PyMC graph and predictive interfaces are
implemented. Centered traits are used throughout. Formal tests do not run posterior MCMC;
small manual sampling checks have run for both branches. Recovery
and complete-workflow results from the earlier noncentered development model
are historical evidence; they do not establish acceptance of the centered graph.
Data validation is independent of model capability. The numerical scope is ranking blocks of sizes
2, 3 and 4, and five-category graded pairs.

## Sources and interpretation

The model attribution is Furr and Fu (2025), *Multidimensional Generalized Partial
Preference Model for Forced-Choice Items*,
[doi:10.1017/psy.2025.10054](https://doi.org/10.1017/psy.2025.10054).
The development baseline is the locally shared FC Tutorial's
`tutorials/05_mgppm.ipynb`, including its graded-pair section. Its Python model
and category-score CSVs were inspected independently of external Stan code.

The tutorial's empirical triplets are attributed to Wetzel, Frick and Brown
(2021), [doi:10.1037/pas0000971](https://doi.org/10.1037/pas0000971). They are not
the nonpublic ETS dataset described in the MGPPM paper. The tutorial's graded
example is simulated; it must not be presented as empirical data.

No empirical response file is currently bundled in PyFCIRT. The source manifest
identifies public OSF inputs, but the redistribution license is not established
by the manifest alone. Keep those inputs in external data storage while their
license and conversion are audited. Source checksums and machine-specific
locations belong to local development records, not package defaults.

## Statement and person parameters

For person `i` and statement `j`, measuring trait `d(j)`, define

```text
loading_j = keying_j * loading_magnitude_j
u_ij      = loading_j * theta_i,d(j) - beta_j
eta_ibc   = sum_j(score_bcj * u_ij)
log p_ibc = eta_ibc - logsumexp_c(eta_ibc)
```

Increasing beta lowers statement utility. Keying is supplied by the design and
does not reverse recorded ranks or graded responses.

The tutorial baseline uses these priors:

- `log(loading_magnitude_j) ~ Normal(0, 2**2)`, equivalently
  `LogNormal(mu=0, sigma=2)` using PyMC's log-scale convention.
- The trait mean is zero, with unit marginal variances and an `LKJ(eta=1)`
  prior for the correlation matrix. A single trait has correlation one and
  needs no LKJ variable.
- Ranking intercepts use a block-wise `ZeroSumNormal(sigma=1)` prior.
- Graded-pair intercepts use independent `Normal(mu=0, sigma=1)` priors.

The lognormal is broad; these are the declared tutorial priors, not a promise of
universal suitability. Prior predictive analysis must precede fitting. Users
can explicitly change supported priors without automatic data-driven scaling.

PyMC 6.3.2's installed `LKJCorr` implementation produces a lower triangular
correlation Cholesky factor. Use `R = L @ L.T` and verify unit diagonal; do not
assume an API returning packed off-diagonal correlations. Implementations must
check the supported version's behavior rather than silently changing the trait
prior. The package uses centered coordinates, matching the tutorial:
`theta ~ MvNormal(mu=0, chol=L)`. Both calibration and new-person conditional
scoring sample the traits directly. The Cholesky factor still defines the
correlation prior; loading signs and intercept identification are unchanged.
Record parameterization and sampling settings before comparing runs.

## Ranking, PICK and MOLE

For each complete preference order, assign score `K-r` to the statement at
one-based rank `r`. Score columns always refer to statement positions, not rank
slots. For example, `[2, 3, 1]` has statement scores `[0, 2, 1]`.

The categorical support contains all `K!` complete orders. PICK retains only the
first slot; MOLE retains the first and last slots. Their observed event
probabilities sum the probabilities of compatible complete orders. Unknown slots
do not remove statements and do not produce additional independent observations.

All complete-ranking score rows have the same sum, `K*(K-1)/2`. A common
intercept shift therefore leaves the response probabilities unchanged. Fix the
location by imposing a zero sum within each ranking block. The stated
`ZeroSumNormal` sigma is the underlying Gaussian scale; its constrained
components are not independent unit-variance Normal variables.

The reference tutorial uses distinct traits within each block. Reused
statements and other loading designs require explicit identification checks;
data validation alone does not guarantee identifiable parameters. In particular,
a shared statement must not receive independently duplicated parameters from
different size groups. Unsupported cross-block zero-sum constraints must be
rejected until they have a validated parameterization.

## Graded pairs

The tutorial uses these score rows, ordered from preference for position 2 to
preference for position 1:

```text
[[0, 2],
 [0, 1],
 [0, 0],
 [1, 0],
 [2, 0]]
```

The middle category has zero utility. Other category utilities are twice or
once the relevant statement utility. Row sums differ, so a common intercept
shift changes the category distribution. Do not apply the ranking zero-sum
constraint to this branch.

The response scale declares the ordered original categories and direction.
Map categories to score rows explicitly; an unobserved category remains in the
support. If higher categories prefer position 2, reverse the score rows relative
to the baseline. Other category counts or score schemes require explicit model
support and tests; the data container can represent them without claiming they
are already fitted by MGPPM.

## Observation and verification targets

One effective person/block response is one observed event. Genuine missing or
unadministered blocks contribute no response log-likelihood. The default
pointwise likelihood conditions on both theta and item parameters and includes
the full category normalization. It excludes priors and Jacobians.

Before fitting, compare fixed utilities, probabilities and gradients with the
short scalar reference in `tests/_reference_mgppm.py`. Test ranking and graded
identification separately, compatible-event sums, label permutations and stable
logarithmic computation. Verify the generator independently of the fitter.
Then use small real fits and controlled recovery; neither a successful sampler
call nor agreement between two implementations sharing the same formula is
sufficient evidence by itself.
