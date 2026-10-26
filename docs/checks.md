# Pearson discrepancy and PPP

The implemented absolute-fit check follows the MGPPM tutorial's categorical
Pearson discrepancy, conditional on each posterior parameter draw:

```python
check = fit.pearson_check(random_seed=42)
print(check.PPP)

by_block = fit.pearson_check(groupby="block", random_seed=42)
print(by_block[["PPP", "n_obs"]])
```

For every actual person-block response, let `p_c` be the normalized probability
of category `c`. The discrepancy sums `(indicator(y == c) - p_c)**2 / p_c` over
categories and effective responses. Replicated responses use the same posterior
parameter draw as the observed discrepancy. `PPP` is the fraction of draws with
`D_rep >= D_obs`, including ties. Values near either extreme indicate poor
reproduction under this particular check; this is not a general convergence test.

The Dataset contains `D_obs`, `D_rep`, `log_D_obs`, `log_D_rep`, `PPP` and `n_obs`.
Discrepancies retain `chain` and `draw`; block grouping adds the original `block`
labels. Overall discrepancies are sums of the block discrepancies before the
comparison. Overall PPP is not the average of block PPPs.

This is the per-response conditional discrepancy used in the tutorial, rather
than a Pearson statistic calculated from category counts pooled over people.
Ranking categories are complete orders. PICK and MOLE use their actual marginal
event categories, preserving unknown ranks; graded uses its declared categories.
Missing and unadministered responses contribute nothing. A block without effective
responses has `n_obs=0`, zero discrepancy and undefined (`NaN`) PPP. A dataset with
no effective responses cannot run the check.

## Numerical calculation and memory

For a normalized categorical distribution, the category sum for response `y`
reduces algebraically to `1 / p_y - 1`. The implementation calculates its logarithm
with `expm1`, then sums discrepancies using log-sum-exp. Consequently it does not
need an arbitrary probability floor or a large one-hot tensor.

PPP comparisons use log discrepancies, which preserve ordering. If a raw
discrepancy exceeds floating-point range, the raw output is infinite and a warning
is emitted; finite log discrepancies and PPP remain available. A zero discrepancy
has log value `-inf`, which is valid. Invalid probability calculations are reported
instead of silently clipped or replaced.

The check clones the native PyMC graph, freezes free posterior parameter draws,
recomputes derived quantities and generates new responses. It retains only two
discrepancy vectors per draw, rather than storing every probability or replicated
response. The fitted graph and DataTree remain unchanged. Use
`fit.posterior_predictive()` when the replicated responses themselves are needed.

The tests compare the algebra against explicit one-hot Pearson arithmetic, verify
the inclusive PPP rule on fixed observed/replicated discrepancies, check block
aggregation and NA exclusion, and test actual overflow without clipping. Existing
small real fits are reused for workflow checks; no extra MCMC is needed merely
to calculate this statistic. Parameter recovery and full tutorial acceptance
remain separate requirements.
