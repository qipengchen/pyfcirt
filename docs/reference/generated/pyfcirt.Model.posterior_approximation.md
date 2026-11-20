# Model.posterior_approximation

```{eval-rst}
.. automethod:: pyfcirt.Model.posterior_approximation
```

Fit a runtime calibration prior from labelled posterior draws for experimental
posterior scoring and recursive calibration.

## Parameters

`var_names` : sequence of str, mapping or None
: Shared native PyMC RV names. A mapping gives explicit groups by parameter
  kind. On the first round, `None` selects all supported free shared RVs
  and pools them into one group by default. On later rounds, `None` reuses
  the incoming approximation's selection. An explicit selection updates only the supplied RVs.

`joint` : bool or None
: Default: `None` selects one group named `joint` on the first round with
  automatic selection. It keeps the grouping for explicit `var_names` or an
  incoming approximation. Use `False` to start with separate groups by kind.
  Use `True` to pool the automatic selection, including inherited members.
  Supply either `joint=True` or explicit `var_names`.
  Full-covariance fits require more retained draws than the joint dimension.

`method` : str
: Default: `"auto"` selects one full-covariance Gaussian or a mixture per group. See the
  [approximation methods](../posterior_approximation.md#approximation-methods).

`components` : sequence of int
: Candidate mixture sizes. Default: `(2, 3)`.

`random_seed` : int or None
: Seed for fitting the approximation.

`transforms` : mapping or None
: Native transform overrides keyed by RV name. `None` uses native
  defaults or inherits the previous round's overrides. `{}` restores
  native defaults on a later round.

## Returns

PosteriorApproximation
: A runtime prior for the selected shared variables and a record of
  covered calibration data. Supply it to
  `score_model(calibration="posterior", approximation=prior)`.

## Notes

Use `Model.fit(retain_calibration=True)` to retain the native draws for
an initial PyFCIRT calibration. The posterior must also include any RVs
needed by the selected parameters' transforms. A 4PL upper asymptote, for example, can need paired draws of
its lower asymptote. Use shared calibration variables; respondent traits are
estimated afresh in the scored model.

## Examples

Use the fitted MGPPM model from the [recursive scoring example](../../tutorials/recursive_scoring.ipynb).
By default, the approximation pools the loadings, intercepts and trait
correlations into one group and selects the density family with `auto`.

```python
prior = calibration.posterior_approximation(random_seed=45)
print(tuple(prior.groups), len(prior.evidence_people))
scoring = calibration.score_model(
    batches[0],
    calibration="posterior",
    approximation=prior,
)
```

```text
('joint',) 50
```

## See also

[PosteriorApproximation](pyfcirt.PosteriorApproximation.md) and
[recursive scoring](../../tutorials/recursive_scoring.ipynb).
