# New-person scoring

Scoring estimates new-person traits conditional on fixed item parameters and a
fixed population correlation matrix. New responses do not update calibration.
The posterior means of the calibrated parameters are used by default.

```python
scores = fitted.score(new_data, random_seed=42)
scores.summary()
scores.diagnostics()
scores.person_status
```

This runs one scoring fit, with the requested number of chains. Trait intervals
reflect uncertainty given that fixed calibration; **calibration uncertainty is
not propagated**. Only fixed calibration is supported. `mode="fixed"` is optional;
`mode="posterior"` and `calibration_draws` are rejected before sampling.

## Calibration points

The calibration Dataset contains `loading_magnitude(statement)`,
`intercept(statement)` and `trait_correlation(trait, trait_aux)`. The Cholesky
factor is calculated from the correlation matrix, rather than averaging Cholesky
factors. A selected joint calibration draw can also be used as a single fixed
point; this does not integrate over calibration uncertainty.

```python
point = fitted.calibration(draw=(0, 20))
scores = fitted.score(new_data, calibration=point, random_seed=42)
```

Known parameters can be supplied without a calibration fit through
`pyfcirt.scoring.score(built, new_data, calibration=point, ...)`.

## Native results and missing responses

`scores.idata` holds native `posterior` and `sample_stats` groups plus the fixed
`calibration` and `person_status`. `scores.samples` stacks chains and draws into
`(sample, person, trait)` for convenient downstream use. `scores.sampling` records
one executed settings dictionary. Summaries and diagnostics delegate to ArviZ.

New data must retain statement identities, positions, traits, keying, response
task and graded scale. A subset of known blocks is allowed. PICK/MOLE retain
their event likelihood without inventing missing ranks.

`scores.person_status` distinguishes response-informed `posterior` traits from
`prior_only` traits for unanswered people. When everyone is unanswered, exact
prior draws replace MCMC and convergence diagnostics are unavailable.

Save with `scores.save(path)` and load with `pyfcirt.load(path)`. The current score
bundle format is version 2. Earlier score bundles used the removed mixture
representation and are explicitly rejected. Calibration-fit bundles are unchanged.
See the [persistence guide](io.md).
