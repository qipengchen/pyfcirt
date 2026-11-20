# Regression

```{eval-rst}
.. autoclass:: pyfcirt.Regression
```

Define a fixed-effect formula and its input table for one parameter target.

## Parameters

`formula` : str
: Right-hand side of a fixed-effect formula, for example `"study_hours + site"`.

`data` : pandas.DataFrame
: One row per entity with the matching ID column(s) and used predictors.

`id` : str or tuple of str
: Key columns identifying a person, statement, item or pair.

`priors` : mapping or None
: Native coefficient priors keyed by term, or `"common"` for non-intercept slopes.
  An explicit term takes precedence. Use `"Intercept"` for a measurement-parameter intercept.

`parameter_residual_sd` : positive float, PyMC RV or None
: Remaining measurement-parameter variation. `None` uses the target default.
  This is on the log scale for LogNormal loading regressions. Theta uses the
  population residual scale instead.

`center_predictors` : bool
: Subtract the training means of numeric predictors. Default: `True`.

`standardize_predictors` : bool
: Divide numeric predictors by their training SDs. Default: `False`.

`coefficient_noncentered` : bool
: Noncenter regression coefficients. Default: `False`.

`deviation_noncentered` : bool or None
: Noncenter remaining measurement deviations where supported.
  `None` uses the target-specific convention.


## Returns

Regression
: A declaration used as the value in a
  `Model.add_regressions({target: declaration})` mapping.

## Notes

Numeric columns are continuous; booleans, strings and pandas categories
are categorical. Use `a * b` for main effects and interaction, or
`a:b` for the interaction alone. Define transformed feature columns
in the DataFrame before using them. Used features must be complete.

### Supported measurement targets

All nine models support person-level `theta` predictors. Measurement targets
below are available when they have free entries:

| Model | Statement/item targets | Pair targets |
| --- | --- | --- |
| TIRT | `fc.loading_magnitude` | `fc.cutpoints` |
| GTUM | `fc.loading_magnitude`, `fc.location` | `fc.cutpoints` |
| MGPPM, TwoPLRank, TIRTRank | `fc.loading_magnitude`, `fc.intercept` | — |
| GGUMRank | `fc.loading_magnitude`, `fc.threshold`, `fc.location` | — |
| DichotomousIRT | `items.loading_magnitude` (2PL–4PL), `items.intercept` | — |
| GPCM | `items.loading_magnitude` | — |
| GGUM | `items.loading_magnitude`, `items.location` | — |

The shared 1PL discrimination has no item-specific predictor. Side-specific
location roles can use distinct regressions. A pair predictor changes the
common location of the cutpoints for that pair.

Loading slopes use a base Normal SD of 0.25; other default slopes use 1.
Automatic continuous-slope priors account for training feature scales. For
log-loading regressions the remaining SD has a HalfNormal(0.2) default.
Leaving a prior in its default does not change the training predictor units.
Theta regression uses the population's mean and covariance and keeps the
unit residual SD by default.

## Examples

### Person and statement predictors


This example binds a person-level predictor to the conditional ability mean. The fitted feature is listed on the theta predictor axis.

```python
import pandas as pd
import pyfcirt as fc

bundle = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=4)
ids = bundle.data.responses.frame["person_id"].drop_duplicates().tolist()
people = pd.DataFrame({"person_id": ids, "study_hours": [1.0, 3.0, 2.0, 6.0]})

model = fc.Model(bundle.data)
declaration = fc.Regression("study_hours", people, id="person_id")
model.add_regressions({"theta": declaration})
print(model.prepared.coords["theta_predictor"].tolist())
```

```text
['study_hours']
```

## See also

[Model.add_regressions()](pyfcirt.Model.add_regressions.md) and
[RegressionOptions](pyfcirt.RegressionOptions.md).
