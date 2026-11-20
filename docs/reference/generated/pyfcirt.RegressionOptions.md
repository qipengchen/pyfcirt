# RegressionOptions

```{eval-rst}
.. autoclass:: pyfcirt.RegressionOptions
```

Specify keyword options for the string form of Model.add_regressions().

## Parameters

`priors` : mapping or None
: Native coefficient priors by term, including `"Intercept"` and
  slope fallback `"common"`.

`parameter_residual_sd` : positive float, PyMC RV or None
: Remaining measurement-parameter SD. `None` uses the target default.

`center_predictors` : bool
: Center numeric predictors. Default when omitted: `True`.

`standardize_predictors` : bool
: Divide numeric predictors by their training SDs. Default: `False`.

`coefficient_noncentered` : bool
: Noncenter the coefficients. Default: `False`.

`deviation_noncentered` : bool or None
: Parameterization of remaining measurement deviations.
  `None` leaves the target default.

## Returns

dict
: A typed options mapping. Pass it under the target key in
  `Model.add_regressions(..., options=...)`. Omitted keys leave their defaults.

## Examples

Standardize the training predictor for a theta formula.

```python
import pandas as pd
import pyfcirt as fc

bundle = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=4)
ids = bundle.data.responses.frame["person_id"].drop_duplicates().tolist()
people = pd.DataFrame({"person_id": ids, "study_hours": [1.0, 3.0, 2.0, 6.0]})

model = fc.Model(bundle.data)
options = fc.RegressionOptions(standardize_predictors=True)
model.add_regressions(
    "theta ~ study_hours",
    person_data=people,
    options={"theta": options},
)
print(model.prepared.coords["theta_predictor"].tolist())
```

```text
['study_hours']
```

## See also

[Regression](pyfcirt.Regression.md) for the same options with an explicit feature table.
