# Model.add_regressions

```{eval-rst}
.. automethod:: pyfcirt.Model.add_regressions
```

Add fixed-effect predictors to a prepared model before estimation.

## Parameters

`formulas` : str or mapping
: One `target ~ predictors` line per target, or a mapping from targets to
  `Regression` objects. In mapping form, each regression supplies its own
  table, IDs and options. Use a tuple such as `("fc", "TIRT", "cutpoints")`
  to select one measurement model in a mixed design.

`person_data`, `statement_data`, `item_data` : pandas.DataFrame or None
: Feature tables keyed by `person_id`, `statement_id` or `item_id`.
  Provide one row per used entity.

`pair_data` : pandas.DataFrame or None
: Features keyed by `(block_id, statement_1, statement_2)`, one row per
  comparison. Reversed pairs align to design order.

`id_columns` : mapping or None
: Alternative key names by entity: `person`, `statement`, `item` or `pair`.

`options` : mapping or None
: `RegressionOptions` per target for string formulas. For a model-specific
  coefficient prior, add a qualified key such as `fc.TIRT.loading_magnitude`
  under `options`, alongside the ordinary `fc.loading_magnitude` formula.

## Returns

None
: Updates an already prepared, unfitted model. A later declaration for the
  same target replaces that regression and preserves other targets and fixed
  references. The feature encoding is retained for scoring.

## Notes

Use [available_covariate_targets()](pyfcirt.Model.available_covariate_targets.md)
to find the regression targets and predictor tables for the current design.

Use `theta` with a person table, `fc.loading_magnitude` with a statement
table, `fc.cutpoints` with a pair table, and item parameters with an item
table. The [supported targets](pyfcirt.Regression.md#supported-measurement-targets)
depend on the response model. Missing values in used predictor columns raise
an error. Formulae support main effects and interactions.

A model-qualified option may override only coefficient priors. A qualified
term-specific prior takes precedence over its `common` slope prior, then
the unqualified option, then the model default. The qualified prior shares
the training feature encoding with the unqualified formula.

## Examples

### String formula


Four people have a continuous predictor and a categorical predictor.
After preparing the model, look up the person-level target before writing
the formula.

```python
import pandas as pd
import pyfcirt as fc

bundle = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=4)
ids = bundle.data.responses.frame["person_id"].drop_duplicates().tolist()
people = pd.DataFrame({"person_id": ids, "study_hours": [1.0, 3.0, 2.0, 6.0]})

people["site"] = ["A", "A", "B", "B"]
model = fc.Model(bundle.data)
```

```python
print(model.available_covariate_targets(level="person").to_string(index=False))
```

```text
target model  level       kind predictor_level
 theta  None person regression          person
```

Use `theta` on the left side of the formula and supply the indicated person table.
The fitted predictor axis includes the interaction.

```python
model.add_regressions("theta ~ study_hours * site", person_data=people)
print(model.prepared.coords["theta_predictor"].tolist())
```

```text
['study_hours', 'site[B]', 'study_hours:site[B]']
```

### Explicit regression declaration


A Regression record keeps a coefficient prior and the input table with its formula.

```python
import pymc as pm

model = fc.Model(bundle.data)
effect = fc.Regression(
    "study_hours",
    people,
    id="person_id",
    priors={"study_hours": pm.Normal.dist(mu=0, sigma=0.3)},
)
model.add_regressions({"theta": effect})
print(model.prepared.coords["theta_predictor"].tolist())
```

```text
['study_hours']
```

## See also

[Regression](pyfcirt.Regression.md) and
[priors for statement covariates](../../tutorials/prior_comparison.ipynb#set-priors-on-a-covariates-intercept-and-slope).
