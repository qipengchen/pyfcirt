# Model.available_covariate_targets

```{eval-rst}
.. automethod:: pyfcirt.Model.available_covariate_targets
```

List the covariate targets available in a prepared design.

## Parameters

`level` : {"person", "statement", "pair", "item", "measurement"} or None
: Filter by the level of the parameter being predicted. By default, show all
  levels. A scalar for a measurement unit, such as GTUM Beta response precision,
  uses `"measurement"`.

`kind` : {"regression", "dif"} or None
: Show ordinary regressions, respondent-level measurement effects (DIF), or
  both by default.

## Returns

pandas.DataFrame
: Columns are `target`, `model`, `level`, `kind`, and `predictor_level`.
  `model` is the measurement model name, or missing for the population target
  `theta`. `predictor_level` identifies the table to supply. Use a string
  regression target on the left side of a formula, a tuple target as a key in
  a mapping of Regression declarations, and a DIF target in `add_dif_terms()`.
  A valid filter with no matches returns an empty table with the same columns.

The query uses the design's free parameters and identification constraints.
It can run before or after fitting. Add the selected covariates before fitting.

## Examples

```python
import pyfcirt as fc

example = fc.load_example("tirt_study", model="TIRT", n_persons=3)
model = fc.Model(example.data)
print(model.available_covariate_targets(level="pair").to_string(index=False))
```

```text
      target model level       kind predictor_level
fc.cutpoints  TIRT  pair regression            pair
fc.cutpoints  TIRT  pair        dif          person
```

## See also

[Add regressions](pyfcirt.Model.add_regressions.md) and
[differential item functioning](../dif.md).
