# information

```{eval-rst}
.. autofunction:: pyfcirt.information
```

Evaluate expected likelihood Fisher information at a parameter point.

## Parameters

`model` : Model
: Prepared model defining responses and parameter coordinates.

`parameters` : xarray.Dataset or None
: Labelled final parameter point with no sample axes. `None` uses the
  fitted posterior point. The explicit route works with non-MCMC estimates
  on the same final coordinates.

`draw` : tuple or None
: One `(chain_label, draw_label)` to use instead of posterior means.
  Applies when reading the point from the model.

`theta` : xarray.DataArray or None
: Optional ability evaluation point labelled `(person, trait)`.
  `None` uses the model's final ability values.

`include_intermediates` : bool
: Include per-observation information and category/shape derivatives.
  Default: `False`.

`store` : bool
: Publish the result to `model.idata.information`. Default: `True`.

`progressbar` : bool
: Display evaluation progress. Default: `True`.

## Returns

xarray.Dataset
: `test_information(person, trait, trait_aux)` sums answered units per person.
  Block and item totals also retain their unit axis.
  `observation_information(obs, trait, trait_aux)` is optional.

## Notes

This is expected likelihood information conditional on calibration. It excludes prior precision. TIRT sums its comparison contributions; the resulting matrix describes its fitted comparison factors.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Inspect the first respondent's test information. The example selects two traits from the full matrix to display.

```python
info = model.information(store=False, progressbar=False)
matrix = info.test_information.isel(
    person=0,
    trait=slice(0, 2),
    trait_aux=slice(0, 2),
)
print(matrix.to_pandas().round(2).to_string())
```

```text
trait_aux        honestyhumility  openness
trait
honestyhumility             4.29     -0.36
openness                   -0.36      3.30
```

## See also

[Information and reliability](../information.md).
