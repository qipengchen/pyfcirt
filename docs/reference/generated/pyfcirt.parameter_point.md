# parameter_point

```{eval-rst}
.. autofunction:: pyfcirt.parameter_point
```

Get a mean parameter point or one labelled joint posterior draw.

## Parameters

`model` : Model
: Prepared model with final parameter draws or fixed values.

`draw` : tuple or None
: One `(chain_label, draw_label)` pair. `None` uses posterior means.

`include_person` : bool
: Include person parameters, such as theta. Default: `True`.

## Returns

xarray.Dataset
: Final values with prepared coordinate labels and no `chain` or `draw`.
  The mean uses the exact prepared fixed value where applicable.
  With `include_person=False` it retains the calibration point for scoring.

## Notes

Posterior means describe a point estimate. For uncertainty use the posterior draws or the coefficient summary.

## Examples

Here model is the fitted binary 2PL calibration in the recursive scoring example. Read final loading means without the person estimates.

```python
import pyfcirt as fc

calibration = fc.parameter_point(model, include_person=False)
print(calibration["loading"].to_series().head(3).round(2).to_string())
```

```text
loading_id
0    1.40
1    1.29
2    1.23
```

## See also

[Prediction and model checks](../prediction.md).
