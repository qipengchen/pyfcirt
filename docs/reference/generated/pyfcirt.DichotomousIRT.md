# DichotomousIRT

```{eval-rst}
.. autoclass:: pyfcirt.DichotomousIRT
```

Binary 1PL–4PL models with a logistic response curve.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.

`kind` : {"1PL", "2PL", "3PL", "4PL"}
: Model type. Default: `"2PL"`. 1PL estimates one common loading. 1PL and 2PL
  fix asymptotes to 0 and 1; 3PL estimates the lower asymptote; 4PL estimates both.

## Returns

DichotomousIRT
: A measurement declaration for `ItemInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Use two consecutive categories in the item declaration. In 4PL the upper
 asymptote's Beta base prior is truncated above the corresponding lower value.
 Custom 1PL priors can relax the default positive loading convention.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0.25, sigma=0.5) |
| `intercept` | Normal(mu=0, sigma=1) |
| `lower_asymptote` | Beta(alpha=1, beta=4) |
| `upper_asymptote` | Beta(alpha=4, beta=1) |

## Examples

The eight items share one discrimination in this 1PL declaration.

```python
import pyfcirt as fc

declaration = fc.DichotomousIRT(kind="1PL")
example = fc.load_example("simulated_mirt", model=declaration, n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print(prepared.constant_data.sizes["loading_id"])
```

```text
1
```

## See also

[Equations and identification](../../models/single_items.md#dichotomous-irt) and
[worked analysis](../../tutorials/binary_ordinal_workflow.ipynb).
