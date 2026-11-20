# Model.build

```{eval-rst}
.. automethod:: pyfcirt.Model.build
```

Get the PyMC graph for an already prepared model.

## Returns

pymc.Model
: The native graph for the prepared data. Repeated calls reuse the graph
  until the model specification changes.

## Notes

Use `with native:` for native PyMC calls. A reconstructed recursive-scoring
round needs the original runtime prior to rebuild its estimation graph.

## Examples

The model has one observed response variable for each of its eight items.

```python
import pyfcirt as fc

bundle = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=3)
model = fc.Model(bundle.data)
native = model.build()
print(len(native.observed_RVs))
```

```text
8
```

## See also

[Native distributions](../distributions.md) and
[estimation backends](../../examples/estimation_backends.md).
