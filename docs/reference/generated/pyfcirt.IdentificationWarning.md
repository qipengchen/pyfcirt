# IdentificationWarning

```{eval-rst}
.. autoclass:: pyfcirt.IdentificationWarning
```

Recognize when a prior or design relaxes a default orientation convention.

## Returns

IdentificationWarning
: A `UserWarning` with the affected orientation convention. Preparation may
  emit it when a custom prior admits values on both sides of zero, or an
  unfolding trait has no signed location anchors.

## Examples

### Watch for a relaxed loading orientation


Use a Normal loading prior to permit a negative loading. The warning can be caught like any Python warning.

```python
import warnings
import pymc as pm
import pyfcirt as fc

declaration = fc.DichotomousIRT(priors={"loading_magnitude": pm.Normal.dist()})
example = fc.load_example("simulated_mirt", model=declaration, n_persons=3)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", fc.IdentificationWarning)
    prepared = fc.PreparedInputs.from_inputs(example.data)

print(
    next(
        type(w.message).__name__ for w in caught if isinstance(w.message, fc.IdentificationWarning)
    )
)
```

```text
IdentificationWarning
```

## See also

[Model.set_priors()](pyfcirt.Model.set_priors.md) for prior overrides.
