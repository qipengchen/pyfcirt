# PosteriorApproximation

```{py:class} pyfcirt.PosteriorApproximation
:canonical: pyfcirt.scoring.approximation.PosteriorApproximation

Experimental runtime approximation to selected native PyMC free variables.
```

## From idata

```{eval-rst}
.. automethod:: pyfcirt.PosteriorApproximation.from_idata
```

### Parameters

`idata` : xarray.DataTree
: Natural-scale posterior with labelled `chain` and `draw` axes.
  Each selected RV's remaining dimensions and labels must match the native model.

`model` : pymc.Model
: Model that owns the selected continuous RVs.

`var_names` : sequence of str or mapping
: Native free-variable names. A sequence gives one density per RV. To group
  multiple RVs in a joint density, map a parameter-kind name to those RV names.

`method` : str
: Default: `"mvn"`. See [approximation methods](../posterior_approximation.md#approximation-methods).

`components` : sequence of int
: Candidate mixture component counts. Default: `(2, 3)`.

`random_seed` : int or None
: Seed for fitting the density.

`transforms` : mapping or None
: PyMC value-transform overrides keyed by RV name. `None` uses the native
  defaults. To override a constrained variable, use a transform whose inverse
  respects its support. Retain paired posterior draws for any free variable
  that supplies its bounds.

### Returns

PosteriorApproximation
: Runtime object with the selected `var_names`, grouping and fitting
  `diagnostics`. It can supply replacement priors in a compatible graph.

## Apply to a native model

```{eval-rst}
.. automethod:: pyfcirt.PosteriorApproximation.apply
```

Pass a PyMC `model` with the same selected RV names, parameter shapes and
coordinate system. The result is a new native `pymc.Model` with the fitted
priors; the input model retains its original priors. For PyFCIRT scoring, use
[Model.posterior_approximation()](pyfcirt.Model.posterior_approximation.md)
to carry design and evidence.

## Examples

Start with the fitted native PyMC graph and posterior from a calibration.
This example uses the MGPPM calibration from the
[recursive scoring analysis](../../tutorials/recursive_scoring.ipynb).
It chooses the native source of the final `loading` variable and applies
the fitted prior to a new graph.

```python
import pyfcirt as fc
from pytensor.graph.traversal import ancestors

native = calibration.build()
(loading_rv,) = [rv for rv in native.free_RVs if rv in ancestors([native["loading"]])]
prior = fc.PosteriorApproximation.from_idata(
    calibration.idata,
    model=native,
    var_names={"loading": [loading_rv.name]},
)
updated = prior.apply(native)
print(updated is not native, tuple(prior.groups))
```

```text
True ('loading',)
```
