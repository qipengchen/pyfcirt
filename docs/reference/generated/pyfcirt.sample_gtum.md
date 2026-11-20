# sample_gtum

```{eval-rst}
.. autofunction:: pyfcirt.sample_gtum
```

Fit a logistic GTUM model with the experimental NUTS and conditional MH sampler.

## Parameters

`model` : Model
: Prepared model with ordinary logistic GTUM only, centered traits, and
  no population groups or covariates. Use the original native graph.

`draws`, `tune`, `chains` : int
: Retained draws and warmup iterations per chain, and number of chains.
  Defaults: 1000, 1000 and 4.

`random_seed` : int or None
: Sampling seed.

`var_names` : sequence of str or None
: Native final variables to retain. Explicit selections must include
  `theta` and required free final measurement and population parameters.

`target_accept` : float
: Target NUTS acceptance. Default: 0.8.

`max_treedepth` : int
: Maximum tree depth. Default: 10.

`initial_step_size` : float
: Initial step size. Default: 1.0.

`dense_mass` : bool
: Use a dense mass matrix. Default: `False`.

`initvals` : mapping, sequence of mappings or None
: Initial values, shared or supplied one mapping per chain.

`progressbar` : bool
: This sampler requires `False` (the default).

`compute_convergence_checks` : bool
: Run native convergence checks. Default: `True`.

`overwrite` : bool
: Replace an existing fit. Default: `False`.

## Returns

xarray.DataTree
: `posterior` contains `theta(chain, draw, person, trait)` and the
  retained calibration variables. `sample_stats` includes NUTS and
  conditional proposal diagnostics. The result becomes `model.idata`.

## Notes

The experimental sampler requires BlackJAX and float64. Warmup uses
full-parameter NUTS. Each retained transition includes a conditional
Metropolis–Hastings update for person traits. Examine rank diagnostics,
NUTS divergences and conditional proposal diagnostics after fitting.

## Examples

Fit the logistic GTUM model with this call:

```python
import pyfcirt as fc

example = fc.load_example("simulated_gtum", model="GTUM", n_persons=50)
model = fc.Model(example.data)
idata = fc.sample_gtum(model, draws=1000, tune=1000, chains=4, random_seed=43)
```

### Check an unsupported model


```python
import pyfcirt as fc

example = fc.load_example("tirt_study", model="TIRT", n_persons=3)
model = fc.Model(example.data)
try:
    fc.sample_gtum(model)
except ValueError as error:
    print(error)
```

```text
Conditional MH requires original logistic GTUM measurements only
```

## See also

[GTUM](pyfcirt.GTUM.md) and [standard NUTS](pyfcirt.Model.fit.md).
