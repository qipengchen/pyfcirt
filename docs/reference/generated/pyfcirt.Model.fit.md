# Model.fit

```{eval-rst}
.. automethod:: pyfcirt.Model.fit
```

Fit a prepared measurement model using PyMC's native sampler.

## Parameters

`draws` : int
: Retained draws per chain. Default: 1000.

`tune` : int
: Warmup iterations per chain. Default: 1000.

`chains` : int or None
: Number of chains. `None` delegates to PyMC.

`target_accept` : float
: Target NUTS acceptance. Default: 0.8.

`nuts_sampler` : str or None
: Default: `"blackjax"`. Select another installed sampler such as
  `"pymc"`, `"numpyro"` or `"nutpie"`, or `None` for native selection.

`random_seed` : int, compatible native seed or None
: Sampling seed passed to PyMC.

`var_names` : sequence of str or None
: Native variables to save. `None` retains final parameters and omits
  names ending in `_raw`. Retain the variables needed by later analyses.

`retain_calibration` : bool
: Default: `False`. Save the native shared calibration draws needed for
  the first posterior-scoring round. This retains the calibration sources
  alongside the ordinary posterior outputs or the supplied `var_names`.

`overwrite` : bool
: Replace current samples and their derived results. Default: `False`.

`sample_kwargs`
: Additional PyMC sampling options. `idata_kwargs={"log_likelihood": True}`
  saves ordinary response factors. Use `log_likelihood()` for marginal
  TIRTRank probabilities.

## Returns

xarray.DataTree
: The fit, also accessible through `model.idata`. `posterior` uses
  `chain` and `draw` plus each variable's parameter axes.
  `sample_stats` contains available sampling diagnostics.

## Notes

BlackJAX and NumPyro default to parallel CPU chains and vectorized GPU chains.
Explicit native options override these choices. PyFCIRT defaults
`compute_convergence_checks` to `False`. Check R-hat, effective sample size
and divergences with ArviZ after fitting.

## Examples

This fit uses 120 respondents and returns scores for six traits.

```python
import pyfcirt as fc

bundle = fc.load_example(
    "hexaco_fc3",
    model="TIRT",
    n_persons=120,
    random_seed=42,
)
model = fc.Model()
model.add_fc(
    bundle.data.design,
    bundle.data.responses.frame,
    format="block_long",
    response_format="mole",
    model="TIRT",
)
model.prepare()
idata = model.fit(draws=1200, tune=1800, chains=4, random_seed=43)
print({axis: idata.posterior.sizes[axis] for axis in ("chain", "draw", "person", "trait")})
```

```text
{'chain': 4, 'draw': 1200, 'person': 120, 'trait': 6}
```

## See also

[Estimation backends](../../examples/estimation_backends.md) and
[the TIRT diagnostics example](../../tutorials/tirt_workflow.ipynb).
