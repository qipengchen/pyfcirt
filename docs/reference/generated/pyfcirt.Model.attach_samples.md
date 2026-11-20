# Model.attach_samples

```{eval-rst}
.. automethod:: pyfcirt.Model.attach_samples
```

Restore compatible saved samples into a freshly prepared model.

## Parameters

`idata` : xarray.DataTree
: Samples for the matching prepared model. Include `posterior`, optional
  `sample_stats`, and `calibration_evidence` when restoring recursive
  scoring. All numeric draws must be finite. Final variables must use the
  prepared coordinates and obey the fixed and joint constraints.

`assume_same_batch` : bool
: Default: `False`. Use `True` only to affirm that untagged external
  draws came from the current prepared answers, predictors, specification and
  any incoming approximation. Recorded batch or prior identities must agree.

## Returns

None
: Installs validated samples in `model.idata`. Recompute predictions and
  checks on the receiving model when needed.

## Notes

Recreate an unfitted model with the same responses, priors, population and
covariates. If the archive contains a fixed calibration snapshot, it must
agree with the receiving model's point. Failed validation leaves its current
results available.

## Examples

Save the fit from the [HEXACO TIRT calibration](pyfcirt.Model.fit.md) using
`model.idata.to_netcdf("fit.nc", engine="h5netcdf")`. The example below
recreates the same design and responses before attaching the saved draws.

```python
import xarray as xr
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
with xr.open_datatree("fit.nc", engine="h5netcdf") as saved:
    model.attach_samples(saved.load())

print(model.idata.posterior["theta"].dims)
```

```text
('chain', 'draw', 'person', 'trait')
```

## See also

[Saving and reusing results](../results.md) and the
[parameter and coordinate contract](../parameter_contract.md).
