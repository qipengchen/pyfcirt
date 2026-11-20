# Saving and reusing results

Save `model.idata` as NetCDF with xarray. It contains the fitted draws and
any explicitly stored outputs. Keep the original input data and
model-building code, including priors and covariates, to recreate the
matching prepared model.

## Save and restore samples

This example uses the fitted [HEXACO TIRT model](generated/pyfcirt.Model.fit.md)
with 120 respondents and six traits. Use a fresh model with the same
prepared inputs when restoring the samples. In a new session, recreate
the design, responses and model settings first.

```python
import xarray as xr
import pyfcirt as fc

model.idata.to_netcdf("analysis.nc", engine="h5netcdf")

restored = fc.Model(model.prepared)
with xr.open_datatree("analysis.nc", engine="h5netcdf") as saved:
    restored.attach_samples(saved.load())

theta = restored.idata.posterior["theta"]
print({axis: theta.sizes[axis] for axis in theta.dims})
```

```text
{'chain': 4, 'draw': 1200, 'person': 120, 'trait': 6}
```

Attachment restores the posterior and available sampling statistics.
Reading the complete saved DataTree directly also gives access to stored
predictions and check results. See [attach_samples()](generated/pyfcirt.Model.attach_samples.md)
for required identities and constraints.

## Inspect or export results

For example, select the Emotionality draws after fitting or restoring
the model. The result retains the respondent, chain and draw coordinates.

```python
theta = model.idata.posterior["theta"]
emotionality = theta.sel(trait="emotionality")
mean_scores = emotionality.mean(("chain", "draw"))
```

- See the [coordinate contract](parameter_contract.md) for `design`,
  `parameter_data` and other labelled groups.
- Use `coef(model)` for a parameter table with study labels. Use pandas
  to export this derived table.
- Use `store=False` on prediction and check calls to keep a returned
  result separate from the main tree.

For a scored batch, reconstruct its prepared responses and use its own
calibration point when attaching saved draws. Keep `calibration_evidence`
with recursive scoring results. See
[scoring and saving](posterior_approximation.md#scoring-and-saving).
