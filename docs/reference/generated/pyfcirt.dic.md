# dic

```{eval-rst}
.. autofunction:: pyfcirt.dic
```

Compute chainwise variance-penalty DIC from response log likelihood.

## Parameters

`log_likelihood` : xarray.DataArray
: Finite MCMC response log likelihood with nonempty labelled `chain`,
  `draw` and `obs` dimensions, as returned by `fc.log_likelihood()`.

## Returns

xarray.Dataset
: `Dbar_chain`, `pV_chain` and `DIC_chain` use the `chain` axis.
  `Dbar`, `pV` and `DIC` are their respective chain averages.

## Notes

For each chain, pV is half the sample variance of the total deviance. There must be at least two retained draws per chain. Compare the same observations and observation unit.

## Examples

Here model is the fitted binary 2PL calibration in the recursive scoring example. Report the per-chain values for the chosen display subset.

```python
import pyfcirt as fc

small = fc.Model(model.prepared)
small.attach_samples(model.idata.isel(draw=slice(0, 32)))

logs = fc.log_likelihood(small, store=False, progressbar=False)

result = fc.dic(logs)
rows = result[["Dbar_chain", "pV_chain", "DIC_chain"]]
print(rows.to_dataframe().round(2).to_string())
```

```text
       Dbar_chain  pV_chain  DIC_chain
chain
0         1268.93    122.49    1391.42
1         1267.15    184.23    1451.37
2         1278.24    142.77    1421.01
3         1272.12    109.75    1381.87
```

## See also

[Prediction and model checks](../prediction.md).
