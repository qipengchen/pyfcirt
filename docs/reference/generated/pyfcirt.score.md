# score

```{eval-rst}
.. autofunction:: pyfcirt.score
```

Prepare and estimate a new-respondent scoring model.

## Parameters

`model` : Model
: Model defining the reference calibration. Supply a fitted model or pass
  its final parameters explicitly when scoring with a fixed point.

`new_responses` : pandas.DataFrame, ResponseTable or mapping
: New answers in the calibration's coding. For a plain DataFrame also pass
  `format`. A mixed design takes a mapping from `"fc"` or `"items"` to
  `ResponseTable`; omit unanswered channels unless you are supplying
  administration information for them.

`format` : str or None
: [Storage layout](pyfcirt.ResponseTable.md) for a single-channel DataFrame.
  Use a `ResponseTable` for custom wide-column mappings.

`person_data` : pandas.DataFrame or None
: Group labels and required predictors for new people, keyed by person ID.
  The fitted groups and encodings are retained.

`administration` : pandas.DataFrame, xarray.Dataset or None
: Optional labelled administration table or prepared schedule. Omission
  uses the supplied response records.

`parameters` : xarray.Dataset or None
: Explicit final calibration point without sample axes. Applies only in
  point mode and takes precedence over the snapshot and posterior mean.

`calibration` : {"point", "posterior"}
: `"point"` (default) fixes the final calibration quantities. The experimental
  `"posterior"` mode updates selected shared variables through an incoming approximation.

`approximation` : PosteriorApproximation or None
: Optional matching runtime prior for posterior calibration. `None` fits
  the default approximation from the reference posterior. To update a subset,
  pass an approximation built with `Model.posterior_approximation(var_names=...)`.

`var_names` : sequence of str or None
: Native output variables to retain through `Model.fit()`. This controls
  the fitted outputs. The approximation controls which parameters update.

`sample_kwargs`
: Native [Model.fit() options](pyfcirt.Model.fit.md), including
  `draws`, `tune`, `chains` and `nuts_sampler`.

## Returns

Model
: Independent fitted scoring model. `scored.idata.posterior.theta` uses
  `(chain, draw, person, trait)`. Point mode conditions on a fixed
  calibration snapshot. Posterior mode retains the selected shared draws
  and incoming history in `calibration_evidence`.

## Notes

Posterior scoring rejects individuals already covered by the incoming
calibration evidence. Point scoring allows rescoring people with their
calibration held fixed. Supply actual new response rows and the necessary
predictors or group labels.

## Examples

### Score with a fixed calibration

Start with a fitted `Model` and a new response table containing the same
items or blocks and response codes. Here `new_responses` is a
[`ResponseTable`](pyfcirt.ResponseTable.md). If you pass a DataFrame, also
supply its storage format, such as `format="block_long"`.

```python
scored = calibration.score(new_responses)
theta = scored.idata.posterior["theta"]
theta_mean = theta.mean(("chain", "draw"))
```

`theta_mean` has one row per new respondent and one column per trait.
The shared parameters are held at their calibration values in this call.

### Update the calibration during scoring

Use `calibration="posterior"` to update shared parameters as well as the
new respondent scores. The original fit must retain the native calibration
draws using `fit(retain_calibration=True)`. This example takes the first
batch of 50 respondents from the [recursive scoring tutorial](../../tutorials/recursive_scoring.ipynb).

```python
scored = calibration.score(
    batches[0], calibration="posterior", draws=1000, tune=1400, chains=4, random_seed=45
)
print(dict(scored.idata.posterior["theta"].sizes))
```

```text
{'chain': 4, 'draw': 1000, 'person': 50, 'trait': 4}
```

## See also

[score_model()](pyfcirt.score_model.md) and
[recursive scoring](../../tutorials/recursive_scoring.ipynb).
