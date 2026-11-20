# score_model

```{eval-rst}
.. autofunction:: pyfcirt.score_model
```

Build an independent scoring model without estimating it.

## Parameters

`reference` : Model
: Model supplying the reference measurement design and frozen feature encoding.

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
  the default approximation from the reference posterior. Pass an explicit
  approximation to select a subset, another method or custom transforms.

## Returns

Model
: Independent, unfitted model for the new responses. Point mode has
  fixed final calibration quantities. Posterior mode can update the
  selected shared variables, fixing other retained calibration values.
  Fit with `scoring.fit()` or use `scoring.build()` for native PyMC.

## Notes

A prepared parameter point uses the same labelled axes as the reference.
Preserve the complete measurement design for new points. For posterior mode
the approximation must cover the reference's existing evidence, and new
respondents must be outside that evidence. On a first calibration fit,
use `retain_calibration=True` to save the native draws for posterior scoring.

## Examples

### Build a model with fixed calibration

Use a fitted `Model` and a [`ResponseTable`](pyfcirt.ResponseTable.md)
containing new responses in the original design and coding. The first
batch in the [recursive scoring tutorial](../../tutorials/recursive_scoring.ipynb)
has 50 respondents.

```python
scoring = calibration.score_model(batches[0])
print(scoring.prepared.constant_data.sizes["person"])
```

```text
50
```

Call `scoring.fit()` when you want PyFCIRT to estimate the scores, or use
`scoring.build()` to work with the native PyMC graph.

### Build a model that updates the calibration

The original fit saved its native draws using `retain_calibration=True`.
Pass `calibration="posterior"` to carry that calibration into the new
model. The returned model records which respondents the prior already covers.

```python
updating = calibration.score_model(batches[0], calibration="posterior")
print(updating.idata.calibration_evidence.sizes["evidence_person"])
```

```text
50
```

## See also

[score()](pyfcirt.score.md),
[parameter_point()](pyfcirt.parameter_point.md) and
[posterior approximation](../posterior_approximation.md).
