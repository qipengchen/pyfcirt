# PreparedInputs.from_inputs

```{eval-rst}
.. automethod:: pyfcirt.PreparedInputs.from_inputs
```

Validate and encode design declarations before building a probability graph.

## Parameters


`data` : FCInput, ItemInput or tuple of both
: One declaration per channel. A two-element tuple combines forced-choice and item responses.

`population` : PopulationSpec or None
: Population groups and priors. `None` uses one population with zero means
  and unit residual scales.

`person_data` : pandas.DataFrame or None
: Group labels or person features, one row per `person_id`.

`regressions` : str, mapping or None
: Formula lines such as `"theta ~ age"` or a mapping of targets to `Regression`
  specifications.

`statement_data`, `item_data` : pandas.DataFrame or None
: Formula inputs with one row per `statement_id` or `item_id`.

`pair_data` : pandas.DataFrame or None
: Formula inputs keyed by `(block_id, statement_1, statement_2)`.
  Reversed pairs are aligned to design order without changing predictor values.

`id_columns` : mapping or None
: Alternative key columns for `person`, `statement`, `item` or `pair`.
  The default uses the keys above.

`regression_options` : mapping or None
: Options for string formulas, including coefficient prior overrides.
  See [add_regressions()](pyfcirt.Model.add_regressions.md).

`purpose` : {"observed", "generate"}
: Validate recorded answers (default), or prepare an unanswered generation schedule.

`theta_parameterization` : {"centered", "noncentered", "conditional"}
: Parameterization of person traits. The default is `"centered"`.

`centered_traits` : tuple, list or None
: For `"conditional"`, trait IDs to center; other traits are noncentered.
  `None` centers FC-measured traits. An empty list centers none.

`fixed_parameters` : xarray.Dataset or None
: Fixed final values with the prepared coordinates and no `chain` or `draw` axis.


## Returns

PreparedInputs
: Validated design and response arrays, coordinates, identifying constraints
  and optional fitted predictor encoders. Pass the result to `fc.Model(prepared)`.

## Notes

Supply feature tables and their key-column options together with formula
regressions. Person data can also be used for grouping without a regression.

## Examples

### Prepare joint responses


The bundled example supplies both channels for the same people. The dimensions below count people and distinct units.

```python
import pyfcirt as fc

example = fc.load_example(
    "mixed_measurement",
    model={"fc": "MGPPM", "items": "GPCM"},
    n_persons=5,
)
prepared = fc.PreparedInputs.from_inputs(example.data)
print({axis: prepared.constant_data.sizes[axis] for axis in ("person", "block", "item")})
```

```text
{'person': 5, 'block': 4, 'item': 4}
```

## See also

[Model.prepare()](pyfcirt.Model.md#prepare-data) and [PreparedInputs](pyfcirt.PreparedInputs.md).
