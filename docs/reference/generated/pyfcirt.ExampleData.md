# ExampleData

```{eval-rst}
.. autoclass:: pyfcirt.ExampleData
```

Read the design, answers and optional companion tables returned by the dataset loader.

## Parameters


`data` : FCInput, ItemInput or tuple of both
: Fresh design and response declarations.

`truth` : xarray.Dataset or None
: Original generating arrays when available and requested.

`metadata` : mapping
: Resource provenance and sample-selection information.

`person_data` : pandas.DataFrame or None
: Matched predictor or grouping table when supplied.

`response_times` : pandas.DataFrame or None
: Matched response-time table when supplied.


## Returns

ExampleData
: The bundle returned by `load_example()`. Use `bundle.data` for
  preparation and join any companion tables by their IDs.

## Examples

```python
import pyfcirt as fc

bundle = fc.load_example("jrt_tirt", model="TIRT", n_persons=2)
print(bundle.response_times.head(3).to_string(index=False))
```

```text
person_id block_id   seconds
     J001       B5 39.040037
     J002       B2 43.381822
     J001       B3 24.876910
```

## See also

[load_example()](pyfcirt.load_example.md).
