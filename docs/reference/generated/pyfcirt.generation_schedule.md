# generation_schedule

```{eval-rst}
.. autofunction:: pyfcirt.generation_schedule
```

Schedule unanswered administrations on a prepared design for new respondents.

## Parameters


`model` : Model
: Prepared reference model. A fitted posterior is optional.

`person_ids` : sequence
: Nonempty unique string or integer IDs, in the desired generation order.

`person_data` : pandas.DataFrame or None
: New people's group labels and predictor values when the reference model
  uses them. Match one row to each requested ID.


## Returns

xarray.Dataset
: New person and `schedule` coordinates. Each `schedule` entry is one
  scheduled block, pair or item event. Pass it to `simulate(administration=...)`.

## Notes

The schedule contains no answers. Events follow the requested person order and
then the prepared unit and pair order. The schedule is independent of the reference model.

## Examples

```python
import pyfcirt as fc

bundle = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=2)
template = fc.Model().prepare(bundle.data)
schedule = fc.generation_schedule(template, person_ids=["new1", "new2"])
print({axis: schedule.sizes[axis] for axis in ("person", "unit", "schedule")})
```

```text
{'person': 2, 'unit': 8, 'schedule': 16}
```

## See also

[simulate()](pyfcirt.simulate.md).
