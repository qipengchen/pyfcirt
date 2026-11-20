# simulate

```{eval-rst}
.. autofunction:: pyfcirt.simulate
```

Simulate one dataset from a declared or calibrated measurement model.

## Parameters

`model` : Model, forced-choice specification, str or None
: A prepared model provides an existing design and population. A model name
  or declaration selects a law for an automatic forced-choice design.
  `None` uses MGPPM for an automatic design.

`source` : {"prior", "fixed", "posterior"}
: Draw generating parameters from the declared priors (default), use a
  supplied point, or use one joint posterior state. An automatic design
  requires the prior source.

`parameters` : xarray.Dataset, mapping or None
: Required for `source="fixed"`. Supply the canonical final parameter
  values on the prepared axes, without `chain` or `draw`. A mapping
  can use arrays or JSON-style lists in prepared order. For the same people,
  include their `theta(person, trait)` and any required local effects.

`draw` : tuple or None
: For the posterior source, use one `(chain_label, draw_label)`. `None`
  selects a joint state using the seed.

`administration` : xarray.Dataset or None
: For an existing design, use `generation_schedule()` for new people.
  Fresh abilities and local effects follow the population model. With
  `None`, keep the existing people and recorded nonresponse.

`random_seed` : int or None
: Reproducible nonnegative generation seed.

`n_persons`, `n_traits`, `n_blocks`, `block_size` : int or None
: Automatic design options. The defaults give 100 people, 3 traits,
  12 blocks and 3 statements per block.

`response_format` : {"rank", "pick", "mole"} or None
: Automatic design option. Joint models default to full ranks; TIRT and
  GTUM generate separate binary comparisons. Use an existing prepared design
  for graded or continuous scores.

## Returns

Model
: Independent, unfitted model with simulated responses. The
  `parameters/generation` group contains the generating values in their
  final axes. Use `response_data()` to display the generated answers and
  `data_summary()` to count them.

## Examples

### Draw from priors


Generate four respondents on two triplets using MGPPM.

```python
import pyfcirt as fc

generated = fc.simulate("MGPPM", n_persons=4, n_blocks=2, random_seed=42)
print(generated.data_summary().to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
MGPPM / rank                  2     3          4           8          0
```

### Fix calibration and generate new people


Use the realized global generating values from the first example. The schedule draws fresh traits for the new IDs.

```python
point = generated.idata["parameters/generation"].to_dataset().drop_vars("theta")
schedule = fc.generation_schedule(generated, person_ids=["P-new1", "P-new2"])
fresh = fc.simulate(
    generated,
    source="fixed",
    parameters=point,
    administration=schedule,
    random_seed=43,
)
print(fresh.prepared.coords["person"].tolist())
```

```text
['P-new1', 'P-new2']
```

### Generate from one posterior state


Here model is the fitted binary 2PL calibration in the recursive scoring example. Record the exact posterior state used to generate these answers.

```python
replicate = fc.simulate(model, source="posterior", draw=(0, 7), random_seed=44)
truth = replicate.idata["parameters/generation"]
print(truth.source_chain.item(), truth.source_draw.item())
```

```text
0 7
```

## See also

[Generation schedules](pyfcirt.generation_schedule.md) and
[Parameter points](pyfcirt.parameter_point.md).
