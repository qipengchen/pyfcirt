# Data and design

Declare what each block or item measures and how the answers are stored.
Register your tables with [`model.add_fc()`](generated/pyfcirt.Model.md#add-a-forced-choice-response-table)
or [`model.add_items()`](generated/pyfcirt.Model.md#add-an-individual-item-response-table),
then call [`model.prepare()`](generated/pyfcirt.Model.md#prepare-data).

## Examples and data summaries

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.load_example
   ~pyfcirt.data_coverage
```

## Design and response options

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.FCDesignMapping
   ~pyfcirt.ItemDesignMapping
   ~pyfcirt.BlockSpec
   ~pyfcirt.ItemOverride
   ~pyfcirt.CategoryScale
   ~pyfcirt.UnitIntervalScale
   ~pyfcirt.ComparisonColumn
   ~pyfcirt.ExampleData
```

## Explicit inputs and prepared data

`FCInput` and `ItemInput` let you assemble input declarations separately and
pass them to `model.prepare()`. Use `PreparedInputs.from_inputs()` if you need
the prepared data independently of the model.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.FCInput
   ~pyfcirt.ItemInput
   ~pyfcirt.ResponseTable
   ~pyfcirt.PreparedInputs
   ~pyfcirt.PreparedInputs.from_inputs
   ~pyfcirt.generation_schedule
```
