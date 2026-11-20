# Data and design

Declare what each block or item measures and how the answers are stored.
`FCInput` and `ItemInput` combine a design with a `ResponseTable`. Pass either
declaration, or both together, to `Model.prepare()`. Use `PreparedInputs.from_inputs()`
to prepare arrays without creating a model. For practice, `load_example()` loads
bundled designs and answers.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.FCInput
   ~pyfcirt.ItemInput
   ~pyfcirt.ResponseTable
   ~pyfcirt.FCDesignMapping
   ~pyfcirt.ItemDesignMapping
   ~pyfcirt.BlockSpec
   ~pyfcirt.ItemOverride
   ~pyfcirt.CategoryScale
   ~pyfcirt.UnitIntervalScale
   ~pyfcirt.ComparisonColumn
   ~pyfcirt.PreparedInputs
   ~pyfcirt.PreparedInputs.from_inputs
   ~pyfcirt.ExampleData
   ~pyfcirt.load_example
   ~pyfcirt.generation_schedule
   ~pyfcirt.data_summary
```
