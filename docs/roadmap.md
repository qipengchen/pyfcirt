# Project scope

The current [API](api.md) uses prepared FC/item inputs, reusable native PyMC graphs,
model-owned measurement distributions and independent new-person scoring.
See the [architecture](architecture.md) for responsibilities and the
[workflow](tutorials/tirt_workflow.ipynb) for supported analysis operations. Experimental methods
are identified on their respective pages; no numerical benchmark or recovery
claim follows from interface availability.

## Planned API names for observed data

- `model.data_coverage(*, by="model", details=False)` replaces
  `model.data_summary()`. Keep counts of administrations, recorded answers
  and missing responses, with the current index and options.
- `fc.response_counts(answers, *, groupby=None)` replaces
  `fc.response_summary()`. Keep the observed category counts, denominators
  and proportions.
- `model.response_data()` remains the entry for extracting labelled observed
  responses. Its long table is the input to `fc.response_counts()`.

When implementing the rename, update the exports, Model method, API pages,
examples and tests together. Preserve the signatures and behavior apart from
the names. Keep the current callable names in runnable documentation until
the corresponding code change.
