# PopulationSpec

```{eval-rst}
.. autoclass:: pyfcirt.PopulationSpec
```

Define population grouping, identifying scales and priors for abilities.

## Parameters

`group` : str or None
: Column name in person data. Provide one row per `person_id` with its
  group label. `None` uses one population.

`reference_group` : str, int or None
: Group with mean zero and residual trait SD fixed at one. `None` selects
  the first group in prepared person order.

`regression_by_group` : bool
: Estimate separate theta slopes by group. Default: `False`.

`theta_residual_scale` : {"unit", "group"}
: `"unit"` fixes SDs to one. `"group"` estimates eligible SDs outside the
  reference group. By default, a grouped population uses `group` and an
  ungrouped population uses `unit`.

`theta_correlation` : {"group", "shared"}
: `"group"` estimates a correlation matrix per group (default).
  `"shared"` estimates one matrix for all groups. This option is independent of
  residual SDs and group means. With no grouping, both options use one matrix.

`priors` : mapping or None
: Native PyMC distributions for free roles.
  Defaults include `trait_correlation` LKJ(1), `population_mean`
  Normal(0, 1) and `population_sd` HalfNormal(1).

## Returns

PopulationSpec
: Population declaration for `Model.prepare(..., population=...)`.
  Group-level arrays use the `population_group` and `trait` coordinates.
  With `theta_correlation="shared"`, `trait_correlation` has dimensions
  `trait, trait_aux` and grouped `covariance` keeps its group dimension.
  Covariates predict conditional means.

## Examples

Assign two groups by person ID before preparation.

```python
import pandas as pd
import pyfcirt as fc

bundle = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=4)
ids = bundle.data.responses.frame["person_id"].drop_duplicates().tolist()

people = pd.DataFrame({"person_id": ids, "site": ["A", "B", "A", "B"]})
population = fc.PopulationSpec(group="site", reference_group="A")
model = fc.Model().prepare(bundle.data, population=population, person_data=people)
print(model.prepared.coords["population_group"].tolist())
```

```text
['A', 'B']
```

## See also

[Trait summaries](pyfcirt.trait_summary.md) and
[multiple groups with MGPPM](../../tutorials/mgppm_groups.ipynb).
