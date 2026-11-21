# Bundled example data

Use `pyfcirt.load_example(name, model=...)`. The case directory gives the name
passed to `load_example()`.

- Each `n<N>.json.gz` stores the design, responses and metadata for a cohort of
  `N` people. Keep companion tables and provenance files in the same directory.
  For example, `custom_tirt/` contains inputs, person covariates and a generation
  record.
- Cases with one cohort use its full sample by default. A case with several
  cohorts declares `default_n_persons` in its `metadata.json`. The
  [`studies/`](studies/README.md) directory holds the multi-cohort cases and their
  source materials.

Paths in payload metadata are relative to this directory. See the
[dataset guide](../../docs/datasets.md) for the examples and their sources.
