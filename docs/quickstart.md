# Quickstart

Install the package into a Python 3.12+ environment. For the complete notebook,
install `pyfcirt[docs,plots,io]`; add `numpyro` only when selecting that sampler.

The package keeps design and response validation separate from sampling:

```python
import pandas as pd
import pyfcirt as fc

design = fc.FCDesign.from_mapping(
    traits={"T1": ["A", "C"], "T2": ["B", "D"]},
    blocks={"pair1": ["A", "B"], "pair2": ["C", "D"]},
)
data = fc.FCData.from_frames(
    pd.DataFrame(
        {
            "person_id": ["P1", "P1", "P2", "P2"],
            "block_id": ["pair1", "pair2", "pair1", "pair2"],
            "response": [[1, 2], [2, 1], [2, 1], [1, 2]],
        }
    ),
    design,
)
model = fc.Model(data)
print(model.prior_summary())
built = model.build()
prior = built.prior_predictive(draws=50, random_seed=42)
print(built.response_frame(prior).head())
```

This tiny dataset demonstrates the interface. It is insufficient for useful item
calibration. For a substantive analysis, use an informative design and enough
responses, check prior predictions, then fit and inspect diagnostics:

```python
fitted = built.fit(draws=500, tune=700, chains=4, cores=1, random_seed=43)
print(fitted.diagnostics())
print(fitted.summary())
```

Assess precision and convergence before interpreting estimates. The full
[MGPPM workflow](tutorials/mgppm_workflow.ipynb) covers empirical ranking and
simulated graded input, plots, prediction, model checks, conditional likelihood
criteria, held-out-person scoring and saved analyses. The model now uses centered trait parameters. Small manual sampling checks have run for both tasks; previous noncentered
recovery results remain historical evidence. The full empirical notebook and
population recovery are optional manual investigations, outside formal tests.

## Prepared tutorial inputs and execution

Raw empirical inputs are external because their redistribution permission has
not been established by the conversion audit. Run the conversion for each
locally authorized source folder, then arrange its output as
`<data-root>/tutorial/mgppm-ranking` and `<data-root>/tutorial/mgppm-graded`:

```shell
python scripts/prepare_tutorial.py --input /path/to/ranking --task rank --output /path/to/data/tutorial/mgppm-ranking
python scripts/prepare_tutorial.py --input /path/to/graded --task graded --output /path/to/data/tutorial/mgppm-graded
```

The script requires the tutorial's `statement_design.csv`, `responses.csv`, and
ranking `selection_orders.csv`. It records conversion and source checksums.
The graded source in this tutorial is simulated; do not describe it as empirical.

On the configured compute worker, execute the full notebook with the installed
environment and external storage. Use a new output directory for each run:

```shell
python scripts/dev.py --storage-root /path/to/storage -- python scripts/execute_tutorial.py --notebook docs/tutorials/mgppm_workflow.ipynb --output /path/to/storage/results/tutorial/run-01
```

The runner creates a temporary kernel specification for its own interpreter,
preserves partial outputs on error, and records installed-source hashes. It does
not register a global kernel. A completed notebook with failed diagnostic gates
returns failure. This protects execution acceptance; it does not establish
population recovery, data licensing or the correctness of arbitrary custom code.

An optional JSON file passed with `--config` can override the displayed workflow
settings. `ranking_initvals` accepts native PyMC initial-value mappings, either
one mapping or one per chain. When explicit starts are supplied, this notebook
disables additional initialization jitter so the recorded starts are reproducible.
Initial values change computation, not the declared prior. They must match the
current model's variables and coordinates; inspect all chains and retain the
same diagnostic checks after changing them.
