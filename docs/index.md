# PyFCIRT

Bayesian forced-choice IRT models using PyMC, PyTensor and ArviZ. MGPPM is the
first implemented family: ranking, PICK/MOLE and five-category graded pairs.
The package retains native labelled arrays and exposes its PyMC graph.

Start with the data and model guides. Current implementation checks do not
establish convergence or recovery. Full empirical tutorial execution and recovery
are optional manual investigations, outside formal tests.

```{toctree}
:maxdepth: 2
:caption: Using PyFCIRT

data
quickstart
model
simulation
checks
comparison
scoring
io
api
tutorials/mgppm_workflow
```

```{toctree}
:maxdepth: 1
:caption: Design and development

roadmap
models/mgppm
architecture
requirements
data_format
graded_response
block_models
core_api
parameter_models
model_comparison
testing
recovery
tutorial_migration
```
