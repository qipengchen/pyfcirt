# Development

Use Python 3.12 or newer in an isolated environment. Keep the interpreter and
environment on a local disk. Data, results, package downloads and compilation
caches can live on a NAS; keep compiled caches separate for each machine,
operating system and dependency environment.

From an activated environment, install the project and the required development
tools:

```shell
python -m pip install -e ".[test,lint,release,io]"
```

`pytest` is the only test framework dependency. Plotting, documentation, IO and
NumPyro are separate extras. R2jags and ArviZ 0.22.0 belong to isolated reference
environments and are not dependencies of the package.

Configure storage for each command with the environment's interpreter. For example:

```powershell
& 'D:\CodexEnvs\pyfcirt\Scripts\python.exe' scripts/dev.py --storage-root 'Z:\PyFCIRT' -- python -m pytest -m 'not slow and not reference'
```

This configures only the child process. The public package does not read these
paths during import. Re-run the setup after changing Python or environment
versions; the launcher separates compiled caches by host, interpreter and relevant
dependency versions. Do not share a Windows environment or compiled cache with Linux.

The launcher also routes pytest, IPython and Jupyter caches and runtime files to
that external cache directory. Pytest configuration is passed to the child
process through `PYTEST_ADDOPTS`; explicit pytest command-line options retain
their normal precedence.

For Windows with UNC storage, the launcher disables PyTensor's optional legacy
C compiler unless `cxx` was explicitly set by the caller. This avoids a UNC
library-path failure during PyMC's internal initial-point construction. PyMC's
default Numba execution remains available. The setting affects only commands
started through the developer launcher; the package does not change global
PyTensor configuration. PyTensor cache paths use forward slashes to preserve
UNC paths through its configuration parser.

Run checks according to the change:

```shell
python -m ruff check .
python -m ruff format --check .
python -m pytest -m "not slow and not reference"
python -m pytest -m reference
```

Formal tests never run posterior MCMC. Sampler forwarding uses mocks; result
checks reuse fixed arrays or supplied saved traces. The optional
`scripts/check_mgppm_sampling.py` is a manual execution check outside pytest and
CI. Use existing sampling evidence rather than repeat a fit for every downstream
change. Full empirical notebook execution and multi-seed recovery are optional
manual investigations, not v0.1 acceptance gates.

## Test design

Follow Bambi's practical testing patterns: share data and model fixtures, test model
construction without MCMC, mock sampler calls when checking argument forwarding,
and reuse representative fitted results for prediction, summaries and persistence.
Use pytest fixtures and monkeypatch or standard-library mocks; do not import PyMC's
internal testing helpers or introduce another test framework.

Keep ordinary CI on the fast command above. Run manual fits only when useful; configured GPU workers are preferred for actual fits where
available, with CPU compatibility retained. An existing saved fit can validate
unchanged downstream analysis without repeating calibration.

Keep fixtures local to a module unless several modules need the same stable object.
Put shared fixtures in conftest.py and ordinary data builders in helper modules when
reuse justifies it. Tests that mutate draws or a native model must copy or rebuild
that object; widening fixture scope must not introduce order-dependent failures.

Parameterize meaningful task, block-size and encoding differences. Avoid repeating
the same contract test for every future model, reproducing dependency test suites,
or testing implementation details without a public behavior or statistical invariant.
Retain independent FC likelihood, normalization, gradient and encoding checks:
these cover mathematics implemented here. Recovery is optional manual research validation, outside the formal test suite;
no simulation grid is required.

Reference: [Bambi tests](https://github.com/bambinos/bambi/tree/main/tests).
Adopt its testing patterns rather than matching file names or test counts.

Build distributions and check them before a release:

```shell
python -m build
python -m twine check dist/*
```

Install the wheel in a fresh environment and import it outside the repository.
Check that `py.typed` and required resources are included. Follow
[the development roadmap](docs/roadmap.md) and [the testing specification](docs/testing.md) for the
model-specific acceptance gates. Publishing and pushing releases are separate
actions.

## Documentation

Install the `docs` extra. Sphinx uses MyST-NB for Markdown and notebooks and
autodoc for installed-package docstrings. Keep conceptual guides and runnable
examples alongside the generated API reference.

```shell
python -m sphinx -W --keep-going -b html docs /path/to/results/docs
```

The default documentation build does not execute notebooks. For an optional manual end-to-end check, execute the complete tutorial from a
fresh kernel on a compute worker using `scripts/execute_tutorial.py`, as
described in the quickstart. Full empirical MCMC is not a v0.1 acceptance gate.
Alternatively, execute through Sphinx with `-D nb_execution_mode=force`.
Choose one execution route; do not repeat MCMC merely to render documentation.
Render the resulting executed notebook with execution disabled when reusing that
run. Keep executed notebooks, temporary files and build output in the configured
external storage directory. A nonexecuting build alone is not workflow acceptance.