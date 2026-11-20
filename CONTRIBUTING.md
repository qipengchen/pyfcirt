# Development

Use Python 3.12 or newer in an isolated environment. Keep the interpreter and
environment on a local disk. Data, results, package downloads and compilation
caches can live on a NAS; keep compiled caches separate for each machine,
operating system and dependency environment.

From an activated environment, install the project and the required development
tools:

```shell
python -m pip install -e ".[test,lint,release,io,plots,covariates]"
```

`pytest` is the only test framework dependency. Plotting, documentation, IO and
NumPyro have separate extras; BlackJAX is a core dependency. R2jags and ArviZ 0.22.0 belong to isolated reference
environments and are not dependencies of the package.

Configure storage for each command with the environment's interpreter. For example:

```powershell
python scripts/dev.py --storage-root 'Z:\PyFCIRT' -- python -m pytest -m 'not slow and not reference and not recovery'
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
python -m pytest -m "not slow and not reference and not recovery"
python -m pytest -m reference
```

Ordinary tests never run posterior MCMC. Sampler forwarding uses mocks; result
checks reuse fixed arrays or supplied saved traces. Long recovery runs require
the recovery marker and explicit `--run-recovery` opt-in, outside ordinary CI.
Use existing sampling evidence rather than repeat a fit for every downstream
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
these cover mathematics implemented here. Long recovery is excluded from ordinary
tests and CI. Use a bounded manual script or the explicit `--run-recovery` opt-in
with the intended condition selector; a full simulation grid is not required.

Reference: [Bambi tests](https://github.com/bambinos/bambi/tree/main/tests).
Adopt its testing patterns rather than matching file names or test counts.

Build distributions and check them before a release:

```shell
python -m build
python -m twine check dist/*
```

Install both wheel and source archive in fresh environments and import outside
the repository. The existing `scripts/install_env.py` automates dependency,
model/prior-prediction and native xarray save/open checks; see
[installation](docs/installation.md).
Its device check does not run posterior MCMC or certify sampler convergence.
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

The default documentation build does not execute notebooks. To execute a tutorial
in a fresh kernel, use `scripts/execute_tutorial.py --notebook <path> --output <new-directory>`.
The runner uses the native asynchronous kernel manager with the current
interpreter and inherited environment, writes
`<source-name>.executed.ipynb` and records hashes, timing and execution status in
`execution.json`, including when a cell fails or times out. It does not consume
`analysis-acceptance.json` or inject workflow-specific environment variables.
`analysis_accepted` remains null: assess convergence and model checks from the
notebook's results. A completed execution alone is not scientific acceptance.
Full empirical MCMC is not a v0.1 acceptance gate.
Alternatively, execute through Sphinx with `-D nb_execution_mode=force`.
Choose one execution route; do not repeat MCMC merely to render documentation.
Render the resulting executed notebook with execution disabled when reusing that
run. Keep executed notebooks, temporary files and build output in the configured
external storage directory. A nonexecuting build alone is not workflow acceptance.
