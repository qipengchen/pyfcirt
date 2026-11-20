# Testing PyFCIRT

Ordinary pytest must not run posterior MCMC. Use deterministic inputs, independent
mathematical references, sampler mocks and reusable saved draws. Calibration and
new-person scoring sampling belong to explicit manual investigations or recovery
conditions enabled with `--run-recovery`. A collected condition is not a completed
run, and a skipped check is not evidence of agreement.

## Routine checks

Run the tests for the code you changed. The CI selector uses an explicit ownership
map in `scripts/ci.py`, including downstream scoring, composition and persistence
checks for shared data/model changes. It prints every changed path and its selection
reason, and retains `ci-plan.json` with the test report. To inspect a selection locally:

```console
python scripts/ci.py --base main --plan ../ci-plan.json
python -m pytest tests/test_summary.py -m "not slow and not reference and not recovery"
python -m unittest discover -s .github/tests -v
```

The selector compares the complete PR against its merge-base. A push to `main`
compares the event's before/head revisions, including deletions and renames. A new
or unavailable baseline, an unmapped path, a deleted test or a shared test helper
change selects the complete ordinary suite and prints the fallback reason. It never
silently skips unknown code. Provide `base` for a manual affected run; otherwise it
also falls back to the ordinary suite. No fallback enables long MCMC.

README and documentation changes check Python example syntax and local link targets
without scientific tests. Notebook magics are parsed using IPython, never executed.
Changes to Sphinx configuration, landing-page structure or templates also build the
documentation without executing notebooks. Packaging and isolated installation run
only for build metadata, bundled resources, the package entry point, or explicit full
checks. Formatting checks target changed Python/notebook files.

Actions run once per PR targeting `main`, and once for each push to `main`; work-branch
pushes do not launch a second copy. Superseded runs for a PR/branch are cancelled.
Manual runs have distinct concurrency groups so an intentional full run is preserved.

To request the complete ordinary suite directly:

```console
python -m pytest -m "not slow and not reference and not recovery"
```

Optional investigations have separate commands; they are not part of affected CI:

```console
python -m pytest -m reference
python -m pytest -m slow
```

Use the installed development dependencies and inspect skips and warnings.
Reference checks may require separately configured reference executables or
frozen outputs; saved-result checks require compatible current-format fixtures.
Keep routine fixtures small and single-process. Posterior sampling must remain
mocked or guarded in ordinary tests, including optional sampler entry points.

Tests should follow responsibilities:

- Data tests cover normalization, typed labels, complete category support,
  missingness, canonical coordinates and identification constraints.
- Model tests independently check fixed-parameter probabilities, complete logp,
  gradients, parameter domains, tail behavior and valid prior overrides.
- Analysis tests use supplied draws for prediction, PPP, information, reliability,
  likelihood and fixed-calibration scoring. Distinguish response units explicitly.
- Persistence tests check current-format save/load and continued analysis, preserving
  coordinates, statistical meaning and existing outputs without calibration MCMC.
- Public API and notebook checks verify documented calls and executable examples.
  Clear old outputs when an example changes; execution metadata must describe the
  same source and inputs as the displayed results.

Expected values must come from independent equations, numerical integration or
versioned external reference outputs. Do not generate a test's expected answer
with the implementation under test. Keep independent scalar references even when
production calculations are vectorized. Preserve category direction, parameter
mapping, observation unit, source versions, checksums and numerical tolerances.
For Gaussian ranking models, distinguish the augmented fitting target from the
marginal observed-event likelihood and predictive generator.

## Numerical agreement and statistical evidence

Check probabilities, log likelihoods, derivatives and information at matched
parameter values before interpreting recovery. Posterior comparisons additionally
require the same likelihood, priors, identification, transformations and data;
compare uncertainty and convergence, not only point estimates. Agreement at
ordinary parameter values does not establish agreement in extreme tails.

A numerical check, a successful sampler call and adequate inference are separate
findings. Record divergences, R-hat, ESS and Monte Carlo error for manual fits.
Prior prediction checks response plausibility; it does not demonstrate recovery.
Information criteria require the same observed events and predictive target.
A product of comparison probabilities is not a normalized joint ranking law.

## Optional recovery

```console
python -m pytest tests/recovery -m recovery --run-recovery
```

This command can launch long sampling jobs. Select a specific intended condition
rather than running the whole grid by default. Specify the population, seeds,
priors, identification, budgets and criteria before execution. Use bounded jobs
and retain source/input checksums, environments, truth, posterior results and
complete diagnostic reports. Preserve failures; do not rerun seeds or relax
criteria after seeing results. See [recovery investigations](recovery.md) for
interpretation and limitations.

## Installation and coverage

Use a separate environment for distribution checks; test installed imports,
resources, documented calls and current-format persistence outside the checkout.
Optional JAX compilation and device checks belong to the relevant backend profile.
They do not establish posterior convergence. See [installation](installation.md).

Report line and branch coverage with test results, skips and source checksums.
The manual **Checks** workflow's `full` input defaults to false. Only an explicit
`full=true` enables full-suite coverage and Codecov upload, in addition to the ordinary
tests. It requires at least 90% overall branch coverage from that full suite;
the gate uses covered/valid branches, not combined statement-and-branch coverage.
Affected runs and automatic full-suite fallbacks retain test reports without the
coverage gate or Codecov upload. A local full selection can be requested with
`python scripts/ci.py --full --plan ../ci-plan.json`; inspect it before execution.
Keep expensive compilation, external references and full worked examples in
bounded batches. Merge coverage only for identical source snapshots; measure
process memory before claiming a supported runner budget. No posterior MCMC
belongs in ordinary CI or ordinary coverage runs.
