"""Write an affected-check plan, falling back to all ordinary tests for unknown scope."""

import argparse
import json
import os
import subprocess
from pathlib import Path

DATA = "tests/data/test_*.py"
DOWNSTREAM = "tests/test_composition.py tests/test_scoring.py tests/test_io.py"
PREDICTION = "tests/test_predictive.py tests/test_checks.py"
INFORMATION = "tests/test_information*.py tests/test_reliability.py"
MODEL = "tests/models/test_*.py tests/test_distributions.py tests/test_model.py"
REGRESSION = (
    f"tests/data/test_covariates.py tests/test_summary.py tests/test_priors.py {DOWNSTREAM}"
)

# Files with narrower or wider consumers override their directory's ordinary scope.
RULES = {
    "pyfcirt/data/examples.py": "tests/test_datasets.py tests/test_simulation.py",
    "pyfcirt/data/grouping.py": f"{REGRESSION} tests/models/test_population.py {PREDICTION}",
    "pyfcirt/data/regressions.py": f"{REGRESSION} tests/models/test_population.py {PREDICTION}",
    "pyfcirt/config.py": f"{MODEL} {REGRESSION}",
    "pyfcirt/analysis/summary.py": "tests/test_summary.py tests/test_checks.py",
    "pyfcirt/analysis/traits.py": "tests/test_summary.py",
    "pyfcirt/analysis/prediction.py": (
        f"{PREDICTION} {INFORMATION} {DOWNSTREAM} tests/test_model_comparison.py"
    ),
    "pyfcirt/analysis/checks.py": "tests/test_checks.py tests/test_plotting.py",
    "pyfcirt/analysis/information.py": INFORMATION,
    "pyfcirt/analysis/reliability.py": "tests/test_reliability.py",
    "pyfcirt/analysis/comparison.py": "tests/test_model_comparison.py",
    "pyfcirt/analysis/__init__.py": (
        f"{PREDICTION} {INFORMATION} tests/test_summary.py tests/test_model_comparison.py"
    ),
    "pyfcirt/models/priors.py": f"{MODEL} {REGRESSION} {PREDICTION}",
    "pyfcirt/models/population.py": f"{MODEL} {REGRESSION} {PREDICTION}",
    "pyfcirt/models/thurstonian/gtum.py": (
        f"{MODEL} {DOWNSTREAM} {PREDICTION} {INFORMATION} tests/test_gtum_*.py"
    ),
    "pyfcirt/analysis/output.py": f"{DOWNSTREAM} {PREDICTION} tests/test_model.py",
    "pyfcirt/model.py": f"{MODEL} {DOWNSTREAM} {PREDICTION} tests/test_summary.py",
    "pyfcirt/__init__.py": f"{MODEL} {DOWNSTREAM} {PREDICTION} tests/test_summary.py",
    "pyfcirt/simulation.py": "tests/test_simulation.py tests/test_datasets.py",
    "pyfcirt/plotting.py": "tests/test_plotting.py",
    "scripts/install_env.py": "tests/test_install_env.py",
    "scripts/execute_tutorial.py": "tests/integration/test_notebook_runner.py",
    "pyproject.toml": "tests/test_install_env.py tests/test_model.py tests/test_io.py",
}
FOLDERS = {
    "pyfcirt/data": (
        f"{DATA} {MODEL} {DOWNSTREAM} {PREDICTION} tests/test_priors.py tests/test_summary.py"
    ),
    "pyfcirt/models": f"{MODEL} {DOWNSTREAM} {PREDICTION} {INFORMATION} tests/test_priors.py",
    "pyfcirt/models/thurstonian": f"{MODEL} {DOWNSTREAM} {PREDICTION} {INFORMATION}",
    "pyfcirt/covariates": f"{REGRESSION} tests/models/test_population.py {PREDICTION}",
    "pyfcirt/scoring": "tests/test_scoring.py tests/test_io.py",
}
CI_FILES = {
    ".github/workflows/ci.yml",
    ".github/workflows/docs.yml",
    "scripts/ci.py",
    "scripts/check_docs.py",
    ".github/tests/test_ci.py",
}


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + root.as_posix(), *args], cwd=root
    )


def changed_files(root, base, head):
    """Use NUL delimiters so filenames never become shell expressions."""
    fields = iter(
        git(root, "diff", "--name-status", "-z", "--find-renames", base, head).split(b"\0")
    )
    changes = []
    for status in fields:
        if not status:
            break
        name = next(fields)
        if status.startswith(b"R"):
            changes.append(("R-old", os.fsdecode(name)))
            name = next(fields)
        changes.append((status.decode(), os.fsdecode(name)))
    return changes


def baseline(root, event, head, explicit=None):
    pull_base = event.get("pull_request", {}).get("base", {}).get("sha")
    value = explicit or pull_base or event.get("before")
    if not value or set(value) == {"0"}:
        raise ValueError("No reliable baseline; use the full ordinary suite")
    base = (
        git(root, "rev-parse", "--verify", "--end-of-options", value + "^{commit}").decode().strip()
    )
    return git(root, "merge-base", base, head).decode().strip() if "pull_request" in event else base


def select(root, changes, *, full=False):
    plan = dict(
        full=full,
        tests=[],
        lint=[],
        docs=[],
        package=full,
        reasons=[],
        fallback=[],
    )
    tests, lint, docs = set(), set(), set()
    for status, name in changes:
        path = root / name
        patterns = RULES.get(name) or FOLDERS.get(Path(name).parent.as_posix())
        if path.is_file() and path.suffix in {".py", ".ipynb"}:
            lint.add(name)
        if name in CI_FILES:
            reason = "CI self-checks only"
        elif name.startswith("tests/"):
            if path.name.startswith("test_") and path.suffix == ".py":
                if path.is_file():
                    tests.add(name)
                elif status != "R-old":
                    plan["fallback"].append(f"Deleted test {name}")
                reason = "changed test (renames select the destination)"
            else:
                plan["fallback"].append(f"Shared test helper {name}")
                reason = "shared helper affects multiple modules"
        elif patterns or name.startswith("pyfcirt/_example_data/"):
            patterns = (
                patterns or "tests/test_datasets.py tests/test_simulation.py tests/test_io.py"
            )
            for pattern in patterns.split():
                matches = list(root.glob(pattern))
                if not matches:
                    plan["fallback"].append(f"Missing test mapping {pattern} for {name}")
                tests.update(p.relative_to(root).as_posix() for p in matches if p.is_file())
            reason = "declared owner and downstream tests"
        elif name.startswith("docs/") or name == "README.md":
            reason = "documentation syntax and local links only"
            if path.is_file() and path.suffix in {".md", ".ipynb"}:
                docs.add(name)
            if status.startswith(("D", "R")):
                docs.update(
                    p.relative_to(root).as_posix()
                    for pattern in ("*.md", "*.ipynb")
                    for p in (root / "docs").rglob(pattern)
                )
                if (root / "README.md").is_file():
                    docs.add("README.md")
        elif name in {"LICENSE", "NOTICE", "pyfcirt/py.typed", ".gitignore"} or name.startswith(
            "LICENSES/"
        ):
            reason = "repository metadata"
        else:
            plan["fallback"].append(f"Unmapped path {name}")
            reason = "unknown impact"
        plan["package"] |= name in {
            "pyproject.toml",
            "pyfcirt/__init__.py",
            "pyfcirt/py.typed",
            "LICENSE",
            "NOTICE",
        } or name.startswith(("pyfcirt/_example_data/", "LICENSES/"))
        plan["reasons"].append(f"{status} {name}: {reason}")
    plan["full"] = full or bool(plan["fallback"])
    plan.update(
        tests=["tests"] if plan["full"] else sorted(tests),
        lint=["."] if full else sorted(lint),
        docs=sorted(docs),
    )
    if plan["full"]:
        plan["reasons"].append("Full ordinary tests; long MCMC remains excluded")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    full = args.full or os.environ.get("FULL_CHECKS") == "true"
    try:
        head = git(root, "rev-parse", "HEAD").decode().strip()
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        event = json.loads(Path(event_path).read_text(encoding="utf-8")) if event_path else {}
        base = (
            None if full else baseline(root, event, head, args.base or os.environ.get("CHECK_BASE"))
        )
        plan = select(root, [] if full else changed_files(root, base, head), full=full)
        plan.update(base=base, head=head)
    except (ValueError, subprocess.CalledProcessError) as error:
        plan = select(root, [], full=False)
        plan.update(full=True, tests=["tests"], fallback=[str(error)])
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as stream:
            for key in ("full", "tests", "lint", "docs", "package"):
                stream.write(f"{key}={str(bool(plan[key])).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
