"""Lightweight CI checks: no scientific dependencies and no real pytest invocation."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ci", ROOT / "scripts/ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class CITests(unittest.TestCase):
    def test_selection_examples(self):
        cases = [
            ("README.md", [], False),
            (
                "pyfcirt/analysis/summary.py",
                ["tests/test_checks.py", "tests/test_summary.py"],
                False,
            ),
            ("tests/test_io.py", ["tests/test_io.py"], False),
            ("scripts/ci.py", [], False),
            (".github/workflows/docs.yml", [], False),
            ("tests/helpers.py", ["tests"], True),
            ("tests/conftest.py", ["tests"], True),
            ("pyfcirt/unknown.py", ["tests"], True),
            ("pyfcirt/common/posterior.py", ["tests"], True),
            ("pyfcirt/inference/conditional.py", ["tests"], True),
        ]
        for name, tests, fallback in cases:
            with self.subTest(name=name):
                plan = ci.select(ROOT, [("M", name)])
                self.assertEqual(plan["tests"], tests)
                self.assertEqual(bool(plan["fallback"]), fallback)
                self.assertFalse(plan["package"])

    def test_shared_data_includes_downstream_without_full(self):
        plan = ci.select(ROOT, [("M", "pyfcirt/data/prepare.py")])
        for name in (
            "tests/data/test_preparation.py",
            "tests/test_composition.py",
            "tests/test_scoring.py",
            "tests/test_io.py",
        ):
            self.assertIn(name, plan["tests"])
        self.assertFalse(plan["full"])
        self.assertEqual(len(plan["tests"]), len(set(plan["tests"])))

    def test_deleted_and_renamed_paths(self):
        data = b"R100\0old file.py\0new\nfile.py\0D\0removed.py\0"
        with patch.object(ci, "git", return_value=data):
            self.assertEqual(
                ci.changed_files(ROOT, "base", "head"),
                [("R-old", "old file.py"), ("R100", "new\nfile.py"), ("D", "removed.py")],
            )
        plan = ci.select(ROOT, [("R-old", "tests/test_old.py"), ("R100", "tests/test_io.py")])
        self.assertEqual(plan["tests"], ["tests/test_io.py"])
        self.assertTrue(ci.select(ROOT, [("D", "tests/test_old.py")])["full"])
        self.assertEqual(
            ci.select(ROOT, [("D", "pyfcirt/analysis/summary.py")])["tests"],
            ["tests/test_checks.py", "tests/test_summary.py"],
        )

    def test_document_removal_rechecks_notebook_links_without_scientific_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/example.ipynb").write_text("{}")
            plan = ci.select(root, [("D", "docs/removed.md")])
            self.assertIn("docs/example.ipynb", plan["docs"])
            self.assertNotIn("README.md", plan["docs"])
            self.assertFalse(plan["tests"] or plan["package"])

    def test_full_and_packaging_controls(self):
        plan = ci.select(ROOT, [], full=True)
        self.assertEqual(plan["tests"], ["tests"])
        self.assertTrue(plan["package"])
        for name in ("pyproject.toml", "pyfcirt/py.typed", "pyfcirt/_example_data/new.json"):
            self.assertTrue(ci.select(ROOT, [("M", name)])["package"])

    def test_baseline_uses_complete_pr_range_or_push_before(self):
        with patch.object(ci, "git", side_effect=[b"base\n", b"common\n"]) as git:
            event = {"before": "previous-pr-head", "pull_request": {"base": {"sha": "base-sha"}}}
            self.assertEqual(ci.baseline(ROOT, event, "head"), "common")
            self.assertEqual(git.call_args_list[0].args[-1], "base-sha^{commit}")
            self.assertEqual(git.call_args.args[1:], ("merge-base", "base", "head"))
        with patch.object(ci, "git", return_value=b"before\n"):
            self.assertEqual(ci.baseline(ROOT, {"before": "old-sha"}, "head"), "before")
        for event in ({}, {"before": "0" * 40}):
            with self.assertRaises(ValueError):
                ci.baseline(ROOT, event, "head")

    def test_real_git_range_keeps_changes_from_multiple_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ci.git(root, "init", "-q")
            ci.git(root, "config", "user.email", "ci@example.invalid")
            ci.git(root, "config", "user.name", "CI test")
            ci.git(root, "config", "commit.gpgsign", "false")
            (root / "first.py").write_text("x = 1\n")
            ci.git(root, "add", ".")
            ci.git(root, "commit", "-qm", "base")
            base = ci.git(root, "rev-parse", "HEAD").decode().strip()
            (root / "first.py").write_text("x = 2\n")
            ci.git(root, "commit", "-qam", "first")
            (root / "second.py").write_text("y = 1\n")
            ci.git(root, "add", ".")
            ci.git(root, "commit", "-qm", "second")
            self.assertEqual(
                {name for _, name in ci.changed_files(root, base, "HEAD")},
                {"first.py", "second.py"},
            )
            # Match checkout's depth-2 PR merge: both parents remain available.
            ci.git(root, "branch", "feature")
            ci.git(root, "checkout", "-qb", "pr-merge", base)
            ci.git(root, "merge", "--no-ff", "feature", "-qm", "PR merge")
            shallow = root / "shallow"
            ci.git(root, "clone", "--quiet", "--depth", "2", root.as_uri(), str(shallow))
            event = {"pull_request": {"base": {"sha": base}}}
            self.assertEqual(ci.baseline(shallow, event, "HEAD"), base)
            self.assertEqual(
                {name for _, name in ci.changed_files(shallow, base, "HEAD")},
                {"first.py", "second.py"},
            )

    def test_cli_reports_missing_baseline_as_full_without_running_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            output = Path(directory) / "outputs"
            with (
                patch.object(ci, "git", return_value=b"head\n"),
                patch.object(sys, "argv", ["ci.py", "--plan", str(plan)]),
                patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}, clear=True),
                patch("builtins.print"),
            ):
                self.assertEqual(ci.main(), 0)
            result = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(result["tests"], ["tests"])
            self.assertTrue(result["fallback"])
            self.assertIn("tests=true", output.read_text(encoding="utf-8"))
            self.assertIn("full=true", output.read_text(encoding="utf-8"))

    def test_workflow_keeps_native_commands_and_no_mcmc(self):
        text = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertEqual(text.count("branches: [main]"), 2)
        for required in (
            "contents: read",
            "cancel-in-progress: true",
            "default: false",
            "env.FULL_CHECKS == 'true'",
            "needs.plan.outputs.package == 'true'",
            "'--dist', 'loadfile'",
            "cache-dependency-path: pyproject.toml",
            "github.event_name == 'push'",
            "--cov-branch",
            "not slow and not reference and not recovery",
        ):
            self.assertIn(required, text)
        self.assertNotIn("pull_request_target", text)
        self.assertNotIn("--run-recovery", text)
        self.assertNotIn("sphinx -", text)
        docs = (ROOT / ".github/workflows/docs.yml").read_text()
        self.assertIn("nb_execution_mode=off", docs)


if __name__ == "__main__":
    unittest.main()
