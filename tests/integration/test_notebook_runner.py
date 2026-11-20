"""Fresh-kernel execution and provenance without a private diagnostic protocol."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize(
    "outcome", ["completed", "legacy_passed", "legacy_failed", "cell_failed", "timed_out"]
)
def test_runner_selects_current_environment_and_preserves_failure(tmp_path, outcome):
    pytest.importorskip("nbclient", reason="Notebook execution requires the docs extra")
    pytest.importorskip("ipykernel", reason="Notebook execution requires the docs extra")
    import nbformat

    root = Path(__file__).resolve().parents[2]
    source = """
import hashlib
import json
import sys
from importlib.util import find_spec
from pathlib import Path

package_file = Path(find_spec("pyfcirt").origin)
Path("kernel-observed.json").write_text(json.dumps({
    "python": sys.executable, "cwd": str(Path.cwd()),
    "package_file": str(package_file),
    "package_sha256": hashlib.sha256(package_file.read_bytes()).hexdigest(),
}))
"""
    if outcome.startswith("legacy_"):
        # Old artifacts never turn execution success into scientific acceptance.
        source += (
            '\nPath("analysis-acceptance.json").write_text(json.dumps({"passed": '
            + ("True" if outcome == "legacy_passed" else "False")
            + "}))\n"
        )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    if outcome == "cell_failed":
        notebook.cells.append(
            nbformat.v4.new_code_cell('raise ValueError("intentional runner fixture failure")')
        )
    elif outcome == "timed_out":
        notebook.cells.append(nbformat.v4.new_code_cell("import time\ntime.sleep(10)"))
    source_path = tmp_path / "runner-fixture.ipynb"
    nbformat.write(notebook, source_path)
    output = tmp_path / "execution"
    env = os.environ.copy()
    for name in ("PYFCIRT_DATA_DIR", "PYFCIRT_RESULTS_DIR"):
        env.pop(name, None)
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "execute_tutorial.py"),
            "--notebook",
            str(source_path),
            "--output",
            str(output),
            "--cell-timeout",
            "2" if outcome == "timed_out" else "90",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    observed = json.loads((output / "kernel-observed.json").read_text())
    assert Path(observed["python"]).resolve() == Path(sys.executable).resolve()
    assert Path(observed["cwd"]).resolve() == output.resolve()
    manifest = json.loads((output / "execution.json").read_text())
    executed = nbformat.read(output / "runner-fixture.executed.ipynb", as_version=4)
    # The kernel resolves its installed package independently of pytest's checkout.
    assert manifest["installed_source_sha256"]["__init__.py"] == observed["package_sha256"]
    assert executed.cells[0].execution_count == 1
    assert manifest["analysis_accepted"] is None
    assert not (output / "mgppm_workflow.executed.ipynb").exists()
    if outcome in {"completed", "legacy_passed", "legacy_failed"}:
        assert process.returncode == 0, process.stderr
        assert manifest["execution_status"] == "completed"
        assert "error" not in manifest
        if outcome == "completed":
            assert not (output / "analysis-acceptance.json").exists()
    else:
        assert process.returncode != 0
        assert manifest["execution_status"] == "failed"
        if outcome == "timed_out":
            assert manifest["error"].startswith("CellTimeoutError:")
        else:
            assert any(out.output_type == "error" for out in executed.cells[1].outputs)
            assert "intentional runner fixture failure" in manifest["error"]


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).resolve().parents[2] / "docs/tutorials").glob("*.ipynb")),
    ids=lambda path: path.stem,
)
def test_public_notebook_has_parseable_source_and_consistent_outputs(path):
    """Validate current notebooks and retained outputs without rerunning their analyses."""
    transform = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    assert "execution" not in notebook["metadata"].get("pyfcirt_example", {})
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert cells
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        compile(transform.transform_cell(source), f"{path.stem}:cell-{index}", "exec")
        if cell["execution_count"] is None:
            assert cell["outputs"] == []
        else:
            assert isinstance(cell["execution_count"], int) and cell["execution_count"] > 0
            assert all(output["output_type"] != "error" for output in cell["outputs"])
