"""Fresh-kernel selection, error retention and diagnostic gating of the docs runner."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("nbclient", reason="Notebook execution requires the docs extra")
pytest.importorskip("ipykernel", reason="Notebook execution requires the docs extra")
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("outcome", ["accepted", "diagnostics_failed", "cell_failed"])
def test_runner_selects_current_environment_and_preserves_failure(tmp_path, outcome):
    import nbformat

    root = Path(__file__).resolve().parents[2]
    source = """
import json
import os
import sys
from pathlib import Path
import pyfcirt

Path("kernel-observed.json").write_text(json.dumps({
    "python": sys.executable, "cwd": str(Path.cwd()),
    "configured_output": os.environ["PYFCIRT_WORKFLOW_DIR"],
    "package_version": pyfcirt.__version__,
}))
"""
    if outcome == "cell_failed":
        source += '\nraise ValueError("intentional runner fixture failure")\n'
    else:
        source += (
            '\nPath("analysis-acceptance.json").write_text(json.dumps({"passed": '
            + ("True" if outcome == "accepted" else "False")
            + "}))\n"
        )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    source_path = tmp_path / "runner-fixture.ipynb"
    nbformat.write(notebook, source_path)
    output = tmp_path / "execution"
    env = os.environ.copy()
    env.update(PYFCIRT_DATA_DIR=str(tmp_path / "unused-data"), PYFCIRT_RESULTS_DIR=str(tmp_path))
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "execute_tutorial.py"),
            "--notebook",
            str(source_path),
            "--output",
            str(output),
            "--cell-timeout",
            "90",
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
    assert observed["configured_output"] == str(output.resolve())
    manifest = json.loads((output / "execution.json").read_text())
    executed = nbformat.read(output / "mgppm_workflow.executed.ipynb", as_version=4)
    assert manifest["installed_source_sha256"]["scoring.py"]
    assert executed.cells[0].execution_count == 1
    if outcome == "accepted":
        assert process.returncode == 0, process.stderr
        assert manifest["execution_status"] == "completed"
        assert manifest["analysis_accepted"] is True
    elif outcome == "diagnostics_failed":
        assert process.returncode != 0
        assert manifest["execution_status"] == "completed"
        assert manifest["analysis_accepted"] is False
        assert "diagnostic acceptance failed" in manifest["error"]
    else:
        assert process.returncode != 0
        assert manifest["execution_status"] == "failed"
        assert any(out.output_type == "error" for out in executed.cells[0].outputs)
        assert "intentional runner fixture failure" in manifest["error"]
