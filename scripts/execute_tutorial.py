"""Execute the complete installed-package tutorial in a fresh, explicit kernel.

Run through scripts/dev.py to configure external caches. This is a heavier
workflow for the configured compute worker, not a default local test command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    import pyfcirt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cell-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.cell_timeout < 1:
        parser.error("--cell-timeout must be positive")
    source = args.notebook.resolve()
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ("PYFCIRT_DATA_DIR", "PYFCIRT_RESULTS_DIR"):
        if name not in env:
            raise ValueError(f"Set {name} before executing the workflow")
    env["PYFCIRT_WORKFLOW_DIR"] = str(output)
    if args.config:
        env["PYFCIRT_WORKFLOW_CONFIG"] = str(args.config.resolve())
    package = Path(pyfcirt.__file__).parent
    manifest = {
        "notebook_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "installed_source_sha256": {
            p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*.py"))
        },
        "python": sys.executable,
        "cell_timeout_seconds": args.cell_timeout,
        "execution_status": "started",
    }
    start = time.perf_counter()
    try:
        # This temporary spec selects this interpreter without global registration.
        with TemporaryDirectory(prefix="kernel-spec-", dir=output) as temporary:
            specs = Path(temporary)
            spec = specs / "python3"
            spec.mkdir()
            (spec / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "PyFCIRT execution environment",
                        "language": "python",
                    }
                ),
                encoding="utf8",
            )
            manager = KernelManager(
                kernel_name="python3",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(specs)]),
            )
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=args.cell_timeout,
                allow_errors=False,
                record_timing=True,
            )
            client.execute(env=env, cwd=str(output), cleanup_kc=True)
        manifest["execution_status"] = "completed"
        gates = json.loads((output / "analysis-acceptance.json").read_text())
        manifest["analysis_accepted"] = gates["passed"]
        if not gates["passed"]:
            raise RuntimeError(
                "Workflow executed but diagnostic acceptance failed; inspect saved gates"
            )
    except Exception as error:
        if manifest["execution_status"] != "completed":
            manifest["execution_status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - start
        nbformat.write(notebook, output / "mgppm_workflow.executed.ipynb")
        (output / "execution.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
