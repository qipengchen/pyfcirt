"""Execute a tutorial with the installed package in a fresh, explicit kernel.

Run through scripts/dev.py to configure external caches. This is a heavier
workflow for the configured compute worker, not a default local test command.
Execution provenance is recorded; scientific diagnostics remain in the notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    import nbformat
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    import pyfcirt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        "analysis_accepted": None,
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
            # Async channel reads let NotebookClient enforce per-cell deadlines.
            manager = AsyncKernelManager(
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
            client.execute(cwd=str(output), cleanup_kc=True)
        manifest["execution_status"] = "completed"
    except Exception as error:
        manifest["execution_status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - start
        nbformat.write(notebook, output / f"{source.stem}.executed.ipynb")
        (output / "execution.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
