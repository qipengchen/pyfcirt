"""Run a development command with data, results and caches in explicit storage.

Example: <venv-python> scripts/dev.py --storage-root <NAS-directory> -- python -m pytest
Only the child process is configured; the package does not depend on this helper.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    storage = args.storage_root.resolve()
    flags = [
        flag
        for flag in os.environ.get("PYTENSOR_FLAGS", "").split(",")
        if flag and not flag.strip().startswith("base_compiledir=")
    ]
    # PyMC initialization still uses legacy Python/C linking internally. A C
    # compiler discovered on Windows may not support UNC build paths. Numba is
    # the supported production backend here; disable only the optional C path.
    if sys.platform == "win32" and str(storage).startswith("\\\\"):
        if not any(flag.strip().startswith("cxx=") for flag in flags):
            flags.append("cxx=")
    installed = {d.metadata["Name"].lower(): d.version for d in metadata.distributions()}
    settings = {
        "environment": sys.prefix,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "pytensor_flags": flags,
        "versions": {
            n: installed.get(n)
            for n in ("numpy", "pytensor", "numba", "llvmlite", "jax", "jaxlib", "numpyro")
        },
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    tag = f"{platform.node()}-{sys.platform}-py{sys.version_info.major}{sys.version_info.minor}"
    cache = storage / "cache" / f"{tag}-{digest}"
    paths = {
        "PYFCIRT_DATA_DIR": storage / "data",
        "PYFCIRT_RESULTS_DIR": storage / "results",
        "UV_CACHE_DIR": storage / "cache" / "downloads" / "uv",
        "PIP_CACHE_DIR": storage / "cache" / "downloads" / "pip",
        "NUMBA_CACHE_DIR": cache / "numba",
        "RUFF_CACHE_DIR": cache / "ruff",
        "JAX_COMPILATION_CACHE_DIR": cache / "jax",
        "XDG_CACHE_HOME": cache / "xdg",
        "MPLCONFIGDIR": cache / "matplotlib",
        "IPYTHONDIR": cache / "ipython",
        "JUPYTER_RUNTIME_DIR": cache / "jupyter" / "runtime",
        "JUPYTER_DATA_DIR": cache / "jupyter" / "data",
        "JUPYTER_CONFIG_DIR": cache / "jupyter" / "config",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
        "TEMP": cache / "tmp",
        "TMP": cache / "tmp",
        "TMPDIR": cache / "tmp",
    }
    env = os.environ.copy()
    env.setdefault("UV_LINK_MODE", "copy")
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    test_cache = cache / "pytest"
    test_cache.mkdir(parents=True, exist_ok=True)
    # Pytest parses this variable with POSIX shlex on Windows as well as Linux.
    env["PYTEST_ADDOPTS"] = " ".join(
        filter(
            None,
            [
                env.get("PYTEST_ADDOPTS", ""),
                "-o",
                shlex.quote(f"cache_dir={test_cache.as_posix()}"),
            ],
        )
    )
    compiledir = cache / "pytensor"
    compiledir.mkdir(parents=True, exist_ok=True)
    # PyTensor parses flags with shell-style escaping: use forward slashes so
    # Windows/UNC backslashes are not consumed as escape characters.
    env["PYTENSOR_FLAGS"] = ",".join([*flags, f"base_compiledir={compiledir.as_posix()}"])
    env["VIRTUAL_ENV"] = sys.prefix
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if command[0] in {"python", "python3"}:
        command[0] = sys.executable
        if "-I" in command[1:]:
            # Isolated Python ignores PYTHONPYCACHEPREFIX from the environment.
            command[1:1] = ["-X", f"pycache_prefix={cache / 'pycache'}"]
    print(f"Interpreter: {sys.executable}\nStorage: {storage}\nCache: {cache}", flush=True)
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
