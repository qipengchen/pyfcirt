"""Create and verify an isolated PyFCIRT environment from a local source artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

SMOKE = r"""
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import xarray as xr
import pyfcirt as fc

backend, output = sys.argv[1], Path(sys.argv[2])
package = Path(fc.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve()), 'Import escaped the new environment'
venv_config = (Path(sys.prefix) / 'pyvenv.cfg').read_text(encoding='utf-8').lower()
assert 'include-system-site-packages = false' in venv_config, 'System packages are enabled'
data = fc.ItemInput(
    design=pd.DataFrame({'item_id': ['a', 'b'], 'trait': ['ability', 'ability']}),
    responses=fc.ResponseTable(pd.DataFrame({'person_id': [1, 1, 2, 2],
        'item_id': ['a', 'b', 'a', 'b'], 'response': [0, 1, 1, 0]}), format='item_long'),
    model='DichotomousIRT',
)
model = fc.Model(fc.PreparedInputs.from_inputs(data))
native = model.build()
point = native.initial_point()
assert np.isfinite(native.compile_logp()(point))
assert np.isfinite(native.compile_dlogp()(point)).all()
forward = fc.prior_predictive(model, draws=2, random_seed=719, store=False)
forward.to_netcdf(output / 'prior.nc', engine='h5netcdf')
with xr.open_datatree(output / 'prior.nc', engine='h5netcdf') as restored:
    xr.testing.assert_equal(restored, forward)

devices = []
default_backend = None
packages = ['pyfcirt', 'pymc', 'pytensor', 'numpy', 'scipy', 'xarray', 'h5netcdf']
if backend != 'pymc':
    import blackjax
    import jax
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_logp
    expected = 'gpu' if 'cuda' in backend else 'cpu'
    default_backend = jax.default_backend()
    assert default_backend == expected, 'Requested default JAX backend unavailable'
    devices = [{'platform': d.platform, 'kind': d.device_kind} for d in jax.devices()]
    assert devices and all(d['platform'] == expected for d in devices), (
        'Requested device unavailable')
    target = jax.jit(jax.value_and_grad(get_jaxified_logp(native)))
    value, gradient = target([jnp.asarray(point[v.name]) for v in native.value_vars])
    jax.block_until_ready((value, gradient))
    assert np.isfinite(value) and all(np.isfinite(part).all() for part in gradient)
    packages += ['blackjax', 'jax', 'jaxlib']
report = {'status': 'passed', 'backend': backend, 'devices': devices,
    'jax_default_backend': default_backend,
    'package_path': str(package), 'isolated': bool(sys.flags.isolated),
    'python_version': platform.python_version(), 'sys_prefix': sys.prefix,
    'system_site_packages': False, 'posterior_mcmc_executed': False,
    'versions': {name: metadata.version(name) for name in packages}}
(output / 'smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
"""


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", type=Path, required=True, help="New or empty local environment path"
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="Local wheel, sdist or project folder"
    )
    parser.add_argument(
        "--storage-root", type=Path, required=True, help="Data, results and cache root"
    )
    parser.add_argument(
        "--backend",
        default="pymc",
        choices=("pymc", "blackjax-cpu", "blackjax-cuda12", "blackjax-cuda13"),
    )
    parser.add_argument(
        "--install-timeout", type=int, default=600, help="Maximum pip-install seconds"
    )
    parser.add_argument("--index-url", help="Explicit PyPI-compatible package index URL")
    parser.add_argument(
        "--no-download-cache",
        action="store_true",
        help="Do not retain downloaded packages in pip's cache",
    )
    args = parser.parse_args(argv)
    args.env, args.source, args.storage_root = (
        path.expanduser().resolve() for path in (args.env, args.source, args.storage_root)
    )
    if args.install_timeout < 1:
        parser.error("--install-timeout must be positive")
    if args.env.exists() and (not args.env.is_dir() or any(args.env.iterdir())):
        parser.error("--env must be new or empty; existing environments are never replaced")
    if args.storage_root == args.env or args.storage_root.is_relative_to(args.env):
        parser.error("--storage-root must be outside the environment")
    if (
        not args.source.exists()
        or (args.source.is_dir() and not (args.source / "pyproject.toml").is_file())
        or (args.source.is_file() and not str(args.source).endswith((".whl", ".tar.gz", ".zip")))
    ):
        parser.error("--source must be an existing local wheel, sdist or pyproject.toml folder")
    if "cuda" in args.backend and (
        sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "aarch64"}
    ):
        parser.error(
            "JAX CUDA wheels require Linux x86_64/aarch64; native Windows GPU is unsupported"
        )
    return args


def environment(args):
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(name, None)
    tag = hashlib.sha256(str(args.env).encode()).hexdigest()[:12]
    cache = args.storage_root / "cache" / f"{platform.node()}-{sys.platform}-{tag}"
    paths = {
        "PIP_CACHE_DIR": args.storage_root / "cache" / "downloads" / "pip",
        "TEMP": cache / "tmp",
        "TMP": cache / "tmp",
        "TMPDIR": cache / "tmp",
        "NUMBA_CACHE_DIR": cache / "numba",
        "JAX_COMPILATION_CACHE_DIR": cache / "jax",
        "CUDA_CACHE_PATH": cache / "cuda",
        "XDG_CACHE_HOME": cache / "xdg",
        "MPLCONFIGDIR": cache / "matplotlib",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
        "PYFCIRT_DATA_DIR": args.storage_root / "data",
        "PYFCIRT_RESULTS_DIR": args.storage_root / "results",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(PIP_CONFIG_FILE=os.devnull, PYTHONNOUSERSITE="1", VIRTUAL_ENV=str(args.env))
    env["PYTENSOR_FLAGS"] = f"base_compiledir={(cache / 'pytensor').as_posix()}"
    if sys.platform == "win32" and str(cache).startswith("\\\\"):
        env["PYTENSOR_FLAGS"] += ",cxx="
    env["JAX_PLATFORMS"] = "cuda,cpu" if "cuda" in args.backend else "cpu"
    env["JAX_ENABLE_X64"] = "true"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return env, cache


def run(command, *, stage, env, directory, timeout):
    print(f"{stage}: running (limit {timeout}s)", flush=True)
    with (directory / f"{stage}.log").open("w", encoding="utf-8") as log:
        group = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            **group,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    check=False,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The owned process group already exited during timeout cleanup.
            process.wait(timeout=10)
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def main(argv=None):
    args = arguments(argv)
    env, _ = environment(args)
    output = Path(tempfile.mkdtemp(prefix="install-", dir=args.storage_root / "results"))
    python = args.env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    dependencies = [f"{args.source}[io]"]
    if args.backend != "pymc":
        dependencies.append(
            "jax" if args.backend == "blackjax-cpu" else f"jax[{args.backend.split('-')[-1]}]"
        )
    cache_options = (
        ["--no-cache-dir"] if args.no_download_cache else ["--cache-dir", env["PIP_CACHE_DIR"]]
    )
    index_options = ["--index-url", args.index_url] if args.index_url is not None else []
    flags = ["-X", f"pycache_prefix={env['PYTHONPYCACHEPREFIX']}", "-I"]
    pip = [str(python), *flags, "-m", "pip", "--isolated", "--disable-pip-version-check"]
    report = {
        "status": "running",
        "backend": args.backend,
        "environment": str(args.env),
        "source": str(args.source),
        "source_type": "unfrozen_directory"
        if args.source.is_dir()
        else ("wheel" if args.source.suffix == ".whl" else "sdist"),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest()
        if args.source.is_file()
        else None,
        "storage_root": str(args.storage_root),
        "download_cache": not args.no_download_cache,
        "custom_package_index": args.index_url is not None,
        "steps": [],
    }
    steps = [
        ("create", [sys.executable, *flags, "-m", "venv", str(args.env)], 120),
        (
            "install",
            [*pip, "install", "--no-input", *cache_options, *index_options, *dependencies],
            args.install_timeout,
        ),
        ("check", [*pip, "check"], 60),
        ("smoke", [str(python), *flags, "-c", SMOKE, args.backend, str(output)], 180),
    ]
    try:
        for stage, command, timeout in steps:
            run(command, stage=stage, env=env, directory=output, timeout=timeout)
            report["steps"].append(stage)
        smoke = json.loads((output / "smoke.json").read_text(encoding="utf-8"))
        if smoke["status"] != "passed":
            raise ValueError("Installed-package verification did not pass")
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        report.update(status="failed", failed_step=stage, error=str(error))
        print(
            f"Installation failed at {stage}. Details: {output / (stage + '.log')}", file=sys.stderr
        )
        code = 1
    else:
        report["status"] = "passed"
        command = [str(python), "-c", "import pyfcirt; print(pyfcirt.__file__)"]
        example = (
            ("& " + " ".join("'" + argument.replace("'", "''") + "'" for argument in command))
            if sys.platform == "win32"
            else shlex.join(command)
        )
        print(f"Environment verified. Run: {example}\nVerification: {output / 'smoke.json'}")
        code = 0
    (output / "installation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
