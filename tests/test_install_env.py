"""Installer command, isolation and failure contracts; never install or use networking."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "install_env_test", Path(__file__).parents[1] / "scripts/install_env.py"
)
installer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(installer)


@pytest.fixture
def paths(tmp_path):
    source = tmp_path / "source with spaces"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='pyfcirt'\n", encoding="utf-8")
    return tmp_path / "environment", source, tmp_path / "storage"


def cli(paths, backend="pymc"):
    env, source, storage = paths
    return [
        "--env",
        str(env),
        "--source",
        str(source),
        "--storage-root",
        str(storage),
        "--backend",
        backend,
    ]


def test_environments_share_downloads_but_keep_compilation_caches_separate(paths):
    first = installer.arguments(cli(paths))
    second_arguments = cli(paths)
    second_arguments[1] = str(paths[0].with_name("second environment"))
    second = installer.arguments(second_arguments)
    first_env, first_cache = installer.environment(first)
    second_env, second_cache = installer.environment(second)
    assert first_env["PIP_CACHE_DIR"] == second_env["PIP_CACHE_DIR"]
    assert Path(first_env["PIP_CACHE_DIR"]) == paths[2] / "cache/downloads/pip"
    assert first_cache != second_cache
    for key in (
        "NUMBA_CACHE_DIR",
        "JAX_COMPILATION_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "PYTHONPYCACHEPREFIX",
    ):
        assert first_env[key] != second_env[key]
        assert Path(first_env[key]).is_relative_to(first_cache)


@pytest.mark.parametrize("backend", ["pymc", "blackjax-cpu", "blackjax-cuda12", "blackjax-cuda13"])
def test_installation_plan_isolated_source_and_device_commands(paths, backend, monkeypatch):
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("PYTHONPATH", "untrusted-path")
    monkeypatch.setenv("PIP_TARGET", "untrusted-target")
    commands = []

    def fake_run(command, *, stage, env, directory, timeout):
        commands.append((stage, command, env, directory, timeout))
        assert "PYTHONPATH" not in env and "PIP_TARGET" not in env
        assert directory.is_relative_to(paths[2]) and directory != paths[1]
        if stage == "smoke":
            device = "gpu" if "cuda" in backend else "cpu"
            (directory / "smoke.json").write_text(
                json.dumps({"status": "passed", "devices": [{"platform": device}]}),
                encoding="utf-8",
            )

    monkeypatch.setattr(installer, "run", fake_run)
    assert installer.main(cli(paths, backend)) == 0
    assert [stage for stage, *_ in commands] == ["create", "install", "check", "smoke"]
    create = commands[0][1]
    assert create[1] == "-X" and create[2].startswith("pycache_prefix=")
    assert create[3:6] == ["-I", "-m", "venv"]
    assert "--system-site-packages" not in create
    install = commands[1][1]
    assert f"{paths[1]}[io]" in install
    assert "--upgrade" not in install and "--isolated" in install
    assert "--cache-dir" in install
    assert "--index-url" not in install
    if backend == "blackjax-cpu":
        assert "jax" in install
    elif "cuda" in backend:
        assert f"jax[{backend.split('-')[-1]}]" in install
    else:
        assert not any(value.startswith("jax") for value in install)
    smoke = commands[-1]
    assert smoke[1][3:5] == ["-I", "-c"]
    assert all(command[1:3] == create[1:3] for _, command, *_ in commands)
    assert smoke[2]["JAX_PLATFORMS"] == ("cuda,cpu" if "cuda" in backend else "cpu")
    assert smoke[2]["JAX_ENABLE_X64"] == "true"
    assert smoke[2]["PYFCIRT_DATA_DIR"] == str(paths[2] / "data")
    assert [timeout for *_, timeout in commands] == [120, 600, 60, 180]
    report = json.loads(next((paths[2] / "results").glob("*/installation.json")).read_text())
    assert report["status"] == "passed"
    assert report["source_type"] == "unfrozen_directory" and report["source_sha256"] is None
    assert report["download_cache"] is True
    assert report["custom_package_index"] is False


def test_no_download_cache_uses_explicit_pip_flag_and_records_policy(paths, monkeypatch):
    commands = []

    def fake_run(command, *, stage, directory, **options):
        commands.append((stage, command))
        if stage == "smoke":
            (directory / "smoke.json").write_text('{"status":"passed"}', encoding="utf-8")

    monkeypatch.setattr(installer, "run", fake_run)
    index = "https://packages.example.org/simple/"
    assert installer.main([*cli(paths), "--no-download-cache", "--index-url", index]) == 0
    install = next(command for stage, command in commands if stage == "install")
    assert "--no-cache-dir" in install and "--cache-dir" not in install
    assert f"{paths[1]}[io]" in install
    assert install[install.index("--index-url") + 1] == index
    report = json.loads(next((paths[2] / "results").glob("*/installation.json")).read_text())
    assert report["download_cache"] is False
    assert report["custom_package_index"] is True


@pytest.mark.parametrize("stage", ["install", "smoke"])
def test_failed_install_or_device_probe_stops_without_replacement(paths, stage, monkeypatch):
    calls = []

    def fail(command, **options):
        calls.append(options["stage"])
        if options["stage"] == "create":
            paths[0].mkdir()
            (paths[0] / "keep.txt").write_text("created environment", encoding="utf-8")
        if options["stage"] == stage:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(installer, "run", fail)
    assert installer.main(cli(paths)) == 1
    assert calls[-1] == stage
    assert (paths[0] / "keep.txt").read_text() == "created environment"
    report = json.loads(next((paths[2] / "results").glob("*/installation.json")).read_text())
    assert report["status"] == "failed" and report["failed_step"] == stage
    with pytest.raises(SystemExit):
        installer.arguments(cli(paths))


def test_run_forwards_timeout_working_directory_and_log(paths, monkeypatch):
    directory = paths[2]
    directory.mkdir()
    called = []
    waits = []
    process = SimpleNamespace(pid=12345, returncode=0, wait=lambda timeout: waits.append(timeout))

    def popen(command, **kw):
        called.append((command, kw))
        return process

    monkeypatch.setattr(installer.subprocess, "Popen", popen)
    installer.run(
        ["python", "probe.py"], stage="smoke", env={"A": "B"}, directory=directory, timeout=17
    )
    command, options = called[0]
    assert command == ["python", "probe.py"]
    assert waits == [17]
    assert options["cwd"] == directory and options["env"] == {"A": "B"}
    assert (directory / "smoke.log").is_file()


@pytest.mark.parametrize("system", ["linux", "win32"])
def test_timeout_kills_only_owned_process_tree_and_waits(paths, system, monkeypatch):
    monkeypatch.setattr(installer.sys, "platform", system)
    monkeypatch.setattr(installer.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr(installer.signal, "SIGKILL", 9, raising=False)
    paths[2].mkdir()
    waits, killed, spawned = [], [], []

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
        return -9

    process = SimpleNamespace(pid=12345, returncode=-9, wait=wait)

    def popen(command, **options):
        spawned.append(options)
        return process

    monkeypatch.setattr(installer.subprocess, "Popen", popen)
    monkeypatch.setattr(
        installer.os, "killpg", lambda pid, sig: killed.append((pid, sig)), raising=False
    )
    monkeypatch.setattr(installer.subprocess, "run", lambda cmd, **kw: killed.append((cmd, kw)))
    with pytest.raises(subprocess.TimeoutExpired):
        installer.run(
            ["python", "worker.py"], stage="install", env={}, directory=paths[2], timeout=17
        )
    assert waits == [17, 10]
    if system == "linux":
        assert spawned[0]["start_new_session"]
        assert killed == [(12345, installer.signal.SIGKILL)]
    else:
        assert spawned[0]["creationflags"] == 512
        assert killed[0][0] == ["taskkill", "/PID", "12345", "/T", "/F"]
        assert killed[0][1]["timeout"] == 10


@pytest.mark.parametrize(
    "problem", ["occupied", "missing-source", "unsupported-source", "nested-storage", "timeout"]
)
def test_invalid_paths_and_limits_rejected_before_subprocess(paths, problem, monkeypatch):
    args = cli(paths)
    if problem == "occupied":
        paths[0].mkdir()
        (paths[0] / "existing.txt").write_text("keep", encoding="utf-8")
    elif problem == "missing-source":
        args[3] = str(paths[1] / "absent")
    elif problem == "unsupported-source":
        invalid = paths[1] / "file.txt"
        invalid.write_text("invalid", encoding="utf-8")
        args[3] = str(invalid)
    elif problem == "nested-storage":
        args[5] = str(paths[0] / "cache")
    else:
        args += ["--install-timeout", "0"]
    monkeypatch.setattr(installer, "run", lambda *a, **kw: pytest.fail("No command expected"))
    with pytest.raises(SystemExit):
        installer.main(args)


def test_native_windows_gpu_is_rejected_before_environment_creation(paths, monkeypatch):
    monkeypatch.setattr(installer.sys, "platform", "win32")
    with pytest.raises(SystemExit):
        installer.arguments(cli(paths, "blackjax-cuda12"))
    assert not paths[0].exists()


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz", ".zip"])
def test_local_artifact_source_selection(paths, suffix):
    artifact = paths[1] / f"pyfcirt{suffix}"
    artifact.write_bytes(b"mock artifact, not installed")
    args = cli(paths)
    args[3] = str(artifact)
    assert installer.arguments(args).source == artifact


def test_archive_checksum_recorded_without_running_an_install(paths, monkeypatch):
    artifact = paths[1] / "pyfcirt.whl"
    artifact.write_bytes(b"fixed artifact fixture")
    args = cli(paths)
    args[3] = str(artifact)

    def run(command, *, stage, directory, **options):
        if stage == "smoke":
            (directory / "smoke.json").write_text('{"status":"passed"}', encoding="utf-8")

    monkeypatch.setattr(installer, "run", run)
    assert installer.main(args) == 0
    report = json.loads(next((paths[2] / "results").glob("*/installation.json")).read_text())
    assert report["source_type"] == "wheel"
    assert report["source_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_printed_powershell_command_quotes_special_path_characters(paths, monkeypatch, capsys):
    monkeypatch.setattr(installer.sys, "platform", "win32")
    env = paths[0].parent / "environment & user's"
    selected = (env, *paths[1:])

    def run(command, *, stage, directory, **options):
        if stage == "smoke":
            (directory / "smoke.json").write_text('{"status":"passed"}', encoding="utf-8")

    monkeypatch.setattr(installer, "run", run)
    assert installer.main(cli(selected)) == 0
    quoted = str(env / "Scripts/python.exe").replace("'", "''")
    assert (
        f"Run: & '{quoted}' '-c' 'import pyfcirt; print(pyfcirt.__file__)'"
        in capsys.readouterr().out
    )
