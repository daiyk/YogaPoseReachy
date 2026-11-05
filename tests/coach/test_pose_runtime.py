"""The coach carries its own pose runtime and needs no other Reachy app."""

import sys
import json
import threading
import subprocess
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest

import reachy_mini_pose_coach.pose.engine as engine_module
from reachy_mini_pose_coach.pose.runtime import environment
from reachy_mini_pose_coach.pose.service import PoseService


PACKAGE = Path(engine_module.__file__).parents[1]


def test_bundled_assets_and_lock_ship_with_the_coach():
    """The Full model, built-in reference and Wireless lock live inside the coach package."""
    assert (engine_module.ASSETS / engine_module.MODEL).stat().st_size > 1_000_000
    assert (engine_module.ASSETS / "reference.jpg").read_bytes()[:2] == b"\xff\xd8"
    assert engine_module.ASSETS.is_relative_to(PACKAGE)
    lock = environment.LOCK_FILE.read_text(encoding="utf-8")
    assert f"mediapipe=={environment.ARM_MEDIAPIPE}" in lock and "numpy==1.26.4" in lock


def test_engine_starts_with_the_compare_app_blocked():
    """A fresh interpreter that cannot import reachy_mini_pose_compare still builds the engine."""
    script = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'reachy_mini_pose_compare':
            raise ImportError('blocked: ' + name)
sys.meta_path.insert(0, Block())
import reachy_mini_pose_coach.pose.engine as e
assert e.cv2 is not None, 'OpenCV missing'
engine = e.PoseEngine()
assert engine.model.is_file() and engine.default_reference
assert not any(m.startswith('reachy_mini_pose_compare') for m in sys.modules)
print('ok', engine.model)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ok") and str(PACKAGE) in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("mediapipe") is None, reason="MediaPipe not installed here")
def test_real_worker_detects_the_bundled_reference():
    """The bundled worker and Full model find one 33-landmark person in the built-in reference."""
    engine = engine_module.PoseEngine()
    engine.prepare(engine.default_reference, "same")
    try:
        assert len(engine.reference_points) == 33 and engine.comparison is not None
    finally:
        engine.close()


def test_desktop_runtime_uses_this_python(monkeypatch):
    """Lite/desktop: the worker runs in the app's own environment, with MediaPipe as a dependency."""
    monkeypatch.delenv(environment.PYTHON_ENV, raising=False)
    monkeypatch.setattr(environment, "isolated", lambda: False)
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: object())
    installs = []
    assert environment.ensure_runtime(lambda: installs.append(1)) == Path(sys.executable)
    assert installs == []
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="MediaPipe is missing"):
        environment.ensure_runtime()


def test_explicit_python_must_exist(monkeypatch, tmp_path):
    """REACHY_POSE_PYTHON is used as-is and never installed into."""
    monkeypatch.setenv(environment.PYTHON_ENV, str(tmp_path / "python"))
    with pytest.raises(RuntimeError, match="missing Python"):
        environment.ensure_runtime()
    (tmp_path / "python").write_bytes(b"")
    assert environment.ensure_runtime() == tmp_path / "python"


@pytest.fixture
def wireless(monkeypatch, tmp_path):
    """Pretend to be ARM64 Linux with the runtime under tmp_path and a recording subprocess.run."""
    root = tmp_path / "reachy-pose-runtime"
    monkeypatch.delenv(environment.PYTHON_ENV, raising=False)
    monkeypatch.setenv(environment.DIRECTORY_ENV, str(root))
    monkeypatch.setattr(environment, "isolated", lambda: True)
    calls = SimpleNamespace(steps=[], probe="0.10.18")

    def run(command, **kwargs):
        if "-c" in command:
            calls.steps.append("probe")
            return SimpleNamespace(stdout=calls.probe + "\n")
        calls.steps.append("venv" if "venv" in command else "pip")
        if "venv" in command:
            (root / "bin").mkdir(parents=True, exist_ok=True)
            (root / "bin" / "python").write_bytes(b"")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(environment.subprocess, "run", run)
    return root, calls


def test_wireless_first_start_installs_once(wireless):
    """Missing runtime: create venv, install the lock, mark it; the next start just reuses it."""
    root, calls = wireless
    installs = []
    python = environment.ensure_runtime(lambda: installs.append(1))
    assert python == root / "bin" / "python" and installs == [1] and calls.steps == ["venv", "pip"]
    marker = json.loads((root / environment.MARKER).read_text(encoding="utf-8"))
    assert marker == {"lock_sha256": environment.lock_digest(), "mediapipe": "0.10.18", "adopted": False}
    calls.steps.clear()
    assert environment.ensure_runtime(lambda: installs.append(1)) == python
    assert installs == [1] and calls.steps == []


def test_wireless_adopts_existing_compatible_runtime(wireless):
    """An existing 0.10.18 environment (made for the compare app) is reused without reinstalling."""
    root, calls = wireless
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python").write_bytes(b"")
    installs = []
    environment.ensure_runtime(lambda: installs.append(1))
    assert calls.steps == ["probe"] and installs == []
    assert json.loads((root / environment.MARKER).read_text(encoding="utf-8"))["adopted"] is True


def test_wireless_replaces_incompatible_runtime(wireless):
    """A runtime with another MediaPipe version is reinstalled from the lock."""
    root, calls = wireless
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python").write_bytes(b"")
    calls.probe = "1.1.1"
    installs = []
    environment.ensure_runtime(lambda: installs.append(1))
    assert calls.steps == ["probe", "venv", "pip"] and installs == [1]


def test_wireless_install_failure_is_reported(wireless, monkeypatch):
    """A failed pip run surfaces as one readable error naming the log."""
    root, calls = wireless

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(environment.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="install.log"):
        environment.ensure_runtime()
    assert not (root / environment.MARKER).exists()


def test_service_reports_installation_then_failure(tmp_path):
    """The panel sees `installing` while the runtime is prepared, then a readable error."""
    seen = []
    done = threading.Event()

    def runtime(on_install):
        on_install()
        seen.append(service.state()["status"])
        raise RuntimeError("Installing the pose runtime failed; see install.log")

    def engine():
        raise AssertionError("engine must not start without a runtime")

    service = PoseService(lambda: None, tmp_path, engine_factory=engine, runtime=runtime)
    original = service._run

    def run():
        original()
        done.set()

    service._run = run  # type: ignore[method-assign]
    service.start()
    assert done.wait(5)
    service.stop()
    assert seen == ["installing"]
    assert service.state()["status"] == "error" and "install.log" in str(service.state()["error"])
