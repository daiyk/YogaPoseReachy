"""Choose, and on Reachy Mini Wireless create, the Python that runs the MediaPipe worker.

* Lite / desktop (Windows, macOS, x86-64 Linux): MediaPipe is an app dependency, the worker uses this Python.
* Wireless (ARM64 Linux): MediaPipe 1.x crashes on the CM4, so the worker runs MediaPipe 0.10.18 in a
  separate virtual environment. The first start creates it from the bundled lock file; later starts reuse it.
  An existing compatible environment (for example one made for the pose-compare app) is adopted as-is.
"""

import os
import sys
import json
import hashlib
import logging
import platform
import subprocess
import importlib.util
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Callable, Iterator


logger = logging.getLogger(__name__)

PYTHON_ENV = "REACHY_POSE_PYTHON"  # explicit interpreter; never installed into
DIRECTORY_ENV = "REACHY_POSE_RUNTIME"  # where the Wireless runtime lives
ARM_MEDIAPIPE = "0.10.18"
LOCK_FILE = Path(__file__).with_name("requirements-arm64.lock.txt")
MARKER = ".reachy-pose-coach.json"
INSTALL_TIMEOUT_S = 1800


def isolated() -> bool:
    """Whether this platform needs the separate MediaPipe 0.10.18 environment."""
    return sys.platform == "linux" and platform.machine().lower() in ("aarch64", "arm64")


def directory() -> Path:
    """Wireless runtime location (default `~/reachy-pose-runtime`, i.e. `/home/pollen/...` on the robot)."""
    return Path(os.environ.get(DIRECTORY_ENV) or Path.home() / "reachy-pose-runtime")


def interpreter() -> Path:
    """Return the Python the worker runs in, without installing anything."""
    override = os.environ.get(PYTHON_ENV)
    if override:
        return Path(override)
    if isolated():
        return directory() / "bin" / "python"
    return Path(sys.executable)


def lock_digest() -> str:
    """Identity of the bundled dependency lock."""
    return hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest()


def ensure_runtime(on_install: Callable[[], None] = lambda: None) -> Path:
    """Return a ready worker interpreter, creating the Wireless environment on first use.

    `on_install` is called once before a (slow, network) installation starts so the UI can say so.
    """
    python = interpreter()
    if os.environ.get(PYTHON_ENV):
        if not python.is_file():
            raise RuntimeError(f"{PYTHON_ENV} points to a missing Python: {python}")
        return python
    if not isolated():
        if importlib.util.find_spec("mediapipe") is None:
            raise RuntimeError("MediaPipe is missing from the app environment; reinstall the pose coach app")
        return python
    root = directory()
    with _locked(root.with_name(root.name + ".lock")):
        if _ready(root, python):
            return python
        on_install()
        _install(root, python)
    return python


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialise installs between app instances (POSIX only; Windows never installs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _ready(root: Path, python: Path) -> bool:
    """Accept a runtime installed from this lock, or adopt an existing compatible one."""
    if not python.is_file():
        return False
    marker = root / MARKER
    try:
        if json.loads(marker.read_text(encoding="utf-8")).get("lock_sha256") == lock_digest():
            return True
        # A managed runtime must apply the entire new lock, even when MediaPipe is unchanged.
        return False
    except FileNotFoundError:
        pass  # Only an unmarked legacy runtime may be adopted.
    except (OSError, ValueError, AttributeError):
        return False
    probe = "import mediapipe, numpy, cv2; print(mediapipe.__version__)"
    try:
        found = subprocess.run(
            [str(python), "-I", "-c", probe], capture_output=True, text=True, timeout=180, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    if found != ARM_MEDIAPIPE:
        logger.warning("Pose runtime has MediaPipe %r, expected %s; reinstalling", found, ARM_MEDIAPIPE)
        return False
    _mark(root, adopted=True)
    logger.info("Adopted existing pose runtime at %s", root)
    return True


def _install(root: Path, python: Path) -> None:
    """Create the virtual environment and install the locked MediaPipe 0.10.18 stack."""
    root.mkdir(parents=True, exist_ok=True)
    log = root / "install.log"
    logger.info("Installing pose runtime into %s (log: %s); this can take several minutes", root, log)
    steps = [
        [sys.executable, "-m", "venv", str(root)],
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-r", str(LOCK_FILE)],
    ]
    with log.open("w", encoding="utf-8") as output:
        for step in steps:
            output.write("$ " + " ".join(step) + "\n")
            output.flush()
            try:
                subprocess.run(step, stdout=output, stderr=subprocess.STDOUT, timeout=INSTALL_TIMEOUT_S, check=True)
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError(f"Installing the pose runtime failed; see {log}") from exc
    _mark(root, adopted=False)
    logger.info("Pose runtime installed at %s", root)


def _mark(root: Path, adopted: bool) -> None:
    (root / MARKER).write_text(
        json.dumps({"lock_sha256": lock_digest(), "mediapipe": ARM_MEDIAPIPE, "adopted": adopted}),
        encoding="utf-8",
    )
