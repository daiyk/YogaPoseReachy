"""A bounded worker process keeps MediaPipe's CPU/NumPy requirements out of the SDK process.

Copied from `reachy_mini_pose_compare`; the interpreter now comes from `environment.interpreter()`.
"""

import sys
import json
import subprocess
from types import SimpleNamespace
from typing import IO, Any
from pathlib import Path
from concurrent.futures import TimeoutError, ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray

from .environment import interpreter


WORKER = Path(__file__).with_name("worker.py")


class PoseDetector:
    """Client for one worker process; every read is bounded by a timeout."""

    def __init__(
        self,
        model_path: Path,
        mode: str,
        python: Path | None = None,
        options: dict[str, int] | None = None,
    ) -> None:
        """Start the worker and wait for its runtime report."""
        python = python if python is not None else interpreter()
        if not python.is_file():
            raise RuntimeError(f"Pose runtime Python is missing: {python}")
        # -I: never import modules from the working directory or PYTHON* variables.
        command = [str(python), "-I", "-u", str(WORKER), str(model_path), mode]
        if options:
            command.append(json.dumps(options))
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, creationflags=flags)
        self.reader = ThreadPoolExecutor(max_workers=1)
        try:
            self.runtime = self.response(45)["runtime"]
        except BaseException:
            self.close()
            raise

    @property
    def _stdin(self) -> IO[bytes]:
        assert self.process.stdin is not None
        return self.process.stdin

    @property
    def _stdout(self) -> IO[bytes]:
        assert self.process.stdout is not None
        return self.process.stdout

    def response(self, timeout: float = 10) -> dict[str, Any]:
        """Read one JSON line or fail after the timeout, killing a stuck worker."""
        future = self.reader.submit(self._stdout.readline, 128 * 1024)
        try:
            data = future.result(timeout=timeout)
        except TimeoutError:
            self.process.kill()
            raise RuntimeError("Pose worker timed out; stop and start the run again.") from None
        if not data:
            raise RuntimeError(f"Pose worker exited (code {self.process.poll()}); check the app log.")
        result: dict[str, Any] = json.loads(data)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def detect(self, rgb: NDArray[np.uint8], timestamp: int | None = None) -> SimpleNamespace:
        """Send one RGB frame and return its landmarks as attribute objects."""
        header = {"shape": list(rgb.shape), "timestamp": timestamp}
        try:
            self._stdin.write(json.dumps(header).encode() + b"\n")
            self._stdin.write(rgb.tobytes())
            self._stdin.flush()
        except (OSError, ValueError) as exc:
            raise RuntimeError("Pose worker stopped accepting camera frames.") from exc
        result = self.response()
        for field in ("pose_landmarks", "pose_world_landmarks"):
            result[field] = [
                [SimpleNamespace(x=p[0], y=p[1], z=p[2], visibility=p[3], presence=p[4]) for p in pose]
                for pose in result[field]
            ]
        return SimpleNamespace(**result)

    def close(self) -> None:
        """Close stdin, wait briefly, then kill; always release the reader thread."""
        try:
            self._stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self._stdout.close()
        self.reader.shutdown(wait=True)

    def __enter__(self) -> "PoseDetector":
        """Use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close on exit."""
        self.close()
