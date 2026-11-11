"""Dependency-free regression checks for installation and runtime updates."""

import json
import unittest
import subprocess
import importlib.util
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    """Load an independent module without importing the robot SDK."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("runtime_distribution", "src/reachy_mini_pose_coach/pose/runtime/environment.py")


class RuntimeUpdateTests(unittest.TestCase):
    """Managed runtimes must apply a changed lock even if MediaPipe is unchanged."""

    def test_lock_change_failure_retry_and_reuse(self):
        """Do not adopt or mark an old lock until installation succeeds."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "bin/python"
            python.parent.mkdir()
            python.touch()
            marker = root / runtime.MARKER
            marker.write_text(json.dumps({"lock_sha256": "previous-lock"}))
            with (
                patch.dict(runtime.os.environ, {runtime.DIRECTORY_ENV: str(root)}, clear=True),
                patch.object(runtime, "isolated", return_value=True),
                patch.object(runtime.subprocess, "run") as run,
            ):
                run.side_effect = subprocess.CalledProcessError(1, "pip")
                with self.assertRaises(RuntimeError):
                    runtime.ensure_runtime()
                self.assertEqual(json.loads(marker.read_text())["lock_sha256"], "previous-lock")
                run.reset_mock(side_effect=True)
                runtime.ensure_runtime()
                self.assertEqual(run.call_count, 2)
                self.assertEqual(json.loads(marker.read_text())["lock_sha256"], runtime.lock_digest())
                run.reset_mock()
                runtime.ensure_runtime()
                run.assert_not_called()

    def test_only_unmarked_legacy_runtime_is_adopted(self):
        """Corrupt managed markers require repair; legacy compatible environments remain reusable."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            python.touch()
            with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(stdout="0.10.18\n")) as run:
                self.assertTrue(runtime._ready(root, python))
                run.assert_called_once()
                (root / runtime.MARKER).write_text("broken json")
                run.reset_mock()
                self.assertFalse(runtime._ready(root, python))
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
