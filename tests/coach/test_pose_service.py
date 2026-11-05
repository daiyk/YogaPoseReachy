"""Live service lifecycle, bounded caching and reference race regressions."""

import time
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import reachy_mini_pose_coach.pose.service as service_module
from reachy_mini_pose_coach.pose.engine import FrameResult
from reachy_mini_pose_coach.coach.sample import Reason, Landmark
from reachy_mini_pose_coach.pose.service import PoseService


class Engine:
    """Controlled worker with real thread coordination and no model dependency."""

    def __init__(self):
        """Start a ready reference with an optionally blocked first detection."""
        self.default_reference = b"builtin"
        self.reference_points = (Landmark(0.3, 0.2, 1), Landmark(0.7, 0.8, 1))
        self.stopping = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = 0
        self.prepares = []
        self.closed = False
        self.fail = False

    def validate_image(self, image):
        """Reject non-image fixture bytes at the upload boundary."""
        if image == b"invalid":
            raise ValueError("Invalid image")
        return image

    def prepare(self, jpeg, side, scope):
        """Record selection and reject an unsupported reference pose."""
        self.prepares.append((jpeg, side))
        self.scope = scope
        if jpeg == b"no-person":
            raise ValueError("No person in reference")

    def detect(self, frame, captured):
        """Block until released so tests can change references or stop inference mid-flight."""
        self.calls += 1
        self.entered.set()
        assert self.release.wait(2)
        if self.fail:
            raise RuntimeError("Worker failed")
        return FrameResult(80.0, (), None, self.reference_points)

    def close(self):
        """Record release of the isolated worker."""
        self.closed = True


def wait_for(predicate):
    """Wait a bounded time for an actual worker transition."""
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.fixture
def running(tmp_path):
    """Provide one service and always stop its worker, including after assertion failures."""
    engine = Engine()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    service = PoseService(lambda: frame, tmp_path, engine_factory=lambda: engine)
    service.start()
    wait_for(lambda: service.state()["reference_ready"])
    try:
        yield service, engine
    finally:
        engine.release.set()
        service.stop()
    assert engine.closed


def test_no_inference_without_lease_and_one_latest_sample(running):
    """Idle startup stays quiet; a lost coordinator cannot keep inference running."""
    service, engine = running
    time.sleep(0.1)
    assert engine.calls == 0 and service.latest() is None
    service.set_rate(6)
    wait_for(lambda: engine.calls >= 3)
    sample = service.latest()
    assert sample.score == 80 and sample.observed_at.utcoffset() is not None
    time.sleep(1.1)
    count = engine.calls
    time.sleep(0.25)
    assert engine.calls == count and service.latest().sample_id >= sample.sample_id


def test_rate_reaches_30_fps_and_rejects_more(running):
    """A 30 FPS lease is honoured within OS timer jitter; higher requests are refused."""
    service, engine = running
    with pytest.raises(ValueError):
        service.set_rate(30.5)
    service.set_rate(30)
    wait_for(lambda: engine.calls > 0)
    start, deadline = engine.calls, time.monotonic() + 2
    while time.monotonic() < deadline:
        service.set_rate(30)
        time.sleep(0.1)
    rate = (engine.calls - start) / 2
    service.set_rate(0)
    assert 25 <= rate <= 31


def test_reference_change_drops_inflight_old_result(running):
    """No old-reference score or skeleton is published after an upload."""
    service, engine = running
    engine.release.clear()
    service.set_rate(6)
    assert engine.entered.wait(1)
    old_id = service.reference_id
    new_id = service.select_reference("uploaded", "opposite", b"new-pose")
    assert new_id != old_id and service.latest() is None
    engine.release.set()
    wait_for(lambda: (b"new-pose", "opposite") in engine.prepares)
    wait_for(lambda: service.latest() is not None)
    assert service.latest().reference_id == new_id


def test_pause_discards_inflight_sample(running):
    """A completed inference captured before playback cannot become new speech evidence."""
    service, engine = running
    engine.release.clear()
    service.set_rate(6)
    assert engine.entered.wait(1)
    service.set_rate(0)
    engine.release.set()
    time.sleep(0.15)
    assert service.latest() is None


def test_worker_restarts_once_and_then_reports_failure(running):
    """Repeated model errors never spin an unbounded subprocess restart loop."""
    service, engine = running
    engine.fail = True
    service.set_rate(6)
    wait_for(lambda: service.state()["status"] == "error")
    assert len(engine.prepares) == 2
    assert service.latest().reason == Reason.WORKER_FAILURE
    calls = engine.calls
    service.set_rate(6)
    time.sleep(0.3)
    assert engine.calls == calls


def test_invalid_reference_and_runtime_unavailable_are_explicit(running, tmp_path):
    """Bad references and missing dependencies leave an explicit unavailable state."""
    service, _ = running
    with pytest.raises(ValueError):
        service.select_reference("uploaded", "same", b"invalid")
    service.select_reference("uploaded", "same", b"no-person")
    wait_for(lambda: service.state()["status"] == "invalid_reference")
    assert service.latest() is None

    def missing():
        raise ImportError("Pose runtime missing")

    other = PoseService(lambda: None, tmp_path, engine_factory=missing)
    other.start()
    wait_for(lambda: other.state()["status"] == "error")
    other.stop()
    assert "missing" in other.state()["error"]


def test_preview_is_demand_only_and_bounded(running, monkeypatch):
    """A closed panel retains no camera frame and never encodes JPEGs."""
    service, _ = running
    calls = []

    def encode(*args):
        calls.append(time.monotonic())
        return True, np.array([1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(service_module, "cv2", SimpleNamespace(imencode=encode, IMWRITE_JPEG_QUALITY=1))
    assert service._frame is None
    assert service.preview() is None
    wait_for(lambda: service._frame is not None)
    assert service.preview()["jpeg"]
    assert service.preview() is None
    assert len(calls) == 1
    time.sleep(1.1)
    wait_for(lambda: service._frame is None)
    assert len(calls) == 1


def test_sleep_shutdown_cannot_be_reenabled_by_coordinator(running):
    """A concurrent activity tick cannot resume inference while the robot goes to sleep."""
    service, engine = running
    service.request_stop()
    service.set_rate(6)
    wait_for(lambda: engine.closed)
    assert engine.calls == 0


def test_body_parts_persist_per_image_and_side_and_reset_on_upload(running, tmp_path):
    """Selections survive restart but never leak to another side or replacement image."""
    service, engine = running
    first = service.select_reference("builtin", "same", parts=["left_arm", "head"])
    assert service.scope_for(first).parts == ("left_arm", "head")
    other = service.select_reference("builtin", "opposite", parts=["right_leg"])
    assert service.scope_for(other).parts == ("left_leg",)
    restored = service.select_reference("builtin", "same")
    assert service.scope_for(restored).parts == ("left_arm", "head")
    assert restored != first
    service.select_reference("uploaded", "same", b"first-upload", parts=["torso"])
    service.select_reference("uploaded", "same", b"second-upload")
    assert len(service.state()["parts"]) == 5
    service.stop()
    restarted = PoseService(lambda: None, tmp_path, engine_factory=Engine)
    restarted.start()
    try:
        wait_for(lambda: restarted.state()["reference_ready"])
        assert restarted.state()["parts"] == ("left_arm", "head")
        reference = restarted.select_reference("builtin", "opposite")
        assert restarted.scope_for(reference).parts == ("left_leg",)
    finally:
        restarted.stop()


def test_scope_change_drops_inflight_sample_and_invalid_selection_is_atomic(running):
    """A new scope on the same image cannot reuse old measurements or overwrite saved state on validation failure."""
    service, engine = running
    engine.release.clear()
    service.set_rate(6)
    assert engine.entered.wait(1)
    previous = service.reference_id
    with pytest.raises(ValueError):
        service.select_reference("builtin", "same", parts=["head"])
    assert service.reference_id == previous
    changed = service.select_reference("builtin", "same", parts=["torso"])
    assert changed != previous and service.latest() is None
    engine.release.set()
    wait_for(lambda: service.latest() is not None)
    assert service.latest().reference_id == changed
    assert engine.scope.parts == ("torso",)


def test_reconnect_bridge_adopts_the_loaded_reference(running):
    """A handler rebuilt after a reference loaded (e.g. new backend URL) must configure, not fail."""
    from unittest.mock import AsyncMock, MagicMock

    from reachy_mini_pose_coach.coach.bridge import CoachBridge
    from reachy_mini_pose_coach.coach.settings import CoachSettings

    service, _engine = running
    reference = service.select_reference("builtin", "same", parts=["left_arm"])
    bridge = CoachBridge(AsyncMock(), AsyncMock(), MagicMock())
    bridge.service = service

    bridge.configure(CoachSettings())

    assert bridge.coach.reference_id == reference
    assert bridge.scope.parts == ("left_arm",)
