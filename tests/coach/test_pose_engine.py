"""The adapter preserves comparison gates while the coach owns temporal evidence."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import reachy_mini_pose_coach.pose.engine as engine_module
from reachy_mini_pose_coach.coach.scope import BodyScope
from reachy_mini_pose_coach.coach.sample import Reason


def test_adapter_timing_and_reason_projection(tmp_path, monkeypatch):
    """Low rates can score without weakening geometry gates or returning stale invalid scores."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "pose_landmarker_full_v1.task").write_bytes(b"model")
    (assets / "reference.jpg").write_bytes(b"jpeg")
    points = [
        SimpleNamespace(x=0.3, y=0.2, presence=1.0, visibility=1.0),
        SimpleNamespace(x=0.7, y=0.8, presence=1.0, visibility=1.0),
    ] * 17
    points = points[:33]
    detector = MagicMock()
    detector.detect.return_value = SimpleNamespace(pose_landmarks=[points], pose_world_landmarks=[points])
    prepare = MagicMock(return_value="prepared")
    comparison = MagicMock()
    comparison.evaluate.return_value = SimpleNamespace(score=80.0, reason_code="", deltas_deg={"left_elbow": -20.0})
    gates = {
        "minimum_frames": 3,
        "minimum_span_seconds": 0.2,
        "smoothing_seconds": 0.5,
        "live_frontal_limit_deg": 30.0,
    }
    modules = {
        "reachy_mini_pose_compare.detector": SimpleNamespace(
            __file__=str(tmp_path / "detector.py"), PoseDetector=MagicMock(return_value=detector)
        ),
        "reachy_mini_pose_compare.comparison": SimpleNamespace(
            SETTINGS=gates, prepare_reference=prepare, ComparisonEngine=lambda ref: comparison
        ),
        "reachy_mini_pose_compare.runner": SimpleNamespace(validate_reference=lambda image: image),
    }
    monkeypatch.setattr(engine_module.importlib, "import_module", modules.__getitem__)
    codec = SimpleNamespace(
        IMREAD_COLOR=1,
        COLOR_BGR2RGB=2,
        imdecode=lambda *_: np.zeros((10, 10, 3), dtype=np.uint8),
        cvtColor=lambda frame, _: frame,
    )
    monkeypatch.setattr(engine_module, "cv2", codec)
    engine = engine_module.PoseEngine()
    engine.prepare(b"jpeg", "same")
    settings = prepare.call_args.kwargs["settings"]
    assert settings["minimum_frames"] == 1 and settings["minimum_span_seconds"] == 0
    assert settings["live_frontal_limit_deg"] == 30 and gates["minimum_frames"] == 3
    sides = [call.args[2] for call in prepare.call_args_list]
    probe = [call.kwargs["settings"].get("relative_yaw_limit_deg") for call in prepare.call_args_list]
    assert sides == ["same", "same", "opposite"] and probe[1:] == [180.0, 180.0] and "relative_yaw" not in str(gates)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = engine.detect(frame, 10)
    assert result.score == 80 and result.errors[0].degrees == -20 and result.reason is None
    assert len(result.landmarks) == 33 and result.mirror is not None and result.mirror.opposite_score == 80
    comparison.evaluate.return_value = SimpleNamespace(score=None, reason_code="unsupported_view", deltas_deg={})
    result = engine.detect(frame, 11)
    assert (result.score, result.errors, result.reason) == (None, (), Reason.UNSUPPORTED_VIEW)
    comparison.evaluate.return_value = SimpleNamespace(score=80.0, reason_code="", deltas_deg={"left_elbow": -20.0})
    points[31] = SimpleNamespace(x=1.05, y=0.5, presence=1.0, visibility=1.0)
    assert engine.detect(frame, 12).score == 80
    points[31].x = 0.7
    # Hidden head: MediaPipe's left/right becomes unreliable, so the head must stay in the image.
    points[0] = SimpleNamespace(x=0.3, y=-0.05, presence=1.0, visibility=1.0)
    result = engine.detect(frame, 12.5)
    assert result.reason == Reason.OUT_OF_FRAME and result.missing[0] == "nose" and result.mirror is None
    points[0] = SimpleNamespace(x=0.3, y=0.2, presence=1.0, visibility=1.0)
    # Swapped hip labels are reported instead of a misleading view or score.
    points[23], points[24] = points[24], points[23]
    assert engine.detect(frame, 12.7).reason == Reason.UNRELIABLE_TRACKING
    points[23], points[24] = points[24], points[23]
    engine.prepare(b"jpeg", "same", BodyScope(("left_arm", "left_foot")))
    assert prepare.call_args.kwargs["features"] == ("left_elbow", "left_upper_arm")
    points[31].visibility = 0
    result = engine.detect(frame, 13)
    assert (result.score, result.errors) == (None, ())
    points[31].visibility = 1
    assert engine.detect(frame, 14).score == 80
    engine.stopping.set()
    previous = modules["reachy_mini_pose_compare.detector"].PoseDetector.call_count
    engine.prepare(b"jpeg", "same")
    assert modules["reachy_mini_pose_compare.detector"].PoseDetector.call_count == previous
    engine.close()
