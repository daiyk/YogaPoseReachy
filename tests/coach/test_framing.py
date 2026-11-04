"""Framing decisions, motor limits and interruption behavior."""

import math
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.rotation import Rotation
from reachy_mini_pose_coach.moves import MovementManager
from reachy_mini_pose_coach.coach.bridge import CoachBridge
from reachy_mini_pose_coach.coach.motion import FramingMove, CameraAngles
from reachy_mini_pose_coach.coach.sample import Landmark
from reachy_mini_pose_coach.coach.framing import Framing, FramingCorrection, landmark_box
from reachy_mini_pose_coach.coach.settings import CoachSettings


def box(left=0.3, right=0.7, top=0.15, bottom=0.85):
    """Make two present extrema without assuming a particular pose."""
    return (Landmark(left, top, 1), Landmark(right, bottom, 1))


def test_safe_arm_changes_do_not_move_camera():
    """A changing arm span within the margins does not produce a correction."""
    framing = Framing()
    for at in np.arange(0, 5, 0.2):
        assert framing.update(float(at), box(0.1 + 0.03 * math.sin(at), 0.9)) is None


def test_offscreen_predictions_use_presence_and_edge_bypasses_hold():
    """Low visibility outside the image must not erase clipping evidence."""
    points = (Landmark(0.5, 0.2, 1), Landmark(1.02, 0.8, 0.8, 0.1), Landmark(-20, 4, 0.1))
    assert landmark_box(points).right == 1.02
    correction = Framing().update(10, points)
    assert correction.edge and correction.x > 0
    assert correction.y == pytest.approx(0)


def test_held_deviation_and_observation_gap():
    """A sideways shift requires a full second, and gaps reset that evidence."""
    framing = Framing()
    for at in (0, 0.3, 0.6, 0.9):
        assert framing.update(at, box(0.6, 0.9)) is None
    correction = framing.update(1.2, box(0.6, 0.9))
    assert not correction.edge and correction.x == pytest.approx(0.2)
    assert framing.update(4, box(0.6, 0.9)) is None
    assert framing.update(4, box(0.6, 0.9)) is None


@pytest.mark.parametrize("points", [box(-0.1, 0.9), box(0.3, 0.7, -0.1, 0.9), (Landmark(0.9, 0.5, 0.2),)])
def test_unfittable_or_missing_person_never_turns(points):
    """Stepping back, rather than turning, is necessary when the person is too large."""
    framing = Framing()
    for at in (0, 1, 2):
        assert framing.update(at, points) is None


@pytest.mark.parametrize(
    "yaw,body,pitch,x,y", [(0, 0, 0, 0.4, 0.4), (24, 0, 24, -0.4, 0.4), (-24, 0, -24, 0.4, -0.4), (84, 60, 0, -0.4, 0)]
)
def test_calibrated_motion_limits_and_endpoint(yaw, body, pitch, x, y):
    """Relative head yaw, body, pitch and each framing step stay bounded."""
    start = (create_head_pose(x=0.02, pitch=pitch, yaw=yaw).astype(np.float32), (0.2, -0.2), math.radians(body))
    move = FramingMove(start, FramingCorrection(x, y, 1, True), CameraAngles(80, 60))
    assert move.duration >= 0.5
    for at in np.linspace(0, move.duration, 21):
        head, antennas, target_body = move.evaluate(float(at))
        roll, target_pitch, target_yaw = Rotation.from_matrix(head[:3, :3]).as_euler("xyz", degrees=True)
        assert abs(target_yaw - math.degrees(target_body)) <= 25.001
        assert abs(target_pitch) <= 25.001 and abs(math.degrees(target_body)) <= 60.001
        assert abs(target_yaw - yaw) <= 8.001
        assert antennas == (0.2, -0.2)
        assert head[0, 3] == pytest.approx(0.02)
    np.testing.assert_allclose(move.evaluate(move.duration)[0], move.target_head, atol=1e-6)


def test_calibration_uses_scaled_sdk_intrinsics():
    """Field-of-view angles come from the real resolution and focal lengths."""
    angles = CameraAngles.from_intrinsics(np.array([[320, 0, 320], [0, 240, 240], [0, 0, 1.0]]), 640, 480)
    assert angles == CameraAngles(90, 90)
    with pytest.raises(ValueError):
        CameraAngles.from_intrinsics(np.eye(3), 640, 480)


def test_listening_halts_move_at_last_command_and_neutral_waits():
    """Interruption holds the actual command; ending practice never stops the motion owner."""
    manager = MovementManager(MagicMock(), autonomous_motion=False)
    now = time.monotonic()
    manager.queue_framing(FramingCorrection(0.25, 0, now, True), CameraAngles(80, 60))
    manager._poll_signals(now)
    manager._manage_move_queue(now)
    move = manager.state.current_move
    assert isinstance(move, FramingMove)
    stopped = move.evaluate(0.2)
    manager._last_commanded_pose = stopped
    manager.set_listening(True)
    manager._poll_signals(now + 0.2)
    assert manager.state.current_move is None and not manager.move_queue
    np.testing.assert_array_equal(manager.state.last_primary_pose[0], stopped[0])
    assert manager.coach_motion_status() == (False, now + 0.2)
    manager.end_practice_motion()
    manager._poll_signals(now + 0.3)
    manager._manage_move_queue(now + 0.3)
    assert manager.state.current_move is None
    manager.set_listening(False)
    manager._poll_signals(now + 0.4)
    manager._manage_move_queue(now + 0.4)
    neutral = manager.state.current_move
    assert neutral.neutral
    np.testing.assert_allclose(neutral.evaluate(neutral.duration)[0], np.eye(4), atol=1e-6)
    assert not manager._stop_event.is_set()


def test_manager_rejects_stale_or_too_frequent_corrections():
    """Queue delay and repeated frames cannot override the two-second motion cooldown."""
    manager = MovementManager(MagicMock(), autonomous_motion=False)
    now = time.monotonic()
    for captured in (now - 2, now):
        manager.queue_framing(FramingCorrection(0.2, 0, captured, True), CameraAngles(80, 60))
        manager._poll_signals(now)
        manager._manage_move_queue(now)
        if captured < now:
            assert manager.state.current_move is None
    assert manager.coach_motion_status()[0]
    manager.clear_move_queue()
    manager._poll_signals(now + 0.6)
    manager.queue_framing(FramingCorrection(0.2, 0, now + 0.7, True), CameraAngles(80, 60))
    manager._poll_signals(now + 0.7)
    assert not manager.move_queue and not manager.coach_motion_status()[0]


def test_bridge_defers_nonedge_motion_until_actual_playback_and_blocks_listening():
    """A generated response alone is insufficient to move the camera."""
    bridge = CoachBridge(AsyncMock(), AsyncMock(), MagicMock())
    bridge.movement = MovementManager(MagicMock(), autonomous_motion=False)
    bridge.configure(CoachSettings(framing_enabled=True, horizontal_fov_deg=80, vertical_fov_deg=60))
    bridge.set_practice(True)
    bridge._framing_pending = FramingCorrection(0.2, 0, time.monotonic(), False)
    bridge._schedule_framing(time.monotonic())
    assert not bridge.movement.coach_motion_status()[0]
    bridge.activity.update(time.monotonic(), listening=True)
    bridge.start_framing_on_playback()
    assert not bridge.movement.coach_motion_status()[0]
    bridge.activity.update(time.monotonic(), listening=False, playback=True)
    bridge.start_framing_on_playback()
    assert bridge.movement.coach_motion_status()[0]
