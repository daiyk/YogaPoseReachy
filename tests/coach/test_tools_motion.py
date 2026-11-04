"""Coach tool behavior and camera-preserving movement policy."""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from . import sample
from reachy_mini_pose_coach.moves import MovementManager
from reachy_mini_pose_coach.tools import core_tools
from reachy_mini_pose_coach.coach.bridge import COACH_TOOLS, CoachBridge
from reachy_mini_pose_coach.tools.core_tools import ToolDependencies
from reachy_mini_pose_coach.tools.end_practice import EndPractice
from reachy_mini_pose_coach.tools.start_practice import StartPractice
from reachy_mini_pose_coach.tools.get_pose_status import GetPoseStatus


@pytest.mark.asyncio
async def test_practice_tools_are_idempotent_and_status_matches_attachment_schema():
    """Tools expose grounded cached status and never synthesize missing pose measurements."""
    bridge = CoachBridge(AsyncMock(), AsyncMock(), MagicMock())
    deps = ToolDependencies(MagicMock(), MagicMock(), coach=bridge)
    bridge.set_reference("tree:v1")
    assert (await StartPractice()(deps))["changed"] is True
    assert (await StartPractice()(deps))["changed"] is False
    status = await GetPoseStatus()(deps)
    assert status["status"] == "unavailable" and status["reason"] == "no_sample"
    bridge.publish(sample(time.monotonic(), 75, sample_id=1))
    status = await GetPoseStatus()(deps)
    assert status["status"] == "valid" and status["score"] == 75
    attached = json.loads(bridge.status_text().split("\n", 1)[1])
    assert status.keys() == attached.keys()
    assert (await EndPractice()(deps))["changed"] is True
    assert (await EndPractice()(deps))["changed"] is False
    assert (await GetPoseStatus()(deps))["reason"] == "practice_stopped"
    deps.movement_manager.end_practice_motion.assert_called()
    await bridge.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [GetPoseStatus, StartPractice, EndPractice])
async def test_pose_tools_report_missing_service(tool):
    """Standalone tools fail explicitly when no coach has been injected."""
    assert "error" in await tool()(ToolDependencies(MagicMock(), MagicMock()))


def test_locked_registry_ignores_saved_toolspaces_and_external_autoload(monkeypatch):
    """Production's allowlist loads all required built-ins without reading installed integrations."""
    monkeypatch.setattr(core_tools, "_TOOL_ALLOWLIST", None)
    monkeypatch.setattr(core_tools, "_TOOLS_SIGNATURE", None)
    monkeypatch.setattr(core_tools, "ALL_TOOLS", {})
    monkeypatch.setattr(core_tools.config, "AUTOLOAD_EXTERNAL_TOOLS", True)
    monkeypatch.setattr(
        core_tools, "read_profile_tool_names", MagicMock(side_effect=AssertionError("saved selection read"))
    )
    monkeypatch.setattr(
        core_tools, "_resolve_remote_tools", MagicMock(side_effect=AssertionError("remote integration read"))
    )
    core_tools.lock_tools(COACH_TOOLS)
    assert {spec["name"] for spec in core_tools.get_tool_specs()} == COACH_TOOLS


def test_coach_cue_preserves_head_body_and_coalesces():
    """Antenna feedback is queued on the movement owner and never resets the camera pose."""
    manager = MovementManager(MagicMock(), autonomous_motion=False)
    head = np.eye(4, dtype=np.float32)
    head[0, 3] = 0.03
    manager._last_commanded_pose = (head, (0.2, -0.2), 0.3)
    manager.queue_coach_cue()
    manager.queue_coach_cue()
    manager._poll_signals(time.monotonic())
    assert len(manager.move_queue) == 1
    cue = manager.move_queue[0]
    for at in (0, 0.15, 0.3, 0.6):
        pose, antennas, body = cue.evaluate(at)
        np.testing.assert_array_equal(pose, head)
        assert body == 0.3
        assert max(abs(antennas[0] - 0.2), abs(antennas[1] + 0.2)) <= 0.101
    assert cue.evaluate(0.6)[1] == pytest.approx((0.2, -0.2))
    manager.clear_move_queue()
    manager.set_listening(True)
    manager.queue_coach_cue()
    manager._poll_signals(time.monotonic())
    assert not manager.move_queue


def test_coach_disables_idle_breathing():
    """Long idle periods do not schedule camera-moving breathing in the coach runtime."""
    robot = MagicMock()
    manager = MovementManager(robot, autonomous_motion=False)
    manager._manage_breathing(time.monotonic() + 100)
    assert not manager.move_queue
    robot.get_current_head_pose.assert_not_called()
