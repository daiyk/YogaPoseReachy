"""Panel controls use the same coach state as spoken tools."""

import json
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from . import sample
from reachy_mini_pose_coach.coach.rate import inference_rate
from reachy_mini_pose_coach.coach.scope import BodyScope
from reachy_mini_pose_coach.coach.state import Activity
from reachy_mini_pose_coach.coach.bridge import CoachBridge
from reachy_mini_pose_coach.coach.routes import register_coach_routes
from reachy_mini_pose_coach.coach.settings import CoachSettings, load_coach_settings, save_coach_settings


def test_settings_accept_up_to_30_fps():
    """Every observation rate can be raised to the Lite camera's 30 FPS."""
    settings = CoachSettings(idle_fps=30, listening_fps=30, thinking_fps=30, searching_fps=30)
    assert (settings.idle_fps, settings.searching_fps) == (30, 30)


@pytest.mark.parametrize(
    "fields",
    [
        {"band_edges": [70, 50, 85]},
        {"band_edges": [50, 52, 85]},
        {"idle_fps": 31},
        {"searching_fps": 30.5},
        {"speech_gap_s": 1},
        {"cooldowns": {}},
        {"horizontal_fov_deg": 80},
        {"thinking_fps": float("nan")},
        {"extra_field": 1},
    ],
)
def test_settings_reject_unsafe_or_incomplete_values(fields):
    """Threshold edits cannot disable the runtime's fixed safety bounds."""
    with pytest.raises(ValidationError):
        CoachSettings(**fields)


def test_settings_persistence_and_configured_rates(tmp_path):
    """Saved values survive restart and all activity modes use the selected rates."""
    settings = CoachSettings(idle_fps=2, listening_fps=1, thinking_fps=2.5, searching_fps=0.5)
    save_coach_settings(tmp_path, settings)
    assert load_coach_settings(tmp_path) == settings
    activity = Activity(practice_active=True)
    assert inference_rate(activity, 0, True, settings) == 2
    activity.update(1, listening=True)
    assert inference_rate(activity, 1, True, settings) == 1
    activity.update(2, listening=False, generating=True)
    assert inference_rate(activity, 2, True, settings) == 2.5
    assert inference_rate(activity, 2, False, settings) == 0.5
    activity.update(3, playback=True)
    assert inference_rate(activity, 3, True, settings) == 0
    (tmp_path / "coach-settings.json").write_text("broken")
    assert load_coach_settings(tmp_path) == CoachSettings()


@pytest.mark.asyncio
async def test_routes_share_state_validate_and_persist(tmp_path):
    """Panel practice and settings changes reach the actual bridge on its owning loop."""
    bridge = CoachBridge(AsyncMock(), AsyncMock(), MagicMock())
    bridge.ready.set()
    app = FastAPI()
    register_coach_routes(app, lambda: bridge, asyncio.get_running_loop, tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://coach", headers={"X-Reachy-Coach": "1"}
    ) as client:
        assert (await client.get("/api/coach/status", headers={"Origin": "http://elsewhere"})).status_code == 403
        assert (await client.get("/api/coach/status", headers={"X-Reachy-Coach": ""})).status_code == 403
        response = await client.post("/api/coach/practice", json={"active": True})
        assert response.status_code == 200 and bridge.activity.practice_active
        assert response.json()["connected"] and response.json()["score"] is None
        assert (await client.post("/api/coach/practice", json={"active": "yes"})).status_code == 422
        assert (await client.put("/api/coach/settings", json={"speech_gap_s": 0})).status_code == 422
        assert not (tmp_path / "coach-settings.json").exists()
        response = await client.put("/api/coach/settings", json={"idle_fps": 2})
        assert response.status_code == 200 and bridge.settings.idle_fps == 2
        assert load_coach_settings(tmp_path).idle_fps == 2
        assert (await client.get("/api/coach/preview")).status_code == 204
        assert (await client.post("/api/coach/reference/select", json={"selection": "builtin"})).status_code == 503
        assert (
            await client.post("/api/coach/reference/upload", content=b"x" * (5 * 1024 * 1024 + 1))
        ).status_code == 413
        await client.post("/api/coach/practice", json={"active": False})
        assert not bridge.activity.practice_active


@pytest.mark.asyncio
async def test_reference_route_invalidates_old_evidence_without_preview_polling(tmp_path):
    """Reference mutation resets coaching, while ordinary status reads never request a JPEG."""
    bridge = CoachBridge(AsyncMock(), AsyncMock(), MagicMock())
    service = MagicMock()
    service.reference_id = "unconfigured"
    service.angles = None
    service.latest.return_value = None
    service.state.return_value = {"status": "ready"}
    bridge.service = service

    def select(selection, side, content, parts):
        if content == b"invalid":
            raise ValueError("Invalid image")
        service.reference_id = "new-reference"
        return service.reference_id

    service.select_reference.side_effect = select
    service.scope_for.return_value = BodyScope()
    app = FastAPI()
    register_coach_routes(app, lambda: bridge, asyncio.get_running_loop, tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://coach", headers={"X-Reachy-Coach": "1"}
    ) as client:
        before = bridge.revision
        response = await client.post("/api/coach/reference/upload", content=b"valid")
        assert response.status_code == 200
        assert bridge.coach.reference_id == "new-reference" and bridge.revision > before
        assert (await client.post("/api/coach/reference/upload", content=b"invalid")).status_code == 400
        response = await client.post(
            "/api/coach/reference/select", json={"selection": "builtin", "parts": ["left_arm"]}
        )
        assert response.status_code == 200
        service.select_reference.assert_called_with("builtin", "same", None, ["left_arm"])
        assert (
            await client.post("/api/coach/reference/select", json={"selection": "builtin", "parts": "torso"})
        ).status_code == 422
        await client.get("/api/coach/status")
        service.preview.assert_not_called()
        service.preview.return_value = {"jpeg": "test", "points": [], "age_s": 0}
        assert (await client.get("/api/coach/preview")).json()["jpeg"] == "test"


@pytest.mark.asyncio
async def test_stopped_loop_rejects_settings_before_writing(tmp_path):
    """An unavailable conversation cannot leave settings changed only on disk."""
    app = FastAPI()
    register_coach_routes(app, lambda: None, lambda: None, tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://coach", headers={"X-Reachy-Coach": "1"}
    ) as client:
        assert (await client.put("/api/coach/settings", json={"idle_fps": 2})).status_code == 503
    assert not (tmp_path / "coach-settings.json").exists()


@pytest.mark.asyncio
async def test_sample_completed_during_observation_is_deferred():
    """A worker completion after the tick's clock read must wait for the next tick."""
    bridge = CoachBridge(AsyncMock(), AsyncMock(), MagicMock())
    bridge.set_reference("tree:v1")
    bridge.set_practice(True)
    now = time.monotonic()
    service = MagicMock()
    service.reference_id = "tree:v1"
    service.latest.return_value = sample(now + 0.01, 80, sample_id=1)
    bridge.service = service
    bridge.observation(now)
    assert bridge.coach.reading is None
    bridge.observation(now + 0.02)
    assert bridge.coach.reading.score == 80


def test_settings_saved_before_new_event_kinds_keep_their_values(tmp_path):
    """An older file without the mirrored cooldown still loads and keeps every saved choice."""
    old = CoachSettings(framing_enabled=True, speech_gap_s=8).model_dump()
    del old["cooldowns"]["mirrored"]
    old["cooldowns"]["got_better"] = 9.0
    (tmp_path / "coach-settings.json").write_text(json.dumps(old), encoding="utf-8")
    loaded = load_coach_settings(tmp_path)
    assert loaded.framing_enabled and loaded.speech_gap_s == 8 and loaded.cooldowns["got_better"] == 9
    assert loaded.cooldowns["mirrored"] == 20
