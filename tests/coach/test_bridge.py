"""Exercise the real receiver, context tasks and serial sender over an in-memory socket."""

import json
import time
import base64
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_pose_coach.huggingface_realtime as hf
from . import sample
from reachy_mini_pose_coach.coach.state import Mode
from reachy_mini_pose_coach.coach.bridge import COACH_TOOLS, CoachProtocolError
from reachy_mini_pose_coach.coach.events import Event, EventKind
from reachy_mini_pose_coach.tools.core_tools import ToolDependencies
from reachy_mini_pose_coach.huggingface_realtime import HuggingFaceRealtimeHandler
from reachy_mini_pose_coach.tools.background_tool_manager import ToolState, ToolNotification


def event(kind, **fields):
    """Build one server-shaped event without an SDK transport dependency."""
    return SimpleNamespace(type=kind, **fields)


async def until(predicate):
    """Yield until an observable protocol transition occurs, with a bounded failure."""
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


class Socket:
    """A duplex event stream that records every client write and allows delayed acknowledgements."""

    def __init__(self, *, auto_ack=True, manual=True):
        """Start an empty connection with configurable server acceptance behavior."""
        self.incoming = asyncio.Queue()
        self.items = []
        self.sessions = []
        self.responses = 0
        self.deleted = []
        self.auto_ack = auto_ack
        self.manual = manual
        self.closed = False
        self.reject_busy = False
        self.before_busy = None
        self.session = SimpleNamespace(update=self.update_session)
        self.response = SimpleNamespace(create=self.create_response, cancel=AsyncMock())
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self.create_item, delete=self.delete_item))
        self.input_audio_buffer = SimpleNamespace(append=AsyncMock())

    async def update_session(self, *, session):
        """Echo the effective settings through the same receive loop as other server events."""
        self.sessions.append(session)
        detection = SimpleNamespace(create_response=not self.manual, interrupt_response=True)
        await self.incoming.put(
            event(
                "session.updated",
                session=SimpleNamespace(
                    type="realtime",
                    audio=SimpleNamespace(input=SimpleNamespace(turn_detection=detection)),
                ),
            )
        )

    async def create_item(self, *, item, event_id=None):
        """Record context before optionally acknowledging its exact client-assigned item ID."""
        self.items.append((event_id, item))
        if self.auto_ack and item.get("id"):
            await self.ack(item["id"])

    async def delete_item(self, *, item_id):
        """Record cancellation history cleanup without deleting unrelated input."""
        self.deleted.append(item_id)

    async def ack(self, item_id):
        """Send an explicit matching or unrelated context acknowledgement."""
        await self.incoming.put(
            event(
                "conversation.item.created",
                item=SimpleNamespace(
                    type="message",
                    role="user",
                    id=item_id,
                ),
            )
        )

    async def create_response(self):
        """Complete ordinary replies or explicitly reject the first request as busy."""
        self.responses += 1
        if self.reject_busy:
            self.reject_busy = False
            if self.before_busy is not None:
                await self.before_busy()
            await self.incoming.put(
                event(
                    "error",
                    error=SimpleNamespace(
                        code="conversation_already_has_active_response",
                        message="busy",
                        event_id=None,
                    ),
                )
            )
            await self.incoming.put(
                event("response.done", response=SimpleNamespace(id="busy", status="completed", metadata=None))
            )
            return
        response = SimpleNamespace(id=f"r{self.responses}", status="completed", metadata=None)
        await self.incoming.put(event("response.created", response=response))
        await self.incoming.put(event("response.done", response=response))

    async def close(self):
        """Close the receiver deterministically."""
        if not self.closed:
            self.closed = True
            await self.incoming.put(None)

    async def __aenter__(self):
        """Return the connected test socket."""
        return self

    async def __aexit__(self, *args):
        """Close any remaining receive wait."""
        await self.close()

    def __aiter__(self):
        """Expose the duplex receiver."""
        return self

    async def __anext__(self):
        """Wait for a real queued event, allowing concurrent context tasks to run."""
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item


@pytest.fixture
def make_session(monkeypatch):
    """Prepare a real handler with only external I/O replaced."""
    monkeypatch.setattr(hf, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(hf, "_RESPONSE_REJECTION_RETRY_DELAY", 0.01)
    monkeypatch.setattr(hf, "get_tool_specs", lambda: [])

    def create(socket=None, handler=None):
        socket = socket or Socket()
        handler = handler or HuggingFaceRealtimeHandler(ToolDependencies(MagicMock(), MagicMock()))
        handler._clear_queue = AsyncMock()
        handler.client = SimpleNamespace(realtime=SimpleNamespace(connect=lambda **kwargs: socket))
        return handler, socket, asyncio.create_task(handler._run_realtime_session())

    return create


async def speak(socket, audio_id="a", text="How am I doing?", lifecycle=True):
    """Deliver one finalized user utterance with optional VAD lifecycle events."""
    if lifecycle:
        await socket.incoming.put(event("input_audio_buffer.speech_started", item_id=audio_id))
        await socket.incoming.put(event("input_audio_buffer.speech_stopped", item_id=audio_id))
    await socket.incoming.put(
        event("conversation.item.input_audio_transcription.completed", item_id=audio_id, transcript=text)
    )


@pytest.mark.asyncio
async def test_audio_snapshot_waits_for_exact_ack_and_initial_reply_is_once(make_session):
    """Audio identity, finalization and accepted context precede the only initial response."""
    handler, socket, task = make_session(Socket(auto_ack=False))
    try:
        await until(handler.coach.ready.is_set)
        await speak(socket)
        await until(lambda: len(socket.items) == 1)
        assert socket.responses == 0
        item = socket.items[0][1]
        context = json.loads(item["content"][0]["text"].split("\n", 1)[1])
        assert context["audio_item_id"] == "a" and context["status"] == "unavailable"
        assert context["reason"] == "practice_stopped"
        assert "How am I doing?" not in item["content"][0]["text"]
        await socket.ack("unrelated")
        await asyncio.sleep(0.01)
        assert socket.responses == 0
        await socket.ack(item["id"])
        await until(lambda: socket.responses == 1)
        await socket.ack(item["id"])
        await speak(socket)
        await asyncio.sleep(0.02)
        assert len(socket.items) == 1 and socket.responses == 1
        assert socket.sessions[0]["audio"]["input"]["turn_detection"] == {
            "type": "server_vad",
            "create_response": False,
            "interrupt_response": True,
        }
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True])
async def test_freezes_speech_stop_sample_and_labels_age_at_submission(make_session, stale):
    """A newer post-turn pose never replaces the frozen observation, even if it becomes stale."""
    handler, socket, task = make_session()
    try:
        await until(handler.coach.ready.is_set)
        handler.coach.set_reference("tree:v1")
        handler.coach.set_practice(True)
        at = time.monotonic()
        handler.coach.publish(sample(at, 60, sample_id=1))
        await socket.incoming.put(event("input_audio_buffer.speech_started", item_id="a"))
        await socket.incoming.put(event("input_audio_buffer.speech_stopped", item_id="a"))
        await until(
            lambda: handler.coach.turns.pending is not None and handler.coach.turns.pending.snapshot is not None
        )
        handler.coach.publish(sample(time.monotonic(), 95, sample_id=2))
        if stale:
            await asyncio.sleep(1.05)
        await speak(socket, lifecycle=False)
        await until(lambda: socket.responses == 1)
        context = json.loads(socket.items[0][1]["content"][0]["text"].split("\n", 1)[1])
        assert context["sample_id"] == 1
        assert context["status"] == ("stale" if stale else "valid")
        assert context["score"] == (None if stale else 60)
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
async def test_missing_lifecycle_is_labelled_fallback(make_session):
    """Final transcription alone still gets accepted context, explicitly labelled as a fallback."""
    handler, socket, task = make_session()
    try:
        await until(handler.coach.ready.is_set)
        await speak(socket, lifecycle=False)
        await until(lambda: socket.responses == 1)
        context = json.loads(socket.items[0][1]["content"][0]["text"].split("\n", 1)[1])
        assert context["snapshot_selection"] == "final_input_fallback"
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["ack", "queued", "retry"])
async def test_new_speech_invalidates_old_context_wait_and_sender_request(make_session, stage):
    """The same guard applies while waiting for context, a previous response, or a busy retry."""
    handler, socket, task = make_session(Socket(auto_ack=stage != "ack"))
    try:
        await until(handler.coach.ready.is_set)
        if stage == "queued":
            handler._response_done_event.clear()
        if stage == "retry":
            socket.reject_busy = True

            async def interrupt_before_rejection():
                await socket.incoming.put(event("input_audio_buffer.speech_started", item_id="new"))

            socket.before_busy = interrupt_before_rejection
        await speak(socket, "old")
        await until(lambda: len(socket.items) == 1)
        old_id = socket.items[0][1]["id"]
        if stage == "retry":
            await until(lambda: handler.coach.turns.pending.token.audio_item_id == "new")
        else:
            await socket.incoming.put(event("input_audio_buffer.speech_started", item_id="new"))
            await until(lambda: handler.coach.turns.pending.token.audio_item_id == "new")
        handler._response_done_event.set()
        await socket.ack(old_id)
        await asyncio.sleep(0.03)
        assert socket.responses == (1 if stage == "retry" else 0)
        socket.auto_ack = True
        await speak(socket, "new", lifecycle=False)
        await until(lambda: socket.responses == (2 if stage == "retry" else 1))
        assert len(socket.items) == 2
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
async def test_busy_rejection_retries_current_initial_response(make_session):
    """An explicitly rejected request can retry once without sending context twice."""
    handler, socket, task = make_session()
    try:
        await until(handler.coach.ready.is_set)
        socket.reject_busy = True
        await speak(socket)
        await until(lambda: socket.responses == 2)
        await asyncio.sleep(0.02)
        assert len(socket.items) == 1 and socket.responses == 2
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["settings", "rejected", "timeout"])
async def test_bad_settings_or_unaccepted_context_fail_closed(make_session, failure):
    """Protocol failures close the session instead of generating an unpaired reply."""
    handler, socket, task = make_session(Socket(auto_ack=False, manual=failure != "settings"))
    if failure != "settings":
        await until(handler.coach.ready.is_set)
        await speak(socket)
        await until(lambda: len(socket.items) == 1)
        if failure == "rejected":
            await socket.incoming.put(
                event(
                    "error",
                    error=SimpleNamespace(
                        code="invalid_request",
                        message="rejected",
                        event_id=socket.items[0][0],
                    ),
                )
            )
    with pytest.raises(CoachProtocolError):
        await asyncio.wait_for(task, timeout=3)
    assert socket.closed and socket.responses == 0
    assert handler.coach.failure is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_empty_or_failed_transcription_creates_no_context_or_reply(make_session, failed):
    """Unusable audio input does not request normal coaching speech."""
    handler, socket, task = make_session()
    try:
        await until(handler.coach.ready.is_set)
        if failed:
            await socket.incoming.put(event("conversation.item.input_audio_transcription.failed", item_id="a"))
        else:
            await speak(socket, text=" ")
        await until(
            lambda: handler.coach.turns.pending is not None and handler.coach.turns.pending.failure is not None
        )
        assert not socket.items and socket.responses == 0
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
async def test_reconnect_reapplies_manual_settings_and_drops_old_ack(make_session):
    """Old context IDs cannot cause a response on a replacement connection."""
    handler, first, task = make_session(Socket(auto_ack=False))
    await until(handler.coach.ready.is_set)
    await speak(first)
    await until(lambda: bool(first.items))
    old_id = first.items[0][1]["id"]
    await first.close()
    await task
    handler, second, task = make_session(Socket(auto_ack=False), handler)
    try:
        await until(handler.coach.ready.is_set)
        await second.ack(old_id)
        await speak(second)
        await until(lambda: bool(second.items))
        assert second.items[0][1]["id"] != old_id and second.responses == 0
        await second.ack(second.items[0][1]["id"])
        await until(lambda: second.responses == 1)
        assert all(not s.sessions[0]["audio"]["input"]["turn_detection"]["create_response"] for s in (first, second))
    finally:
        await second.close()
        await task


@pytest.mark.asyncio
async def test_new_final_preserves_tool_result_and_batches_it_with_initial_reply(make_session, monkeypatch):
    """A new transcript must not clear the preceding tool call or generate a duplicate tool reply."""
    handler, socket, task = make_session()
    monkeypatch.setattr(
        type(handler.tool_manager), "start_tool", AsyncMock(return_value=SimpleNamespace(tool_id="status-c1"))
    )
    try:
        await until(handler.coach.ready.is_set)
        await socket.incoming.put(
            event("response.function_call_arguments.done", name="get_pose_status", arguments="{}", call_id="c1")
        )
        await until(lambda: "c1" in handler._in_flight_tool_calls)
        await speak(socket)
        await until(lambda: bool(socket.items))
        assert "c1" in handler._in_flight_tool_calls and socket.responses == 0
        await handler._handle_tool_result(
            ToolNotification(
                id="c1",
                tool_name="get_pose_status",
                is_idle_tool_call=False,
                status=ToolState.COMPLETED,
                result={"status": "unavailable"},
            )
        )
        await until(lambda: socket.responses == 1)
        await asyncio.sleep(0.1)
        assert socket.responses == 1
        assert any(item.get("call_id") == "c1" for _, item in socket.items)
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
async def test_tool_followup_after_initial_answer_remains_allowed(make_session):
    """The one-initial-response guard does not suppress required function-result follow-ups."""
    handler, socket, task = make_session()
    try:
        await until(handler.coach.ready.is_set)
        await speak(socket)
        await until(lambda: socket.responses == 1 and handler._response_done_event.is_set())
        handler._in_flight_tool_calls.add("c1")
        await handler._handle_tool_result(
            ToolNotification(
                id="c1",
                tool_name="get_pose_status",
                is_idle_tool_call=False,
                status=ToolState.COMPLETED,
                result={"status": "unavailable"},
            )
        )
        await until(lambda: socket.responses == 2)
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
async def test_generation_and_tool_only_responses_do_not_mark_physical_speech(make_session):
    """Playback begins at the local push and remains active through queued/draining audio."""
    handler, socket, task = make_session()
    try:
        await until(handler.coach.ready.is_set)
        await socket.incoming.put(event("response.created", response=SimpleNamespace(id="r", status="in_progress")))
        await until(lambda: handler.coach.activity.mode == Mode.THINKING)
        assert not handler.coach.activity.playback
        handler.playback_started()
        handler.playback_until = time.monotonic() + 1
        await socket.incoming.put(
            event("response.done", response=SimpleNamespace(id="r", status="completed", metadata=None))
        )
        await until(handler._response_done_event.is_set)
        await asyncio.sleep(0.06)
        assert handler.coach.activity.mode == Mode.SPEAKING
        handler.playback_until = 0
        handler.playback_pending_frames = 1
        await asyncio.sleep(0.06)
        assert handler.coach.activity.playback
        handler.playback_pending_frames = 0
        await until(lambda: not handler.coach.activity.playback)
        assert handler.coach.activity.settle_until > time.monotonic()
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
async def test_disconnect_flushes_pending_physical_audio(make_session):
    """A replacement connection cannot inherit audio queued by the disconnected session."""
    handler, socket, task = make_session()
    await until(handler.coach.ready.is_set)
    handler.playback_started()
    handler.playback_pending_frames = 2
    await socket.close()
    await task
    handler._clear_queue.assert_awaited_once()


def test_session_tools_exclude_camera_motion_and_external_tools():
    """Even a stale saved tool selection cannot expose images or arbitrary motion to the coach."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(MagicMock(), MagicMock()))
    specs = [
        {"type": "function", "name": name, "description": name, "parameters": {}}
        for name in (*COACH_TOOLS, "camera", "dance", "external__tool")
    ]
    effective = handler._get_session_config(specs)["tools"]
    assert {tool["name"] for tool in effective} == COACH_TOOLS


def test_empty_tool_configuration_omits_tool_choice():
    """The hosted model rejects tool_choice when no tools are supplied."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(MagicMock(), MagicMock()))
    session = handler._get_session_config([])
    assert session["tools"] == [] and "tool_choice" not in session


@pytest.mark.asyncio
async def test_late_created_response_is_cancelled_without_deleting_new_audio(make_session):
    """Interruption between the send and response.created cannot play old audio or erase the new turn."""
    handler, socket, task = make_session()

    async def late_response():
        socket.responses += 1
        await socket.incoming.put(event("input_audio_buffer.speech_started", item_id="new"))
        await socket.ack("new")
        await socket.incoming.put(event("response.created", response=SimpleNamespace(id="late", status="in_progress")))
        await socket.incoming.put(
            event("response.output_audio.delta", response_id="late", delta=base64.b64encode(b"\0\0").decode())
        )
        await socket.incoming.put(
            event("response.done", response=SimpleNamespace(id="late", status="cancelled", metadata=None))
        )

    socket.response.create = late_response
    try:
        await until(handler.coach.ready.is_set)
        await socket.ack("old")
        await speak(socket, "old")
        await until(lambda: "old" in socket.deleted)
        assert "new" not in socket.deleted
        assert socket.items[0][1]["id"] in socket.deleted
        socket.response.cancel.assert_awaited_once_with(response_id="late")
        assert handler.playback_pending_frames == 0
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
async def test_late_stop_and_final_do_not_release_newer_listening_turn(make_session):
    """Old lifecycle callbacks cannot turn off listening for a newer identified utterance."""
    handler, socket, task = make_session(Socket(auto_ack=False))
    try:
        await until(handler.coach.ready.is_set)
        await speak(socket, "old")
        await until(lambda: bool(socket.items))
        await socket.incoming.put(event("input_audio_buffer.speech_started", item_id="new"))
        await until(lambda: handler.coach.turns.pending.token.audio_item_id == "new")
        await socket.incoming.put(event("input_audio_buffer.speech_stopped", item_id="old"))
        await speak(socket, "old", lifecycle=False)
        await asyncio.sleep(0.02)
        assert handler._server_speech and handler.coach.activity.mode == Mode.LISTENING
        assert socket.responses == 0
    finally:
        await socket.close()
        await task


@pytest.mark.asyncio
async def test_proactive_event_ack_dispatch_and_listening_discard(make_session):
    """Event context uses the same coordinator and a dropped reservation cannot stall later events."""
    handler, socket, task = make_session(Socket(auto_ack=False))
    try:
        await until(handler.coach.ready.is_set)
        handler.coach.set_reference("tree:v1")
        handler.coach.set_practice(True)
        handler.coach.publish(sample(time.monotonic(), 95, sample_id=1))
        await until(lambda: handler.coach.coach.reading is not None)
        now = time.monotonic()
        handler.coach.coach.delivery.offer(
            (Event(EventKind.POSE_ACHIEVED, now, "tree:v1", None),), handler.coach.activity
        )
        await until(lambda: bool(socket.items))
        first = socket.items[0][1]
        assert first["content"][0]["text"].startswith("[pose_event]")
        handler.coach.activity.update(time.monotonic(), listening=True)
        await socket.ack(first["id"])
        await asyncio.sleep(0.02)
        assert socket.responses == 0
        handler.coach.activity.update(time.monotonic(), listening=False)
        now = time.monotonic()
        handler.coach.coach.delivery.offer(
            (Event(EventKind.POSE_ACHIEVED, now, "tree:v1", None),), handler.coach.activity
        )
        await until(lambda: len(socket.items) == 2)
        await socket.ack(socket.items[1][1]["id"])
        await until(lambda: socket.responses == 1)
    finally:
        await socket.close()
        await task
