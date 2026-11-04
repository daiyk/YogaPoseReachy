"""Attach cached pose observations without blocking the realtime receiver."""

import json
import time
import asyncio
import logging
import threading
from typing import TYPE_CHECKING
from collections import deque
from dataclasses import dataclass
from collections.abc import Callable, Awaitable

from websockets.exceptions import ConnectionClosed
from openai.resources.realtime.realtime import AsyncRealtimeConnection

from .rate import inference_rate
from .scope import BodyScope
from .state import Activity
from .turns import Turns, TurnToken, ResponseIntent
from .events import Event, EventKind
from .motion import CameraAngles
from .sample import PoseSample
from .framing import Framing, FramingCorrection
from .settings import CoachSettings
from .snapshot import Snapshot, Observation
from .controller import Coach


if TYPE_CHECKING:
    from ..moves import MovementManager
    from ..pose.service import PoseService


logger = logging.getLogger(__name__)
COACH_TOOLS = frozenset(
    ("get_pose_status", "start_practice", "end_practice", "go_to_sleep", "task_status", "task_cancel")
)


class CoachProtocolError(RuntimeError):
    """A connection cannot safely produce a paired response."""


@dataclass(frozen=True)
class ResponseRequest:
    """Connection-scoped work retained through the serial sender and explicit rejection retries."""

    epoch: int
    revision: int
    initial: ResponseIntent | None = None
    event: Event | None = None
    tool_followup: bool = False


class CoachBridge:
    """Own cached samples, pose activity, context ACK tasks and response identity."""

    def __init__(
        self,
        enqueue: Callable[[ResponseRequest], Awaitable[None]],
        report_error: Callable[[str], Awaitable[None]],
        cue: Callable[[], None],
    ) -> None:
        """Start without a reference or live sample provider."""
        self.coach = Coach("unconfigured")
        self.activity = Activity()
        self.turns = Turns()
        self.revision = 0
        self.failure: str | None = None
        self.connection: AsyncRealtimeConnection | None = None
        self.ready = asyncio.Event()
        self._enqueue = enqueue
        self._report_error = report_error
        self._cue = cue
        self._sample_lock = threading.Lock()
        self._latest: PoseSample | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._ack_waiters: dict[str, asyncio.Future[None]] = {}
        self._event_pending = False
        self._event_sequence = 0
        self.settings = CoachSettings()
        self.movement: MovementManager | None = None
        self.service: PoseService | None = None
        self.framing = Framing()
        self.scope = BodyScope()
        self._framing_pending: FramingCorrection | None = None
        self._framing_sample_id = -1
        self._motion_ended_at = 0.0
        self.recent_events: deque[tuple[str, float]] = deque(maxlen=30)

    def configure(self, settings: CoachSettings) -> None:
        """Apply a validated configuration and reset evidence tied to the old thresholds."""
        self.settings = settings.model_copy(deep=True)
        self.coach.settings = self.settings
        # A bridge built for a reconnect starts "unconfigured" while the shared service already holds a reference.
        self.set_reference(self.service.reference_id if self.service is not None else self.coach.reference_id)
        if self.movement is not None:
            self.movement.clear_move_queue()

    def camera_angles(self) -> CameraAngles | None:
        """Prefer an explicit measurement, otherwise use the connected camera's SDK calibration."""
        if self.settings.horizontal_fov_deg is not None and self.settings.vertical_fov_deg is not None:
            return CameraAngles(self.settings.horizontal_fov_deg, self.settings.vertical_fov_deg)
        return self.service.angles if self.service is not None else None

    def panel_state(self) -> dict[str, object]:
        """Project fresh coach state for the user panel without transport IDs or raw context."""
        now = time.monotonic()
        snapshot: dict[str, object] = json.loads(self.status_text().split("\n", 1)[1])
        keys = (
            "practice_active",
            "status",
            "reason",
            "age_s",
            "score",
            "band",
            "trend",
            "held_s",
            "top_corrections",
            "matched_parts",
            "mirrored",
            "scored_parts",
            "visibility_only_parts",
        )
        return {
            **{key: snapshot[key] for key in keys},
            "activity": self.activity.mode.value,
            "moving": self.activity.moving,
            "connected": self.ready.is_set() and self.failure is None,
            "events": [
                {"kind": kind, "seconds_ago": max(0, round(now - at))} for kind, at in reversed(self.recent_events)
            ],
            "pose": self.service.state()
            if self.service is not None
            else {"status": "unavailable", "error": "Live pose service is unavailable"},
            "settings": self.settings.model_dump(),
            "framing_calibrated": self.camera_angles() is not None,
        }

    def start_framing_on_playback(self) -> None:
        """Run a fresh pending correction when actual playback starts, never during user speech."""
        self._schedule_framing(time.monotonic(), speaking=True)

    def _schedule_framing(self, now: float, *, speaking: bool = False) -> None:
        correction = self._framing_pending
        angles = self.camera_angles()
        if (
            correction is not None
            and self.movement is not None
            and angles is not None
            and self.settings.framing_enabled
            and self.activity.practice_active
            and not self.activity.listening
            and not self.activity.moving
            and now >= self.activity.settle_until
            and 0 <= now - correction.captured_at <= 1
            and (speaking or (correction.edge and self.activity.can_dispatch(now)))
        ):
            self.movement.queue_framing(correction, angles)
            self._framing_pending = None

    def _sync_motion(self, now: float) -> None:
        if self.movement is not None:
            moving, ended_at = self.movement.coach_motion_status()
            self.activity.update(now, moving=moving)
            if ended_at > self._motion_ended_at:
                self._motion_ended_at = ended_at
                self.activity.settle_until = max(self.activity.settle_until, ended_at + 0.5)

    @property
    def user_pending(self) -> bool:
        """Report an unfinished initial user reply, including the pre-transcription interval."""
        pending = self.turns.pending
        return pending is not None and not pending.responded and pending.failure is None

    async def connect(self, connection: AsyncRealtimeConnection) -> None:
        """Start a fresh epoch before accepting callbacks from a new socket."""
        await self.disconnect()
        self.connection = connection
        self.failure = None
        self.ready.clear()

    async def disconnect(self) -> None:
        """Cancel tasks and waiters, invalidate queued requests and detach the old socket."""
        self.connection = None
        self.turns.reconnect()
        self.revision += 1
        self.ready.clear()
        self._cancel_pending()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self.coach.delivery.clear()
        self.activity.update(time.monotonic(), listening=False, generating=False, tool_pending=False)
        if self.service is not None:
            self.service.set_rate(0)
        if self.movement is not None:
            self.movement.clear_move_queue()
        self.framing.reset()
        self._framing_pending = None

    def publish(self, sample: PoseSample) -> None:
        """Accept a completed immutable sample from the pose worker thread."""
        with self._sample_lock:
            if sample.reference_id != self.coach.reference_id:
                raise ValueError("Sample belongs to another reference")
            if self._latest is None or sample.sample_id > self._latest.sample_id:
                self._latest = sample

    def set_reference(self, reference_id: str) -> None:
        """Change reference on the event loop and invalidate advice from the old practice scope."""
        self.invalidate()
        if self.service is not None:
            self.scope = self.service.scope_for(reference_id)
        if self.movement is not None:
            self.movement.clear_move_queue()
        with self._sample_lock:
            self._latest = None
            self.coach.reset(reference_id)
        self.framing.reset()
        self._framing_pending = None
        self._framing_sample_id = -1
        self.recent_events.clear()

    def set_practice(self, active: bool) -> bool:
        """Start/end practice idempotently without interrupting required tool completion."""
        if active == self.activity.practice_active:
            return False
        self.invalidate()
        with self._sample_lock:
            self._latest = None
        self.activity.update(time.monotonic(), practice_active=active)
        self.coach.update(time.monotonic(), self.activity)
        self.framing.reset()
        self._framing_pending = None
        if not active and self.movement is not None:
            self.movement.end_practice_motion()
        return True

    def invalidate(self) -> None:
        """Invalidate pending pose requests on reference/practice changes."""
        self.revision += 1
        self.turns.invalidate()
        self._cancel_pending()
        self.coach.delivery.clear()

    def observation(self, now: float) -> Observation:
        """Read one coherent cached window without camera, inference or network work."""
        self._observe(now)
        return Observation(
            self.coach.reference_id,
            self.coach.reading,
            self.activity.practice_active,
            self.coach.unavailable,
            self.scope,
        )

    def status_text(self) -> str:
        """Return the same bounded status schema used by automatic audio attachments."""
        now = time.monotonic()
        return Snapshot("pose_status", self.turns.epoch, None, self.observation(now)).to_text(now)

    def speech_started(self, audio_item_id: str) -> bool:
        """Give new identified user speech precedence over old waits and queued event turns."""
        previous = self.turns.pending
        token = self.turns.speech_started(self.turns.epoch, audio_item_id)
        if token is None or (previous is not None and token == previous.token):
            return False
        self.revision += 1
        self._cancel_pending()
        self.coach.delivery.clear()
        self.activity.update(time.monotonic(), listening=True)
        return True

    def speech_stopped(self, audio_item_id: str) -> bool:
        """Freeze the cached end-of-turn observation; this signal never requests a response."""
        now = time.monotonic()
        if self.turns.pending is None:
            self.turns.speech_started(self.turns.epoch, audio_item_id)
        pending = self.turns.pending
        if pending is None or pending.token.audio_item_id != audio_item_id or pending.snapshot is not None:
            return False
        self.turns.speech_stopped(TurnToken(self.turns.epoch, audio_item_id), now, self.observation(now))
        self.activity.update(now, listening=False, generating=True)
        return True

    def finalized(self, audio_item_id: str, transcript: str | None) -> None:
        """Launch a cancellable context task after final input, keeping the receiver free for ACKs."""
        now = time.monotonic()
        token = self.turns.finalized(self.turns.epoch, audio_item_id, now, transcript, self.observation(now))
        if token is not None:
            revision = self.revision
            self._spawn(lambda: self._attach_user(token, revision))

    def acknowledge(self, item_id: str) -> None:
        """Resolve only the registered waiter for this exact context item."""
        waiter = self._ack_waiters.get(item_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    def reject(self, event_id: str | None) -> bool:
        """Match a context creation rejection to its client event ID."""
        if event_id is None or not event_id.startswith("ctx_"):
            return False
        waiter = self._ack_waiters.get(event_id[4:])
        if waiter is None or waiter.done():
            return False
        waiter.set_exception(CoachProtocolError("Pose context was rejected by the backend"))
        return True

    def current(self, request: ResponseRequest) -> bool:
        """Reject stale initial/event work while allowing required tool follow-ups in this epoch."""
        if request.epoch != self.turns.epoch or self.failure is not None or not self.ready.is_set():
            return False
        if request.tool_followup:
            return True
        if request.revision != self.revision:
            return False
        if request.initial is not None:
            turn = self.turns.pending
            return (
                turn is not None
                and turn.token == request.initial.token
                and turn.failure is None
                and not turn.responded
            )
        return not self.user_pending and not self.activity.listening

    def claim(self, request: ResponseRequest) -> bool:
        """Revalidate and claim a response immediately before the serial sender writes it."""
        if not self.current(request):
            self.discard(request)
            return False
        now = time.monotonic()
        if request.initial is not None:
            return self.turns.dispatch(request.initial, now)
        if request.event is not None:
            claimed = self.coach.delivery.dispatch(request.event, now, self.coach.reading, self.activity)
            self._event_pending = False
            return claimed
        return True

    def discard(self, request: ResponseRequest) -> None:
        """Release superseded event reservations so later coaching can resume."""
        if request.event is not None:
            self.coach.delivery.discard(request.event)
            if request.epoch == self.turns.epoch and request.revision == self.revision:
                self._event_pending = False

    def retry_rejected(self, request: ResponseRequest) -> None:
        """Allow retry only after an explicit active-response rejection, never unknown acceptance."""
        if request.initial is not None:
            self.turns.retry_rejected(request.initial)
        # Event advice may have expired during the rejected send; it is deliberately not retried.

    async def fail(self, message: str) -> None:
        """Surface protocol failure and close the socket so startup can recover a fresh session."""
        if self.failure is not None:
            return
        self.failure = message
        logger.error("Pose context: %s", message)
        await self._report_error(message)
        if self.connection is not None:
            try:
                await self.connection.close()
            except (ConnectionClosed, OSError) as exc:
                logger.warning("Connection already unavailable during recovery: %s", exc)

    async def tick(self, *, generating: bool, tools_pending: bool, requests_pending: bool = False) -> None:
        """Advance observation and event state without inference, then consider one proactive turn."""
        now = time.monotonic()
        self.activity.update(now, generating=generating or self.user_pending, tool_pending=tools_pending)
        self._observe(now)
        self._schedule_framing(now)
        self._sync_motion(now)
        if self.activity.moving:
            self.coach.update(now, self.activity)
        failure = self.turns.expire(now)
        if failure is not None:
            await self.fail(failure)
            return
        if (
            not self.ready.is_set()
            or self.user_pending
            or self._event_pending
            or self.failure is not None
            or requests_pending
        ):
            return
        event = self.coach.delivery.reserve(now, self.coach.reading, self.activity)
        if event is not None:
            self._event_pending = True
            self._event_sequence += 1
            revision = self.revision
            self._spawn(lambda: self._attach_event(event, revision))

    def _observe(self, now: float) -> None:
        self._sync_motion(now)
        if self.service is not None:
            if self.service.reference_id != self.coach.reference_id:
                self.set_reference(self.service.reference_id)
            incoming = self.service.latest()
            if (
                incoming is not None
                and incoming.reference_id == self.coach.reference_id
                and incoming.completed_at <= now
            ):
                self.publish(incoming)
            present = incoming.person_present if incoming is not None else None
            self.service.set_rate(
                inference_rate(self.activity, now, present, self.settings) if self.ready.is_set() else 0
            )
        with self._sample_lock:
            latest = self._latest
        events = self.coach.update(now, self.activity, latest)
        self.recent_events.extend((event.kind.value, now) for event in events)
        if not self.activity.can_sample(now):
            self.framing.reset()
            self._framing_pending = None
        elif latest is not None and latest.sample_id > self._framing_sample_id and 0 <= now - latest.captured_at <= 1:
            self._framing_sample_id = latest.sample_id
            self._framing_pending = self.framing.update(
                latest.captured_at, self.scope.framing_points(latest.landmarks)
            )
        if self.activity.can_sample(now) and not self.activity.listening:
            if any(event.kind in (EventKind.GOT_BETTER, EventKind.POSE_ACHIEVED) for event in events):
                self._cue()

    def _spawn(self, operation: Callable[[], Awaitable[None]]) -> None:
        async def run() -> None:
            try:
                await operation()
            except (TimeoutError, OSError, RuntimeError, ValueError, ConnectionClosed) as exc:
                await self.fail(str(exc) or "Pose context acknowledgement timed out")

        task = asyncio.create_task(run(), name="pose-context")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _cancel_pending(self) -> None:
        for task in self._tasks:
            task.cancel()
        for waiter in self._ack_waiters.values():
            waiter.cancel()
        self._event_pending = False

    async def _send_context(self, snapshot: Snapshot, *, event: Event | None = None) -> None:
        connection = self.connection
        if connection is None:
            raise CoachProtocolError("Pose context has no active connection")
        text = snapshot.to_text(time.monotonic(), event=event)
        waiter = asyncio.get_running_loop().create_future()
        self._ack_waiters[snapshot.snapshot_id] = waiter
        try:
            async with asyncio.timeout(2.0):
                await connection.conversation.item.create(
                    event_id="ctx_" + snapshot.snapshot_id,
                    item={
                        "id": snapshot.snapshot_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                )
                await waiter
                logger.info(
                    "Pose context accepted: snapshot=%s audio=%s selection=%s",
                    snapshot.snapshot_id,
                    snapshot.audio_item_id,
                    snapshot.selection.value,
                )
        finally:
            self._ack_waiters.pop(snapshot.snapshot_id, None)

    async def _attach_user(self, token: TurnToken, revision: int) -> None:
        await self.ready.wait()
        snapshot = self.turns.submit_context(token, time.monotonic())
        if snapshot is None:
            return
        await self._send_context(snapshot)
        intent = self.turns.acknowledge(token.epoch, snapshot.snapshot_id, time.monotonic())
        if intent is not None:
            await self._enqueue(ResponseRequest(token.epoch, revision, initial=intent))

    async def _attach_event(self, event: Event, revision: int) -> None:
        epoch = self.turns.epoch
        now = time.monotonic()
        snapshot = Snapshot(f"pose_event_{epoch}_{self._event_sequence}", epoch, None, self.observation(now))
        await self._send_context(snapshot, event=event)
        await self._enqueue(ResponseRequest(epoch, revision, event=event))
