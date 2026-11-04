"""Pure initial-response guards; socket sends and legitimate tool follow-ups belong to the bridge."""

import math
from dataclasses import dataclass

from .snapshot import Snapshot, Selection, Observation


@dataclass(frozen=True)
class TurnToken:
    """Connection and audio identity carried through every queued initial response."""

    epoch: int
    audio_item_id: str


@dataclass(frozen=True)
class ResponseIntent:
    """An acknowledged initial response that still needs a sender-time guard."""

    token: TurnToken
    snapshot_id: str


@dataclass
class PendingTurn:
    """Only the current audio turn can submit context or an initial response."""

    token: TurnToken
    snapshot_id: str
    snapshot: Snapshot | None = None
    stopped_at: float | None = None
    finalized: bool = False
    sent_at: float | None = None
    acknowledged: bool = False
    responded: bool = False
    failure: str | None = None


class Turns:
    """Pair each finalized audio turn with one accepted snapshot and one initial response."""

    def __init__(self) -> None:
        """Start an empty connection epoch without transport side effects."""
        self.epoch = 0
        self.pending: PendingTurn | None = None
        self._seen: set[str] = set()
        self._sequence = 0
        self._last_now = 0.0

    def reconnect(self) -> int:
        """Invalidate all old callbacks and identities before using a new connection."""
        self.epoch += 1
        self.pending = None
        self._seen.clear()
        return self.epoch

    def invalidate(self) -> None:
        """Cancel unsent pose requests after reference changes or practice termination."""
        self.pending = None

    def speech_started(self, epoch: int, audio_item_id: str) -> TurnToken | None:
        """Supersede older unsent work, ignoring duplicates and old-connection notifications."""
        if epoch != self.epoch:
            return None
        if not audio_item_id or len(audio_item_id.encode()) > 128:
            raise ValueError("Invalid audio identity")
        if audio_item_id in self._seen:
            return (
                self.pending.token
                if self.pending is not None and self.pending.token.audio_item_id == audio_item_id
                else None
            )
        self._seen.add(audio_item_id)
        self._sequence += 1
        token = TurnToken(epoch, audio_item_id)
        self.pending = PendingTurn(token, f"pose_{epoch}_{self._sequence}")
        return token

    def speech_stopped(self, token: TurnToken, now: float, observation: Observation) -> None:
        """Freeze the speech-stop observation without requesting a response."""
        self._check_time(now)
        turn = self.pending
        if turn is None or turn.token != token or turn.snapshot is not None or turn.failure is not None:
            return
        turn.stopped_at = now
        turn.snapshot = Snapshot(
            turn.snapshot_id, token.epoch, token.audio_item_id, observation, Selection.SPEECH_STOP
        )

    def finalized(
        self,
        epoch: int,
        audio_item_id: str,
        now: float,
        transcript: str | None,
        observation: Observation,
    ) -> TurnToken | None:
        """Accept final input, using a labelled fallback only if speech-stop context is missing."""
        self._check_time(now)
        if epoch != self.epoch:
            return None
        turn = self.pending
        if turn is None or (
            (turn.responded or turn.failure is not None) and turn.token.audio_item_id != audio_item_id
        ):
            token = self.speech_started(epoch, audio_item_id)
            turn = self.pending if token is not None else None
        if turn is None or turn.token.audio_item_id != audio_item_id or turn.finalized or turn.failure is not None:
            return None
        if turn.stopped_at is not None and now - turn.stopped_at > 10:
            turn.failure = "transcription_timeout"
            return None
        turn.finalized = True
        if transcript is None or not transcript.strip():
            turn.failure = "transcription_failed" if transcript is None else "empty_transcription"
            return None
        if turn.snapshot is None:
            turn.snapshot = Snapshot(
                turn.snapshot_id,
                epoch,
                audio_item_id,
                observation,
                Selection.FINAL_INPUT_FALLBACK,
            )
        return turn.token

    def submit_context(self, token: TurnToken, now: float) -> Snapshot | None:
        """Register pending acceptance before the bridge sends the unique context item."""
        self._check_time(now)
        turn = self.pending
        if (
            turn is None
            or turn.token != token
            or not turn.finalized
            or turn.snapshot is None
            or turn.failure is not None
            or turn.sent_at is not None
        ):
            return None
        turn.sent_at = now
        return turn.snapshot

    def acknowledge(self, epoch: int, snapshot_id: str, now: float) -> ResponseIntent | None:
        """Accept only the matching context ACK within two seconds, once."""
        self._check_time(now)
        turn = self.pending
        if (
            turn is None
            or epoch != self.epoch
            or turn.snapshot_id != snapshot_id
            or turn.sent_at is None
            or turn.acknowledged
            or turn.failure is not None
        ):
            return None
        if now - turn.sent_at > 2:
            turn.failure = "context_ack_timeout"
            return None
        turn.acknowledged = True
        return ResponseIntent(turn.token, snapshot_id)

    def dispatch(self, intent: ResponseIntent, now: float) -> bool:
        """Claim the initial response immediately before the sender writes response.create."""
        self._check_time(now)
        turn = self.pending
        if (
            turn is None
            or turn.token != intent.token
            or turn.snapshot_id != intent.snapshot_id
            or not turn.acknowledged
            or not turn.finalized
            or turn.responded
            or turn.failure is not None
        ):
            return False
        turn.responded = True
        return True

    def reject(self, epoch: int, snapshot_id: str) -> bool:
        """Fail closed on an explicit context rejection; the bridge must report and recover."""
        turn = self.pending
        if turn is None or epoch != self.epoch or turn.snapshot_id != snapshot_id or turn.responded:
            return False
        turn.failure = "context_rejected"
        return True

    def retry_rejected(self, intent: ResponseIntent) -> None:
        """Release a claimed initial send only when the backend explicitly rejected it as busy."""
        turn = self.pending
        if turn is not None and turn.token == intent.token and turn.snapshot_id == intent.snapshot_id:
            turn.responded = False

    def expire(self, now: float) -> str | None:
        """Return a connection/input error to surface, without blind retries or an unpaired reply."""
        self._check_time(now)
        turn = self.pending
        if turn is None or turn.failure is not None or turn.responded:
            return None
        if not turn.finalized and turn.stopped_at is not None and now - turn.stopped_at > 10:
            turn.failure = "transcription_timeout"
        elif turn.sent_at is not None and not turn.acknowledged and now - turn.sent_at > 2:
            turn.failure = "context_ack_timeout"
        return turn.failure

    def _check_time(self, now: float) -> None:
        if not math.isfinite(now) or now < self._last_now:
            raise ValueError("Turn time must be finite and monotonic")
        self._last_now = now
