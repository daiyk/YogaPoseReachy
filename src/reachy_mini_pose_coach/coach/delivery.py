"""Bounded proactive speech queue; user and tool replies never enter this gate."""

from .state import Mode, Activity
from .events import RULES, Event, EventKind
from .scorer import Reading


class Delivery:
    """Keep the newest event per type and revalidate at reservation and actual dispatch."""

    def __init__(self, speech_gap_s: float = 5.0) -> None:
        """Start an empty queue with no proactive speech cooldown."""
        self.pending: dict[EventKind, Event] = {}
        self.speech_gap_s = speech_gap_s
        self._reserved: Event | None = None
        self._last_dispatch = -float("inf")

    def clear(self) -> None:
        """Invalidate queued and reserved events on listening, practice reset or observation gaps."""
        self.pending.clear()
        self._reserved = None

    def offer(self, events: tuple[Event, ...], activity: Activity) -> None:
        """Coalesce events while busy and discard them while the user is speaking."""
        if activity.mode == Mode.LISTENING or not activity.practice_active:
            self.clear()
        else:
            for event in events:
                previous = self.pending.get(event.kind)
                if previous is None or event.emitted_at >= previous.emitted_at:
                    self.pending[event.kind] = event

    def reserve(self, now: float, reading: Reading | None, activity: Activity) -> Event | None:
        """Select one event for context submission without committing a spoken turn."""
        if activity.mode == Mode.LISTENING or not activity.practice_active:
            self.clear()
        if not activity.can_dispatch(now) or self._reserved is not None:
            return None
        self.pending = {
            kind: event
            for kind, event in self.pending.items()
            if reading is not None
            and (reading.score is None or reading.confirmed)
            and 0 <= now - reading.sample.captured_at <= 1.0
            and 0 <= now - event.emitted_at <= 3.0
            and event.still_true(reading)
        }
        if now - self._last_dispatch < self.speech_gap_s or not self.pending:
            return None
        selected = min(self.pending.values(), key=lambda event: (RULES[event.kind].priority, -event.emitted_at))
        self.pending.pop(selected.kind)
        self._reserved = selected
        return selected

    def dispatch(self, event: Event, now: float, reading: Reading | None, activity: Activity) -> bool:
        """Commit at most one proactive response every five seconds, after its context ACK."""
        if self._reserved != event:
            return False
        self._reserved = None
        valid = (
            activity.can_dispatch(now)
            and now - self._last_dispatch >= self.speech_gap_s
            and 0 <= now - event.emitted_at <= 3
            and reading is not None
            and 0 <= now - reading.sample.captured_at <= 1
            and (reading.score is None or reading.confirmed)
            and event.still_true(reading)
        )
        if valid:
            self._last_dispatch = now
        return valid

    def mark_spoken(self, now: float) -> None:
        """Anchor the proactive gap to actual playback start when generation was delayed."""
        self._last_dispatch = max(self._last_dispatch, now)

    def discard(self, event: Event) -> None:
        """Release a reserved event whose context or queued response was superseded."""
        if self._reserved == event:
            self._reserved = None
