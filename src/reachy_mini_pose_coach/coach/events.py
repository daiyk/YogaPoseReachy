"""Held conditions, event priorities and cooldowns."""

from enum import StrEnum
from dataclasses import dataclass

from .scorer import BANDS, Band, Reading, Baseline


class EventKind(StrEnum):
    """Seven coaching rules, with separate person and framing loss observations."""

    PERSON_LOST = "person_lost"
    FRAMING_BAD = "framing_bad"
    POSE_ACHIEVED = "pose_achieved"
    GOT_WORSE = "got_worse"
    GOT_BETTER = "got_better"
    CORRECTION_CHANGED = "correction_changed"
    STALLED = "stalled"
    MIRRORED = "mirrored"


@dataclass(frozen=True)
class Rule:
    """Required continuous observation, cooldown and dispatch priority."""

    hold_s: float
    cooldown_s: float
    priority: int


RULES = {
    EventKind.PERSON_LOST: Rule(1.5, 10, 1),
    EventKind.FRAMING_BAD: Rule(1.5, 10, 1),
    EventKind.POSE_ACHIEVED: Rule(2, 15, 2),
    EventKind.GOT_WORSE: Rule(1.5, 7, 3),
    EventKind.GOT_BETTER: Rule(1, 5, 4),
    EventKind.CORRECTION_CHANGED: Rule(2, 10, 5),
    EventKind.STALLED: Rule(10, 15, 6),
    EventKind.MIRRORED: Rule(0.5, 20, 2),
}
CHANGE_EVENTS = frozenset((EventKind.GOT_WORSE, EventKind.GOT_BETTER, EventKind.CORRECTION_CHANGED))


@dataclass(frozen=True)
class Event:
    """An observation and its original comparison baseline for dispatch-time validation."""

    kind: EventKind
    emitted_at: float
    reference_id: str
    baseline: Baseline | None
    joint: str | None = None
    stable_score: float | None = None

    def still_true(self, reading: Reading) -> bool:
        """Check the original condition against current evidence, never a changed baseline."""
        if reading.sample.reference_id != self.reference_id:
            return False
        if self.kind == EventKind.MIRRORED:
            return reading.mirrored
        if reading.mirrored:
            # Per-joint progress is meaningless until the person switches sides.
            return False
        if self.kind == EventKind.PERSON_LOST:
            return reading.sample.person_present is False
        if self.kind == EventKind.FRAMING_BAD:
            return reading.sample.person_present is True and reading.sample.framing_ok is False
        if reading.score is None or reading.band is None:
            return False
        if self.kind == EventKind.POSE_ACHIEVED:
            return reading.band == Band.MATCH
        if self.kind == EventKind.STALLED:
            return (
                reading.band != Band.MATCH
                and self.stable_score is not None
                and abs(reading.score - self.stable_score) < 3
            )
        baseline = self.baseline
        if baseline is None:
            return False
        if self.kind == EventKind.GOT_WORSE:
            return BANDS.index(reading.band) < BANDS.index(baseline.band)
        if self.kind == EventKind.GOT_BETTER:
            return BANDS.index(reading.band) > BANDS.index(baseline.band) or (
                baseline.correction is not None
                and baseline.correction.amount_deg >= 10
                and reading.correction is not None
                and reading.correction.joint == baseline.correction.joint
                and reading.correction.amount_deg < 10
            )
        return (
            baseline.correction is not None
            and reading.correction is not None
            and reading.correction.joint != baseline.correction.joint
            and reading.correction.joint == self.joint
            and reading.band != Band.MATCH
        )


@dataclass
class Evidence:
    """Distinct-frame support for one continuous condition."""

    since: float
    frames: int = 1
    joint: str | None = None


class EventDetector:
    """Evaluate event rules once per frame, without crediting gaps or paused inference."""

    def __init__(self, cooldowns: dict[str, float] | None = None) -> None:
        """Start with no holds, announcements or cooldowns."""
        self._evidence: dict[EventKind, Evidence] = {}
        self.cooldowns = dict(cooldowns or {kind.value: rule.cooldown_s for kind, rule in RULES.items()})
        self._last_emitted: dict[EventKind, float] = {}
        self._last_id = -1
        self._achieved = False
        self._stable_low: float | None = None
        self._stable_high: float | None = None

    def pause(self) -> None:
        """Reset holds while preserving cooldowns and an already announced match."""
        self._evidence.clear()
        self._stable_low = self._stable_high = None

    def update(self, reading: Reading, baseline: Baseline | None, now: float) -> tuple[Event, ...]:
        """Emit eligible conditions in priority order and retain their original baselines."""
        if reading.sample.sample_id <= self._last_id:
            return ()
        self._last_id = reading.sample.sample_id
        if reading.score is not None:
            low = min(self._stable_low, reading.score) if self._stable_low is not None else reading.score
            high = max(self._stable_high, reading.score) if self._stable_high is not None else reading.score
            if high - low >= 3:
                self._evidence.pop(EventKind.STALLED, None)
                low = high = reading.score
            self._stable_low, self._stable_high = low, high
            if reading.band != Band.MATCH:
                self._achieved = False
        else:
            self._stable_low = self._stable_high = None
        emitted: list[Event] = []
        for kind, rule in RULES.items():
            event = Event(
                kind,
                now,
                reading.sample.reference_id,
                baseline,
                reading.correction.joint if reading.correction else None,
                self._stable_low,
            )
            if not event.still_true(reading):
                self._evidence.pop(kind, None)
                continue
            evidence = self._evidence.get(kind)
            if evidence is None or (kind == EventKind.CORRECTION_CHANGED and evidence.joint != event.joint):
                evidence = Evidence(reading.sample.captured_at, joint=event.joint)
                self._evidence[kind] = evidence
            else:
                evidence.frames += 1
            held = reading.sample.captured_at - evidence.since
            ready = held >= rule.hold_s - 1e-9
            if kind in CHANGE_EVENTS and reading.after_pause and evidence.frames >= 3:
                ready = True
            if reading.score is not None and not reading.confirmed:
                ready = False
            if kind == EventKind.POSE_ACHIEVED and self._achieved:
                ready = False
            if ready and now - self._last_emitted.get(kind, -float("inf")) >= self.cooldowns[kind.value]:
                emitted.append(event)
                self._last_emitted[kind] = now
                if kind == EventKind.POSE_ACHIEVED:
                    self._achieved = True
        return tuple(emitted)


def event_details(event: Event, reading: Reading | None) -> dict[str, object]:
    """Describe what changed for the AI by comparing the event's baseline with the current reading."""
    details: dict[str, object] = {"kind": event.kind.value}
    if reading is None or reading.score is None or reading.band is None:
        return details
    baseline, current = event.baseline, reading.correction
    if event.kind in (EventKind.GOT_BETTER, EventKind.GOT_WORSE) and baseline is not None:
        if baseline.band != reading.band:
            details.update(from_band=baseline.band.value, to_band=reading.band.value)
        if baseline.correction is not None and current is not None and current.joint == baseline.correction.joint:
            details.update(
                part=current.part,
                measurement=current.joint,
                from_deg=round(baseline.correction.amount_deg, 1),
                to_deg=round(current.amount_deg, 1),
            )
    elif event.kind == EventKind.CORRECTION_CHANGED and current is not None:
        details.update(part=current.part, measurement=current.joint, action=current.action)
    elif event.kind == EventKind.POSE_ACHIEVED:
        details.update(score=round(reading.score, 1))
    return details
