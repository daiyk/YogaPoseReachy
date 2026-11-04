"""Offline composition of sample validation, scoring, events and delivery."""

import math
from dataclasses import replace

from .state import Mode, Activity
from .events import Event, EventDetector
from .mirror import MirrorDetector
from .sample import Reason, PoseSample
from .scorer import Scorer, Reading
from .delivery import Delivery
from .settings import CoachSettings


class Coach:
    """Consume completed numeric samples; callers own camera, clocks, playback and transport."""

    def __init__(self, reference_id: str, max_gap_s: float = 1.5, settings: CoachSettings | None = None) -> None:
        """Configure the longest continuous observation interval, including one-FPS searching."""
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("Invalid maximum observation gap")
        self.max_gap_s = max_gap_s
        self.settings = settings or CoachSettings()
        self.reset(reference_id)

    def reset(self, reference_id: str) -> None:
        """Start a new practice/reference scope with no inherited baseline, cooldown or queued advice."""
        if not reference_id or len(reference_id.encode()) > 128:
            raise ValueError("Invalid reference identity")
        self.reference_id = reference_id
        self.scorer = Scorer(self.settings.band_edges)
        self.events = EventDetector(self.settings.cooldowns)
        self.delivery = Delivery(self.settings.speech_gap_s)
        self.mirror = MirrorDetector()
        self.reading: Reading | None = None
        self.unavailable: Reason | None = Reason.NO_SAMPLE
        self._paused = False
        self._playback = False
        self._active = False
        self._last_now = 0.0
        self._last_frame = -float("inf")
        self._last_id = -1
        self._accept_after = 0.0

    def update(self, now: float, activity: Activity, sample: PoseSample | None = None) -> tuple[Event, ...]:
        """Apply activity even without frames, then process at most one fresh completed sample."""
        if not math.isfinite(now) or now < self._last_now:
            raise ValueError("Coach time must be finite and monotonic")
        self._last_now = now
        if activity.practice_active and not self._active:
            self.scorer = Scorer(self.settings.band_edges)
            self.events = EventDetector(self.settings.cooldowns)
            self.delivery.clear()
            self.mirror = MirrorDetector()
            self.reading = None
            self.unavailable = Reason.NO_SAMPLE
            self._paused = False
            self._last_id = -1
            self._last_frame = -float("inf")
            self._accept_after = now
        self._active = activity.practice_active
        if activity.playback and not self._playback:
            self.scorer.checkpoint()
        self._playback = activity.playback
        if activity.mode == Mode.LISTENING:
            self.delivery.clear()
        if not activity.can_sample(now):
            self.unavailable = (
                Reason.PRACTICE_STOPPED
                if not activity.practice_active
                else Reason.SETTLING
                if activity.moving or now < activity.settle_until
                else Reason.PAUSED
            )
            self._pause()
            self._accept_after = max(now, activity.settle_until)
            return ()
        if now - self._last_frame > self.max_gap_s and self.reading is not None:
            self._pause()
        if sample is None:
            if self.reading is None:
                self.unavailable = Reason.NO_SAMPLE
            return ()
        if sample.reference_id != self.reference_id:
            raise ValueError("Sample belongs to a different reference revision")
        if sample.completed_at > now:
            raise ValueError("Sample has not completed yet")
        if sample.sample_id <= self._last_id or sample.captured_at <= self._last_frame:
            return ()
        self._last_id = sample.sample_id
        if sample.captured_at < max(self._accept_after, activity.settle_until) or now - sample.captured_at > 1:
            return ()
        reading = self.scorer.update(sample)
        if reading is None:
            return ()
        if self.mirror.update(sample.captured_at, sample.mirror):
            reading = replace(reading, mirrored=True)
        self._last_frame = sample.captured_at
        self._paused = False
        self.reading = reading
        self.unavailable = sample.reason
        emitted = self.events.update(reading, self.scorer.baseline, now)
        if emitted:
            self.scorer.checkpoint()
        self.delivery.offer(emitted, activity)
        return emitted

    def _pause(self) -> None:
        if not self._paused:
            self.scorer.pause()
            self.events.pause()
            self.mirror.pause()
            self.delivery.clear()
            self._paused = True
