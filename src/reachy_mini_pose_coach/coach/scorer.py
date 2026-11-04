"""Elapsed-time smoothing, hysteresis and fresh evidence after a pause."""

import math
from enum import StrEnum
from dataclasses import dataclass

from .sample import PoseSample
from .corrections import Correction, Corrections


class Band(StrEnum):
    """Reference similarity bands, in ascending order."""

    FAR = "far"
    GETTING_CLOSER = "getting_closer"
    CLOSE = "close"
    MATCH = "match"


BANDS = tuple(Band)
EDGES = (50, 70, 85)


class Trend(StrEnum):
    """Change relative to the stored event/speech baseline."""

    STEADY = "steady"
    IMPROVING = "improving"
    WORSENING = "worsening"


def band_for(score: float, previous: Band | None = None, edges: tuple[float, float, float] = EDGES) -> Band:
    """Apply three-point hysteresis to an existing band, or classify an initial score."""
    if previous is None:
        return BANDS[sum(score >= edge for edge in edges)]
    index = BANDS.index(previous)
    while index < 3 and score >= edges[index] + 3:
        index += 1
    while index > 0 and score < edges[index - 1] - 3:
        index -= 1
    return BANDS[index]


@dataclass(frozen=True)
class Baseline:
    """Last observed state at an event or the start of robot speech."""

    score: float
    band: Band
    correction: Correction | None


@dataclass(frozen=True)
class Reading:
    """Score and correction derived together from one completed observation window."""

    sample: PoseSample
    score: float | None = None
    band: Band | None = None
    trend: Trend | None = None
    held_s: float = 0.0
    correction: Correction | None = None
    confirmed: bool = False
    after_pause: bool = False
    second: Correction | None = None
    matched_parts: tuple[str, ...] = ()
    mirrored: bool = False

    def __post_init__(self) -> None:
        """Reject malformed derived readings before they reach model context."""
        if not math.isfinite(self.held_s) or self.held_s < 0:
            raise ValueError("Invalid held duration")
        if self.second is not None and (self.correction is None or self.second.part == self.correction.part):
            raise ValueError("A second correction needs a primary correction on another body part")
        if self.score is None:
            if self.band is not None or self.trend is not None or self.correction is not None or self.matched_parts:
                raise ValueError("Unavailable readings cannot carry score-derived fields")
        elif (
            self.sample.score is None
            or not math.isfinite(self.score)
            or not 0 <= self.score <= 100
            or not isinstance(self.band, Band)
            or not isinstance(self.trend, Trend)
        ):
            raise ValueError("Invalid score, band or trend")


class Scorer:
    """Retain the old EMA as baseline but confirm resumed changes using only new frames."""

    def __init__(self, edges: tuple[float, float, float] = EDGES) -> None:
        """Start with no pose evidence or comparison baseline."""
        self.baseline: Baseline | None = None
        self.edges = edges
        self.reading: Reading | None = None
        self.ema: float | None = None
        self._band: Band | None = None
        self._window_ema: float | None = None
        self._last_at: float | None = None
        self._last_id = -1
        self._held_since = 0.0
        self._candidate: Band | None = None
        self._frames = 0
        self._after_pause = False
        self._corrections = Corrections()

    def checkpoint(self) -> None:
        """Store the current coherent reading before an event or speech."""
        reading = self.reading
        if reading is not None and reading.confirmed and reading.score is not None and reading.band is not None:
            self.baseline = Baseline(reading.score, reading.band, reading.correction)
            self._after_pause = False

    def pause(self) -> None:
        """Keep the baseline/EMA and reset all continuous observation evidence."""
        self._after_pause = True
        self._last_at = None
        self._window_ema = None
        self._frames = 0
        self._candidate = None
        self._corrections.pause()

    def update(self, sample: PoseSample) -> Reading | None:
        """Process a distinct new frame; duplicate or out-of-order frames add no evidence."""
        if sample.sample_id <= self._last_id or (self._last_at is not None and sample.captured_at <= self._last_at):
            return None
        self._last_id = sample.sample_id
        if sample.score is None:
            self.pause()
            self.reading = Reading(sample)
            return self.reading
        previous_at = self._last_at
        dt = sample.captured_at - previous_at if previous_at is not None else 0.0
        self._last_at = sample.captured_at
        if self._window_ema is None:
            self._window_ema = sample.score
        else:
            self._window_ema += (1 - math.exp(-dt)) * (sample.score - self._window_ema)
        candidate = band_for(self._window_ema, self._band, self.edges)
        if candidate != self._candidate or previous_at is None:
            self._candidate = candidate
            self._frames = 1
            self._held_since = sample.captured_at
        else:
            self._frames += 1
        confirmed = not self._after_pause or self._frames >= 3
        if confirmed:
            self.ema = self._window_ema
            self._band = candidate
        correction = self._corrections.update(sample.captured_at, sample.errors, sample.components)
        trend = Trend.STEADY
        if self.baseline is not None:
            difference = self._window_ema - self.baseline.score
            trend = Trend.IMPROVING if difference >= 3 else Trend.WORSENING if difference <= -3 else Trend.STEADY
        self.reading = Reading(
            sample,
            self._window_ema,
            candidate,
            trend,
            sample.captured_at - self._held_since,
            correction,
            confirmed,
            self._after_pause,
            self._corrections.second,
            self._corrections.matched_parts,
        )
        if self.baseline is None and confirmed:
            self.checkpoint()
        return self.reading
