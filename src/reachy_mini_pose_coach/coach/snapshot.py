"""Compact immutable observations for an audio turn or status tool."""

import json
import math
from enum import StrEnum
from datetime import timezone
from dataclasses import dataclass

from .scope import DEFAULT_PARTS, BodyScope
from .events import Event, event_details
from .sample import Reason
from .scorer import Band, Reading
from .corrections import THRESHOLD_DEG, Correction


class Selection(StrEnum):
    """The local lifecycle signal at which the observation was frozen."""

    SPEECH_STOP = "speech_stop"
    FINAL_INPUT_FALLBACK = "final_input_fallback"
    CURRENT = "current"


@dataclass(frozen=True)
class Observation:
    """One reference-scoped cached reading and the availability at selection time."""

    reference_id: str
    reading: Reading | None
    practice_active: bool
    unavailable: Reason | None = None
    scope: BodyScope = BodyScope()

    def __post_init__(self) -> None:
        """Enforce one reference revision and an enumerated availability reason."""
        if not self.reference_id or len(self.reference_id.encode()) > 128:
            raise ValueError("Invalid reference identity")
        if self.reading is not None and self.reading.sample.reference_id != self.reference_id:
            raise ValueError("Observation mixes reference revisions")
        if self.unavailable is not None and not isinstance(self.unavailable, Reason):
            raise ValueError("Unknown observation reason")


@dataclass(frozen=True)
class Snapshot:
    """A fixed reading paired locally with an audio item, with age computed at submission."""

    snapshot_id: str
    session_epoch: int
    audio_item_id: str | None
    observation: Observation
    selection: Selection = Selection.CURRENT

    def __post_init__(self) -> None:
        """Bound metadata and validate connection/selection identity."""
        if self.session_epoch < 0 or not isinstance(self.selection, Selection):
            raise ValueError("Invalid snapshot epoch or selection")
        for identity in (self.snapshot_id, self.audio_item_id):
            if identity is not None and (not identity or len(identity.encode()) > 128):
                raise ValueError("Invalid snapshot/audio identity")

    def to_text(self, now: float, max_age_s: float = 1.0, event: Event | None = None) -> str:
        """Serialize at most 2 KiB of JSON; stale/unavailable readings never carry a current score."""
        if not math.isfinite(now) or now < 0 or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Invalid snapshot timing")
        observation = self.observation
        reading = observation.reading
        sample = reading.sample if reading is not None else None
        age = now - sample.captured_at if sample is not None else None
        if sample is not None and (sample.completed_at > now or age is None or age < 0):
            raise ValueError("Cannot submit a future observation")
        reason = observation.unavailable
        if not observation.practice_active:
            reason = Reason.PRACTICE_STOPPED
        elif reading is None:
            reason = reason or Reason.NO_SAMPLE
        elif sample is not None and sample.reason is not None:
            reason = reason or sample.reason
        elif not reading.confirmed:
            reason = reason or Reason.STABILIZING
        if reading is not None and reading.mirrored and reason in (None, Reason.UNSUPPORTED_VIEW):
            reason = Reason.MIRRORED
        status = "unavailable" if reason is not None else "stale" if age is not None and age > max_age_s else "valid"
        valid = status == "valid" and reading is not None
        # A matched pose is the success state, so leftover joint differences are not sent as corrections.
        matched = valid and reading is not None and reading.band == Band.MATCH
        corrections = (reading.correction, reading.second) if valid and reading is not None and not matched else ()
        payload: dict[str, object] = {
            "schema_version": 2,
            "snapshot_id": self.snapshot_id,
            "session_epoch": self.session_epoch,
            "audio_item_id": self.audio_item_id,
            "reference_id": observation.reference_id,
            "scored_parts": [part for part in observation.scope.parts if part in DEFAULT_PARTS],
            "visibility_only_parts": [part for part in observation.scope.parts if part not in DEFAULT_PARTS],
            "missing_landmarks": observation.scope.missing(sample.landmarks)
            if sample is not None and sample.landmarks
            else [],
            "snapshot_selection": self.selection.value,
            "sample_id": sample.sample_id if sample is not None else None,
            "observed_at": sample.observed_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if sample is not None
            else None,
            "age_s": round(age, 3) if age is not None else None,
            "practice_active": observation.practice_active,
            "status": status,
            "reason": "sample_too_old" if status == "stale" else reason.value if reason else None,
            "person_present": sample.person_present if sample is not None else None,
            "framing_ok": sample.framing_ok if sample is not None else None,
            "score": round(reading.score, 2) if valid and reading is not None and reading.score is not None else None,
            "band": reading.band.value if valid and reading is not None and reading.band is not None else None,
            "trend": reading.trend if valid and reading is not None else None,
            "held_s": round(reading.held_s, 3) if valid and reading is not None else None,
            "top_corrections": [
                _correction(correction)
                for correction in corrections
                if correction is not None and correction.amount_deg >= THRESHOLD_DEG
            ],
            "matched_parts": list(reading.matched_parts) if valid and reading is not None else [],
            "mirrored": reason == Reason.MIRRORED,
        }
        if event is not None:
            payload["event"] = event_details(event, reading if valid else None)
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > 2048:
            raise ValueError("Pose snapshot exceeds 2 KiB")
        return ("[pose_event]\n" if event is not None else "[pose_context]\n") + encoded


def _correction(correction: Correction) -> dict[str, object]:
    return {
        "part": correction.part,
        "measurement": correction.joint,
        "issue": correction.issue,
        "action": correction.action,
        "action_deg": correction.action_deg,
        "amount_deg": round(correction.amount_deg, 1),
    }
