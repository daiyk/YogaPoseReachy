"""Validated numeric projection of a completed pose comparison."""

import math
from enum import StrEnum
from datetime import datetime
from dataclasses import dataclass


class Reason(StrEnum):
    """Reasons a pose cannot be used as a current measurement."""

    NO_PERSON = "no_person"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    TOO_CLOSE = "too_close"
    UNSUPPORTED_VIEW = "unsupported_view"
    INVALID_GEOMETRY = "invalid_geometry"
    STABILIZING = "stabilizing"
    WORKER_FAILURE = "worker_failure"
    NO_SAMPLE = "no_sample"
    SETTLING = "settling"
    PAUSED = "paused"
    PRACTICE_STOPPED = "practice_stopped"
    UNRELIABLE_TRACKING = "unreliable_tracking"
    MIRRORED = "mirrored"


JOINTS = frozenset(
    (
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
        "left_upper_arm",
        "right_upper_arm",
        "left_thigh",
        "right_thigh",
        "torso",
        "shoulders",
    )
)
BENDS = frozenset(("left_elbow", "right_elbow", "left_knee", "right_knee"))
DIRECTIONS = JOINTS - BENDS


@dataclass(frozen=True)
class JointError:
    """Signed live-minus-reference bend or unsigned direction error in degrees."""

    joint: str
    degrees: float

    def __post_init__(self) -> None:
        """Reject unknown joints and invalid angle magnitudes."""
        if self.joint not in JOINTS or not math.isfinite(self.degrees) or abs(self.degrees) > 180:
            raise ValueError("Invalid joint error")
        if self.joint not in BENDS and self.degrees < 0:
            raise ValueError("Direction errors must be unsigned")


@dataclass(frozen=True)
class Component:
    """Signed, depth-free live-minus-reference angle of one direction measurement."""

    measurement: str
    degrees: float
    reliable: bool

    def __post_init__(self) -> None:
        """Accept only finite, wrapped angles of known direction measurements."""
        if self.measurement not in DIRECTIONS or not math.isfinite(self.degrees) or abs(self.degrees) > 180:
            raise ValueError("Invalid direction component")


@dataclass(frozen=True)
class MirrorEvidence:
    """Ungated similarity to the reference and to its mirror image for one frame."""

    same_score: float
    opposite_score: float

    def __post_init__(self) -> None:
        """Keep both similarity scores finite and bounded."""
        if not all(math.isfinite(v) and 0 <= v <= 100 for v in (self.same_score, self.opposite_score)):
            raise ValueError("Invalid mirror evidence")


@dataclass(frozen=True)
class Landmark:
    """An image-normalized landmark; off-screen coordinates remain useful for framing."""

    x: float
    y: float
    presence: float
    visibility: float = 1.0

    def __post_init__(self) -> None:
        """Reject nonfinite coordinates and invalid confidence values."""
        if not all(math.isfinite(v) for v in (self.x, self.y, self.presence, self.visibility)):
            raise ValueError("Landmarks must be finite")
        if not 0 <= self.presence <= 1 or not 0 <= self.visibility <= 1:
            raise ValueError("Invalid landmark confidence")


@dataclass(frozen=True)
class PoseSample:
    """One immutable comparison result with its original camera capture times."""

    sample_id: int
    reference_id: str
    captured_at: float
    completed_at: float
    observed_at: datetime
    person_present: bool | None
    framing_ok: bool | None
    score: float | None = None
    errors: tuple[JointError, ...] = ()
    reason: Reason | None = None
    landmarks: tuple[Landmark, ...] = ()
    components: tuple[Component, ...] = ()
    mirror: MirrorEvidence | None = None

    def __post_init__(self) -> None:
        """Validate identities, capture timing and measurement availability together."""
        if self.sample_id < 0 or not self.reference_id or len(self.reference_id.encode()) > 128:
            raise ValueError("Invalid sample/reference identity")
        if not isinstance(self.landmarks, tuple) or len(self.landmarks) > 33:
            raise ValueError("Expected at most 33 immutable landmarks")
        if not all(math.isfinite(t) and t >= 0 for t in (self.captured_at, self.completed_at)):
            raise ValueError("Capture and completion times must be finite and nonnegative")
        if self.completed_at < self.captured_at or self.observed_at.utcoffset() is None:
            raise ValueError("Invalid completion time or timezone-naive observation")
        if not isinstance(self.errors, tuple) or len({error.joint for error in self.errors}) != len(self.errors):
            raise ValueError("Joint errors must be immutable and unique")
        if self.reason is not None and not isinstance(self.reason, Reason):
            raise ValueError("Unknown unavailability reason")
        if not isinstance(self.components, tuple) or len({c.measurement for c in self.components}) != len(
            self.components
        ):
            raise ValueError("Direction components must be immutable and unique")
        if self.mirror is not None and self.reason not in (None, Reason.UNSUPPORTED_VIEW):
            raise ValueError("Mirror evidence needs a usable or view-gated frame")
        if self.score is None:
            if self.reason is None or self.errors or self.components:
                raise ValueError("Unavailable samples need a reason and no corrections")
        elif (
            not math.isfinite(self.score)
            or not 0 <= self.score <= 100
            or self.person_present is not True
            or self.framing_ok is not True
            or self.reason is not None
        ):
            raise ValueError("A score requires finite, visible, valid pose evidence")
