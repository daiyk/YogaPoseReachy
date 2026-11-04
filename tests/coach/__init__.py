"""Synthetic numeric observations for offline coach tests; no recorded human poses."""

from datetime import datetime, timezone, timedelta

from reachy_mini_pose_coach.coach.sample import Reason, JointError, PoseSample
from reachy_mini_pose_coach.coach.scorer import Scorer, Reading


def sample(
    at: float = 0,
    score: float | None = 60,
    *,
    sample_id: int | None = None,
    errors: tuple[JointError, ...] = (),
    reason: Reason | None = None,
    person: bool | None = True,
    framing: bool | None = True,
    reference: str = "tree:v1",
) -> PoseSample:
    """Create a synthetic frame with aligned UTC and monotonic capture times."""
    return PoseSample(
        int(round(at * 1000)) if sample_id is None else sample_id,
        reference,
        at,
        at,
        datetime(2026, 10, 26, tzinfo=timezone.utc) + timedelta(seconds=at),
        person,
        framing,
        score,
        errors,
        reason,
    )


def reading(at: float = 0, score: float | None = 60, *, reason: Reason | None = None) -> Reading:
    """Create an independent synthetic reading without a preceding smoothing window."""
    result = Scorer().update(sample(at, score, reason=reason, person=reason != Reason.NO_PERSON))
    assert result is not None
    return result
