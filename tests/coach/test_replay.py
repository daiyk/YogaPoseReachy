"""Replay synthetic scored sequences; these do not validate real pose accuracy."""

import json
from pathlib import Path

import pytest

from . import sample
from reachy_mini_pose_coach.coach.state import Activity
from reachy_mini_pose_coach.coach.sample import Reason, JointError
from reachy_mini_pose_coach.coach.controller import Coach


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parent / "fixtures").glob("*.jsonl")), ids=lambda path: path.stem
)
def test_synthetic_sequence_replay(path):
    """A full numeric stream produces the specified event sequence without a camera or backend."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    metadata, frames = rows[0], rows[1:]
    assert metadata["provenance"] == "synthetic_numeric_fixture"
    coach, activity = Coach("tree:v1"), Activity(practice_active=True)
    observed = []
    for index, frame in enumerate(frames):
        at = frame["at"]
        current = sample(
            at,
            frame["score"],
            sample_id=index,
            errors=tuple(JointError(joint, error) for joint, error in frame["errors"].items()),
            reason=Reason(frame["reason"]) if frame["reason"] else None,
            person=frame["person_present"],
            framing=frame["framing_ok"],
        )
        observed.extend(event.kind.value for event in coach.update(at, activity, current))
    assert observed == metadata["expected_events"]
