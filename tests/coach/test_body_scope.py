"""Selected measurements, visibility, framing and feedback share one fixed scope."""

import copy
import json
import importlib
from types import SimpleNamespace
from dataclasses import replace

import pytest

from . import sample
from reachy_mini_pose_coach.coach.scope import PARTS, DEFAULT_PARTS, BodyScope
from reachy_mini_pose_coach.coach.state import Activity
from reachy_mini_pose_coach.coach.sample import Landmark, JointError
from reachy_mini_pose_coach.coach.framing import Framing
from reachy_mini_pose_coach.coach.snapshot import Snapshot, Observation
from reachy_mini_pose_coach.coach.controller import Coach


@pytest.fixture
def comparison(monkeypatch):
    """Use the bundled production geometry without loading a detector or model."""
    return importlib.import_module("reachy_mini_pose_coach.pose.runtime.comparison")


def pose():
    """Build a finite asymmetric pose with all image points comfortably inside view."""
    image = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0, presence=1.0) for _ in range(33)]
    world = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(33)]
    coordinates = {
        11: (0.25, -0.6),
        12: (-0.25, -0.6),
        13: (0.65, -0.6),
        14: (-0.65, -0.6),
        15: (1.05, -0.6),
        16: (-1.05, -0.6),
        23: (0.2, 0),
        24: (-0.2, 0),
        25: (0.6, 0.05),
        26: (-0.45, 0.4),
        27: (0.6, 0.6),
        28: (-0.7, 0.8),
    }
    for i, (x, y) in coordinates.items():
        world[i] = SimpleNamespace(x=x, y=y, z=0.0)
    return image, world


@pytest.mark.parametrize("parts", [(), ("head",), ("left_hand", "right_foot"), ("unknown",), ("torso", "torso")])
def test_selection_requires_real_scoring_parts(parts):
    """Invalid selections cannot create empty or accidentally weighted averages."""
    with pytest.raises(ValueError):
        BodyScope(parts)


@pytest.mark.parametrize("part", DEFAULT_PARTS)
@pytest.mark.parametrize("side", ["same", "opposite"])
def test_partial_reference_and_live_only_need_selected_measurements(comparison, part, side):
    """Invisible or invalid excluded limbs never reach score, errors or corrections."""
    scope = BodyScope((part,))
    image, world = pose()
    for i in range(33):
        if i not in scope.required:
            image[i].visibility = 0
            world[i].x = float("nan")
    settings = {**comparison.SETTINGS, "minimum_frames": 1, "minimum_span_seconds": 0}
    reference = comparison.prepare_reference(image, world, side, settings=settings, features=scope.features)
    if side == "opposite":
        original_image, original_world = copy.deepcopy(image), copy.deepcopy(world)
        for i in comparison.KEY_JOINTS:
            source = comparison.mirror_joint(i)
            image[i] = original_image[source]
            xyz = original_world[source]
            world[i] = SimpleNamespace(x=-xyz.x, y=xyz.y, z=xyz.z)
    engine = comparison.ComparisonEngine(reference)
    result = engine.evaluate(image, world, 1)
    assert result.score == pytest.approx(100)
    assert set(result.deltas_deg) == set(scope.mapped(side).features)
    for i in scope.mapped(side).required:
        old = image[i].visibility
        image[i].visibility = 0
        assert engine.evaluate(image, world, 2).score is None
        image[i].visibility = old


def test_excluded_measurements_do_not_change_events_or_snapshot(comparison):
    """Large unselected leg errors never cause lower scores or leg coaching."""
    image, world = pose()
    scope = BodyScope(("left_arm",))
    reference = comparison.prepare_reference(
        image,
        world,
        features=scope.features,
        settings={**comparison.SETTINGS, "minimum_frames": 1, "minimum_span_seconds": 0},
    )
    engine = comparison.ComparisonEngine(reference)
    coach = Coach("partial:v1")
    activity = Activity(practice_active=True)
    kinds = set()
    for frame in range(80):
        world[27].x = frame * 50
        result = engine.evaluate(image, world, frame / 3)
        errors = tuple(JointError(name, value) for name, value in result.deltas_deg.items())
        kinds.update(
            e.kind.value
            for e in coach.update(
                frame / 3, activity, sample(frame / 3, result.score, errors=errors, reference="partial:v1")
            )
        )
    payload = json.loads(
        Snapshot("test", 0, None, Observation("partial:v1", coach.reading, True, scope=scope))
        .to_text(79 / 3)
        .split("\n")[1]
    )
    assert payload["score"] == 100 and payload["scored_parts"] == ["left_arm"]
    assert payload["top_corrections"] == []
    assert "got_worse" not in kinds and "correction_changed" not in kinds


@pytest.mark.parametrize("part,index", [("left_hand", 19), ("right_hand", 20), ("left_foot", 31), ("right_foot", 32)])
def test_optional_edges_only_affect_visibility_and_framing(part, index):
    """Turning off an extremity ignores its clipped point without dropping wrist/ankle scores."""
    basic = BodyScope()
    extended = BodyScope((*DEFAULT_PARTS, part))
    points = tuple(Landmark(0.5, 0.5, 1, 1) for _ in range(33))
    clipped = list(points)
    clipped[index] = Landmark(1.1, 0.5, 1, 1)
    clipped = tuple(clipped)
    assert basic.features == extended.features
    assert not basic.missing(clipped) and extended.missing(clipped)
    assert Framing().update(1, basic.framing_points(clipped)) is None
    assert Framing().update(1, extended.framing_points(clipped)) is not None
    assert {15, 16, 27, 28, 23, 24}.issubset(basic.required)


def test_head_is_always_required_in_view_and_framed():
    """MediaPipe's left/right depends on the face, so the head counts even when it is not selected."""
    scope = BodyScope()
    points = [Landmark(0.5, 0.5, 1, 1) for _ in range(33)]
    assert scope.head_in_view(tuple(points)) and not scope.missing(tuple(points))
    points[0] = Landmark(0.5, -0.1, 1, 1)
    assert not scope.head_in_view(tuple(points)) and scope.missing(tuple(points)) == ("nose",)
    assert Framing().update(1, scope.framing_points(tuple(points))) is not None
    points[0] = Landmark(0.5, 0.5, 0.2, 1)
    assert not scope.head_in_view(tuple(points))
    # Selecting the head does not list the nose twice.
    assert BodyScope((*DEFAULT_PARTS, "head")).missing(tuple(points)).count("nose") == 1


def test_snapshot_with_every_missing_point_stays_bounded():
    """Maximum scope metadata remains inside the existing conversation payload limit."""
    from reachy_mini_pose_coach.coach.scorer import Scorer

    scope = BodyScope(tuple(PARTS))
    frame = replace(sample(), landmarks=tuple(Landmark(0.5, 0.5, 0, 0) for _ in range(33)))
    reading = Scorer().update(frame)
    text = Snapshot("x" * 128, 1, "y" * 128, Observation("tree:v1", reading, True, scope=scope)).to_text(0)
    assert len(text.split("\n")[1].encode()) <= 2048
