"""Depth-free signed angles, their reliability and the movement phrases they produce."""

from types import SimpleNamespace

import pytest

from reachy_mini_pose_coach.coach.sample import Landmark
from reachy_mini_pose_coach.coach.geometry import (
    Angle,
    wrap,
    rounded,
    describe,
    mirrored,
    components,
    action_text,
    frontal_angles,
    labels_reversed,
)


def world(**overrides: tuple[float, float, float]) -> list[SimpleNamespace]:
    """Build a camera-facing standing pose (x to the person's left, y down) with arms and legs hanging."""
    points = dict.fromkeys(range(33), (0.0, 0.0, 0.0))
    points.update(
        {
            11: (0.2, -0.5, 0.0),
            12: (-0.2, -0.5, 0.0),
            13: (0.25, -0.2, 0.0),
            14: (-0.25, -0.2, 0.0),
            23: (0.1, 0.0, 0.0),
            24: (-0.1, 0.0, 0.0),
            25: (0.1, 0.4, 0.0),
            26: (-0.1, 0.4, 0.0),
        }
    )
    for name, value in overrides.items():
        points[int(name[1:])] = value
    return [SimpleNamespace(x=x, y=y, z=z) for _, (x, y, z) in sorted(points.items())]


def test_limb_angles_measure_from_hanging_through_side_to_overhead_on_both_sides():
    """Outward is the person's left for left limbs and their right for right limbs."""
    out = frontal_angles(world(p13=(0.5, -0.5, 0.0), p14=(-0.5, -0.5, 0.0)))
    assert out["left_upper_arm"].degrees == pytest.approx(90) and out["right_upper_arm"].degrees == pytest.approx(90)
    up = frontal_angles(world(p13=(0.2, -0.8, 0.0)))
    assert up["left_upper_arm"].degrees == pytest.approx(180)
    hanging = frontal_angles(world(p13=(0.2, -0.2, 0.0), p25=(0.1, 0.4, 0.0)))
    assert hanging["left_upper_arm"].degrees == pytest.approx(0) and hanging["left_thigh"].degrees == pytest.approx(0)
    knee_out = frontal_angles(world(p26=(-0.4, 0.3, 0.0)))
    assert knee_out["right_thigh"].degrees > 30


def test_torso_lean_and_shoulder_tilt_signs():
    """Positive lean is toward the person's left; positive tilt means the left shoulder is higher."""
    lean = frontal_angles(world(p11=(0.4, -0.5, 0.0), p12=(0.0, -0.5, 0.0)))
    assert lean["torso"].degrees == pytest.approx(21.8, abs=0.1)
    tilt = frontal_angles(world(p11=(0.2, -0.6, 0.0)))
    assert tilt["shoulders"].degrees > 10
    assert frontal_angles(world())["shoulders"].degrees == pytest.approx(0)


@pytest.mark.parametrize(
    "measurement,delta,issue,action",
    [
        ("left_upper_arm", 20, "higher than reference", "lower your left arm about 20°"),
        ("left_upper_arm", -20, "lower than reference", "raise your left arm about 20°"),
        ("right_thigh", 15, "knee further out than reference", "bring your right knee inward about 15°"),
        ("right_thigh", -15, "knee further in than reference", "move your right knee outward about 15°"),
        (
            "torso",
            12,
            "leaning further toward your left than reference",
            "lean your upper body toward your right about 10°",
        ),
        (
            "torso",
            -12,
            "leaning further toward your right than reference",
            "lean your upper body toward your left about 10°",
        ),
        ("shoulders", 11, "left shoulder higher than reference", "lower your left shoulder about 10°"),
        ("shoulders", -11, "right shoulder higher than reference", "lower your right shoulder about 10°"),
        ("left_knee", -30, "more bent than reference", "straighten your left knee about 30°"),
        ("right_elbow", 30, "straighter than reference", "bend your right elbow more about 30°"),
    ],
)
def test_action_sign_table(measurement, delta, issue, action):
    """Each signed difference maps to the movement that reduces it, with the rounded amount."""
    assert describe(measurement, delta)[0] == issue and action_text(measurement, delta) == action


def test_rounding_and_wrapping():
    """Amounts round to five degrees and never below ten; differences wrap into (-180, 180]."""
    assert [rounded(v) for v in (3, 12, 12.5, 23, -23, 178)] == [10, 10, 15, 25, 25, 180]
    assert [wrap(v) for v in (190, -190, 180, -180, 360, 10)] == [-170, 170, 180, 180, 0, 10]


def test_reliability_excludes_foreshortened_overhead_and_crossing_limbs():
    """Only segments seen clearly in the camera's plane, on their outer side, get a direction."""
    reference = frontal_angles(world(p13=(0.5, -0.5, 0.0)))
    side = components(frontal_angles(world(p13=(0.45, -0.3, 0.0))), reference, ["left_upper_arm"])
    assert side[0].reliable and side[0].degrees < 0
    toward_camera = frontal_angles(world(p13=(0.25, -0.45, -0.3)))
    assert not components(toward_camera, reference, ["left_upper_arm"])[0].reliable
    crossing = frontal_angles(world(p13=(0.0, -0.3, 0.0)))
    assert not components(crossing, reference, ["left_upper_arm"])[0].reliable
    assert components(reference, reference, ["left_elbow", "unknown"]) == ()


def test_mirrored_angles_equal_angles_of_mirrored_points():
    """Swapping sides and negating x gives the same angles as the mapping used for opposite-side references."""
    points = world(p13=(0.5, -0.4, 0.0), p14=(-0.3, -0.1, 0.0), p11=(0.25, -0.55, 0.0), p25=(0.3, 0.35, 0.0))
    flipped = [SimpleNamespace(x=-p.x, y=p.y, z=p.z) for p in points]
    for left in range(11, 33, 2):
        flipped[left], flipped[left + 1] = flipped[left + 1], flipped[left]
    expected = frontal_angles(flipped)
    for name, angle in mirrored(frontal_angles(points)).items():
        assert angle.degrees == pytest.approx(expected[name].degrees) and angle.ratio == pytest.approx(
            expected[name].ratio
        )
    assert mirrored({"torso": Angle(10, 1)})["torso"].degrees == -10


def test_left_right_labels_reversed_in_the_image():
    """A camera-facing person shows their left side at larger image x; either swapped pair is flagged."""
    points = [Landmark(0.5, 0.5, 1) for _ in range(33)]
    points[11], points[12], points[23], points[24] = (
        Landmark(0.6, 0.3, 1),
        Landmark(0.4, 0.3, 1),
        Landmark(0.55, 0.6, 1),
        Landmark(0.45, 0.6, 1),
    )
    assert not labels_reversed(points)
    hips = list(points)
    hips[23], hips[24] = hips[24], hips[23]
    assert labels_reversed(hips)
    shoulders = list(points)
    shoulders[11], shoulders[12] = shoulders[12], shoulders[11]
    assert labels_reversed(shoulders)
    uncertain = list(hips)
    uncertain[23] = Landmark(0.45, 0.6, 0.2)
    assert not labels_reversed(uncertain) and not labels_reversed(points[:20])
