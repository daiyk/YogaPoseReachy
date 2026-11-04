"""Timing, smoothing and correction behavior with no hardware dependencies."""

import math
from dataclasses import replace

import pytest

from . import sample
from reachy_mini_pose_coach.coach.rate import inference_rate
from reachy_mini_pose_coach.coach.state import Mode, Activity
from reachy_mini_pose_coach.coach.sample import Reason, Component, JointError
from reachy_mini_pose_coach.coach.scorer import Band, Scorer, band_for
from reachy_mini_pose_coach.coach.controller import Coach
from reachy_mini_pose_coach.coach.corrections import Correction, Corrections


@pytest.mark.parametrize("score,band", [(49, Band.FAR), (50, Band.GETTING_CLOSER), (70, Band.CLOSE), (85, Band.MATCH)])
def test_initial_band(score, band):
    """Initial bands use nominal boundaries."""
    assert band_for(score) == band


@pytest.mark.parametrize(
    "edge,lower,upper",
    [(50, Band.FAR, Band.GETTING_CLOSER), (70, Band.GETTING_CLOSER, Band.CLOSE), (85, Band.CLOSE, Band.MATCH)],
)
def test_hysteresis_both_directions(edge, lower, upper):
    """Each boundary keeps its band until the three-point margin is crossed."""
    assert band_for(edge + 2.99, lower) == lower
    assert band_for(edge + 3, lower) == upper
    assert band_for(edge - 3, upper) == upper
    assert band_for(edge - 3.01, upper) == lower


def test_ema_depends_on_elapsed_time_not_frame_count():
    """Different cadences converge to the same one-second step response."""
    for steps in ((1,), (0.1, 0.3, 0.6, 1)):
        scorer = Scorer()
        scorer.update(sample(score=20))
        for at in steps:
            scorer.update(sample(at, 80))
        assert scorer.ema == pytest.approx(80 - 60 * math.exp(-1))


@pytest.mark.parametrize("spacing", [0.2, 1.0])
def test_resume_uses_three_distinct_new_frames_and_preserves_baseline(spacing):
    """Old EMA values and repeated frames cannot confirm a new pose after speech."""
    scorer = Scorer()
    scorer.update(sample(score=40))
    scorer.checkpoint()
    baseline = scorer.baseline
    scorer.pause()
    assert scorer.ema == 40
    for index in range(3):
        frame = sample(20 + index * spacing, 95)
        result = scorer.update(frame)
        assert result.score == 95
        assert result.confirmed is (index == 2)
        assert scorer.update(frame) is None
    assert scorer.baseline == baseline
    assert scorer.ema == 95
    assert result.held_s == pytest.approx(2 * spacing)


def test_resume_requires_consistent_band():
    """A changing new pose restarts confirmation rather than borrowing the old band."""
    scorer = Scorer()
    scorer.update(sample(score=40))
    scorer.pause()
    assert not scorer.update(sample(2, 95)).confirmed
    assert not scorer.update(sample(3, 0)).confirmed
    assert not scorer.update(sample(4, 0)).confirmed
    assert scorer.update(sample(5, 0)).confirmed


def test_invalid_sample_clears_measurements_and_holds():
    """Occlusion is unknown rather than zero or a fresh copy of the preceding score."""
    scorer = Scorer()
    scorer.update(sample(score=95))
    missing = scorer.update(sample(1, None, reason=Reason.OCCLUDED, framing=False))
    assert missing.score is None and missing.correction is None
    resumed = scorer.update(sample(10, 95))
    assert resumed.held_s == 0 and not resumed.confirmed


def test_correction_switch_margin_and_anatomical_wording():
    """A new joint must exceed the current error by ten degrees, preserving left/right."""
    corrections = Corrections()
    first = corrections.update(0, (JointError("left_elbow", -20), JointError("right_knee", 25)))
    assert first.joint == "right_knee" and first.issue == "straighter than reference"
    corrections.pause()
    second = corrections.update(2, (JointError("left_elbow", -34.9), JointError("right_knee", 25)))
    assert second.joint == "right_knee"
    corrections.pause()
    third = corrections.update(3, (JointError("left_elbow", -35), JointError("right_knee", 25)))
    assert third.joint == "left_elbow" and third.issue == "more bent than reference"


def test_correction_window_is_time_weighted_and_sign_changes_do_not_cancel():
    """Irregular sampling and bend reversals must not fabricate a zero-angle error."""
    corrections = Corrections()
    corrections.update(0, (JointError("left_elbow", -20),))
    corrections.update(0.8, (JointError("left_elbow", 40),))
    result = corrections.update(1, (JointError("left_elbow", 40),))
    assert result.amount_deg == pytest.approx(24)
    assert result.issue == "angle differs from reference"
    assert corrections.update(2, (JointError("left_elbow", 40),)).amount_deg == 40
    assert corrections.update(2.1, ()) is None


def test_unsigned_direction_has_no_invented_movement_hint():
    """Unsigned orientation differences cannot justify left/right/up/down advice."""
    correction = Corrections().update(0, (JointError("left_upper_arm", 30),))
    assert correction.issue == "direction differs from reference" and correction.action is None


def test_measured_direction_gives_movement_with_amount_and_second_correction_on_other_part():
    """A reliable, consistently signed component becomes an action; the second slot uses another part."""
    corrections = Corrections()
    errors = (
        JointError("left_upper_arm", 24),
        JointError("left_elbow", -18),
        JointError("right_knee", 15),
        JointError("torso", 4),
        JointError("shoulders", 3),
    )
    parts = (Component("left_upper_arm", -21, True), Component("torso", 2, True), Component("shoulders", 1, True))
    for at in (0, 0.33, 0.66):
        primary = corrections.update(at, errors, parts)
    assert primary.joint == "left_upper_arm" and primary.part == "left_arm"
    assert primary.issue == "lower than reference" and primary.action == "raise your left arm about 20°"
    assert primary.action_deg == 20 and primary.amount_deg == pytest.approx(24)
    second = corrections.second
    assert second.joint == "right_knee" and second.action == "bend your right knee more about 15°"
    assert corrections.matched_parts == ("torso",)


def test_unreliable_or_inconsistent_direction_keeps_the_correction_without_a_direction():
    """Depth-heavy or sign-flipping evidence never produces a guessed movement."""
    unreliable = Corrections().update(0, (JointError("left_thigh", 30),), (Component("left_thigh", -25, False),))
    assert unreliable.action is None and unreliable.issue == "direction differs from reference"
    flipping = Corrections()
    flipping.update(0, (JointError("left_thigh", 30),), (Component("left_thigh", -25, True),))
    result = flipping.update(0.5, (JointError("left_thigh", 30),), (Component("left_thigh", 25, True),))
    assert result.action is None
    small = Corrections().update(0, (JointError("left_thigh", 30),), (Component("left_thigh", -6, True),))
    assert small.action is None  # the error is mostly forward/backward, which is not coached


def test_second_correction_requires_its_own_part_and_threshold():
    """Two cues never repeat one body part, and matches count only fully matching parts."""
    corrections = Corrections()
    corrections.update(0, (JointError("left_elbow", -30), JointError("left_upper_arm", 25), JointError("torso", 9)))
    assert corrections.second is None and corrections.matched_parts == ("torso",)
    corrections.update(1, (JointError("left_elbow", -30), JointError("right_elbow", -9.9)))
    assert corrections.second is None and corrections.matched_parts == ("right_arm",)
    with pytest.raises(ValueError):
        Correction("left_elbow", "more bent than reference", 20, "straighten your left elbow about 5°", 5)
    with pytest.raises(ValueError):
        Correction("left_elbow", "more bent than reference", 20, "straighten your left elbow")


def test_activity_tracks_physical_drain_and_later_motion_end():
    """Server completion and listening do not bypass physical pause or settling."""
    activity = Activity(practice_active=True)
    activity.update(1, generating=True)
    assert activity.mode == Mode.THINKING and inference_rate(activity, 1, True) == 5.6
    activity.update(2, playback=True, moving=True)
    activity.update(3, generating=False)
    assert activity.mode == Mode.SPEAKING and inference_rate(activity, 3, True) == 0
    activity.update(4, listening=True)
    assert activity.mode == Mode.LISTENING and inference_rate(activity, 4, True) == 0
    activity.update(5, playback=False)
    activity.update(6, moving=False)
    assert inference_rate(activity, 6.49, True) == 0
    assert inference_rate(activity, 6.5, True) == 3
    assert inference_rate(activity, 6.5, False) == 1
    activity.update(7, listening=False, tool_pending=True)
    assert activity.mode == Mode.THINKING and not activity.can_dispatch(7)
    activity.update(8, tool_pending=False)
    assert activity.can_dispatch(8)
    activity.update(9, practice_active=False)
    assert inference_rate(activity, 9, True) == 0


def test_controller_rejects_duplicate_stale_and_wrong_reference_samples():
    """Only new, current-reference, completed frames enter the coach."""
    activity = Activity(practice_active=True)
    coach = Coach("tree:v1")
    coach.update(0, activity, sample())
    baseline = coach.reading
    assert coach.update(0.1, activity, sample()) == ()
    assert coach.reading == baseline
    coach.update(4, activity, sample(1))
    assert coach.reading == baseline
    with pytest.raises(ValueError, match="reference"):
        coach.update(4, activity, sample(4, reference="other:v1"))
    with pytest.raises(ValueError, match="completed"):
        coach.update(4, activity, sample(5))


def test_controller_pause_freezes_baseline_and_excludes_motion_frames():
    """Speech saves the latest baseline and held time restarts after the later settle end."""
    coach, activity = Coach("tree:v1"), Activity(practice_active=True)
    coach.update(0, activity, sample(score=40))
    coach.update(1, activity, sample(1, 60))
    score_before = coach.reading.score
    activity.update(2, playback=True)
    coach.update(2, activity)
    assert coach.scorer.baseline.score == score_before
    activity.update(10, playback=False)
    coach.update(10, activity)
    coach.update(10.6, activity, sample(10.4, 90))
    assert coach.reading.sample.captured_at == 1
    coach.update(10.6, activity, sample(10.6, 90))
    assert coach.reading.held_s == 0
    assert not coach.reading.confirmed


def test_reference_change_and_practice_restart_clear_evidence():
    """A different reference or restarted practice cannot inherit an old score baseline."""
    coach, activity = Coach("tree:v1"), Activity(practice_active=True)
    coach.update(0, activity, sample(score=95))
    coach.reset("warrior:v2")
    assert coach.reading is None and coach.scorer.baseline is None
    coach.update(1, activity, sample(1, 30, reference="warrior:v2"))
    activity.update(2, practice_active=False)
    coach.update(2, activity)
    activity.update(3, practice_active=True)
    coach.update(3, activity, sample(3, 80, sample_id=0, reference="warrior:v2"))
    assert coach.scorer.baseline.score == 80


@pytest.mark.parametrize(
    "changes",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": -1},
        {"score": 101},
        {"captured_at": -1},
        {"completed_at": -1},
        {"observed_at": sample().observed_at.replace(tzinfo=None)},
        {"reference_id": "x" * 129},
        {"reason": "unknown"},
        {"score": None},
        {"person_present": False},
        {"errors": (JointError("left_elbow", 1), JointError("left_elbow", 2))},
    ],
)
def test_sample_boundary_validation(changes):
    """Malformed numeric results fail at the data boundary rather than becoming model context."""
    with pytest.raises(ValueError):
        replace(sample(), **changes)


@pytest.mark.parametrize(
    "joint,degrees", [("unknown", 1), ("left_elbow", float("nan")), ("torso", -1), ("left_knee", 181)]
)
def test_joint_validation(joint, degrees):
    """Error geometry follows the comparison module's signed/unsigned contract."""
    with pytest.raises(ValueError):
        JointError(joint, degrees)
