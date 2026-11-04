"""Event rules and delivery gates exercised with explicit monotonic time."""

from dataclasses import replace

import pytest

from . import sample, reading
from reachy_mini_pose_coach.coach.state import Activity
from reachy_mini_pose_coach.coach.events import RULES, Event, EventKind, EventDetector
from reachy_mini_pose_coach.coach.sample import Reason
from reachy_mini_pose_coach.coach.scorer import Band, Reading, Baseline
from reachy_mini_pose_coach.coach.delivery import Delivery
from reachy_mini_pose_coach.coach.controller import Coach
from reachy_mini_pose_coach.coach.corrections import Correction


BASELINE = Baseline(60, Band.GETTING_CLOSER, Correction("left_elbow", "more bent than reference", 20))


def condition(kind: EventKind, at: float) -> Reading:
    """Provide independent evidence for each event's trigger."""
    if kind == EventKind.PERSON_LOST:
        return reading(at, None, reason=Reason.NO_PERSON)
    if kind == EventKind.FRAMING_BAD:
        return Reading(sample(at, None, reason=Reason.OCCLUDED, framing=False))
    if kind == EventKind.CORRECTION_CHANGED:
        return replace(reading(at), correction=Correction("right_knee", "straighter than reference", 35))
    if kind == EventKind.MIRRORED:
        return replace(reading(at, 60), mirrored=True)
    score = {EventKind.POSE_ACHIEVED: 95, EventKind.GOT_WORSE: 40, EventKind.GOT_BETTER: 80}.get(kind, 60)
    return reading(at, score)


@pytest.mark.parametrize("kind", list(EventKind))
def test_every_event_requires_its_hold_and_honors_cooldown(kind):
    """Each rule waits its full duration and cannot fire again inside its cooldown."""
    detector, rule = EventDetector(), RULES[kind]
    emitted = []
    for index in range(int(rule.hold_s * 4) + 1):
        at = index / 4
        emitted.extend(event for event in detector.update(condition(kind, at), BASELINE, at) if event.kind == kind)
    assert [event.emitted_at for event in emitted] == [rule.hold_s]
    for index in range(1, int(rule.cooldown_s * 4)):
        at = rule.hold_s + index / 4
        assert kind not in [event.kind for event in detector.update(condition(kind, at), BASELINE, at)]
    at = rule.hold_s + rule.cooldown_s
    again = [event.kind for event in detector.update(condition(kind, at), BASELINE, at)]
    assert (kind in again) is (kind != EventKind.POSE_ACHIEVED)


def test_pose_achieved_rearms_only_after_leaving_match():
    """An already achieved pose stays quiet even after fifteen seconds."""
    detector = EventDetector()
    for at in (0, 1, 2, 17, 18):
        result = detector.update(reading(at, 95), BASELINE, at)
    assert EventKind.POSE_ACHIEVED not in [event.kind for event in result]
    detector.update(reading(19, 40), BASELINE, 19)
    detector.update(reading(20, 95), BASELINE, 20)
    result = detector.update(reading(22, 95), BASELINE, 22)
    assert EventKind.POSE_ACHIEVED in [event.kind for event in result]


def test_correction_improvement_requires_same_known_joint():
    """A missing or different correction is not evidence that the original one improved."""
    event = Event(EventKind.GOT_BETTER, 0, "tree:v1", BASELINE)
    assert not event.still_true(reading())
    assert event.still_true(replace(reading(), correction=Correction("left_elbow", "more bent than reference", 9.9)))
    assert not event.still_true(
        replace(reading(), correction=Correction("right_knee", "straighter than reference", 5))
    )


def test_change_after_pause_can_confirm_in_three_frames():
    """Fresh confirmation can finish before a full second, but repeated frame IDs cannot help."""
    detector = EventDetector()
    detector.pause()
    for index, at in enumerate((20, 20.2, 20.4)):
        current = replace(reading(at, 80), after_pause=True, confirmed=index == 2)
        result = detector.update(current, BASELINE, at)
        assert (EventKind.GOT_BETTER in [event.kind for event in result]) is (index == 2)
        assert detector.update(current, BASELINE, at) == ()


def test_stall_hold_restarts_after_pause_and_score_range_change():
    """A long unobserved pause and a three-point score change cannot count as a ten-second stall."""
    detector = EventDetector()
    for at in range(6):
        detector.update(reading(at), BASELINE, at)
    detector.pause()
    for at in range(100, 110):
        assert EventKind.STALLED not in [event.kind for event in detector.update(reading(at), BASELINE, at)]
    result = detector.update(reading(110), BASELINE, 110)
    assert EventKind.STALLED in [event.kind for event in result]
    detector = EventDetector()
    for at in range(20):
        result = detector.update(reading(at, 60 + 3 * (at // 9)), BASELINE, at)
        assert EventKind.STALLED not in [event.kind for event in result]


def test_match_never_stalls_or_changes_correction():
    """Normal small error movement while matching does not produce nagging corrections."""
    detector = EventDetector()
    for at in range(30):
        current = replace(reading(at, 95), correction=Correction("right_knee", "straighter than reference", 12))
        result = detector.update(current, Baseline(95, Band.MATCH, BASELINE.correction), at)
        assert not {EventKind.STALLED, EventKind.CORRECTION_CHANGED} & {event.kind for event in result}


def test_controller_gap_resets_stall_and_frame_holds():
    """A missing stream resets continuous evidence even if no explicit pause signal arrived."""
    coach, activity = Coach("tree:v1"), Activity(practice_active=True)
    for at in range(6):
        coach.update(at, activity, sample(at))
    for at in range(20, 30):
        assert EventKind.STALLED not in [event.kind for event in coach.update(at, activity, sample(at))]
    assert EventKind.STALLED in [event.kind for event in coach.update(30, activity, sample(30))]


def test_lost_and_framing_are_observations_not_worker_failures():
    """Only actual absence/visibility evidence can emit person/framing loss."""
    detector = EventDetector()
    for at in range(20):
        current = Reading(sample(at, None, person=None, framing=None, reason=Reason.WORKER_FAILURE))
        assert detector.update(current, BASELINE, at) == ()


def test_delivery_coalesces_and_prioritizes_while_busy():
    """Only the latest event per kind remains, and match outranks improvement."""
    delivery = Delivery()
    activity = Activity(practice_active=True, generating=True)
    old = Event(EventKind.GOT_BETTER, 0, "tree:v1", BASELINE)
    newer = replace(old, emitted_at=1)
    achieved = Event(EventKind.POSE_ACHIEVED, 1, "tree:v1", BASELINE)
    delivery.offer((newer, old, achieved), activity)
    assert len(delivery.pending) == 2 and delivery.pending[old.kind] == newer
    assert delivery.reserve(1, reading(1, 95), activity) is None
    activity.update(1, generating=False)
    selected = delivery.reserve(1, reading(1, 95), activity)
    assert selected == achieved
    assert delivery.reserve(1, reading(1, 95), activity) is None
    assert delivery.dispatch(selected, 1, reading(1, 95), activity)
    assert not delivery.dispatch(selected, 1, reading(1, 95), activity)


@pytest.mark.parametrize("failure", ["ttl", "condition", "stale", "listening", "reference", "unconfirmed"])
def test_delivery_revalidates_after_context_ack(failure):
    """An event eligible at selection can still be rejected immediately before response dispatch."""
    delivery, activity = Delivery(), Activity(practice_active=True)
    event = Event(EventKind.GOT_BETTER, 0, "tree:v1", BASELINE)
    delivery.offer((event,), activity)
    selected = delivery.reserve(0, reading(0, 80), activity)
    now = 3.01 if failure == "ttl" else 1.01 if failure == "stale" else 0.5
    current = reading(0 if failure == "stale" else now, 40 if failure == "condition" else 80)
    if failure == "listening":
        activity.update(now, listening=True)
    if failure == "reference":
        current = replace(current, sample=replace(current.sample, reference_id="other:v2"))
    if failure == "unconfirmed":
        current = replace(current, confirmed=False)
    assert not delivery.dispatch(selected, now, current, activity)


def test_listening_drops_reserved_and_queued_events():
    """Beginning to listen clears existing advice as well as new events."""
    delivery, activity = Delivery(), Activity(practice_active=True)
    event = Event(EventKind.GOT_BETTER, 0, "tree:v1", BASELINE)
    delivery.offer((event,), activity)
    selected = delivery.reserve(0, reading(0, 80), activity)
    activity.update(0.1, listening=True)
    delivery.offer((event,), activity)
    activity.update(0.2, listening=False)
    assert not delivery.pending
    assert not delivery.dispatch(selected, 0.2, reading(0.2, 80), activity)


def test_proactive_gap_uses_actual_speech_start_and_drops_expired_events():
    """Generation delay cannot shorten the five-second speech gap."""
    delivery, activity = Delivery(), Activity(practice_active=True)
    event = Event(EventKind.GOT_BETTER, 0, "tree:v1", BASELINE)
    delivery.offer((event,), activity)
    selected = delivery.reserve(0, reading(0, 80), activity)
    assert delivery.dispatch(selected, 0, reading(0, 80), activity)
    delivery.mark_spoken(2)
    delivery.offer((replace(event, emitted_at=5),), activity)
    assert delivery.reserve(6.99, reading(6.99, 80), activity) is None
    selected = delivery.reserve(7, reading(7, 80), activity)
    assert selected is not None
    assert delivery.dispatch(selected, 7, reading(7, 80), activity)
    delivery.offer((replace(event, emitted_at=8),), activity)
    assert delivery.reserve(12, reading(12, 80), activity) is None
    assert not delivery.pending


def test_mirrored_reading_suppresses_joint_events_and_reports_once():
    """While mirrored, only the mirrored cue may fire; per-joint progress would mislead."""
    detector = EventDetector()
    kinds = []
    for index in range(12):
        at = index / 4
        current = replace(reading(at, 95), mirrored=True)
        kinds.extend(event.kind for event in detector.update(current, BASELINE, at))
    assert kinds == [EventKind.MIRRORED]
    assert Event(EventKind.MIRRORED, 0, "tree:v1", None).still_true(
        replace(reading(0, None, reason=Reason.UNSUPPORTED_VIEW), mirrored=True)
    )
