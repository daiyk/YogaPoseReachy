"""Audio identity, context acceptance and immutable snapshot contracts."""

import json
from dataclasses import replace

import pytest

from . import reading
from reachy_mini_pose_coach.coach.turns import Turns, TurnToken, ResponseIntent
from reachy_mini_pose_coach.coach.sample import Reason
from reachy_mini_pose_coach.coach.scorer import Band
from reachy_mini_pose_coach.coach.snapshot import Snapshot, Selection, Observation
from reachy_mini_pose_coach.coach.corrections import Correction


def observation(at=0, score=60):
    """Create a fresh synthetic practice observation."""
    return Observation("tree:v1", reading(at, score), True)


def payload(snapshot, at):
    """Inspect the actual text that a bridge would put into an input_text item."""
    text = snapshot.to_text(at)
    assert text.startswith("[pose_context]\n")
    return json.loads(text.split("\n", 1)[1])


def ready_turn(turns, at=0, audio_id="audio_1"):
    """Finalize, submit and acknowledge one synthetic user turn without dispatching it."""
    token = turns.speech_started(turns.epoch, audio_id)
    turns.speech_stopped(token, at, observation(at))
    assert turns.finalized(turns.epoch, audio_id, at, "How am I doing?", observation(at)) == token
    context = turns.submit_context(token, at)
    intent = turns.acknowledge(turns.epoch, context.snapshot_id, at)
    assert intent is not None
    return token, context, intent


def test_snapshot_recomputes_age_from_capture_and_stale_has_no_measurements():
    """Inference completion and serialization time cannot make an old camera frame fresh."""
    current = reading(10, 75)
    current = replace(current, sample=replace(current.sample, completed_at=10.8))
    snapshot = Snapshot("pose_1", 2, "audio_1", Observation("tree:v1", current, True))
    fresh = payload(snapshot, 10.9)
    assert fresh["age_s"] == 0.9 and fresh["score"] == 75 and fresh["status"] == "valid"
    stale = payload(snapshot, 11.1)
    assert stale["sample_id"] == fresh["sample_id"]
    assert stale["status"] == "stale" and stale["score"] is None and stale["top_corrections"] == []
    assert stale["reason"] == "sample_too_old"
    assert stale["observed_at"] == "2026-10-26T00:00:10Z"


@pytest.mark.parametrize("reason", list(Reason))
def test_unavailable_context_has_explicit_reason_and_no_fabricated_score(reason):
    """Every unavailable state remains useful context without invented pose estimates."""
    snapshot = Snapshot("pose_1", 0, "audio_1", replace(observation(), unavailable=reason))
    result = payload(snapshot, 0)
    assert result["status"] == "unavailable" and result["reason"] == reason.value
    assert result["score"] is None and result["band"] is None and result["trend"] is None
    assert result["held_s"] is None and not result["top_corrections"]


def test_no_person_missing_sample_stopped_practice_and_unconfirmed():
    """Absence, no estimate and a new unconfirmed segment are distinct unavailable states."""
    absent = Observation("tree:v1", reading(0, None, reason=Reason.NO_PERSON), True)
    result = payload(Snapshot("p", 0, None, absent), 0)
    assert result["person_present"] is False and result["score"] is None
    empty = Observation("tree:v1", None, True)
    assert payload(Snapshot("p", 0, None, empty), 1)["reason"] == "no_sample"
    stopped = replace(observation(), practice_active=False)
    assert payload(Snapshot("p", 0, None, stopped), 0)["reason"] == "practice_stopped"
    unconfirmed = replace(observation(), reading=replace(reading(), confirmed=False))
    assert payload(Snapshot("p", 0, None, unconfirmed), 0)["reason"] == "stabilizing"


def test_payload_is_bounded_and_contains_at_most_two_corrections_no_raw_pose():
    """The context contract has a fixed whitelist, two grounded corrections and matched parts."""
    first = Correction("left_elbow", "more bent than reference", 23, "straighten your left elbow about 25°", 25)
    second = Correction("right_thigh", "direction differs from reference", 12)
    current = replace(reading(), correction=first, second=second, matched_parts=("torso", "right_arm"))
    context = Snapshot("p" * 128, 2, "a" * 128, Observation("tree:v1", current, True))
    text = context.to_text(0)
    assert len(text.split("\n", 1)[1].encode()) <= 2048
    result = payload(context, 0)
    assert [c["measurement"] for c in result["top_corrections"]] == ["left_elbow", "right_thigh"]
    assert result["top_corrections"][0] == {
        "part": "left_arm",
        "measurement": "left_elbow",
        "issue": "more bent than reference",
        "action": "straighten your left elbow about 25°",
        "action_deg": 25,
        "amount_deg": 23.0,
    }
    assert result["top_corrections"][1]["action"] is None and result["matched_parts"] == ["torso", "right_arm"]
    assert result["mirrored"] is False
    assert not {"landmarks", "image", "frame", "settings", "transcript"} & result.keys()
    assert result["schema_version"] == 2 and result["session_epoch"] == 2


def test_matched_pose_sends_no_corrections():
    """Once the band is match, leftover joint differences no longer ask the person to move."""
    leftover = Correction("left_elbow", "more bent than reference", 12, "straighten your left elbow about 10°", 10)
    matched = replace(reading(score=95), correction=leftover, matched_parts=("torso",))
    result = payload(Snapshot("p", 0, None, Observation("tree:v1", matched, True)), 0)
    assert result["band"] == "match" and result["top_corrections"] == [] and result["matched_parts"] == ["torso"]


def test_mirrored_observation_hides_joint_advice_and_events_carry_details():
    """A mirrored attempt reports only that it is mirrored; event detail stays inside the size cap."""
    from reachy_mini_pose_coach.coach.events import Event, EventKind
    from reachy_mini_pose_coach.coach.scorer import Baseline

    correction = Correction("left_elbow", "more bent than reference", 23, "straighten your left elbow about 25°", 25)
    mirrored = replace(reading(), correction=correction, mirrored=True)
    result = payload(Snapshot("p", 0, None, Observation("tree:v1", mirrored, True)), 0)
    assert result["mirrored"] is True and result["reason"] == "mirrored" and result["status"] == "unavailable"
    assert result["top_corrections"] == [] and result["score"] is None
    view = replace(observation(), unavailable=Reason.UNSUPPORTED_VIEW)
    view = replace(view, reading=replace(view.reading, mirrored=True))
    assert payload(Snapshot("p", 0, None, view), 0)["reason"] == "mirrored"
    before = Correction("left_elbow", "more bent than reference", 30, "straighten your left elbow about 30°", 30)
    improved = replace(reading(at=0, score=80), correction=replace(before, amount_deg=8, action=None, action_deg=None))
    event = Event(EventKind.GOT_BETTER, 0, "tree:v1", Baseline(60, Band.GETTING_CLOSER, before))
    text = Snapshot("e" * 128, 0, None, Observation("tree:v1", improved, True)).to_text(0, event=event)
    assert text.startswith("[pose_event]\n") and len(text.split("\n", 1)[1].encode()) <= 2048
    detail = json.loads(text.split("\n", 1)[1])["event"]
    assert detail == {
        "kind": "got_better",
        "from_band": "getting_closer",
        "to_band": "close",
        "part": "left_arm",
        "measurement": "left_elbow",
        "from_deg": 30,
        "to_deg": 8,
    }


def test_snapshot_rejects_bad_identity_reference_and_future_reading():
    """Invalid metadata and mixed reference observations cannot enter model context."""
    with pytest.raises(ValueError):
        Snapshot("x" * 129, 0, "a", observation())
    with pytest.raises(ValueError):
        Observation("other:v2", reading(), True)
    with pytest.raises(ValueError):
        Snapshot("p", 0, "a", observation(2)).to_text(1)
    with pytest.raises(ValueError):
        replace(reading(), band="other")
    with pytest.raises(ValueError):
        replace(reading(), trend="maybe")
    with pytest.raises(ValueError):
        replace(reading(), held_s=float("nan"))
    with pytest.raises(ValueError):
        Snapshot("p", 0, "a", observation()).to_text(0, max_age_s=float("inf"))


def test_stop_then_final_then_matching_ack_then_one_initial_response():
    """Neither speech stop nor final input alone permits an initial response."""
    turns = Turns()
    token = turns.speech_started(0, "audio_1")
    frozen = observation(1, 40)
    turns.speech_stopped(token, 1, frozen)
    assert turns.submit_context(token, 1) is None
    assert turns.acknowledge(0, "pose_0_1", 1) is None
    assert not turns.dispatch(ResponseIntent(token, "pose_0_1"), 1)
    assert turns.finalized(0, "audio_1", 1.3, "How am I doing?", observation(1.3, 95)) == token
    context = turns.submit_context(token, 1.3)
    assert context.observation == frozen and payload(context, 1.3)["score"] == 40
    assert context.selection == Selection.SPEECH_STOP
    assert turns.acknowledge(0, "unrelated_item", 1.4) is None
    assert not turns.dispatch(ResponseIntent(token, context.snapshot_id), 1.4)
    intent = turns.acknowledge(0, context.snapshot_id, 1.5)
    assert turns.dispatch(intent, 1.5)
    assert not turns.dispatch(intent, 1.6)


def test_duplicate_lifecycle_notifications_cannot_refreeze_or_resend():
    """Duplicate starts/stops/finals/ACKs never create a second initial response."""
    turns = Turns()
    token, context, intent = ready_turn(turns)
    assert turns.speech_started(0, "audio_1") == token
    turns.speech_stopped(token, 0.1, observation(0.1, 95))
    assert turns.finalized(0, "audio_1", 0.1, "repeat", observation(0.1)) is None
    assert turns.submit_context(token, 0.1) is None
    assert turns.acknowledge(0, context.snapshot_id, 0.1) is None
    assert turns.dispatch(intent, 0.1)
    assert turns.pending.snapshot == context


@pytest.mark.parametrize("start_seen", [False, True])
def test_missing_lifecycle_uses_labelled_final_input_fallback(start_seen):
    """Missing speech-stop events do not silently borrow a different utterance's pose."""
    turns = Turns()
    if start_seen:
        turns.speech_started(0, "a")
    token = turns.finalized(0, "a", 1, "hello", observation(1))
    context = turns.submit_context(token, 1)
    assert context.selection == Selection.FINAL_INPUT_FALLBACK
    assert payload(context, 1)["audio_item_id"] == "a"


@pytest.mark.parametrize("stage", ["waiting_ack", "queued_response"])
def test_new_speech_invalidates_old_unsent_work(stage):
    """User interruption wins both during context acceptance and after queueing a reply."""
    turns = Turns()
    token, context, intent = ready_turn(turns)
    if stage == "waiting_ack":
        turns = Turns()
        token = turns.speech_started(0, "audio_1")
        turns.finalized(0, "audio_1", 0, "old", observation())
        context = turns.submit_context(token, 0)
    newer = turns.speech_started(0, "audio_2")
    assert turns.acknowledge(0, context.snapshot_id, 0.1) is None
    assert not turns.dispatch(intent, 0.1)
    assert turns.finalized(0, "audio_1", 0.1, "late", observation()) is None
    assert turns.speech_started(0, "audio_1") is None
    assert turns.pending.token == newer
    assert turns.finalized(0, "audio_2", 0.2, "new", observation(0.2)) == newer


def test_late_unknown_final_cannot_steal_newer_listening_turn():
    """An unrelated final notification does not replace the actively identified utterance."""
    turns = Turns()
    token = turns.speech_started(0, "new")
    assert turns.finalized(0, "unknown_old", 1, "late", observation(1)) is None
    assert turns.pending.token == token


def test_reconnect_clears_waits_and_allows_reused_audio_ids_without_old_callbacks():
    """Connection epochs prevent old context ACKs or queued responses from crossing sessions."""
    turns = Turns()
    old_token, old_context, old_intent = ready_turn(turns)
    assert turns.reconnect() == 1
    assert turns.acknowledge(0, old_context.snapshot_id, 1) is None
    assert turns.finalized(0, "audio_1", 1, "late", observation(1)) is None
    assert not turns.dispatch(old_intent, 1)
    token, context, intent = ready_turn(turns, 1)
    assert token != old_token and context.snapshot_id != old_context.snapshot_id
    assert turns.dispatch(intent, 1)


@pytest.mark.parametrize(
    "cause", ["rejected", "ack_timeout", "transcription_timeout", "empty", "failed", "practice_end"]
)
def test_failures_never_fall_back_to_unpaired_response(cause):
    """Unknown context acceptance and unusable transcripts fail closed with an explicit reason."""
    turns = Turns()
    token = turns.speech_started(0, "a")
    turns.speech_stopped(token, 0, observation())
    if cause == "transcription_timeout":
        assert turns.expire(10.01) == "transcription_timeout"
        assert turns.finalized(0, "a", 10.02, "late", observation(10)) is None
        return
    transcript = "" if cause == "empty" else None if cause == "failed" else "hi"
    turns.finalized(0, "a", 0, transcript, observation())
    context = turns.submit_context(token, 0)
    if cause in ("empty", "failed"):
        assert context is None and turns.pending.failure is not None
        return
    if cause == "rejected":
        assert turns.reject(0, context.snapshot_id)
    elif cause == "ack_timeout":
        assert turns.expire(2.01) == "context_ack_timeout"
    else:
        turns.invalidate()
    assert turns.acknowledge(0, context.snapshot_id, 2.02) is None
    assert not turns.dispatch(ResponseIntent(token, context.snapshot_id), 2.02)


def test_timeout_is_checked_even_when_timer_callback_is_late():
    """A late final or ACK cannot win a race against the explicit timeout callback."""
    turns = Turns()
    token = turns.speech_started(0, "a")
    turns.speech_stopped(token, 0, observation())
    assert turns.finalized(0, "a", 11, "late", observation(11)) is None
    token = turns.speech_started(0, "b")
    turns.finalized(0, "b", 12, "hi", observation(12))
    context = turns.submit_context(token, 12)
    assert turns.acknowledge(0, context.snapshot_id, 14.01) is None


def test_next_user_answer_has_no_proactive_five_second_delay():
    """Only proactive delivery has a speech-rate gate; distinct user turns can answer immediately."""
    turns = Turns()
    _, _, first = ready_turn(turns)
    assert turns.dispatch(first, 0)
    _, _, second = ready_turn(turns, 0.2, "audio_2")
    assert turns.dispatch(second, 0.2)
    assert not turns.dispatch(ResponseIntent(TurnToken(0, "audio_1"), first.snapshot_id), 0.2)


def test_snapshot_band_is_an_enum_value():
    """The model sees the documented band spelling, not an arbitrary Python object."""
    result = payload(Snapshot("p", 0, "a", observation(0, 75)), 0)
    assert result["band"] == Band.CLOSE.value
