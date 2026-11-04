"""Mirrored-pose evidence must be clear and held, and it clears once the person switches sides."""

from reachy_mini_pose_coach.coach.mirror import MirrorDetector
from reachy_mini_pose_coach.coach.sample import MirrorEvidence


MIRRORED = MirrorEvidence(55, 92)
CORRECT = MirrorEvidence(95, 60)


def run(detector, start, stop, evidence, step=0.33):
    """Feed evidence at about 3 FPS and return the last state."""
    at, state = start, detector.mirrored
    while at <= stop + 1e-9:
        state = detector.update(at, evidence)
        at += step
    return state


def test_held_clear_votes_report_mirroring_and_correct_poses_never_do():
    """Two seconds and three frames of clearly better mirrored fit are required."""
    detector = MirrorDetector()
    assert not run(detector, 0, 1.9, MIRRORED)
    assert run(detector, 2.0, 2.3, MIRRORED)
    assert not run(MirrorDetector(), 0, 5, CORRECT)
    assert not run(MirrorDetector(), 0, 5, MirrorEvidence(60, 70))  # too small a margin
    assert not run(MirrorDetector(), 0, 5, MirrorEvidence(40, 65))  # mirrored fit itself too poor


def test_mirroring_clears_after_two_seconds_of_correct_side():
    """Switching sides clears the report; a single noisy frame does not."""
    detector = MirrorDetector()
    run(detector, 0, 2.4, MIRRORED)
    assert detector.update(2.7, CORRECT)
    assert run(detector, 3.0, 4.0, CORRECT)
    assert not run(detector, 4.4, 5.0, CORRECT)


def test_missing_evidence_neither_votes_nor_clears_and_gaps_reset():
    """Frames without usable evidence are ignored; a gap or pause forgets old evidence."""
    detector = MirrorDetector()
    run(detector, 0, 2.4, MIRRORED)
    assert detector.update(3.0, None) and detector.update(3.3, None)
    assert not detector.update(5.0, MIRRORED)  # 2.3 s since the last evidence
    run(detector, 5.3, 7.4, MIRRORED)
    assert detector.mirrored
    detector.pause()
    assert not detector.mirrored and not detector.update(7.6, MIRRORED)
