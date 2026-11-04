"""Held evidence that the person is doing the mirror image of the reference."""

import math

from .sample import MirrorEvidence


# Fixed so a panel settings save can never reset them.
VOTE_SCORE = 70.0
VOTE_MARGIN = 15.0
HOLD_S = 2.0
MIN_FRAMES = 3
CLEAR_S = 2.0
MAX_GAP_S = 1.5


class MirrorDetector:
    """Report mirroring once the mirrored reference clearly fits better for a while; report only."""

    def __init__(self) -> None:
        """Start without evidence."""
        self.mirrored = False
        self._vote_since: float | None = None
        self._votes = 0
        self._last_vote: float | None = None
        self._last_at: float | None = None

    def pause(self) -> None:
        """Forget evidence; paused time is unobserved."""
        self.mirrored = False
        self._vote_since = self._last_vote = self._last_at = None
        self._votes = 0

    def update(self, captured_at: float, evidence: MirrorEvidence | None) -> bool:
        """Consume one frame; frames without evidence neither vote nor clear."""
        if not math.isfinite(captured_at):
            raise ValueError("Mirror evidence time must be finite")
        if self._last_at is not None and captured_at - self._last_at > MAX_GAP_S:
            self.pause()
        if evidence is None:
            return self.mirrored
        self._last_at = captured_at
        vote = evidence.opposite_score >= VOTE_SCORE and evidence.opposite_score - evidence.same_score >= VOTE_MARGIN
        if vote:
            self._last_vote = captured_at
            if self._vote_since is None:
                self._vote_since, self._votes = captured_at, 0
            self._votes += 1
            if captured_at - self._vote_since >= HOLD_S - 1e-9 and self._votes >= MIN_FRAMES:
                self.mirrored = True
        else:
            self._vote_since, self._votes = None, 0
            if self.mirrored and self._last_vote is not None and captured_at - self._last_vote >= CLEAR_S - 1e-9:
                self.mirrored = False
        return self.mirrored
