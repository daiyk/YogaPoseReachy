"""Activity precedence and the physical playback/motion settling interval."""

import math
from enum import StrEnum
from dataclasses import dataclass


class Mode(StrEnum):
    """User-facing conversation activity."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class Activity:
    """Independent activity flags; playback ends only on physical drain evidence."""

    practice_active: bool = False
    listening: bool = False
    generating: bool = False
    tool_pending: bool = False
    playback: bool = False
    moving: bool = False
    settle_until: float = 0.0
    _last_update: float = 0.0

    def update(
        self,
        now: float,
        *,
        practice_active: bool | None = None,
        listening: bool | None = None,
        generating: bool | None = None,
        tool_pending: bool | None = None,
        playback: bool | None = None,
        moving: bool | None = None,
    ) -> None:
        """Apply lifecycle signals at a monotonic time without conflating server and speaker completion."""
        if not math.isfinite(now) or now < self._last_update:
            raise ValueError("Activity time must be finite and monotonic")
        self._last_update = now
        if (self.playback and playback is False) or (self.moving and moving is False):
            self.settle_until = now + 0.5
        for name, value in (
            ("practice_active", practice_active),
            ("listening", listening),
            ("generating", generating),
            ("tool_pending", tool_pending),
            ("playback", playback),
            ("moving", moving),
        ):
            if value is not None:
                setattr(self, name, value)

    @property
    def mode(self) -> Mode:
        """Give listening precedence during interruption and real playback precedence over generation."""
        if self.listening:
            return Mode.LISTENING
        if self.playback:
            return Mode.SPEAKING
        if self.generating or self.tool_pending:
            return Mode.THINKING
        return Mode.IDLE

    def can_sample(self, now: float) -> bool:
        """Allow measurements only after practice, playback, motion and settling guards pass."""
        return self.practice_active and not self.playback and not self.moving and now >= self.settle_until

    def can_dispatch(self, now: float) -> bool:
        """Allow autonomous event speech only while idle and settled."""
        return self.can_sample(now) and self.mode == Mode.IDLE
