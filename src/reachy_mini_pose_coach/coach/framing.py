"""Presence-based framing decisions, independent of robot and camera I/O."""

import math
from collections import deque
from dataclasses import dataclass

from .sample import Landmark


@dataclass(frozen=True)
class Box:
    """Bounds of present landmarks, including predictions outside the image."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def too_large(self) -> bool:
        """Report when turning cannot fit the person into the camera frame."""
        return self.right - self.left > 0.9 or self.bottom - self.top > 0.9

    @property
    def edge(self) -> bool:
        """Detect clipping risk within five percent of any image boundary."""
        return self.left <= 0.05 or self.top <= 0.05 or self.right >= 0.95 or self.bottom >= 0.95


def landmark_box(points: tuple[Landmark, ...]) -> Box | None:
    """Use presence rather than on-screen visibility so clipped predictions survive."""
    present = [point for point in points if point.presence >= 0.5]
    if len(present) < 2:
        return None
    return Box(
        min(p.x for p in present), min(p.y for p in present), max(p.x for p in present), max(p.y for p in present)
    )


@dataclass(frozen=True)
class FramingCorrection:
    """A bounded-lifetime image-space correction, never an absolute motor command."""

    x: float
    y: float
    captured_at: float
    edge: bool


class Framing:
    """Average one second of observations and require a held deviation except near an edge."""

    def __init__(self) -> None:
        """Start with no history or pending hold."""
        self._window: deque[tuple[float, float, float]] = deque(maxlen=64)
        self._outside_since: float | None = None
        self._direction = (0, 0)

    def reset(self) -> None:
        """Discard framing evidence after a reference change, movement or observation gap."""
        self._window.clear()
        self._outside_since = None
        self._direction = (0, 0)

    def update(self, captured_at: float, points: tuple[Landmark, ...]) -> FramingCorrection | None:
        """Return a correction only from new finite capture times and sufficiently present landmarks."""
        if not math.isfinite(captured_at) or captured_at < 0:
            raise ValueError("Invalid framing capture time")
        if self._window and captured_at <= self._window[-1][0]:
            return None
        if self._window and captured_at - self._window[-1][0] > 1.5:
            self.reset()
        box = landmark_box(points)
        if box is None or box.too_large:
            self.reset()
            return None
        x, y = (box.left + box.right) / 2 - 0.5, (box.top + box.bottom) / 2 - 0.5
        self._window.append((captured_at, x, y))
        while len(self._window) > 1 and self._window[1][0] <= captured_at - 1:
            self._window.popleft()
        entries = tuple(self._window)
        # Integrate elapsed time; a burst of fast frames must not dominate a slow stream.
        weights = [entries[i + 1][0] - max(t, captured_at - 1) for i, (t, _, _) in enumerate(entries[:-1])]
        total = sum(weights)
        if total > 0:
            mean_x = sum(entries[i][1] * weight for i, weight in enumerate(weights)) / total
            mean_y = sum(entries[i][2] * weight for i, weight in enumerate(weights)) / total
        else:
            mean_x, mean_y = x, y
        direction = (int(mean_x > 0.15) - int(mean_x < -0.15), int(mean_y > 0.12) - int(mean_y < -0.12))
        if direction != self._direction:
            self._outside_since = captured_at if direction != (0, 0) else None
            self._direction = direction
        held = self._outside_since is not None and captured_at - self._outside_since >= 1
        if not box.edge and not held:
            return None
        # Near an edge use the latest box; an average can still point toward the old side.
        dx, dy = (x, y) if box.edge else (mean_x if direction[0] else 0.0, mean_y if direction[1] else 0.0)
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return None
        return FramingCorrection(0.8 * dx, 0.8 * dy, captured_at, box.edge)
