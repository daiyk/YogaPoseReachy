"""Time-weighted joint errors and stable, evidence-based correction descriptions."""

import math
from collections import deque
from dataclasses import dataclass

from .scope import PARTS, PART_OF
from .sample import BENDS, JOINTS, Component, JointError
from .geometry import SIGNED_ISSUES, rounded, describe, action_text


# Differences at or below this match the reference and raise no cue (comparison full-score error).
THRESHOLD_DEG = 10.0
GENERIC_ISSUES = frozenset(("angle differs from reference", "direction differs from reference"))


@dataclass(frozen=True)
class Correction:
    """One anatomical mismatch and, only when its sign is measured, the movement that reduces it."""

    joint: str
    issue: str
    amount_deg: float
    action: str | None = None
    action_deg: float | None = None

    def __post_init__(self) -> None:
        """Keep correction fields finite, bounded and anatomically named."""
        if self.joint not in JOINTS or not math.isfinite(self.amount_deg) or not 0 <= self.amount_deg <= 180:
            raise ValueError("Invalid correction")
        if self.issue not in SIGNED_ISSUES | GENERIC_ISSUES:
            raise ValueError("Unknown correction description")
        if (self.action is None) != (self.action_deg is None) or (
            self.action_deg is not None and not THRESHOLD_DEG <= self.action_deg <= 180
        ):
            raise ValueError("A correction action needs a bounded movement amount")

    @property
    def part(self) -> str:
        """The selectable body part this measurement belongs to."""
        return PART_OF[self.joint]


@dataclass(frozen=True)
class _Signed:
    """Window evidence for one signed quantity."""

    mean: float
    signs: frozenset[bool]
    reliable: bool


class Corrections:
    """Select two corrections from the last second with a ten-degree switching margin."""

    def __init__(self) -> None:
        """Start a bounded, one-second correction window."""
        self._history: deque[tuple[float, tuple[JointError, ...], tuple[Component, ...]]] = deque(maxlen=256)
        self._joint: str | None = None
        self.second: Correction | None = None
        self.matched_parts: tuple[str, ...] = ()

    def pause(self) -> None:
        """Discard unobserved window duration while retaining the previous joint preference."""
        self._history.clear()
        self.second = None
        self.matched_parts = ()

    def update(
        self, captured_at: float, errors: tuple[JointError, ...], components: tuple[Component, ...] = ()
    ) -> Correction | None:
        """Use elapsed time weights, preserving anatomical left and right; also set second and matched parts."""
        self._history.append((captured_at, errors, components))
        while len(self._history) > 1 and self._history[1][0] <= captured_at - 1.0:
            self._history.popleft()
        entries = list(self._history)
        # Each frame's value holds until the next frame; the newest frame has no duration yet.
        weights = [end - max(start, captured_at - 1.0) for (start, _, _), (end, _, _) in zip(entries, entries[1:])]
        averaged = self._averaged(
            [{error.joint: (error.degrees, True) for error in values} for _, values, _ in entries], weights, True
        )
        directions = self._averaged(
            [{c.measurement: (c.degrees, c.reliable) for c in values} for _, _, values in entries], weights, False
        )
        self.second, self.matched_parts = None, ()
        if not averaged:
            self._joint = None
            return None
        magnitude = {joint: abs(value.mean) for joint, value in averaged.items()}
        candidate = max(magnitude, key=lambda joint: magnitude[joint])
        if self._joint not in magnitude or magnitude[candidate] >= magnitude[self._joint] + 10:
            self._joint = candidate
        primary = self._joint
        others = sorted(
            (j for j in magnitude if PART_OF[j] != PART_OF[primary] and magnitude[j] >= THRESHOLD_DEG),
            key=lambda joint: magnitude[joint],
            reverse=True,
        )
        self.second = self._describe(others[0], averaged, directions) if others else None
        present = {PART_OF[joint] for joint in magnitude}
        self.matched_parts = tuple(
            part
            for part in PARTS
            if part in present
            and all(value < THRESHOLD_DEG for joint, value in magnitude.items() if PART_OF[joint] == part)
        )
        return self._describe(primary, averaged, directions)

    @staticmethod
    def _averaged(
        frames: list[dict[str, tuple[float, bool]]], weights: list[float], magnitude: bool
    ) -> dict[str, _Signed]:
        """Time-weighted means over the window for every key of the newest frame.

        With `magnitude`, values are averaged unsigned (3D errors) and the sign is kept only when every
        frame agrees; otherwise signed values (components) are averaged directly.
        """
        result = {}
        for key, (latest, latest_reliable) in frames[-1].items():
            total = weight_sum = 0.0
            signs, reliable = {latest < 0}, latest_reliable
            for frame, weight in zip(frames, weights):
                if key in frame:
                    value, frame_reliable = frame[key]
                    total += (abs(value) if magnitude else value) * weight
                    weight_sum += weight
                    signs.add(value < 0)
                    reliable = reliable and frame_reliable
            mean = total / weight_sum if weight_sum > 0 else (abs(latest) if magnitude else latest)
            if magnitude and signs == {True}:
                mean = -mean
            result[key] = _Signed(mean, frozenset(signs), reliable)
        return result

    @staticmethod
    def _describe(joint: str, averaged: dict[str, _Signed], directions: dict[str, _Signed]) -> Correction:
        error = averaged[joint]
        amount = abs(error.mean)
        if joint in BENDS:
            if len(error.signs) > 1:
                return Correction(joint, "angle differs from reference", amount)
            issue = describe(joint, error.mean)[0]
            if amount < THRESHOLD_DEG:
                return Correction(joint, issue, amount)
            return Correction(joint, issue, amount, action_text(joint, error.mean), rounded(error.mean))
        direction = directions.get(joint)
        if (
            direction is not None
            and direction.reliable
            and len(direction.signs) == 1
            and abs(direction.mean) >= THRESHOLD_DEG
        ):
            issue = describe(joint, direction.mean)[0]
            return Correction(joint, issue, amount, action_text(joint, direction.mean), rounded(direction.mean))
        return Correction(joint, "direction differs from reference", amount)
