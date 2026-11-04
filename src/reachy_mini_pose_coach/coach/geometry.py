"""Depth-free signed body angles in the camera's frontal plane and the coaching phrases they support.

Inputs are MediaPipe world landmarks: x points toward image right (the person's left when they face the
camera) and y points down. Only x and y set an angle; z only rates how well a segment shows in that plane.
"""

import math
from typing import Protocol
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from .sample import BENDS, Landmark, Component


class XY(Protocol):
    """A landmark with at least frontal-plane coordinates."""

    @property
    def x(self) -> float:
        """Toward image right."""
        ...

    @property
    def y(self) -> float:
        """Downward."""
        ...


# measurement -> (proximal landmark, distal landmark, outward sign along x)
LIMBS = {
    "left_upper_arm": (11, 13, 1.0),
    "right_upper_arm": (12, 14, -1.0),
    "left_thigh": (23, 25, 1.0),
    "right_thigh": (24, 26, -1.0),
}
# Below half its 3D length a segment points mostly toward or away from the camera.
MIN_RATIO = 0.5
# Normalized image width by which left/right landmarks may overlap before labels count as swapped.
FLIP_MARGIN = 0.01


@dataclass(frozen=True)
class Angle:
    """One frontal-plane angle in degrees and the share of its segment visible in that plane."""

    degrees: float
    ratio: float


@dataclass(frozen=True)
class _Point:
    x: float
    y: float
    z: float


def wrap(degrees: float) -> float:
    """Map an angle difference into (-180, 180]."""
    result = math.fmod(degrees, 360.0)
    if result <= -180:
        return result + 360.0
    return result - 360.0 if result > 180 else result


def _segment(a: XY, b: XY) -> tuple[float, float, float] | None:
    dx, dy = b.x - a.x, b.y - a.y
    dz = float(getattr(b, "z", 0.0)) - float(getattr(a, "z", 0.0))
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not all(math.isfinite(v) for v in (dx, dy, dz)) or length < 1e-8:
        return None
    return dx, dy, math.hypot(dx, dy) / length


def _midpoint(a: XY, b: XY) -> _Point:
    return _Point((a.x + b.x) / 2, (a.y + b.y) / 2, (float(getattr(a, "z", 0.0)) + float(getattr(b, "z", 0.0))) / 2)


def frontal_angles(points: Sequence[XY]) -> dict[str, Angle]:
    """Limb angle from hanging down (0) through out to the side (90) to overhead (180), torso lean, shoulder tilt."""
    if len(points) <= 26:
        return {}
    angles: dict[str, Angle] = {}
    for name, (start, end, side) in LIMBS.items():
        segment = _segment(points[start], points[end])
        if segment is not None:
            dx, dy, ratio = segment
            angles[name] = Angle(math.degrees(math.atan2(side * dx, dy)), ratio)
    torso = _segment(_midpoint(points[23], points[24]), _midpoint(points[11], points[12]))
    if torso is not None:
        # Positive: the upper body leans toward the person's left.
        angles["torso"] = Angle(math.degrees(math.atan2(torso[0], -torso[1])), torso[2])
    shoulders = _segment(points[12], points[11])
    if shoulders is not None:
        # Positive: the left shoulder is higher than the right.
        angles["shoulders"] = Angle(math.degrees(math.atan2(-shoulders[1], shoulders[0])), shoulders[2])
    return angles


def mirrored(angles: dict[str, Angle]) -> dict[str, Angle]:
    """Angles of the mirror image: limb sides swap with equal values; lean and tilt change sign."""
    result: dict[str, Angle] = {}
    for name, angle in angles.items():
        if name.startswith("left_"):
            result["right_" + name[5:]] = angle
        elif name.startswith("right_"):
            result["left_" + name[6:]] = angle
        else:
            result[name] = Angle(-angle.degrees, angle.ratio)
    return result


def components(
    live: dict[str, Angle], reference: dict[str, Angle], measurements: Iterable[str]
) -> tuple[Component, ...]:
    """Signed live-minus-reference angles; reliable only where both poses show the segment clearly."""
    result = []
    for name in measurements:
        if name in BENDS or name not in live or name not in reference:
            continue
        current, target = live[name], reference[name]
        reliable = min(current.ratio, target.ratio) >= MIN_RATIO
        if name in LIMBS:
            # Past overhead or across the midline, "raise" and "lower" stop meaning one direction.
            reliable = reliable and all(0 <= angle.degrees <= 180 for angle in (current, target))
        elif name == "torso":
            reliable = reliable and all(abs(angle.degrees) < 90 for angle in (current, target))
        result.append(Component(name, wrap(current.degrees - target.degrees), reliable))
    return tuple(result)


def labels_reversed(points: Sequence[Landmark]) -> bool:
    """Detect swapped labels: a camera-facing person shows their left hip and shoulder at larger image x."""
    if len(points) <= 24:
        return False
    for left, right in ((23, 24), (11, 12)):
        if (
            min(points[left].presence, points[right].presence) >= 0.5
            and points[left].x < points[right].x - FLIP_MARGIN
        ):
            return True
    return False


def rounded(degrees: float) -> int:
    """Amount to move, to the nearest 5 degrees and never below the 10-degree correction threshold."""
    return max(10, 5 * math.floor(abs(degrees) / 5 + 0.5))


def describe(measurement: str, degrees: float) -> tuple[str, str]:
    """Return (mismatch description, movement) for a signed live-minus-reference difference."""
    more = degrees > 0
    if measurement in BENDS:
        side, joint = measurement.split("_")
        if more:
            return "straighter than reference", f"bend your {side} {joint} more"
        return "more bent than reference", f"straighten your {side} {joint}"
    if measurement.endswith("_upper_arm"):
        side = measurement.split("_")[0]
        if more:
            return "higher than reference", f"lower your {side} arm"
        return "lower than reference", f"raise your {side} arm"
    if measurement.endswith("_thigh"):
        side = measurement.split("_")[0]
        if more:
            return "knee further out than reference", f"bring your {side} knee inward"
        return "knee further in than reference", f"move your {side} knee outward"
    if measurement == "torso":
        if more:
            return "leaning further toward your left than reference", "lean your upper body toward your right"
        return "leaning further toward your right than reference", "lean your upper body toward your left"
    if measurement == "shoulders":
        if more:
            return "left shoulder higher than reference", "lower your left shoulder"
        return "right shoulder higher than reference", "lower your right shoulder"
    raise ValueError("Unknown measurement")


def action_text(measurement: str, degrees: float) -> str:
    """Return a movement with its rounded amount, e.g. "raise your left arm about 20°"."""
    return f"{describe(measurement, degrees)[1]} about {rounded(degrees)}°"


SIGNED_ISSUES = frozenset(
    describe(name, sign)[0]
    for name in ("left_elbow", "left_upper_arm", "left_thigh", "torso", "shoulders")
    for sign in (1.0, -1.0)
)
