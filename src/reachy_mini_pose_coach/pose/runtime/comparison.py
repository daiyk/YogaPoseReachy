"""Reference matching geometry and state only, no camera or model I/O.

Copied from `reachy_mini_pose_compare` (body-part version, 2026-11-02) so the coach runs without that app.
Only type hints were added; scoring behaviour must stay identical (see repository `checks/`).
"""

import math
import statistics
from typing import Any, Protocol
from collections import deque
from dataclasses import field, dataclass
from collections.abc import Iterable, Sequence


KEY_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
JOINT_NAMES = dict(
    zip(
        KEY_JOINTS,
        (
            "left shoulder",
            "right shoulder",
            "left elbow",
            "right elbow",
            "left wrist",
            "right wrist",
            "left hip",
            "right hip",
            "left knee",
            "right knee",
            "left ankle",
            "right ankle",
        ),
    )
)
JOINT_ANGLES = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
}
DIRECTIONS = {
    "left_upper_arm": (11, 13),
    "right_upper_arm": (12, 14),
    "left_thigh": (23, 25),
    "right_thigh": (24, 26),
}
# Default thresholds (the compare app's `parameters.py` comparison group).
SETTINGS: dict[str, float | int] = {
    "version": 1,
    "visibility_threshold": 0.6,
    "reference_frontal_limit_deg": 20.0,
    "live_frontal_limit_deg": 30.0,
    "relative_yaw_limit_deg": 20.0,
    "full_score_error_deg": 10.0,
    "zero_score_error_deg": 45.0,
    "smoothing_seconds": 0.5,
    "minimum_frames": 3,
    "minimum_span_seconds": 0.2,
    "message_persistence_seconds": 1.0,
}

Vector = tuple[float, ...]
Settings = dict[str, float | int]


class ImagePoint(Protocol):
    """Normalised image landmark with MediaPipe's confidence fields."""

    x: float
    y: float
    visibility: float
    presence: float


class WorldPoint(Protocol):
    """Metric 3D landmark."""

    x: float
    y: float
    z: float


def joint_visible(point: ImagePoint, threshold: float) -> bool:
    """Return whether one image landmark is finite, inside the image and confidently visible."""
    return (
        all(math.isfinite(v) for v in (point.x, point.y, point.visibility, point.presence))
        and 0 <= point.x <= 1
        and 0 <= point.y <= 1
        and point.visibility >= threshold
        and point.presence >= threshold
    )


def subtract(a: Vector, b: Vector) -> Vector:
    """Component-wise difference."""
    return tuple(x - y for x, y in zip(a, b))


def midpoint(a: Vector, b: Vector) -> Vector:
    """Component-wise mean."""
    return tuple((x + y) / 2 for x, y in zip(a, b))


def unit(vector: Vector) -> Vector:
    """Normalise a vector, rejecting zero-length or non-finite segments."""
    length = math.hypot(*vector)
    if not math.isfinite(length) or length < 1e-8:
        raise ValueError("Landmarks contain a zero-length or invalid body segment")
    return tuple(v / length for v in vector)


def angle(a: Vector, b: Vector) -> float:
    """Angle between two vectors in degrees."""
    dot = sum(x * y for x, y in zip(unit(a), unit(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


class UnusablePose(ValueError):
    """A frame that cannot be scored, with a machine-readable reason code."""

    def __init__(self, code: str, message: str) -> None:
        """Keep the reason code next to the human message."""
        super().__init__(message)
        self.code = code


def canonical_pose(
    image_landmarks: Sequence[ImagePoint],
    world_landmarks: Sequence[WorldPoint],
    threshold: float,
    required_joints: tuple[int, ...] = KEY_JOINTS,
) -> tuple[dict[int, Vector], float]:
    """Return landmarks in a body-centered frame plus hip yaw in degrees.

    Logic:
    1) Validate required joints in both image/world landmarks.
    2) Use hip midpoint (left/right hips) as translation origin.
    3) Build a horizontal body axis from right hip -> left hip:
         hip_axis = left_hip - right_hip
    4) Estimate yaw in the local x-z plane with atan2(z, x).
    5) Rotate all joints around the vertical axis by -yaw so the body is
       normalized ("canonicalized") before scoring.

    Note: yaw sign is a convention defined by this axis choice + rotation
    formula. Absolute world handedness does not matter for matching because
    reference and live poses are normalized the same way.
    """
    if not image_landmarks:
        raise UnusablePose("no_pose", "No person detected")
    if len(image_landmarks) <= max(required_joints) or len(world_landmarks) <= max(required_joints):
        raise UnusablePose("missing_landmarks", "Complete image and 3D body landmarks are required")
    points: dict[int, Vector] = {}
    for i in required_joints:
        if not joint_visible(image_landmarks[i], threshold):
            raise UnusablePose("joint_not_visible", f"Keep the {JOINT_NAMES[i]} visible and inside the image")
        point = world_landmarks[i]
        xyz = (point.x, point.y, point.z)
        if not all(math.isfinite(v) for v in xyz):
            raise UnusablePose("invalid_geometry", "3D body landmarks are not finite")
        points[i] = xyz
    center = midpoint(points[23], points[24])
    hip_axis = subtract(points[23], points[24])  # Right-to-left points right in a frontal camera image.
    if math.hypot(hip_axis[0], hip_axis[2]) < 1e-8:
        raise UnusablePose("invalid_geometry", "Cannot determine horizontal orientation from the hips")
    yaw = math.atan2(hip_axis[2], hip_axis[0])
    c, s = math.cos(yaw), math.sin(yaw)
    normalized: dict[int, Vector] = {}
    for i, point_xyz in points.items():
        x, y, z = subtract(point_xyz, center)
        normalized[i] = (c * x + s * z, y, -s * x + c * z)
    return normalized, math.degrees(yaw)


def measurements(points: dict[int, Vector], features: Iterable[str] | None = None) -> dict[str, Any]:
    """Joint bends (degrees) and unit limb/body directions for the selected features."""
    selected = set(features) if features is not None else set(JOINT_ANGLES) | set(DIRECTIONS) | {"torso", "shoulders"}
    # Joint bends: angle at the middle joint (shoulder-elbow-wrist, hip-knee-ankle), in degrees.
    values: dict[str, Any] = {
        name: angle(subtract(points[a], points[b]), subtract(points[c], points[b]))
        for name, (a, b, c) in JOINT_ANGLES.items()
        if name in selected
    }
    # Limb directions: normalized vectors (shoulder->elbow, hip->knee) for orientation comparison.
    values.update(
        {name: unit(subtract(points[b], points[a])) for name, (a, b) in DIRECTIONS.items() if name in selected}
    )
    # Global body directions: hip-center->shoulder-center (torso) and right-shoulder->left-shoulder.
    if "torso" in selected:
        values["torso"] = unit(subtract(midpoint(points[11], points[12]), midpoint(points[23], points[24])))
    if "shoulders" in selected:
        values["shoulders"] = unit(subtract(points[11], points[12]))
    return values


@dataclass
class Reference:
    """Prepared reference measurements and the gates used against it."""

    values: dict[str, Any]
    yaw_deg: float
    side: str
    threshold: float
    settings: Settings = field(default_factory=lambda: SETTINGS)
    required_joints: tuple[int, ...] = KEY_JOINTS


def feature_joints(features: Sequence[str]) -> tuple[int, ...]:
    """Joints needed by the selected measurements, always including both hips."""
    dependencies: dict[str, tuple[int, ...]] = {
        **JOINT_ANGLES,
        **DIRECTIONS,
        "torso": (11, 12, 23, 24),
        "shoulders": (11, 12),
    }
    if not features or any(name not in dependencies for name in features):
        raise ValueError("Select at least one known scoring measurement")
    return tuple(sorted({23, 24}.union(*(dependencies[name] for name in features))))


def mirror_joint(index: int) -> int:
    """MediaPipe's left/right partner of a body landmark."""
    return index + 1 if index % 2 else index - 1


def prepare_reference(
    image_landmarks: Sequence[ImagePoint],
    world_landmarks: Sequence[WorldPoint],
    side: str = "same",
    threshold: float = 0.6,
    settings: Settings | None = None,
    features: Sequence[str] | None = None,
) -> Reference:
    """Validate and canonicalise a reference, mirroring it for opposite-side practice."""
    settings = settings or SETTINGS
    if side not in ("same", "opposite") or not 0 <= threshold <= 1:
        raise ValueError("Invalid reference side or visibility threshold")
    selected = (
        tuple(features) if features is not None else tuple(JOINT_ANGLES) + tuple(DIRECTIONS) + ("torso", "shoulders")
    )
    required = feature_joints(selected)
    points, yaw = canonical_pose(image_landmarks, world_landmarks, threshold, required)
    limit = settings["reference_frontal_limit_deg"]
    if abs(yaw) > limit + 1e-9:
        raise ValueError(
            f"Reference hip yaw is {yaw:.1f} deg which is too large; use a reference within {limit:g} deg of frontal"
        )
    if side == "opposite":
        opposite: dict[int, Vector] = {}
        for source, (x, y, z) in points.items():
            opposite[mirror_joint(source)] = (-x, y, z)
        selected = tuple(
            "right_" + name[5:]
            if name.startswith("left_")
            else "left_" + name[6:]
            if name.startswith("right_")
            else name
            for name in selected
        )
        required = tuple(sorted(mirror_joint(i) for i in required))
        points = opposite
        yaw = -yaw
    return Reference(measurements(points, selected), yaw, side, threshold, settings, required)


@dataclass
class ComparisonResult:
    """One frame's score or the reason it cannot be scored."""

    score: float | None = None
    reason_code: str = ""
    reason: str = ""
    deltas_deg: dict[str, float] = field(default_factory=dict)
    messages: tuple[str, ...] = ()
    yaw_deg: float | None = None


def similarity(deltas: dict[str, float], settings: Settings = SETTINGS) -> float:
    """Mean per-feature score between the full-score and zero-score error limits."""
    low, high = settings["full_score_error_deg"], settings["zero_score_error_deg"]
    return statistics.mean(100 * max(0.0, 1 - max(0.0, abs(delta) - low) / (high - low)) for delta in deltas.values())


def body_part(name: str) -> str:
    """Group a measurement into torso, left/right arm or left/right leg."""
    if name in ("torso", "shoulders"):
        return "torso"
    return name.split("_")[0] + ("_arm" if "elbow" in name or "arm" in name else "_leg")


def difference_message(name: str, delta: float) -> str:
    """Short human description of one measurement difference."""
    label = name.replace("_", " ").capitalize()
    if name in JOINT_ANGLES:
        return f"{label} is {'more bent' if delta < 0 else 'straighter'} than reference ({abs(delta):.0f} deg)"
    label = {"torso": "Torso direction", "shoulders": "Shoulder line"}.get(name, label + " direction")
    return f"{label} differs from reference ({abs(delta):.0f} deg)"


class ComparisonEngine:
    """Compare live frames to one prepared reference with view gates and temporal smoothing."""

    def __init__(self, reference: Reference) -> None:
        """Start with empty history and no shown cues."""
        self.reference = reference
        self.history: deque[tuple[float, dict[str, float]]] = deque()
        self.pending: tuple[tuple[str, int], ...] = ()
        self.shown: tuple[tuple[str, int], ...] = ()
        self.pending_since: float | None = None

    def unavailable(self, code: str, message: str) -> ComparisonResult:
        """Reset temporal state and report why the frame cannot be scored."""
        self.history.clear()
        self.pending = self.shown = ()
        self.pending_since = None
        return ComparisonResult(reason_code=code, reason=message)

    def evaluate(
        self,
        image_landmarks: Sequence[ImagePoint],
        world_landmarks: Sequence[WorldPoint],
        timestamp_seconds: float,
    ) -> ComparisonResult:
        """Compare one live frame to the prepared reference.

        Pipeline:
        - Canonicalize current frame (same geometry convention as reference).
        - Reject unsupported viewpoints:
            * too far from frontal
            * too far from reference yaw
        - Compute per-feature deltas (joint-angle signed error, direction
          angular error), smooth over a short time window, and score.
        - Persist top correction cues only if they stay stable for a minimum
          duration to avoid flicker.
        """
        if not math.isfinite(timestamp_seconds):
            raise ValueError("Comparison timestamp must be finite")
        settings = self.reference.settings
        try:
            points, yaw = canonical_pose(
                image_landmarks, world_landmarks, self.reference.threshold, self.reference.required_joints
            )
            if abs(yaw) > settings["live_frontal_limit_deg"] + 1e-9:
                return self.unavailable(
                    "unsupported_view", f"Turn closer to frontal (within {settings['live_frontal_limit_deg']:g} deg)"
                )
            if abs(yaw - self.reference.yaw_deg) > settings["relative_yaw_limit_deg"] + 1e-9:
                return self.unavailable(
                    "unsupported_view",
                    f"Turn closer to the reference view (within {settings['relative_yaw_limit_deg']:g} deg)",
                )
            live = measurements(points, self.reference.values)
        except UnusablePose as exc:
            return self.unavailable(exc.code, str(exc))
        except ValueError as exc:
            return self.unavailable("invalid_geometry", str(exc))
        deltas: dict[str, float] = {
            name: live[name] - target if name in JOINT_ANGLES else angle(live[name], target)
            for name, target in self.reference.values.items()
        }
        if self.history and (
            timestamp_seconds <= self.history[-1][0]
            or timestamp_seconds - self.history[-1][0] > settings["smoothing_seconds"]
        ):
            self.unavailable("stabilizing", "Collecting stable body measurements")
        self.history.append((timestamp_seconds, deltas))
        while timestamp_seconds - self.history[0][0] > settings["smoothing_seconds"] + 1e-9:
            self.history.popleft()
        if (
            len(self.history) < settings["minimum_frames"]
            or timestamp_seconds - self.history[0][0] < settings["minimum_span_seconds"] - 1e-9
        ):
            return ComparisonResult(
                reason_code="stabilizing",
                yaw_deg=yaw,
                reason=f"Collecting body measurements ({settings['minimum_span_seconds']:g}s minimum)",
            )
        smoothed = {name: statistics.median(item[1][name] for item in self.history) for name in deltas}
        candidates: list[tuple[str, int]] = []
        groups: set[str] = set()
        for name in sorted(smoothed, key=lambda name: abs(smoothed[name]), reverse=True):
            if abs(smoothed[name]) > settings["full_score_error_deg"] and body_part(name) not in groups:
                candidates.append((name, -1 if smoothed[name] < 0 else 1))
                groups.add(body_part(name))
            if len(candidates) == 2:
                break
        candidate = tuple(candidates)
        if candidate != self.pending:
            self.pending, self.pending_since = candidate, timestamp_seconds
        if (
            self.pending_since is not None
            and timestamp_seconds - self.pending_since >= settings["message_persistence_seconds"] - 1e-9
        ):
            self.shown = candidate
        # Do not keep an old cue once its difference disappears or changes direction.
        self.shown = tuple(
            (name, sign)
            for name, sign in self.shown
            if abs(smoothed[name]) > settings["full_score_error_deg"] and smoothed[name] * sign > 0
        )
        messages = tuple(difference_message(name, smoothed[name]) for name, _ in self.shown)
        return ComparisonResult(similarity(smoothed, settings), deltas_deg=smoothed, messages=messages, yaw_deg=yaw)
