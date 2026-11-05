"""Reference-anatomical selections and their fixed measurement dependencies."""

from dataclasses import dataclass

from .sample import Landmark


PARTS = {
    "torso": ((11, 12, 23, 24), ("torso", "shoulders")),
    "left_arm": ((11, 13, 15), ("left_elbow", "left_upper_arm")),
    "right_arm": ((12, 14, 16), ("right_elbow", "right_upper_arm")),
    "left_leg": ((23, 25, 27), ("left_knee", "left_thigh")),
    "right_leg": ((24, 26, 28), ("right_knee", "right_thigh")),
    "head": ((0, 7, 8), ()),
    "left_hand": ((17, 19, 21), ()),
    "right_hand": ((18, 20, 22), ()),
    "left_foot": ((29, 31), ()),
    "right_foot": ((30, 32), ()),
}
DEFAULT_PARTS = tuple(name for name, (_, features) in PARTS.items() if features)
PART_OF = {feature: part for part, (_, features) in PARTS.items() for feature in features}
# MediaPipe's left/right and facing estimates depend on the face: hiding it can swap hip labels.
NOSE = 0
LANDMARK_NAMES = {
    0: "nose",
    7: "left ear",
    8: "right ear",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index",
}


@dataclass(frozen=True)
class BodyScope:
    """A validated fixed selection; sides refer to the person in the reference."""

    parts: tuple[str, ...] = DEFAULT_PARTS

    def __post_init__(self) -> None:
        """Reject unknown, duplicate or visibility-only selections."""
        if (
            not isinstance(self.parts, tuple)
            or any(not isinstance(part, str) or part not in PARTS for part in self.parts)
            or len(set(self.parts)) != len(self.parts)
            or not any(part in DEFAULT_PARTS for part in self.parts)
        ):
            raise ValueError("Select at least one torso, arm or leg; use each known body part only once")
        object.__setattr__(self, "parts", tuple(part for part in PARTS if part in self.parts))

    def mapped(self, side: str) -> "BodyScope":
        """Map reference sides to live anatomical sides."""
        if side not in ("same", "opposite"):
            raise ValueError("Invalid comparison side")
        return BodyScope(
            tuple(
                (
                    "right_" + part[5:]
                    if part.startswith("left_")
                    else "left_" + part[6:]
                    if part.startswith("right_")
                    else part
                )
                if side == "opposite"
                else part
                for part in self.parts
            )
        )

    @property
    def features(self) -> tuple[str, ...]:
        """Return only measurements belonging to selected scoring parts."""
        return tuple(feature for part in self.parts for feature in PARTS[part][1])

    @property
    def required(self) -> tuple[int, ...]:
        """Include both hip anchors even when no leg or torso score is selected."""
        return tuple(sorted({23, 24}.union(*(PARTS[part][0] for part in self.parts))))

    def framing_points(self, points: tuple[Landmark, ...]) -> tuple[Landmark, ...]:
        """Frame the selected parts plus the head; unselected hands and feet may leave the image."""
        return tuple(points[i] for i in (NOSE, *self.required) if i < len(points))

    @staticmethod
    def head_in_view(points: tuple[Landmark, ...]) -> bool:
        """Require the nose inside the image; visibility stays high even when the face is hidden."""
        if len(points) <= NOSE:
            return False
        nose = points[NOSE]
        return 0 <= nose.x <= 1 and 0 <= nose.y <= 1 and nose.presence >= 0.5

    def missing(self, points: tuple[Landmark, ...], threshold: float = 0.6) -> tuple[str, ...]:
        """Name required image landmarks, and the head, that cannot support a comparison."""
        missing = tuple(
            LANDMARK_NAMES[i]
            for i in self.required
            if i >= len(points)
            or not (
                0 <= points[i].x <= 1
                and 0 <= points[i].y <= 1
                and points[i].presence >= threshold
                and points[i].visibility >= threshold
            )
        )
        if self.head_in_view(points) or LANDMARK_NAMES[NOSE] in missing:
            return missing
        return (LANDMARK_NAMES[NOSE], *missing)
