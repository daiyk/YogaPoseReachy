"""Camera-calibrated, clamped framing moves evaluated by the existing movement owner."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.rotation import Rotation
from reachy_mini.utils.interpolation import linear_pose_interpolation
from .framing import FramingCorrection


Pose = tuple[NDArray[np.float32], tuple[float, float], float]


@dataclass(frozen=True)
class CameraAngles:
    """Horizontal and vertical angles derived from SDK calibration or an explicit measurement."""

    horizontal: float
    vertical: float

    def __post_init__(self) -> None:
        """Reject missing, nonfinite and implausible calibration."""
        if not 20 <= self.horizontal <= 160 or not 15 <= self.vertical <= 130:
            raise ValueError("Camera field of view is outside supported limits")

    @classmethod
    def from_intrinsics(cls, matrix: NDArray[np.float64], width: int, height: int) -> "CameraAngles":
        """Use the SDK's intrinsics scaled to the actual camera resolution."""
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or width <= 0 or height <= 0:
            raise ValueError("Invalid camera intrinsics")
        fx, fy, cx, cy = float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])
        if fx <= 0 or fy <= 0 or not 0 < cx < width or not 0 < cy < height:
            raise ValueError("Uncalibrated camera intrinsics")
        return cls(
            math.degrees(math.atan(cx / fx) + math.atan((width - cx) / fx)),
            math.degrees(math.atan(cy / fy) + math.atan((height - cy) / fy)),
        )


class FramingMove:
    """Interpolate a small camera correction without exceeding head or body limits."""

    def __init__(self, start: Pose, correction: FramingCorrection | None, angles: CameraAngles | None = None) -> None:
        """Build a framing step, or a gentle neutral return when correction is absent."""
        self.start = (start[0].copy(), start[1], start[2])
        self.neutral = correction is None
        roll, pitch, yaw = Rotation.from_matrix(start[0][:3, :3]).as_euler("xyz", degrees=True)
        body = math.degrees(start[2])
        if correction is None:
            target_roll = target_pitch = target_yaw = target_body = 0.0
            translation = (0.0, 0.0, 0.0)
        else:
            if angles is None:
                raise ValueError("Framing requires calibrated camera angles")
            # Camera right/down corresponds to negative world yaw / positive head pitch.
            dx = math.degrees(math.atan(2 * correction.x * math.tan(math.radians(angles.horizontal) / 2)))
            dy = math.degrees(math.atan(2 * correction.y * math.tan(math.radians(angles.vertical) / 2)))
            target_yaw = float(np.clip(yaw - np.clip(dx, -8, 8), -85, 85))
            target_pitch = float(np.clip(pitch + np.clip(dy, -8, 8), -25, 25))
            target_roll = float(np.clip(roll, -25, 25))
            target_body = float(np.clip(target_yaw - np.clip(target_yaw - body, -25, 25), -60, 60))
            translation = (float(start[0][0, 3]), float(start[0][1, 3]), float(start[0][2, 3]))
        self.duration = max(
            0.5,
            max(abs(target_yaw - yaw), abs(target_pitch - pitch), abs(target_body - body), abs(target_roll - roll))
            / 15,
        )
        self.target_head = np.asarray(
            create_head_pose(
                *translation, roll=target_roll, pitch=target_pitch, yaw=target_yaw, degrees=True, mm=False
            ),
            dtype=np.float32,
        )
        self.target_body = math.radians(target_body)

    def evaluate(self, t: float) -> Pose:
        """Ease between clamped endpoints while retaining antenna positions."""
        progress = min(1.0, max(0.0, t / self.duration))
        blend = progress * progress * (3 - 2 * progress)
        head = np.asarray(linear_pose_interpolation(self.start[0], self.target_head, blend), dtype=np.float32)
        body = self.start[2] + blend * (self.target_body - self.start[2])
        return head, self.start[1], body
