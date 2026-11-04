"""Typed boundary to the already-installed pose comparison app and its isolated worker."""

import importlib
import threading
from typing import Protocol
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..coach.scope import BodyScope
from ..coach.sample import Reason, Landmark, Component, JointError, MirrorEvidence
from ..coach.framing import landmark_box
from ..coach.geometry import Angle, mirrored, components, frontal_angles, labels_reversed


try:
    import cv2
except ImportError:
    cv2 = None


Frame = NDArray[np.uint8]
__all__ = ["Frame", "FrameResult", "PoseEngine", "cv2"]


@dataclass(frozen=True)
class FrameResult:
    """One frame's coaching evidence: score or reason, plus direction and mirror evidence."""

    score: float | None
    errors: tuple[JointError, ...]
    reason: Reason | None
    landmarks: tuple[Landmark, ...]
    components: tuple[Component, ...] = ()
    mirror: MirrorEvidence | None = None
    missing: tuple[str, ...] = ()


class Point(Protocol):
    """Numeric landmark attributes returned by the existing worker client."""

    x: float
    y: float
    presence: float
    visibility: float


class Detection(Protocol):
    """The two landmark coordinate systems returned by MediaPipe."""

    pose_landmarks: list[list[Point]]
    pose_world_landmarks: list[list[Point]]


class Detector(Protocol):
    """Existing bounded worker client operations."""

    def detect(self, rgb: Frame, timestamp: int | None = None) -> Detection:
        """Read one completed numeric result."""
        ...

    def close(self) -> None:
        """Close the isolated process and its bounded reader."""
        ...


class DetectorFactory(Protocol):
    """The installed PoseDetector constructor signature used here."""

    def __call__(self, model_path: Path, mode: str, *, options: dict[str, int]) -> Detector:
        """Create one worker without changing the SDK environment."""
        ...


class ComparisonResult(Protocol):
    """Validated comparison fields projected into the coach sample contract."""

    score: float | None
    reason_code: str
    deltas_deg: dict[str, float]


class Comparison(Protocol):
    """Existing comparison engine retaining its visibility, geometry and temporal gates."""

    def evaluate(self, points: list[Point], world: list[Point], at: float) -> ComparisonResult:
        """Compare one frame against the selected reference."""
        ...


class PrepareReference(Protocol):
    """Reference preparation with explicit comparison timing settings."""

    def __call__(
        self,
        points: list[Point],
        world: list[Point],
        side: str,
        *,
        settings: dict[str, float | int],
        features: tuple[str, ...],
    ) -> object:
        """Retain geometry gates while delegating temporal evidence to the coach."""
        ...


class PoseEngine:
    """Reuse the installed comparison implementation without copying or downgrading its runtime."""

    def __init__(self) -> None:
        """Resolve the existing full model, worker and reference validator."""
        if cv2 is None:
            raise RuntimeError("Install the existing pose-comparison runtime, including OpenCV")
        module = importlib.import_module("reachy_mini_pose_compare.detector")
        comparison = importlib.import_module("reachy_mini_pose_compare.comparison")
        runner = importlib.import_module("reachy_mini_pose_compare.runner")
        if module.__file__ is None:
            raise RuntimeError("The installed pose-comparison app has no assets directory")
        assets = Path(module.__file__).parent / "assets"
        self.model = assets / "pose_landmarker_full_v1.task"
        if not self.model.is_file():
            raise RuntimeError("Install the existing pose-comparison app with its Full model assets")
        self.default_reference = (assets / "reference.jpg").read_bytes()
        self.validate_image: Callable[[bytes], bytes] = getattr(runner, "validate_reference")
        self._detector_factory: DetectorFactory = getattr(module, "PoseDetector")
        self._prepare: PrepareReference = getattr(comparison, "prepare_reference")
        self.comparison_settings: dict[str, float | int] = dict(getattr(comparison, "SETTINGS"))
        # The coach owns smoothing/holds; the old three-frame gate cannot score at 3 FPS.
        self.comparison_settings.update(smoothing_seconds=0.05, minimum_frames=1, minimum_span_seconds=0.0)
        self._comparison_factory: Callable[[object], Comparison] = getattr(comparison, "ComparisonEngine")
        self.detector: Detector | None = None
        self.comparison: Comparison | None = None
        # Same-side and mirrored references without the relative-yaw gate, used only to detect mirroring.
        self.probes: tuple[Comparison, Comparison] | None = None
        self.reference_angles: dict[str, Angle] = {}
        self.reference_points: tuple[Landmark, ...] = ()
        self.stopping = threading.Event()
        self.scope = BodyScope()
        self.missing_landmarks: tuple[str, ...] = ()

    def prepare(self, jpeg: bytes, side: str, scope: BodyScope = BodyScope()) -> None:
        """Validate exactly one person using the same Full model and existing view gates."""
        assert cv2 is not None
        self.close()
        self.scope = scope.mapped(side)
        if self.stopping.is_set():
            return
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode the reference image")
        detector = self._detector_factory(self.model, "IMAGE", options={"num_poses": 2, "inference_threads": 1})
        try:
            result = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(result.pose_landmarks) != 1 or len(result.pose_world_landmarks) != 1:
                raise ValueError("The reference must contain exactly one visible person")
            self.reference_points = tuple(
                Landmark(p.x, p.y, p.presence, p.visibility) for p in result.pose_landmarks[0]
            )
            missing = scope.missing(self.reference_points)
            if missing:
                raise ValueError("Reference needs visible: " + ", ".join(missing))
            image, world = result.pose_landmarks[0], result.pose_world_landmarks[0]
            reference = self._prepare(image, world, side, settings=self.comparison_settings, features=scope.features)
            # A mirrored attempt fails the active relative-yaw gate, so both probes leave it open.
            probe_settings = {**self.comparison_settings, "relative_yaw_limit_deg": 180.0}
            same, opposite = (
                self._comparison_factory(
                    self._prepare(image, world, probe_side, settings=probe_settings, features=scope.features)
                )
                for probe_side in (side, "opposite" if side == "same" else "same")
            )
            self.comparison, self.probes = self._comparison_factory(reference), (same, opposite)
            angles = frontal_angles(world)
            self.reference_angles = mirrored(angles) if side == "opposite" else angles
        finally:
            detector.close()
        if self.stopping.is_set():
            return
        self.detector = self._detector_factory(self.model, "VIDEO", options={"num_poses": 1, "inference_threads": 1})

    def detect(self, frame: Frame, captured_at: float) -> FrameResult:
        """Project a real comparison into finite bounded coaching measurements."""
        assert cv2 is not None
        if self.detector is None or self.comparison is None:
            raise RuntimeError("Pose reference is not ready")
        result = self.detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), round(captured_at * 1000))
        points = result.pose_landmarks[0] if result.pose_landmarks else []
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
        landmarks = tuple(Landmark(p.x, p.y, p.presence, p.visibility) for p in points)
        box = landmark_box(self.scope.framing_points(landmarks))
        self.missing_landmarks = self.scope.missing(landmarks) if landmarks else ()
        comparison = self.comparison.evaluate(points, world, captured_at)
        reason = {
            "no_pose": Reason.NO_PERSON,
            "missing_landmarks": Reason.OCCLUDED,
            "joint_not_visible": Reason.OCCLUDED,
            "unsupported_view": Reason.UNSUPPORTED_VIEW,
            "invalid_geometry": Reason.INVALID_GEOMETRY,
            "stabilizing": Reason.STABILIZING,
        }.get(comparison.reason_code)
        # Later checks take priority: too close > out of frame (box or head) > swapped labels > comparison.
        if self.missing_landmarks and reason is None:
            reason = Reason.OCCLUDED
        if landmarks and labels_reversed(landmarks):
            reason = Reason.UNRELIABLE_TRACKING
        if landmarks and not self.scope.head_in_view(landmarks):
            reason = Reason.OUT_OF_FRAME
        if box is not None and box.too_large:
            reason = Reason.TOO_CLOSE
        elif box is not None and (box.left < 0 or box.top < 0 or box.right > 1 or box.bottom > 1):
            reason = Reason.OUT_OF_FRAME
        if comparison.score is None and reason is None:
            reason = Reason.INVALID_GEOMETRY
        mirror = None
        if reason in (None, Reason.UNSUPPORTED_VIEW) and self.probes is not None:
            same, opposite = (probe.evaluate(points, world, captured_at).score for probe in self.probes)
            if same is not None and opposite is not None:
                mirror = MirrorEvidence(same, opposite)
        if reason is not None:
            return FrameResult(None, (), reason, landmarks, mirror=mirror, missing=self.missing_landmarks)
        assert comparison.score is not None
        return FrameResult(
            comparison.score,
            tuple(JointError(name, value) for name, value in comparison.deltas_deg.items()),
            None,
            landmarks,
            components(frontal_angles(world), self.reference_angles, comparison.deltas_deg),
            mirror,
            self.missing_landmarks,
        )

    def close(self) -> None:
        """Release the subprocess before replacing a reference or stopping the app."""
        if self.detector is not None:
            self.detector.close()
            self.detector = None
