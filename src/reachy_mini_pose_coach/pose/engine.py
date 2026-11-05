"""Typed boundary to the coach's own pose runtime: Full model, isolated worker and comparison geometry."""

import threading
from types import SimpleNamespace
from typing import Any, Protocol
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .runtime import comparison
from ..coach.scope import BodyScope
from ..coach.sample import Reason, Landmark, Component, JointError, MirrorEvidence
from ..coach.framing import landmark_box
from ..coach.geometry import Angle, mirrored, components, frontal_angles, labels_reversed
from .runtime.detector import PoseDetector
from .runtime.reference import validate_reference


try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


Frame = NDArray[np.uint8]
ASSETS = Path(__file__).with_name("assets")
MODEL = "pose_landmarker_full_v1.task"
__all__ = ["ASSETS", "Frame", "FrameResult", "PoseEngine", "cv2"]


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


class Detector(Protocol):
    """Worker client operations (a fake replaces it in tests)."""

    def detect(self, rgb: NDArray[Any], timestamp: int | None = None) -> SimpleNamespace:
        """Read one completed numeric result."""
        ...

    def close(self) -> None:
        """Close the isolated process and its bounded reader."""
        ...


class DetectorFactory(Protocol):
    """The PoseDetector constructor signature used here."""

    def __call__(self, model_path: Path, mode: str, *, options: dict[str, int]) -> Detector:
        """Create one worker without changing the SDK environment."""
        ...


class Comparison(Protocol):
    """Comparison engine retaining its visibility, geometry and temporal gates."""

    def evaluate(
        self, points: Sequence[comparison.ImagePoint], world: Sequence[comparison.WorldPoint], at: float
    ) -> comparison.ComparisonResult:
        """Compare one frame against the selected reference."""
        ...


class PoseEngine:
    """Run the bundled Full model and comparison without any other Reachy app installed."""

    def __init__(self, assets: Path = ASSETS) -> None:
        """Resolve the bundled full model, worker and reference validator."""
        if cv2 is None:
            raise RuntimeError("OpenCV is missing from the app environment; reinstall the pose coach app")
        self.model = assets / MODEL
        if not self.model.is_file():
            raise RuntimeError(f"Pose model is missing: {self.model}; reinstall the pose coach app")
        self.default_reference = (assets / "reference.jpg").read_bytes()
        # Module globals are read here so tests can replace the worker and geometry.
        self.validate_image: Callable[[bytes], bytes] = validate_reference
        self._detector_factory: DetectorFactory = PoseDetector
        self._prepare: Callable[..., comparison.Reference] = comparison.prepare_reference
        self.comparison_settings: dict[str, float | int] = dict(comparison.SETTINGS)
        # The coach owns smoothing/holds; the old three-frame gate cannot score at 3 FPS.
        self.comparison_settings.update(smoothing_seconds=0.05, minimum_frames=1, minimum_span_seconds=0.0)
        self._comparison_factory: Callable[[comparison.Reference], Comparison] = comparison.ComparisonEngine
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
        compared = self.comparison.evaluate(points, world, captured_at)
        reason = {
            "no_pose": Reason.NO_PERSON,
            "missing_landmarks": Reason.OCCLUDED,
            "joint_not_visible": Reason.OCCLUDED,
            "unsupported_view": Reason.UNSUPPORTED_VIEW,
            "invalid_geometry": Reason.INVALID_GEOMETRY,
            "stabilizing": Reason.STABILIZING,
        }.get(compared.reason_code)
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
        if compared.score is None and reason is None:
            reason = Reason.INVALID_GEOMETRY
        mirror = None
        if reason in (None, Reason.UNSUPPORTED_VIEW) and self.probes is not None:
            same, opposite = (probe.evaluate(points, world, captured_at).score for probe in self.probes)
            if same is not None and opposite is not None:
                mirror = MirrorEvidence(same, opposite)
        if reason is not None:
            return FrameResult(None, (), reason, landmarks, mirror=mirror, missing=self.missing_landmarks)
        assert compared.score is not None
        return FrameResult(
            compared.score,
            tuple(JointError(name, value) for name, value in compared.deltas_deg.items()),
            None,
            landmarks,
            components(frontal_angles(world), self.reference_angles, compared.deltas_deg),
            mirror,
            self.missing_landmarks,
        )

    def close(self) -> None:
        """Release the subprocess before replacing a reference or stopping the app."""
        if self.detector is not None:
            self.detector.close()
            self.detector = None
