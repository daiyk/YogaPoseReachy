"""One latest frame/sample, a rate-controlled worker and demand-only JPEG preview."""

import json
import time
import base64
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Callable

from .engine import Frame, PoseEngine, cv2
from ..coach.scope import PARTS, BodyScope
from ..coach.motion import CameraAngles
from ..coach.sample import Reason, Landmark, PoseSample


logger = logging.getLogger(__name__)
# A present person whose framing the coach should ask them to fix.
FRAMING_REASONS = frozenset((Reason.OUT_OF_FRAME, Reason.TOO_CLOSE, Reason.OCCLUDED, Reason.UNRELIABLE_TRACKING))


class PoseService:
    """Keep camera/inference off the realtime loop and report failures without stopping speech."""

    def __init__(
        self,
        read_frame: Callable[[], Frame | None],
        directory: Path | None,
        angles: CameraAngles | None = None,
        engine_factory: Callable[[], PoseEngine] = PoseEngine,
    ) -> None:
        """Start without inference or preview until the worker thread is launched."""
        self._read_frame = read_frame
        self.directory = directory
        self.angles = angles
        self._engine_factory = engine_factory
        self._lock = threading.RLock()
        self._preview_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._engine: PoseEngine | None = None
        self._rate = 0.0
        self._rate_until = 0.0
        self._preview_until = 0.0
        self._frame: tuple[int, float, Frame, tuple[tuple[float, float, bool], ...]] | None = None
        self._preview_at = -float("inf")
        self._preview_id = -1
        self._latest: PoseSample | None = None
        self._reference = b""
        self._builtin = b""
        self._uploaded = b""
        self._side = "same"
        self._selection = "builtin"
        self._revision = 0
        self._scope = BodyScope()
        self._saved_scopes: dict[str, tuple[str, BodyScope]] = {}
        self.missing_landmarks: tuple[str, ...] = ()
        saved_scopes = directory / "coach-reference-parts.json" if directory is not None else None
        if saved_scopes is not None and saved_scopes.exists():
            try:
                records = json.loads(saved_scopes.read_text(encoding="utf-8"))
                if not isinstance(records, dict):
                    raise ValueError("Invalid saved reference selections")
                for key, record in records.items():
                    if key not in ("builtin:same", "builtin:opposite", "uploaded:same", "uploaded:opposite"):
                        raise ValueError("Unknown saved reference selection")
                    if (
                        not isinstance(record, dict)
                        or not isinstance(record.get("image"), str)
                        or not isinstance(record.get("parts"), list)
                    ):
                        raise ValueError("Invalid saved body parts")
                    self._saved_scopes[key] = (record["image"], BodyScope(tuple(record["parts"])))
            except (ValueError, OSError) as exc:
                logger.warning("Saved body selections unavailable: %s", exc)
                self._saved_scopes.clear()
        self.reference_id = "unconfigured"
        self.status = "starting"
        self.error: str | None = None
        self.reference_points: tuple[tuple[float, float, bool], ...] = ()

    def start(self) -> None:
        """Start one worker thread; repeated calls do not create another camera consumer."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="coach-pose", daemon=True)
            self._thread.start()

    def request_stop(self) -> None:
        """Stop accepting work before the robot's sleep animation begins."""
        self._stop.set()
        with self._lock:
            if self._engine is not None:
                self._engine.stopping.set()

    def stop(self) -> None:
        """Stop sampling and wait for the bounded worker operation to release its process."""
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=55)
            if self._thread.is_alive():
                logger.error("Pose service did not finish within its shutdown deadline")

    def set_rate(self, rate: float) -> None:
        """Lease the requested rate briefly; a stopped coordinator cannot leave inference running."""
        if not 0 <= rate <= 6:
            raise ValueError("Invalid inference rate")
        with self._lock:
            self._rate = rate
            self._rate_until = time.monotonic() + 1

    def latest(self) -> PoseSample | None:
        """Read the latest immutable completed sample without camera or worker I/O."""
        with self._lock:
            return self._latest

    def reference_image(self) -> bytes:
        """Return the selected, validated reference image."""
        with self._lock:
            return self._reference

    def select_reference(
        self, selection: str, side: str, uploaded: bytes | None = None, parts: list[str] | None = None
    ) -> str:
        """Change the reference atomically and discard readings from its predecessor."""
        if selection not in ("builtin", "uploaded") or side not in ("same", "opposite"):
            raise ValueError("Invalid reference selection")
        with self._lock:
            engine = self._engine
        if engine is None:
            raise RuntimeError("Pose runtime is not available")
        jpeg = engine.validate_image(uploaded) if uploaded is not None else None
        with self._lock:
            selected = jpeg if jpeg is not None else self._builtin if selection == "builtin" else self._uploaded
            if not selected:
                raise ValueError("Upload a reference first")
            image_hash = hashlib.sha256(selected).hexdigest()
            records = dict(self._saved_scopes)
            if jpeg is not None:
                records = {key: record for key, record in records.items() if not key.startswith("uploaded:")}
            key = selection + ":" + side
            saved = records.get(key)
            scope = (
                BodyScope(tuple(parts))
                if parts is not None
                else (saved[1] if saved is not None and saved[0] == image_hash else BodyScope())
            )
            records[key] = (image_hash, scope)
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                scope_file = self.directory / "coach-reference-parts.json"
                temporary_scope = scope_file.with_suffix(".json.tmp")
                temporary_scope.write_text(
                    json.dumps(
                        {key: {"image": digest, "parts": value.parts} for key, (digest, value) in records.items()}
                    ),
                    encoding="utf-8",
                )
                temporary_scope.replace(scope_file)
            if jpeg is not None and self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = self.directory / "coach-reference.jpg.tmp"
                temporary.write_bytes(jpeg)
                temporary.replace(self.directory / "coach-reference.jpg")
            if jpeg is not None:
                self._uploaded = jpeg
            self._reference, self._side, self._selection = selected, side, selection
            self._scope, self._saved_scopes = scope, records
            self._revision += 1
            scope_hash = hashlib.sha256(",".join(scope.parts).encode()).hexdigest()[:12]
            self.reference_id = f"ref_{image_hash[:20]}_{side}_{scope_hash}_{self._revision}"
            self.missing_landmarks = ()
            self._latest = self._frame = None
            self.reference_points = ()
            self.status, self.error = "loading", None
            return self.reference_id

    def scope_for(self, reference_id: str) -> BodyScope:
        """Read live-side scope only for the requested revision."""
        with self._lock:
            if reference_id != self.reference_id:
                raise ValueError("Reference changed while reading body selection")
            return self._scope.mapped(self._side)

    def state(self) -> dict[str, object]:
        """Expose availability and reference controls without internal protocol identities."""
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "reference": self._selection,
                "side": self._side,
                "parts": self._scope.parts,
                "part_options": tuple(PARTS),
                "required_reference_landmarks": self._scope.required,
                "required_live_landmarks": self._scope.mapped(self._side).required,
                "missing_landmarks": self.missing_landmarks,
                "revision": self.reference_id,
                "uploaded_available": bool(self._uploaded),
                "reference_ready": self.status == "ready",
                "reference_points": self.reference_points,
                "camera_calibrated": self.angles is not None,
            }

    def preview(self) -> dict[str, object] | None:
        """Encode at most five latest images per second, only while a visible panel requests them."""
        with self._lock:
            now = time.monotonic()
            self._preview_until = now + 1
        if not self._preview_lock.acquire(blocking=False):
            return None
        try:
            with self._lock:
                frame = self._frame
                if frame is None or now - self._preview_at < 0.2 or frame[0] == self._preview_id:
                    return None
                self._preview_at, self._preview_id = now, frame[0]
            if cv2 is None:
                raise RuntimeError("Camera preview requires the existing OpenCV runtime")
            ok, jpeg = cv2.imencode(".jpg", frame[2], [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise RuntimeError("Cannot encode camera preview")
            return {
                "jpeg": base64.b64encode(jpeg).decode("ascii"),
                "points": frame[3],
                "age_s": max(0, now - frame[1]),
            }
        finally:
            self._preview_lock.release()

    def _run(self) -> None:
        engine = None
        try:
            engine = self._engine_factory()
            with self._lock:
                self._engine = engine
                self._builtin = engine.default_reference
                saved = self.directory / "coach-reference.jpg" if self.directory is not None else None
                if saved is not None and saved.exists():
                    try:
                        self._uploaded = engine.validate_image(saved.read_bytes())
                    except (ValueError, OSError) as exc:
                        logger.warning("Saved pose reference is unavailable: %s", exc)
            self.select_reference("builtin", "same")
            revision = -1
            frame_id = 0
            next_inference = next_preview = 0.0
            retries = 0
            while not self._stop.is_set():
                with self._lock:
                    current_revision = self._revision
                    reference, side, reference_id, scope = self._reference, self._side, self.reference_id, self._scope
                    now = time.monotonic()
                    rate = self._rate if now < self._rate_until else 0
                    preview = now < self._preview_until
                    if not preview:
                        self._frame = None
                if current_revision != revision:
                    try:
                        engine.prepare(reference, side, scope)
                    except ValueError as exc:
                        with self._lock:
                            if self._revision == current_revision:
                                self.status, self.error = "invalid_reference", str(exc)
                        logger.warning("Reference rejected: %s", exc)
                        revision = current_revision
                        continue
                    with self._lock:
                        if self._revision == current_revision:
                            self.status, self.error = "ready", None
                            self.reference_points = tuple(
                                (p.x, p.y, p.presence >= 0.5 and p.visibility >= 0.5) for p in engine.reference_points
                            )
                    revision, retries = current_revision, 0
                    continue
                infer = rate > 0 and now >= next_inference and self.status == "ready"
                if not infer and not (preview and rate == 0 and now >= next_preview):
                    self._stop.wait(0.03)
                    continue
                frame = self._read_frame()
                captured = time.monotonic()
                observed = datetime.now(timezone.utc)
                if frame is None:
                    self._stop.wait(0.03)
                    continue
                frame_id += 1
                if frame.shape[1] > 640:
                    if cv2 is None:
                        raise RuntimeError("Camera processing requires the existing OpenCV runtime")
                    frame = cv2.resize(
                        frame,
                        (640, max(1, round(frame.shape[0] * 640 / frame.shape[1]))),
                        interpolation=cv2.INTER_AREA,
                    )
                landmarks: tuple[Landmark, ...] = ()
                missing: tuple[str, ...] = ()
                if infer:
                    try:
                        result = engine.detect(frame, captured)
                        landmarks, missing, reason = result.landmarks, result.missing, result.reason
                        sample = PoseSample(
                            frame_id,
                            reference_id,
                            captured,
                            time.monotonic(),
                            observed,
                            False if reason == Reason.NO_PERSON else bool(landmarks) or None,
                            reason not in FRAMING_REASONS if landmarks else None,
                            result.score,
                            result.errors,
                            reason,
                            landmarks,
                            result.components,
                            result.mirror,
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        logger.error("Pose worker failed: %s", exc)
                        sample = PoseSample(
                            frame_id,
                            reference_id,
                            captured,
                            time.monotonic(),
                            observed,
                            None,
                            None,
                            reason=Reason.WORKER_FAILURE,
                        )
                        with self._lock:
                            self.error = str(exc)
                        if retries < 1:
                            retries += 1
                            engine.prepare(reference, side, scope)
                        else:
                            with self._lock:
                                self.status = "error"
                    with self._lock:
                        if (
                            self._revision == current_revision
                            and time.monotonic() < self._rate_until
                            and self._rate > 0
                        ):
                            self._latest = sample
                            self.missing_landmarks = missing
                            if sample.reason != Reason.WORKER_FAILURE:
                                self.error = None
                    next_inference = captured + 1 / rate
                with self._lock:
                    if self._revision == current_revision and preview:
                        points = tuple((p.x, p.y, p.presence >= 0.5 and p.visibility >= 0.5) for p in landmarks)
                        self._frame = (frame_id, captured, frame, points)
                next_preview = captured + 0.2
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.error("Pose service unavailable: %s", exc)
            with self._lock:
                self.status, self.error = "error", str(exc)
        finally:
            if engine is not None:
                engine.close()
            with self._lock:
                self._engine = None
                self._frame = None
