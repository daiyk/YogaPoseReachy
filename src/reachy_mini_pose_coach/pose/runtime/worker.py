"""Private stdin/stdout protocol: one RGB frame in, one numeric pose result out.

Runs as a standalone script inside the pose runtime's Python (see `environment.py`), so it imports nothing
from this package. Copied from `reachy_mini_pose_compare`.
"""

import sys
import json
import time
from typing import Any, BinaryIO


DEFAULT_OPTIONS: dict[str, float] = {
    "num_poses": 1,
    "inference_threads": 2,
    "min_pose_detection_confidence": 0.5,
    "min_pose_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}


def cpu_base_options(mp: Any, model_path: str, threads: int) -> tuple[Any, int | str]:
    """MediaPipe's CPU path runs one thread on Linux; 0.10.x lets us request XNNPACK threads instead."""
    base = mp.tasks.BaseOptions(model_asset_path=model_path, delegate=mp.tasks.BaseOptions.Delegate.CPU)
    if threads < 1 or not hasattr(base, "to_pb2"):  # 1.x builds a C struct with no thread option.
        return base, "default"
    build = base.to_pb2

    def to_pb2() -> Any:
        proto = build()
        proto.acceleration.xnnpack.num_threads = threads
        return proto

    base.to_pb2 = to_pb2
    return base, threads


def send(value: dict[str, Any]) -> None:
    """Write one JSON line to the parent process."""
    sys.stdout.buffer.write(json.dumps(value, allow_nan=False).encode() + b"\n")
    sys.stdout.buffer.flush()


def read_frame(stream: BinaryIO, shape: object) -> bytes:
    """Read exactly one bounded RGB frame."""
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(v) is not int for v in shape)
        or shape[2] != 3
        or not all(0 < v <= 4096 for v in shape[:2])
        or shape[0] * shape[1] > 12_000_000
    ):
        raise ValueError("Invalid frame dimensions.")
    size = shape[0] * shape[1] * shape[2]
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Incomplete camera frame.")
    return data


def main() -> None:
    """Create one landmarker and answer frames until stdin closes."""
    import numpy as np
    import mediapipe as mp

    mode = sys.argv[2]
    if mode not in ("IMAGE", "VIDEO"):
        raise ValueError("Invalid detector mode.")
    settings = {**DEFAULT_OPTIONS, **(json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})}
    base, threads = cpu_base_options(mp, sys.argv[1], int(settings["inference_threads"]))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=getattr(mp.tasks.vision.RunningMode, mode),
        num_poses=int(settings["num_poses"]),
        min_pose_detection_confidence=settings["min_pose_detection_confidence"],
        min_pose_presence_confidence=settings["min_pose_presence_confidence"],
        min_tracking_confidence=settings["min_tracking_confidence"],
        output_segmentation_masks=False,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        send(
            {
                "runtime": {
                    "mediapipe": mp.__version__,
                    "numpy": np.__version__,
                    "python": sys.version.split()[0],
                    **settings,
                    "inference_threads": threads,
                }
            }
        )
        while line := sys.stdin.buffer.readline(4096):
            header = json.loads(line)
            shape = header["shape"]
            data = read_frame(sys.stdin.buffer, shape)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.frombuffer(data, np.uint8).reshape(shape))
            started = time.perf_counter()
            result = (
                detector.detect(image) if mode == "IMAGE" else detector.detect_for_video(image, header["timestamp"])
            )
            inference_ms = (time.perf_counter() - started) * 1000
            payload = {
                field: [
                    [[p.x, p.y, p.z, p.visibility or 0.0, p.presence or 0.0] for p in pose]
                    for pose in getattr(result, field)
                ]
                for field in ("pose_landmarks", "pose_world_landmarks")
            }
            send({**payload, "model_inference_ms": inference_ms})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        send({"error": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
