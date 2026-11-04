"""Real MediaPipe 0.10.18 Full landmarks of the bundled reference drive the real comparison geometry.

The fixture holds the landmarks of `reference.jpg` (Warrior II, resized to 640 px) and of its horizontal
flip, which is a true mirror image of the pose. No camera, model or detector runs in these tests.
"""

import json
import math
import importlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import reachy_mini_pose_coach.pose.engine as engine_module
from reachy_mini_pose_coach.coach.mirror import MirrorDetector
from reachy_mini_pose_coach.coach.sample import Reason
from reachy_mini_pose_coach.coach.corrections import Corrections


FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "reference_landmarks.json").read_text(encoding="utf-8"))
FRAME = np.zeros((360, 640, 3), dtype=np.uint8)


def landmarks(name):
    """Image and world landmark lists in MediaPipe's attribute shape."""
    data = FIXTURE[name]
    image = [SimpleNamespace(x=x, y=y, visibility=v, presence=p) for x, y, v, p in data["image"]]
    world = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in data["world"]]
    return image, world


class Detector:
    """Return the reference for the IMAGE-mode reference check and a chosen scene for VIDEO mode."""

    def __init__(self, mode, scene):
        """Remember the mode and the shared scene selector."""
        self.mode, self.scene = mode, scene

    def detect(self, rgb, timestamp=None):
        """Return one person's landmarks like the worker client."""
        image, world = landmarks("reference") if self.mode == "IMAGE" else self.scene["value"]
        return SimpleNamespace(pose_landmarks=[image], pose_world_landmarks=[world])

    def close(self):
        """Nothing to release."""


@pytest.fixture
def engine(monkeypatch):
    """Build a PoseEngine on the shared production comparison module and the fixture landmarks."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "reachy_mini_pose_compare"))
    comparison = importlib.import_module("reachy_mini_pose_compare.comparison")
    assets = Path(comparison.__file__).parent / "assets"
    scene = {"value": landmarks("reference")}
    modules = {
        "reachy_mini_pose_compare.detector": SimpleNamespace(
            __file__=str(assets.parent / "detector.py"),
            PoseDetector=lambda model, mode, options: Detector(mode, scene),
        ),
        "reachy_mini_pose_compare.comparison": comparison,
        "reachy_mini_pose_compare.runner": SimpleNamespace(validate_reference=lambda image: image),
    }
    monkeypatch.setattr(engine_module.importlib, "import_module", modules.__getitem__)
    codec = SimpleNamespace(
        IMREAD_COLOR=1, COLOR_BGR2RGB=2, imdecode=lambda *_: FRAME, cvtColor=lambda frame, _: frame
    )
    monkeypatch.setattr(engine_module, "cv2", codec)
    pose = engine_module.PoseEngine()
    pose.prepare(b"jpeg", "same")
    pose.scene = scene
    yield pose
    pose.close()


def rotate_left_arm(degrees):
    """Swing the whole left arm about the shoulder in the camera-facing plane (positive = upward/outward)."""
    image, world = landmarks("reference")
    shoulder = world[11]
    turn = math.radians(degrees)
    for index in (13, 15, 17, 19, 21):
        dx, dy = world[index].x - shoulder.x, world[index].y - shoulder.y
        angle, radius = math.atan2(dx, dy) + turn, math.hypot(dx, dy)
        world[index] = SimpleNamespace(
            x=shoulder.x + radius * math.sin(angle), y=shoulder.y + radius * math.cos(angle), z=world[index].z
        )
    return image, world


def yawed(degrees):
    """Return the same pose with the hips turned by `degrees` about the vertical axis."""
    image, world = landmarks("reference")
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return image, [SimpleNamespace(x=p.x * c - p.z * s, y=p.y, z=p.x * s + p.z * c) for p in world]


def test_fixture_conventions_match_the_geometry_assumptions():
    """World x points toward the person's left (image right) and world y points down, in both images."""
    for name in ("reference", "mirrored"):
        image, world = landmarks(name)
        assert image[23].x > image[24].x and world[23].x > world[24].x
        assert world[11].y < world[23].y < world[25].y


def test_reference_matches_itself_without_mirror_votes(engine):
    """The reference scores fully against itself and is clearly not its own mirror image."""
    result = engine.detect(FRAME, 1)
    assert result.reason is None and result.score == pytest.approx(100, abs=0.5)
    assert all(abs(c.degrees) < 1 for c in result.components)
    assert result.mirror.same_score > result.mirror.opposite_score + 15


@pytest.mark.parametrize(
    "degrees,action", [(20, "lower your left arm about 20°"), (-20, "raise your left arm about 20°")]
)
def test_swinging_the_left_arm_gives_the_opposite_movement_with_its_amount(engine, degrees, action):
    """A 20° frontal swing of the real arm is coached back by about 20° on the anatomical left."""
    engine.scene["value"] = rotate_left_arm(degrees)
    corrections = Corrections()
    for at in (1, 1.33, 1.66):
        result = engine.detect(FRAME, at)
        assert result.reason is None
        primary = corrections.update(at, result.errors, result.components)
    assert primary.joint == "left_upper_arm" and primary.action == action and primary.action_deg == 20
    assert "torso" in corrections.matched_parts and "right_arm" in corrections.matched_parts


def test_mirror_image_is_reported_as_mirrored(engine):
    """The flipped photo fails the active view gate but fits the mirrored reference, for two seconds."""
    engine.scene["value"] = landmarks("mirrored")
    detector, at = MirrorDetector(), 1.0
    while at < 3.5:
        result = engine.detect(FRAME, at)
        assert result.reason in (None, Reason.UNSUPPORTED_VIEW) and result.mirror is not None
        detector.update(at, result.mirror)
        at += 0.33
    assert detector.mirrored and result.mirror.opposite_score >= 70


def test_correct_pose_with_square_hips_is_not_called_mirrored(engine):
    """A correct pose turned to +2° fails the 20° relative-yaw gate, yet must not look mirrored."""
    reference_yaw = engine.comparison.reference.yaw_deg
    engine.scene["value"] = yawed(2 - reference_yaw)
    detector, at = MirrorDetector(), 1.0
    while at < 3.5:
        result = engine.detect(FRAME, at)
        detector.update(at, result.mirror)
        at += 0.33
    assert result.reason == Reason.UNSUPPORTED_VIEW and not detector.mirrored
    assert result.mirror.same_score > result.mirror.opposite_score
