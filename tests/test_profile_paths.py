"""Single-profile prompts and actual wheel packaging."""

import sys
import shutil
import zipfile
import subprocess
from pathlib import Path, PurePosixPath

import pytest

import reachy_mini_pose_coach.config as config_mod
from reachy_mini_pose_coach.prompts import get_session_voice, get_session_instructions, get_session_greeting_prompt
from reachy_mini_pose_coach.profile_store import read_profile


WINDOWS_PATH_BUDGET = 130
WINDOWS_WHEEL_PATH_BUDGET = 71


def _source_files(root: Path) -> list[Path]:
    """Include package inputs even in a newly scaffolded, uncommitted checkout."""
    files = [root / name for name in ("pyproject.toml", "setup.py", "README.md", "MANIFEST.in")]
    for folder in ("src", "profiles"):
        files.extend(
            p
            for p in (root / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and not any(part.endswith(".egg-info") for part in p.parts)
        )
    return [p for p in files if p.is_file()]


def test_single_coach_profile_loads_prompts_and_tools(monkeypatch):
    """The only bundled personality supplies instructions, voice and exit tool."""
    monkeypatch.setattr(config_mod.config, "MEMORY_ENABLED", False)
    monkeypatch.setattr(config_mod.config, "REALTIME_TRANSCRIPTION_LANGUAGE", "auto")
    profile = read_profile(None)
    assert get_session_instructions() == profile.instructions
    assert get_session_voice()
    assert get_session_greeting_prompt()
    assert "go_to_sleep" in profile.default_tools
    assert list(config_mod.DEFAULT_PROFILES_DIRECTORY.glob("*/profile.md")) == [
        config_mod.DEFAULT_PROFILES_DIRECTORY / config_mod.LOCKED_PROFILE / "profile.md"
    ]


def test_packaged_profiles_win_outside_source_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Installed builds should use packaged profiles, not an unrelated sibling folder."""
    unrelated_profiles = tmp_path / "profiles"
    unrelated_profiles.mkdir()
    packaged_profiles = tmp_path / "package_data" / "profiles"
    packaged_profiles.mkdir(parents=True)

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "_packaged_profiles_directory", lambda: packaged_profiles)

    assert config_mod._resolve_default_profiles_directory() == packaged_profiles


def test_project_file_paths_stay_within_windows_budget() -> None:
    """Packaged project file paths should stay below the agreed Windows budget."""
    project_root = Path(__file__).parents[1].resolve()
    project_files = _source_files(project_root)

    violations = []
    for path in project_files:
        relative = str(Path(project_root.name) / path.relative_to(project_root))
        length = len(relative)
        if length > WINDOWS_PATH_BUDGET:
            violations.append(
                f"Windows path budget exceeded ({WINDOWS_PATH_BUDGET}): {relative} is {length} characters long"
            )

    assert not violations, "\n".join(violations)


def test_wheel_file_paths_stay_within_windows_budget(tmp_path: Path) -> None:
    """Built wheel paths should stay below the agreed Windows budget."""
    project_root = Path(__file__).parents[1].resolve()
    source_checkout = tmp_path / "checkout"
    dist_dir = tmp_path / "dist"

    for source_file in _source_files(project_root):
        target_file = source_checkout / source_file.relative_to(project_root)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(dist_dir),
                ".",
            ],
            cwd=source_checkout,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        details = exc.stderr if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        pytest.fail(f"Wheel build failed while checking Windows path budget: {details}")

    wheel_files = list(dist_dir.glob("*.whl"))
    assert len(wheel_files) == 1, f"Expected exactly one built wheel in {dist_dir}, found: {wheel_files}"

    with zipfile.ZipFile(wheel_files[0]) as archive:
        assert f"reachy_talk_data/profiles/{config_mod.LOCKED_PROFILE}/profile.md" in archive.namelist()
        assert archive.read("reachy_mini_pose_coach/pose/assets/reference.jpg").startswith(b"\xff\xd8")
        assert len(archive.read("reachy_mini_pose_coach/pose/assets/pose_landmarker_full_v1.task")) > 1_000_000
        assert b"mediapipe==0.10.18" in archive.read("reachy_mini_pose_coach/pose/runtime/requirements-arm64.lock.txt")
        for info in archive.infolist():
            if not info.is_dir():
                assert not archive.read(info).startswith(b"version https://git-lfs.github.com/spec/v1"), info.filename
        archived_paths = [PurePosixPath(info.filename) for info in archive.infolist() if not info.is_dir()]

    violations = []
    for path in archived_paths:
        length = len(path.as_posix())
        if length > WINDOWS_WHEEL_PATH_BUDGET:
            violations.append(
                f"Windows wheel path budget exceeded ({WINDOWS_WHEEL_PATH_BUDGET}): "
                f"{path.as_posix()} is {length} characters long"
            )

    assert not violations, "\n".join(violations)
