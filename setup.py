from shutil import copy2
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


PROJECT_ROOT = Path(__file__).parent.resolve()
SOURCE_PROFILES_DIR = PROJECT_ROOT / "profiles"
TARGET_PACKAGE = "reachy_talk_data"
TARGET_SUBDIR = "profiles"


class BuildPyWithProfiles(build_py):
    """Copy built-in profiles into the wheel data package at build time."""

    def run(self) -> None:
        """Build Python modules, then copy root-level profiles into reachy_talk_data."""
        # Source archives/clones without Git LFS must not produce a broken wheel.
        for asset in (PROJECT_ROOT / "src").rglob("*"):
            if asset.is_file():
                with asset.open("rb") as handle:
                    if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                        raise RuntimeError(f"Unresolved Git LFS asset: {asset}. Run git lfs install and git lfs pull.")
        super().run()

        target_root = Path(self.build_lib) / TARGET_PACKAGE / TARGET_SUBDIR
        for profile_document in SOURCE_PROFILES_DIR.glob("*/profile.md"):
            target_directory = target_root / profile_document.parent.name
            target_directory.mkdir(parents=True, exist_ok=True)
            copy2(profile_document, target_directory / profile_document.name)


setup(
    cmdclass={"build_py": BuildPyWithProfiles},
)
