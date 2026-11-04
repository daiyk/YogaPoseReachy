"""A single-profile app must report a broken profile rather than switch personas."""

import pytest

import reachy_mini_pose_coach.profile_store as store
from reachy_mini_pose_coach.tools import core_tools
from reachy_mini_pose_coach.config import LOCKED_PROFILE, config
from reachy_mini_pose_coach.prompts import get_session_instructions


@pytest.mark.parametrize("document", [None, "invalid front matter"])
def test_broken_coach_fails_clearly(tmp_path, monkeypatch, document):
    """Both prompts and tools reject a missing or corrupt authored profile."""
    directory = tmp_path / LOCKED_PROFILE
    directory.mkdir()
    if document is not None:
        (directory / "profile.md").write_text(document)
    monkeypatch.setattr(store, "DEFAULT_PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", LOCKED_PROFILE)
    with pytest.raises((FileNotFoundError, store.ProfileFormatError)):
        get_session_instructions()
    with pytest.raises(RuntimeError, match="Failed to read tools"):
        core_tools.initialize_tools(force=True)
    assert config.REACHY_MINI_CUSTOM_PROFILE == LOCKED_PROFILE
