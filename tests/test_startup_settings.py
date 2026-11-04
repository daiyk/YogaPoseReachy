"""Startup voice persistence, including old files with an obsolete profile field."""

import pytest

from reachy_mini_pose_coach.startup_settings import StartupSettings, read_startup_settings, write_startup_settings


def test_voice_round_trip_and_clear(tmp_path):
    """Voice overrides persist and clearing them restores the profile default."""
    write_startup_settings(tmp_path, voice="Serena")
    assert read_startup_settings(tmp_path) == StartupSettings(voice="Serena")
    write_startup_settings(tmp_path, voice=None)
    assert read_startup_settings(tmp_path) == StartupSettings()


def test_old_profile_selection_is_ignored(tmp_path):
    """A legacy startup personality cannot replace the coach."""
    (tmp_path / "startup_settings.json").write_text('{"profile": "sorry_bro", "voice": "Serena"}')
    assert read_startup_settings(tmp_path) == StartupSettings(voice="Serena")


@pytest.mark.parametrize("content", ["{", "[]", '{"voice": 123}', '{"voice": " "}'])
def test_invalid_startup_voice_is_ignored(tmp_path, content):
    """Malformed persisted values leave the authored voice in place."""
    (tmp_path / "startup_settings.json").write_text(content)
    assert read_startup_settings(tmp_path) == StartupSettings()
