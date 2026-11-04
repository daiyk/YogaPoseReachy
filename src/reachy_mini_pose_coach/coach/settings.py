"""Validated coaching controls shared by the panel, scoring and pose service."""

import json
import math
import logging
from pathlib import Path

from pydantic import Field, BaseModel, ConfigDict, model_validator


logger = logging.getLogger(__name__)
DEFAULT_COOLDOWNS = {
    "person_lost": 10.0,
    "framing_bad": 10.0,
    "pose_achieved": 15.0,
    "got_worse": 7.0,
    "got_better": 5.0,
    "correction_changed": 10.0,
    "stalled": 15.0,
    "mirrored": 20.0,
}


class CoachSettings(BaseModel):
    """Bound tunable thresholds while keeping listening and movement safety guards fixed."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    band_edges: tuple[float, float, float] = (50.0, 70.0, 85.0)
    cooldowns: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_COOLDOWNS))
    speech_gap_s: float = Field(default=5.0, ge=5, le=60)
    idle_fps: float = Field(default=3.0, ge=0.5, le=6)
    listening_fps: float = Field(default=3.0, ge=0.5, le=6)
    thinking_fps: float = Field(default=3.0, ge=0.5, le=6)
    searching_fps: float = Field(default=1.0, ge=0.5, le=3)
    framing_enabled: bool = False
    horizontal_fov_deg: float | None = Field(default=None, ge=20, le=160)
    vertical_fov_deg: float | None = Field(default=None, ge=15, le=130)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "CoachSettings":
        """Require ordered bands, complete cooldowns and paired measured camera angles."""
        a, b, c = self.band_edges
        if not (5 <= a and a + 6 <= b and b + 6 <= c <= 95):
            raise ValueError("Band boundaries must increase by at least six points within 5–95")
        if set(self.cooldowns) != set(DEFAULT_COOLDOWNS):
            raise ValueError("Provide a cooldown for each coaching event")
        if any(not math.isfinite(value) or not 2 <= value <= 120 for value in self.cooldowns.values()):
            raise ValueError("Event cooldowns must be between 2 and 120 seconds")
        if (self.horizontal_fov_deg is None) != (self.vertical_fov_deg is None):
            raise ValueError("Provide both measured camera field-of-view angles")
        return self


def load_coach_settings(directory: Path | None) -> CoachSettings:
    """Read local settings without letting a corrupt file prevent conversation startup."""
    if directory is None or not (directory / "coach-settings.json").exists():
        return CoachSettings()
    try:
        data = json.loads((directory / "coach-settings.json").read_text(encoding="utf-8"))
        cooldowns = data.get("cooldowns") if isinstance(data, dict) else None
        if isinstance(cooldowns, dict) and cooldowns:
            # Files saved before an event kind existed keep their values and gain its default cooldown.
            data["cooldowns"] = {**DEFAULT_COOLDOWNS, **cooldowns}
        return CoachSettings.model_validate(data)
    except (OSError, ValueError) as exc:
        logger.warning("Cannot load coach settings; using defaults: %s", exc)
        return CoachSettings()


def save_coach_settings(directory: Path, settings: CoachSettings) -> None:
    """Replace validated settings atomically in the app's existing instance directory."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "coach-settings.json.tmp"
    temporary.write_text(json.dumps(settings.model_dump(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "coach-settings.json")
