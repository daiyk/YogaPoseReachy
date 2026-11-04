"""Requested inference cadence; actual scored throughput still needs hardware validation."""

from .state import Mode, Activity
from .settings import CoachSettings


def inference_rate(
    activity: Activity, now: float, person_present: bool | None, settings: CoachSettings | None = None
) -> float:
    """Apply activity guards and configured rates, preserving the offline default policy."""
    if not activity.can_sample(now):
        return 0.0
    if person_present is False:
        return settings.searching_fps if settings else 1.0
    if settings is None:
        return 3.0 if activity.mode == Mode.LISTENING else 5.6
    return {Mode.LISTENING: settings.listening_fps, Mode.THINKING: settings.thinking_fps}.get(
        activity.mode, settings.idle_fps
    )
