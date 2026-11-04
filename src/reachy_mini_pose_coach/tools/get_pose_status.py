"""Refresh the same numeric pose context that accompanies audio turns."""

import json

from .core_tools import Tool, ToolDependencies


class GetPoseStatus(Tool):
    """Return the current cached status without camera or inference I/O."""

    name = "get_pose_status"
    description = "Refresh pose status when the attached observation is missing or stale. Never captures an image."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, deps: ToolDependencies, **kwargs: object) -> dict[str, object]:
        """Return the validated context schema or an explicit unavailable-service error."""
        if deps.coach is None:
            return {"error": "Pose coaching service is unavailable"}
        status = json.loads(deps.coach.status_text().split("\n", 1)[1])
        if not isinstance(status, dict):
            raise ValueError("Pose status must be an object")
        return status
