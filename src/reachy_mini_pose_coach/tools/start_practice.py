"""Start the local practice state without claiming that a live pose was detected."""

from .core_tools import Tool, ToolDependencies


class StartPractice(Tool):
    """Start practice idempotently and request one normal tool-batch acknowledgement."""

    name = "start_practice"
    description = "Start pose practice. A started session may still have no valid pose observation."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    needs_response = True

    async def __call__(self, deps: ToolDependencies, **kwargs: object) -> dict[str, object]:
        """Enable practice and report whether its state changed."""
        if deps.coach is None:
            return {"error": "Pose coaching service is unavailable"}
        changed = deps.coach.set_practice(True)
        return {"practice_active": True, "changed": changed, "pose_context": deps.coach.status_text()}
