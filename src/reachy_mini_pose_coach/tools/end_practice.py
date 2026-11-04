"""End practice while keeping ordinary conversation and tool results available."""

from .core_tools import Tool, ToolDependencies


class EndPractice(Tool):
    """Stop practice idempotently, with one normal tool-batch acknowledgement."""

    name = "end_practice"
    description = "End pose practice and clear pending coaching feedback; ordinary conversation continues."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    needs_response = True

    async def __call__(self, deps: ToolDependencies, **kwargs: object) -> dict[str, object]:
        """Disable practice and clear queued coach cues through the movement manager."""
        if deps.coach is None:
            return {"error": "Pose coaching service is unavailable"}
        changed = deps.coach.set_practice(False)
        if deps.coach.movement is None:
            deps.movement_manager.end_practice_motion()
        return {"practice_active": False, "changed": changed}
