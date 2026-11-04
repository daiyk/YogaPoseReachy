"""Catalog of built-in, external and installed tools."""

import logging
from typing import Literal, TypedDict
from pathlib import Path

from reachy_mini_pose_coach.config import (
    config,
    list_tool_module_names,
)
from reachy_mini_pose_coach.tool_spaces import read_installed_tool_spaces
from reachy_mini_pose_coach.tools.tool_constants import SystemTool


logger = logging.getLogger(__name__)


class AvailableTool(TypedDict):
    """Tool metadata exposed by the read-only coach inventory."""

    id: str
    kind: Literal["shared", "external", "tool_space"]
    source: str
    description: str


def available_tool_catalog() -> list[AvailableTool]:
    """List configurable tools and their source."""
    catalog: dict[str, AvailableTool] = {}
    excluded_modules = {"__init__", "core_tools", "background_tool_manager", "tool_constants"}
    excluded_modules.update(tool.value for tool in SystemTool)
    for tool_name in list_tool_module_names(Path(__file__).parent / "tools"):
        if tool_name in excluded_modules:
            continue
        catalog[tool_name] = {
            "id": tool_name,
            "kind": "shared",
            "source": "Built-in",
            "description": "",
        }

    for tool_name in list_tool_module_names(config.TOOLS_DIRECTORY):
        catalog[tool_name] = {
            "id": tool_name,
            "kind": "external",
            "source": "External",
            "description": "",
        }

    try:
        for space in read_installed_tool_spaces(config.INSTANCE_PATH).spaces:
            for tool in space.tools:
                catalog[tool.local_name] = {
                    "id": tool.local_name,
                    "kind": "tool_space",
                    "source": space.slug,
                    "description": tool.description,
                }
    except (RuntimeError, ValueError) as exc:
        logger.warning("Failed to list installed Tool Space tools: %s", exc)
    return [catalog[tool_id] for tool_id in sorted(catalog)]
