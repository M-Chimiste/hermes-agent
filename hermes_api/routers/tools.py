"""Tool listing REST endpoints."""

from typing import List

from fastapi import APIRouter, Depends, Query

from hermes_api.auth import verify_token
from hermes_api.models.tools import ToolDefinition, ToolInfo, ToolsetInfo

router = APIRouter(dependencies=[Depends(verify_token)])


@router.get("", response_model=List[ToolInfo])
async def list_tools():
    import os
    from tools.registry import registry

    tools = []
    for name, entry in sorted(registry._tools.items()):
        available = True
        missing_env = []
        if entry.check_fn:
            try:
                available = entry.check_fn()
            except Exception:
                available = False
        if entry.requires_env:
            for var in entry.requires_env:
                if not os.getenv(var):
                    missing_env.append(var)
        desc = None
        if entry.schema and isinstance(entry.schema, dict):
            desc = entry.schema.get("description")
        elif entry.description:
            desc = entry.description
        tools.append(ToolInfo(
            name=name,
            description=desc,
            toolset=entry.toolset,
            available=available,
            missing_env=missing_env or None,
        ))
    return tools


@router.get("/definitions", response_model=List[ToolDefinition])
async def get_tool_definitions(
    enabled: List[str] = Query(None, description="Toolsets to enable"),
    disabled: List[str] = Query(None, description="Toolsets to disable"),
):
    from model_tools import get_tool_definitions

    defs = get_tool_definitions(
        enabled_toolsets=enabled,
        disabled_toolsets=disabled,
    )
    return [ToolDefinition(**d) for d in defs]


@router.get("/toolsets", response_model=List[ToolsetInfo])
async def list_toolsets():
    from toolsets import TOOLSETS

    result = []
    for name, entry in sorted(TOOLSETS.items()):
        if isinstance(entry, dict):
            tool_list = entry.get("tools", [])
            description = entry.get("description")
        else:
            tool_list = list(entry) if isinstance(entry, (list, set, tuple)) else [str(entry)]
            description = None
        result.append(ToolsetInfo(
            name=name,
            tools=tool_list,
            description=description,
        ))
    return result
