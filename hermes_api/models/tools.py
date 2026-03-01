"""Pydantic schemas for tool endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolInfo(BaseModel):
    name: str
    description: Optional[str] = None
    toolset: Optional[str] = None
    available: bool = True
    missing_env: Optional[List[str]] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: Dict[str, Any]


class ToolsetInfo(BaseModel):
    name: str
    tools: List[str]
    description: Optional[str] = None
