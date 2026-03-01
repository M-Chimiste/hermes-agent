"""Pydantic schemas for config endpoints."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class ConfigResponse(BaseModel):
    config: Dict[str, Any]


class ConfigPatchRequest(BaseModel):
    """Partial config update — keys are dotted paths (e.g. 'tts.provider')."""
    updates: Dict[str, Any]


class ConfigStatusResponse(BaseModel):
    current_version: int
    latest_version: int
    missing_fields: List[Dict[str, Any]]
