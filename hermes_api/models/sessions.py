"""Pydantic schemas for session endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SessionSummary(BaseModel):
    id: str
    source: str
    model: Optional[str] = None
    started_at: float
    ended_at: Optional[float] = None
    end_reason: Optional[str] = None
    message_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class SessionDetail(SessionSummary):
    user_id: Optional[str] = None
    model_config_data: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    parent_session_id: Optional[str] = None


class SessionMessage(BaseModel):
    id: int
    session_id: str
    role: str
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[Any] = None
    tool_name: Optional[str] = None
    timestamp: float
    token_count: Optional[int] = None
    finish_reason: Optional[str] = None


class SessionSearchRequest(BaseModel):
    query: str
    source_filter: Optional[List[str]] = None
    role_filter: Optional[List[str]] = None
    limit: int = 20
    offset: int = 0


class SessionSearchResult(BaseModel):
    id: int
    session_id: str
    role: str
    snippet: str
    timestamp: float
    tool_name: Optional[str] = None
    source: str
    model: Optional[str] = None
    context: Optional[List[Dict[str, Any]]] = None
