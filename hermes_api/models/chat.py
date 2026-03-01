"""Pydantic schemas for WebSocket chat protocol."""

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel


class ServerEventType(str, Enum):
    STATUS = "status"
    TOOL_PROGRESS = "tool_progress"
    TOOL_RESULT = "tool_result"
    RESPONSE = "response"
    CLARIFY = "clarify"
    SESSION_INFO = "session_info"
    ERROR = "error"


class ClientMessageType(str, Enum):
    MESSAGE = "message"
    INTERRUPT = "interrupt"
    CLARIFY_RESPONSE = "clarify_response"


class ClientMessage(BaseModel):
    type: ClientMessageType
    content: Optional[str] = None
    session_id: Optional[str] = None
    answer: Optional[str] = None


class ServerEvent(BaseModel):
    type: ServerEventType
    # status
    status: Optional[str] = None
    # tool_progress / tool_result
    tool_name: Optional[str] = None
    preview: Optional[str] = None
    result: Optional[str] = None
    success: Optional[bool] = None
    # response
    content: Optional[str] = None
    api_calls: Optional[int] = None
    completed: Optional[bool] = None
    interrupted: Optional[bool] = None
    # clarify
    question: Optional[str] = None
    choices: Optional[List[str]] = None
    # session_info
    session_id: Optional[str] = None
    message_count: Optional[int] = None
    resumed: Optional[bool] = None
    # error
    message: Optional[str] = None
