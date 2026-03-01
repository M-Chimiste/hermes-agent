"""Pydantic schemas for memory endpoints."""

from typing import List, Optional

from pydantic import BaseModel


class MemoryResponse(BaseModel):
    target: str
    content: str
    entries: List[str]
    usage: str
    entry_count: int


class MemoryUpdateRequest(BaseModel):
    content: str


class MemoryAddRequest(BaseModel):
    content: str


class MemoryRemoveRequest(BaseModel):
    old_text: str


class MemoryMutationResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    entries: Optional[List[str]] = None
    usage: Optional[str] = None
    entry_count: Optional[int] = None
