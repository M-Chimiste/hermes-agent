"""Session management REST endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from hermes_api.auth import verify_token
from hermes_api.dependencies import get_session_db
from hermes_api.models.sessions import (
    SessionDetail,
    SessionMessage,
    SessionSearchRequest,
    SessionSearchResult,
    SessionSummary,
)

router = APIRouter(dependencies=[Depends(verify_token)])


@router.get("", response_model=List[SessionSummary])
async def list_sessions(
    source: Optional[str] = Query(None, description="Filter by source (cli, telegram, discord, etc.)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    db = get_session_db()
    rows = db.search_sessions(source=source, limit=limit, offset=offset)
    return [SessionSummary(**row) for row in rows]


@router.get("/count")
async def session_count(
    source: Optional[str] = Query(None),
):
    db = get_session_db()
    return {"count": db.session_count(source=source)}


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    db = get_session_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Rename model_config to avoid Pydantic conflict
    if "model_config" in session:
        session["model_config_data"] = session.pop("model_config")
    return SessionDetail(**session)


@router.get("/{session_id}/messages", response_model=List[SessionMessage])
async def get_session_messages(session_id: str):
    db = get_session_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = db.get_messages(session_id)
    return [SessionMessage(**msg) for msg in messages]


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    db = get_session_db()
    deleted = db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}


@router.post("/search", response_model=List[SessionSearchResult])
async def search_messages(req: SessionSearchRequest):
    db = get_session_db()
    results = db.search_messages(
        query=req.query,
        source_filter=req.source_filter,
        role_filter=req.role_filter,
        limit=req.limit,
        offset=req.offset,
    )
    return [SessionSearchResult(**r) for r in results]


@router.get("/{session_id}/export")
async def export_session(session_id: str):
    db = get_session_db()
    data = db.export_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data
