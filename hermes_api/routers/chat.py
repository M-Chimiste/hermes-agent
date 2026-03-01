"""WebSocket chat endpoint."""

import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from hermes_api.auth import verify_ws_token
from hermes_api.bridge import AgentBridge
from hermes_api.dependencies import get_model_catalog, get_session_db
from hermes_api.models.chat import ClientMessageType, ServerEventType

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared bridge instance (created lazily)
_bridge = None


def _get_bridge() -> AgentBridge:
    global _bridge
    if _bridge is None:
        _bridge = AgentBridge(
            session_db=get_session_db(),
            model_catalog=get_model_catalog(),
        )
    return _bridge


@router.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()

    # Authenticate
    if not await verify_ws_token(websocket):
        return

    bridge = _get_bridge()
    current_session_id = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": ServerEventType.ERROR,
                    "message": "Invalid JSON",
                })
                continue

            msg_type = msg.get("type")

            if msg_type == ClientMessageType.MESSAGE:
                content = msg.get("content", "").strip()
                if not content:
                    continue

                # Determine session
                session_id = msg.get("session_id") or current_session_id
                if not session_id:
                    session_id = f"web_{int(time.time())}_{id(websocket) % 10000:04d}"
                current_session_id = session_id

                # Load history if resuming an existing session
                conversation_history = None
                if msg.get("session_id"):
                    try:
                        db = get_session_db()
                        conversation_history = db.get_messages_as_conversation(session_id)
                    except Exception:
                        conversation_history = None

                # Check if we're interrupting an active session
                session = bridge._active_sessions.get(session_id)
                if session and session.is_processing:
                    bridge.interrupt_session(session_id, content)
                    continue

                await bridge.run_conversation(
                    session_id=session_id,
                    user_message=content,
                    send_json=websocket.send_json,
                    conversation_history=conversation_history,
                )

            elif msg_type == ClientMessageType.INTERRUPT:
                if current_session_id:
                    bridge.interrupt_session(current_session_id, msg.get("content"))

            elif msg_type == ClientMessageType.CLARIFY_RESPONSE:
                if current_session_id:
                    bridge.respond_to_clarify(current_session_id, msg.get("answer", ""))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await websocket.send_json({
                "type": ServerEventType.ERROR,
                "message": str(exc),
            })
        except Exception:
            pass
