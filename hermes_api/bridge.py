"""Sync AIAgent → async WebSocket bridge.

Adapts the proven gateway pattern (gateway/run.py:1454-1610):
  - AIAgent created per conversation turn
  - tool_progress_callback pushes events to a queue.Queue
  - clarify_callback blocks agent thread, waits for WS response
  - Agent runs in thread pool via loop.run_in_executor()
  - Async task polls queue and sends JSON to WebSocket
"""

import asyncio
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_api.models.chat import ServerEventType

logger = logging.getLogger(__name__)

# Shared thread pool for agent executions
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hermes-agent")


@dataclass
class AgentSession:
    """Tracks an active agent conversation over WebSocket."""
    session_id: str
    agent: Optional[Any] = None
    event_queue: queue.Queue = field(default_factory=queue.Queue)
    is_processing: bool = False
    clarify_response_queue: queue.Queue = field(default_factory=queue.Queue)


class AgentBridge:
    """Bridges sync AIAgent to async WebSocket with real-time streaming."""

    def __init__(self, session_db=None, model_catalog=None):
        self._session_db = session_db
        self._model_catalog = model_catalog
        self._active_sessions: Dict[str, AgentSession] = {}

    def get_or_create_session(self, session_id: str) -> AgentSession:
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = AgentSession(session_id=session_id)
        return self._active_sessions[session_id]

    def interrupt_session(self, session_id: str, message: Optional[str] = None):
        session = self._active_sessions.get(session_id)
        if session and session.agent:
            session.agent.interrupt(message)

    def respond_to_clarify(self, session_id: str, answer: str):
        session = self._active_sessions.get(session_id)
        if session:
            session.clarify_response_queue.put(answer)

    async def run_conversation(
        self,
        session_id: str,
        user_message: str,
        send_json,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Optional[Dict]:
        """Run an agent conversation, streaming events via send_json callback.

        Args:
            session_id: Session identifier (new or existing for resume)
            user_message: The user's message
            send_json: Async callable that sends a dict as JSON to the client
            conversation_history: Optional prior messages for session resume
        """
        session = self.get_or_create_session(session_id)
        session.is_processing = True
        event_queue = session.event_queue

        # --- Callbacks invoked from the agent thread ---

        def progress_callback(tool_name: str, preview: str = None):
            event_queue.put({
                "type": ServerEventType.TOOL_PROGRESS,
                "tool_name": tool_name,
                "preview": preview,
            })

        def clarify_callback(question: str, choices: list = None) -> str:
            event_queue.put({
                "type": ServerEventType.CLARIFY,
                "question": question,
                "choices": choices,
            })
            try:
                return session.clarify_response_queue.get(timeout=300)
            except queue.Empty:
                return ""

        # --- Sync function to run in thread pool ---

        def run_sync():
            from hermes_cli.config import load_config
            from run_agent import AIAgent

            config = load_config()
            model = config.get("model", "anthropic/claude-opus-4.6")
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            base_url = config.get("base_url")
            enabled_toolsets = config.get("toolsets", ["hermes-cli"])
            max_iterations = config.get("max_turns", 60)

            # Resolve Nous Portal credentials if needed
            provider = config.get("provider", "")
            if provider == "nous":
                try:
                    from hermes_cli.auth import resolve_nous_runtime_credentials
                    creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=5 * 60)
                    api_key = creds.get("api_key", api_key)
                    base_url = creds.get("base_url", base_url)
                except Exception:
                    pass

            event_queue.put({"type": ServerEventType.STATUS, "status": "initializing"})

            agent = AIAgent(
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_iterations=max_iterations,
                quiet_mode=True,
                enabled_toolsets=enabled_toolsets,
                session_id=session_id,
                tool_progress_callback=progress_callback,
                clarify_callback=clarify_callback,
                platform="web",
                session_db=self._session_db,
                model_catalog=self._model_catalog,
            )
            session.agent = agent

            event_queue.put({"type": ServerEventType.STATUS, "status": "processing"})

            result = agent.run_conversation(
                user_message,
                conversation_history=conversation_history,
            )

            event_queue.put({
                "type": ServerEventType.RESPONSE,
                "content": result.get("final_response"),
                "api_calls": result.get("api_calls"),
                "completed": result.get("completed"),
                "interrupted": result.get("interrupted", False),
            })

            if result.get("error"):
                event_queue.put({
                    "type": ServerEventType.ERROR,
                    "message": result["error"],
                })

            return result

        # --- Async event forwarder ---

        async def forward_events():
            while True:
                try:
                    event = event_queue.get_nowait()
                    await send_json(event)
                except queue.Empty:
                    await asyncio.sleep(0.1)
                except Exception:
                    break

        # --- Orchestrate ---

        await send_json({
            "type": ServerEventType.SESSION_INFO,
            "session_id": session_id,
            "resumed": conversation_history is not None and len(conversation_history) > 0,
            "message_count": len(conversation_history) if conversation_history else 0,
        })

        forward_task = asyncio.create_task(forward_events())
        result = None

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_executor, run_sync)
        except Exception as exc:
            await send_json({
                "type": ServerEventType.ERROR,
                "message": str(exc),
            })
        finally:
            # Drain remaining events
            await asyncio.sleep(0.3)
            forward_task.cancel()
            session.is_processing = False
            session.agent = None

            await send_json({"type": ServerEventType.STATUS, "status": "done"})

        return result
