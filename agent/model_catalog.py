"""Model Catalog — Multi-server model registry with health checking.

Registers multiple local LLM servers (LMStudio, etc.) so the orchestrator
agent can delegate tasks to specific models via ``delegate_task``.

The catalog is loaded from ``~/.hermes/model_catalog.yaml`` at startup.
When the file doesn't exist or contains no entries, the entire feature
is invisible — no tools registered, no system prompt injection, no threads.

Usage::

    from agent.model_catalog import ModelCatalog, get_catalog, set_catalog

    catalog = ModelCatalog()
    n = catalog.load()          # returns number of entries loaded
    if n > 0:
        catalog.check_health_sync()
        catalog.start_periodic_health_check()
    set_catalog(catalog)        # make available as singleton

    # Later, from tool handlers:
    cat = get_catalog()
    entry = cat.get_entry("local-qwen-32b")
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    """One registered model server."""

    id: str
    url: str
    model: str
    api_key: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    # Runtime-only (never serialized back to YAML):
    status: str = "pending"        # "pending" | "available" | "not_reachable"
    last_check: float = 0.0       # time.monotonic() of last health check
    error: str = ""                # last error message if not_reachable


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_catalog_instance: Optional["ModelCatalog"] = None


def get_catalog() -> Optional["ModelCatalog"]:
    """Return the global ModelCatalog singleton (None if not initialised)."""
    return _catalog_instance


def set_catalog(catalog: "ModelCatalog") -> None:
    """Set the global ModelCatalog singleton."""
    global _catalog_instance
    _catalog_instance = catalog


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ModelCatalog:
    """Manages a catalog of local LLM servers."""

    def __init__(self, catalog_path: Optional[Path] = None):
        if catalog_path is None:
            from hermes_cli.config import get_hermes_home
            catalog_path = get_hermes_home() / "model_catalog.yaml"
        self._path = catalog_path
        self._entries: Dict[str, CatalogEntry] = {}
        self._lock = threading.Lock()

        # Health-check settings (overridden from YAML)
        self._interval: int = 300       # seconds between periodic checks
        self._timeout: int = 5          # HTTP timeout per server
        self._health_method: str = "models"   # "models" or "completions"

        # Background health-check thread
        self._health_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> int:
        """Load catalog from YAML.  Returns number of entries loaded.

        Idempotent — can be called again to reload.  If the file doesn't
        exist or is empty, returns 0 silently.
        """
        if not self._path.exists():
            return 0

        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read model catalog %s: %s", self._path, exc)
            return 0

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            logger.warning("Invalid YAML in model catalog %s: %s", self._path, exc)
            return 0

        if not isinstance(data, dict):
            return 0

        # Health-check config
        hc = data.get("health_check", {})
        if isinstance(hc, dict):
            self._interval = int(hc.get("interval_seconds", self._interval))
            self._timeout = int(hc.get("timeout_seconds", self._timeout))
            method = hc.get("method", self._health_method)
            if method in ("models", "completions"):
                self._health_method = method

        # Model entries
        models_raw = data.get("models", [])
        if not isinstance(models_raw, list):
            return 0

        entries: Dict[str, CatalogEntry] = {}
        for item in models_raw:
            if not isinstance(item, dict):
                continue
            entry_id = item.get("id", "").strip()
            url = item.get("url", "").strip()
            model = item.get("model", "").strip()
            if not entry_id or not url or not model:
                logger.warning("Skipping catalog entry with missing id/url/model: %s", item)
                continue
            entries[entry_id] = CatalogEntry(
                id=entry_id,
                url=url.rstrip("/"),
                model=model,
                api_key=str(item.get("api_key", "") or ""),
                tags=item.get("tags", {}),
                description=str(item.get("description", "")),
            )

        with self._lock:
            self._entries = entries

        return len(entries)

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    def _ping_server(self, entry: CatalogEntry) -> Tuple[str, str]:
        """Ping a single server.  Returns (status, error_msg)."""
        headers = {}
        if entry.api_key:
            headers["Authorization"] = f"Bearer {entry.api_key}"

        try:
            if self._health_method == "models":
                resp = requests.get(
                    f"{entry.url}/models",
                    headers=headers,
                    timeout=self._timeout,
                )
                if resp.status_code == 200:
                    return ("available", "")
                return ("not_reachable", f"HTTP {resp.status_code}")
            else:
                # Tiny completions probe
                resp = requests.post(
                    f"{entry.url}/chat/completions",
                    headers={**headers, "Content-Type": "application/json"},
                    json={
                        "model": entry.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                    timeout=self._timeout,
                )
                if resp.status_code == 200:
                    return ("available", "")
                return ("not_reachable", f"HTTP {resp.status_code}")
        except requests.ConnectionError:
            return ("not_reachable", "Connection refused")
        except requests.Timeout:
            return ("not_reachable", f"Timeout ({self._timeout}s)")
        except Exception as exc:
            return ("not_reachable", str(exc))

    def check_health_sync(self, entry_id: Optional[str] = None) -> Dict[str, str]:
        """Check health of one or all entries.  Returns ``{id: status}``."""
        with self._lock:
            if entry_id:
                targets = [self._entries[entry_id]] if entry_id in self._entries else []
            else:
                targets = list(self._entries.values())

        results: Dict[str, str] = {}
        for entry in targets:
            status, error = self._ping_server(entry)
            with self._lock:
                entry.status = status
                entry.error = error
                entry.last_check = time.monotonic()
            results[entry.id] = status

        return results

    def start_periodic_health_check(self) -> None:
        """Start a daemon thread that checks all entries periodically."""
        if self._health_thread and self._health_thread.is_alive():
            return
        self._stop_event.clear()
        self._health_thread = threading.Thread(
            target=self._health_loop,
            name="model-catalog-health",
            daemon=True,
        )
        self._health_thread.start()

    def stop_periodic_health_check(self) -> None:
        """Signal the background health-check thread to stop."""
        self._stop_event.set()
        if self._health_thread:
            self._health_thread.join(timeout=2)
            self._health_thread = None

    def _health_loop(self) -> None:
        """Background loop: check all entries, sleep, repeat."""
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    targets = list(self._entries.values())
                for entry in targets:
                    if self._stop_event.is_set():
                        break
                    status, error = self._ping_server(entry)
                    with self._lock:
                        entry.status = status
                        entry.error = error
                        entry.last_check = time.monotonic()
            except Exception:
                logger.debug("Health check loop error", exc_info=True)
            self._stop_event.wait(timeout=self._interval)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def has_entries(self) -> bool:
        """True if the catalog has at least one entry."""
        with self._lock:
            return len(self._entries) > 0

    def get_entry(self, entry_id: str) -> Optional[CatalogEntry]:
        """Get a single entry by ID."""
        with self._lock:
            return self._entries.get(entry_id)

    def get_available(self) -> List[CatalogEntry]:
        """Return entries whose status is 'available'."""
        with self._lock:
            return [e for e in self._entries.values() if e.status == "available"]

    def get_all(self) -> List[CatalogEntry]:
        """Return all entries (any status)."""
        with self._lock:
            return list(self._entries.values())

    def find_by_tags(self, **kwargs) -> List[CatalogEntry]:
        """Find entries matching tag criteria.

        Examples::

            find_by_tags(capabilities=["code"])
            find_by_tags(size="32b", family="qwen")
        """
        with self._lock:
            candidates = list(self._entries.values())

        matches = []
        for entry in candidates:
            tags = entry.tags or {}
            match = True
            for key, value in kwargs.items():
                tag_val = tags.get(key)
                if tag_val is None:
                    match = False
                    break
                # List membership: all requested values must be present
                if isinstance(value, list) and isinstance(tag_val, list):
                    if not all(v in tag_val for v in value):
                        match = False
                        break
                elif isinstance(value, list) and not isinstance(tag_val, list):
                    match = False
                    break
                elif value != tag_val:
                    match = False
                    break
            if match:
                matches.append(entry)
        return matches

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_system_prompt(self) -> str:
        """Compact summary for system prompt injection.

        Returns empty string if no entries — caller should skip injection.
        """
        entries = self.get_all()
        if not entries:
            return ""

        lines = [
            "## Available Local Models",
            "",
            'Use delegate_task with catalog_model="<id>" to route work to these servers.',
            "Use the model_catalog tool for detailed capabilities and live status checks.",
            "",
            "| ID | Model | Status | Capabilities |",
            "|----|-------|--------|--------------|",
        ]
        for e in entries:
            caps = e.tags.get("capabilities", [])
            caps_str = ", ".join(caps) if isinstance(caps, list) else str(caps)
            size = e.tags.get("size", "")
            quant = e.tags.get("quantization", "")
            extras = " — ".join(filter(None, [size, quant]))
            if extras:
                caps_str = f"{caps_str} ({extras})" if caps_str else extras
            # Truncate model string for compactness
            model_short = e.model.split("/")[-1] if "/" in e.model else e.model
            lines.append(f"| {e.id} | {model_short} | {e.status} | {caps_str} |")

        return "\n".join(lines)

    def format_for_tool_response(
        self,
        entries: Optional[List[CatalogEntry]] = None,
        verbose: bool = False,
    ) -> str:
        """JSON response for the model_catalog tool."""
        if entries is None:
            entries = self.get_all()

        result = []
        for e in entries:
            item: Dict[str, Any] = {
                "id": e.id,
                "model": e.model,
                "status": e.status,
                "capabilities": e.tags.get("capabilities", []),
            }
            if verbose:
                item["url"] = e.url
                item["description"] = e.description
                item["tags"] = e.tags
                item["error"] = e.error or None
            result.append(item)

        return json.dumps({"models": result, "total": len(result)}, indent=2)
