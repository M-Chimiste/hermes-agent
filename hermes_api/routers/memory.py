"""Memory read/write REST endpoints."""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from hermes_api.auth import verify_token
from hermes_api.models.memory import (
    MemoryAddRequest,
    MemoryMutationResponse,
    MemoryRemoveRequest,
    MemoryResponse,
    MemoryUpdateRequest,
)

router = APIRouter(dependencies=[Depends(verify_token)])

MEMORY_DIR = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "memories"
ENTRY_DELIMITER = "\n§\n"


def _validate_target(target: str):
    if target not in ("memory", "user"):
        raise HTTPException(status_code=400, detail="target must be 'memory' or 'user'")


def _read_memory(target: str) -> tuple[str, list[str]]:
    filename = "MEMORY.md" if target == "memory" else "USER.md"
    filepath = MEMORY_DIR / filename
    if not filepath.exists():
        return "", []
    content = filepath.read_text(encoding="utf-8").strip()
    if not content:
        return "", []
    entries = [e.strip() for e in content.split("§") if e.strip()]
    return content, entries


@router.get("")
async def get_all_memory():
    results = {}
    for target in ("memory", "user"):
        content, entries = _read_memory(target)
        results[target] = {
            "content": content,
            "entries": entries,
            "entry_count": len(entries),
        }
    return results


@router.get("/{target}", response_model=MemoryResponse)
async def get_memory(target: str):
    _validate_target(target)
    from hermes_cli.config import load_config
    config = load_config()
    mem_cfg = config.get("memory", {})
    char_limit = mem_cfg.get("memory_char_limit", 2200) if target == "memory" else mem_cfg.get("user_char_limit", 1375)

    content, entries = _read_memory(target)
    used = len(content)
    usage = f"{round(used / char_limit * 100)}% — {used}/{char_limit} chars" if char_limit else "N/A"
    return MemoryResponse(
        target=target,
        content=content,
        entries=entries,
        usage=usage,
        entry_count=len(entries),
    )


@router.put("/{target}")
async def replace_memory(target: str, req: MemoryUpdateRequest):
    _validate_target(target)
    filename = "MEMORY.md" if target == "memory" else "USER.md"
    filepath = MEMORY_DIR / filename
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    filepath.write_text(req.content, encoding="utf-8")
    return {"success": True, "target": target}


@router.post("/{target}/entries", response_model=MemoryMutationResponse)
async def add_entry(target: str, req: MemoryAddRequest):
    _validate_target(target)
    from tools.memory_tool import MemoryStore
    from hermes_cli.config import load_config

    config = load_config()
    mem_cfg = config.get("memory", {})
    store = MemoryStore(
        memory_char_limit=mem_cfg.get("memory_char_limit", 2200),
        user_char_limit=mem_cfg.get("user_char_limit", 1375),
    )
    store.load_from_disk()
    result = store.add(target, req.content)
    return MemoryMutationResponse(**result)


@router.delete("/{target}/entries", response_model=MemoryMutationResponse)
async def remove_entry(target: str, req: MemoryRemoveRequest):
    _validate_target(target)
    from tools.memory_tool import MemoryStore
    from hermes_cli.config import load_config

    config = load_config()
    mem_cfg = config.get("memory", {})
    store = MemoryStore(
        memory_char_limit=mem_cfg.get("memory_char_limit", 2200),
        user_char_limit=mem_cfg.get("user_char_limit", 1375),
    )
    store.load_from_disk()
    result = store.remove(target, req.old_text)
    return MemoryMutationResponse(**result)
