"""Model Catalog Tool — Query the local model catalog.

Lets the orchestration agent discover available local LLM servers,
check their health, and find models by capability tags before routing
work to them via ``delegate_task(catalog_model=...)``.

The tool is gated by ``check_fn``: when no catalog is loaded (file
missing or empty), the tool is never sent to the LLM and the feature
is completely invisible.
"""

import json
from tools.registry import registry


def _check_catalog_available() -> bool:
    """Return True only when a catalog with entries is loaded."""
    from agent.model_catalog import get_catalog
    cat = get_catalog()
    return cat is not None and cat.has_entries()


def _handle_model_catalog(args: dict, **kwargs) -> str:
    from agent.model_catalog import get_catalog

    catalog = get_catalog()
    if catalog is None or not catalog.has_entries():
        return json.dumps({"error": "Model catalog not configured. Create ~/.hermes/model_catalog.yaml"})

    action = args.get("action", "list")

    if action == "list":
        return catalog.format_for_tool_response(verbose=True)

    elif action == "status":
        model_id = args.get("model_id")
        result = catalog.check_health_sync(model_id)
        # Also return full entry info for the queried models
        if model_id:
            entry = catalog.get_entry(model_id)
            if entry is None:
                return json.dumps({"error": f"Unknown model ID: {model_id}"})
            return catalog.format_for_tool_response(entries=[entry], verbose=True)
        return catalog.format_for_tool_response(verbose=True)

    elif action == "find":
        tags = args.get("tags", {})
        if not isinstance(tags, dict):
            return json.dumps({"error": "tags must be an object, e.g. {\"capabilities\": [\"code\"]}"})
        matches = catalog.find_by_tags(**tags)
        if not matches:
            return json.dumps({"models": [], "total": 0, "message": "No models match the given tags."})
        return catalog.format_for_tool_response(entries=matches, verbose=True)

    else:
        return json.dumps({"error": f"Unknown action: {action}. Use 'list', 'status', or 'find'."})


MODEL_CATALOG_SCHEMA = {
    "name": "model_catalog",
    "description": (
        "Query the local model catalog to find available LLM servers. "
        "Use this to discover which local models are available, their status, "
        "capabilities, and connection details before delegating tasks to them.\n\n"
        "WHEN TO USE:\n"
        "- Before choosing a model for delegate_task (catalog_model parameter)\n"
        "- When you need to check if a specific model is online\n"
        "- To find models with specific capabilities (e.g. code, reasoning)\n\n"
        "The catalog only contains LOCAL models (LMStudio, etc). "
        "Your own model (the orchestrator) is separate and not listed here."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "status", "find"],
                "description": (
                    "list: Show all registered models with full status, tags, and description. "
                    "status: Re-check the health of a specific model (model_id) or all models. "
                    "find: Search for models matching tag criteria."
                ),
            },
            "model_id": {
                "type": "string",
                "description": "Specific model ID to query (for 'status' action).",
            },
            "tags": {
                "type": "object",
                "description": (
                    "Tag filters for 'find' action. "
                    'E.g. {"capabilities": ["code"]} or {"size": "32b", "family": "qwen"}'
                ),
            },
        },
        "required": ["action"],
    },
}


# --- Registry ---
registry.register(
    name="model_catalog",
    toolset="catalog",
    schema=MODEL_CATALOG_SCHEMA,
    handler=_handle_model_catalog,
    check_fn=_check_catalog_available,
    description="Query the local model catalog for available LLM servers",
)
