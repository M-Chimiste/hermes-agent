"""Model catalog REST endpoints."""

from fastapi import APIRouter, Depends

from hermes_api.auth import verify_token
from hermes_api.dependencies import get_model_catalog

router = APIRouter(dependencies=[Depends(verify_token)])


@router.get("/catalog")
async def list_catalog():
    catalog = get_model_catalog()
    if catalog is None:
        return {"configured": False, "models": []}

    models = []
    for entry in catalog._entries:
        models.append({
            "id": entry.get("id"),
            "name": entry.get("name"),
            "server_url": entry.get("server_url"),
            "model_id": entry.get("model_id"),
            "healthy": entry.get("healthy", False),
            "tags": entry.get("tags", []),
            "description": entry.get("description", ""),
        })
    return {"configured": True, "models": models}


@router.post("/catalog/health")
async def check_catalog_health():
    catalog = get_model_catalog()
    if catalog is None:
        return {"configured": False}
    catalog.check_health_sync()
    return {"configured": True, "checked": True}
