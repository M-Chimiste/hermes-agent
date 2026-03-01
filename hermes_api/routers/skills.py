"""Skills browsing REST endpoints."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, Query

from hermes_api.auth import verify_token

router = APIRouter(dependencies=[Depends(verify_token)])


@router.get("/categories")
async def skills_categories():
    from tools.skills_tool import skills_categories
    result = skills_categories()
    return json.loads(result)


@router.get("")
async def list_skills(
    category: Optional[str] = Query(None, description="Filter by category"),
):
    from tools.skills_tool import skills_list
    result = skills_list(category=category) if category else skills_list()
    return json.loads(result)


@router.get("/{name}")
async def get_skill(name: str):
    from tools.skills_tool import skill_view
    result = skill_view(name=name)
    return json.loads(result)
