"""API authentication — auto-generated bearer token stored in config."""

import secrets
from typing import Optional

from fastapi import HTTPException, Security, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_security = HTTPBearer(auto_error=False)

_cached_token: Optional[str] = None


def get_or_create_api_token() -> str:
    """Return the API token, creating one if it doesn't exist yet."""
    global _cached_token
    if _cached_token:
        return _cached_token

    from hermes_cli.config import load_config, save_config

    config = load_config()
    token = config.get("api", {}).get("token")
    if not token:
        token = secrets.token_urlsafe(32)
        if "api" not in config:
            config["api"] = {}
        config["api"]["token"] = token
        save_config(config)
    _cached_token = token
    return token


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(_security),
) -> str:
    """FastAPI dependency that validates the bearer token on REST endpoints."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization header")
    expected = get_or_create_api_token()
    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return credentials.credentials


async def verify_ws_token(websocket: WebSocket) -> bool:
    """Validate token passed as a query parameter on WebSocket connections."""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token query parameter")
        return False
    expected = get_or_create_api_token()
    if not secrets.compare_digest(token, expected):
        await websocket.close(code=4003, reason="Invalid token")
        return False
    return True
