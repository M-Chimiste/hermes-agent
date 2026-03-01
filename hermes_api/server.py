"""Uvicorn server startup — called by `hermes serve`."""

import os


def start_server(
    host: str = "127.0.0.1",
    port: int = 8642,
    reload: bool = False,
    cors_origins: str = None,
):
    """Start the Hermes API server with uvicorn."""
    import uvicorn

    if cors_origins:
        os.environ["HERMES_API_CORS_ORIGINS"] = cors_origins

    # Print startup info
    from hermes_api.auth import get_or_create_api_token
    token = get_or_create_api_token()
    print(f"\n  Hermes Web API starting on http://{host}:{port}")
    print(f"  API docs:  http://{host}:{port}/api/docs")
    print(f"  API token: {token}\n")

    uvicorn.run(
        "hermes_api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
    )
