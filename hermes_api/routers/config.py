"""Configuration read/write REST endpoints."""

from fastapi import APIRouter, Depends

from hermes_api.auth import verify_token
from hermes_api.models.config import ConfigPatchRequest, ConfigResponse, ConfigStatusResponse

router = APIRouter(dependencies=[Depends(verify_token)])


@router.get("", response_model=ConfigResponse)
async def get_config():
    from hermes_cli.config import load_config
    config = load_config()
    # Redact API token from response
    if "api" in config and "token" in config["api"]:
        config["api"]["token"] = "***"
    return ConfigResponse(config=config)


@router.patch("")
async def patch_config(req: ConfigPatchRequest):
    from hermes_cli.config import load_config, save_config

    config = load_config()
    for dotted_key, value in req.updates.items():
        _set_nested(config, dotted_key, value)
    save_config(config)
    return {"updated": list(req.updates.keys())}


@router.get("/status", response_model=ConfigStatusResponse)
async def config_status():
    from hermes_cli.config import check_config_version, get_missing_config_fields

    current, latest = check_config_version()
    missing = get_missing_config_fields()
    return ConfigStatusResponse(
        current_version=current,
        latest_version=latest,
        missing_fields=missing,
    )


def _set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict using a dotted key path."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
