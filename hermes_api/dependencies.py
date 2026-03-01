"""Shared FastAPI dependencies — singletons for SessionDB, config, model catalog."""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache()
def get_session_db():
    """Return the shared SessionDB instance (created once)."""
    from hermes_state import SessionDB
    return SessionDB()


def get_model_catalog():
    """Return the ModelCatalog if configured, else None."""
    try:
        from agent.model_catalog import ModelCatalog
        catalog = ModelCatalog()
        if catalog.load() > 0:
            catalog.check_health_sync()
            return catalog
    except Exception as exc:
        logger.debug("Model catalog not available: %s", exc)
    return None


def get_config() -> dict:
    """Load and return the current config from disk."""
    from hermes_cli.config import load_config
    return load_config()
