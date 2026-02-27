"""Tests for the model catalog system."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from agent.model_catalog import ModelCatalog, CatalogEntry, get_catalog, set_catalog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CATALOG_YAML = {
    "health_check": {
        "interval_seconds": 60,
        "timeout_seconds": 3,
        "method": "models",
    },
    "models": [
        {
            "id": "test-qwen",
            "url": "http://localhost:1234/v1",
            "model": "qwen2.5-coder-32b",
            "api_key": "test-key",
            "tags": {
                "size": "32b",
                "family": "qwen",
                "capabilities": ["code", "reasoning"],
            },
            "description": "Test Qwen model",
        },
        {
            "id": "test-llama",
            "url": "http://10.0.0.5:1234/v1",
            "model": "llama-3.1-70b",
            "tags": {
                "size": "70b",
                "family": "llama",
                "capabilities": ["general", "reasoning", "code"],
            },
            "description": "Test Llama model",
        },
    ],
}


@pytest.fixture()
def catalog_file(tmp_path):
    """Create a temporary model catalog YAML file."""
    path = tmp_path / "model_catalog.yaml"
    path.write_text(yaml.dump(SAMPLE_CATALOG_YAML))
    return path


@pytest.fixture()
def catalog(catalog_file):
    """Return a loaded ModelCatalog."""
    cat = ModelCatalog(catalog_path=catalog_file)
    cat.load()
    return cat


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------

class TestCatalogLoading:
    def test_load_valid_yaml(self, catalog_file):
        cat = ModelCatalog(catalog_path=catalog_file)
        n = cat.load()
        assert n == 2

    def test_load_missing_file(self, tmp_path):
        cat = ModelCatalog(catalog_path=tmp_path / "nonexistent.yaml")
        n = cat.load()
        assert n == 0
        assert not cat.has_entries()

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cat = ModelCatalog(catalog_path=path)
        n = cat.load()
        assert n == 0

    def test_load_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("{{invalid yaml::")
        cat = ModelCatalog(catalog_path=path)
        n = cat.load()
        assert n == 0

    def test_load_skips_entries_without_required_fields(self, tmp_path):
        data = {
            "models": [
                {"id": "valid", "url": "http://localhost:1234/v1", "model": "test"},
                {"id": "no-url", "model": "test"},       # missing url
                {"url": "http://x/v1", "model": "test"},  # missing id
                {"id": "no-model", "url": "http://x/v1"}, # missing model
            ]
        }
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.dump(data))
        cat = ModelCatalog(catalog_path=path)
        n = cat.load()
        assert n == 1
        assert cat.get_entry("valid") is not None

    def test_load_health_check_config(self, catalog):
        assert catalog._interval == 60
        assert catalog._timeout == 3
        assert catalog._health_method == "models"

    def test_load_is_idempotent(self, catalog_file):
        cat = ModelCatalog(catalog_path=catalog_file)
        cat.load()
        cat.load()  # second call should not fail
        assert cat.has_entries()

    def test_optional_api_key_defaults_empty(self, tmp_path):
        data = {
            "models": [
                {"id": "no-key", "url": "http://localhost/v1", "model": "test"}
            ]
        }
        path = tmp_path / "nokey.yaml"
        path.write_text(yaml.dump(data))
        cat = ModelCatalog(catalog_path=path)
        cat.load()
        entry = cat.get_entry("no-key")
        assert entry.api_key == ""


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

class TestCatalogQueries:
    def test_get_entry(self, catalog):
        entry = catalog.get_entry("test-qwen")
        assert entry is not None
        assert entry.model == "qwen2.5-coder-32b"

    def test_get_entry_not_found(self, catalog):
        assert catalog.get_entry("nonexistent") is None

    def test_get_all(self, catalog):
        all_entries = catalog.get_all()
        assert len(all_entries) == 2

    def test_has_entries(self, catalog):
        assert catalog.has_entries()

    def test_get_available_initially_pending(self, catalog):
        # All entries start as "pending" status
        available = catalog.get_available()
        assert len(available) == 0

    def test_find_by_capabilities(self, catalog):
        matches = catalog.find_by_tags(capabilities=["code"])
        assert len(matches) == 2  # both have code

    def test_find_by_family(self, catalog):
        matches = catalog.find_by_tags(family="qwen")
        assert len(matches) == 1
        assert matches[0].id == "test-qwen"

    def test_find_by_size(self, catalog):
        matches = catalog.find_by_tags(size="70b")
        assert len(matches) == 1
        assert matches[0].id == "test-llama"

    def test_find_by_multiple_tags(self, catalog):
        matches = catalog.find_by_tags(family="llama", capabilities=["general"])
        assert len(matches) == 1
        assert matches[0].id == "test-llama"

    def test_find_no_match(self, catalog):
        matches = catalog.find_by_tags(family="mistral")
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_check_health_marks_available(self, catalog):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("agent.model_catalog.requests.get", return_value=mock_resp):
            results = catalog.check_health_sync()
        assert results["test-qwen"] == "available"
        assert results["test-llama"] == "available"
        assert catalog.get_entry("test-qwen").status == "available"

    def test_check_health_marks_not_reachable(self, catalog):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("agent.model_catalog.requests.get", return_value=mock_resp):
            results = catalog.check_health_sync()
        assert results["test-qwen"] == "not_reachable"
        assert "HTTP 500" in catalog.get_entry("test-qwen").error

    def test_check_health_connection_error(self, catalog):
        import requests as req
        with patch("agent.model_catalog.requests.get", side_effect=req.ConnectionError("refused")):
            results = catalog.check_health_sync()
        assert results["test-qwen"] == "not_reachable"
        assert "Connection refused" in catalog.get_entry("test-qwen").error

    def test_check_health_timeout(self, catalog):
        import requests as req
        with patch("agent.model_catalog.requests.get", side_effect=req.Timeout("timed out")):
            results = catalog.check_health_sync()
        assert results["test-qwen"] == "not_reachable"
        assert "Timeout" in catalog.get_entry("test-qwen").error

    def test_check_health_single_entry(self, catalog):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("agent.model_catalog.requests.get", return_value=mock_resp):
            results = catalog.check_health_sync("test-qwen")
        assert "test-qwen" in results
        assert "test-llama" not in results

    def test_check_health_completions_method(self, catalog_file):
        # Override method to completions
        data = yaml.safe_load(catalog_file.read_text())
        data["health_check"]["method"] = "completions"
        catalog_file.write_text(yaml.dump(data))
        cat = ModelCatalog(catalog_path=catalog_file)
        cat.load()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("agent.model_catalog.requests.post", return_value=mock_resp):
            results = cat.check_health_sync("test-qwen")
        assert results["test-qwen"] == "available"

    def test_check_health_sends_auth_header(self, catalog):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("agent.model_catalog.requests.get", return_value=mock_resp) as mock_get:
            catalog.check_health_sync("test-qwen")
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-key"


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_for_system_prompt_empty(self, tmp_path):
        cat = ModelCatalog(catalog_path=tmp_path / "nope.yaml")
        cat.load()
        assert cat.format_for_system_prompt() == ""

    def test_format_for_system_prompt_content(self, catalog):
        prompt = catalog.format_for_system_prompt()
        assert "Available Local Models" in prompt
        assert "test-qwen" in prompt
        assert "test-llama" in prompt
        assert "delegate_task" in prompt
        assert "model_catalog" in prompt

    def test_format_for_tool_response(self, catalog):
        resp = json.loads(catalog.format_for_tool_response())
        assert resp["total"] == 2
        assert len(resp["models"]) == 2

    def test_format_for_tool_response_verbose(self, catalog):
        resp = json.loads(catalog.format_for_tool_response(verbose=True))
        model = resp["models"][0]
        assert "url" in model
        assert "description" in model
        assert "tags" in model

    def test_format_for_tool_response_filtered(self, catalog):
        entries = catalog.find_by_tags(family="qwen")
        resp = json.loads(catalog.format_for_tool_response(entries=entries))
        assert resp["total"] == 1


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_set_catalog(self, catalog):
        set_catalog(catalog)
        assert get_catalog() is catalog
        # Clean up
        set_catalog(None)

    def test_get_catalog_returns_none_by_default(self):
        original = get_catalog()
        set_catalog(None)
        assert get_catalog() is None
        # Restore
        if original:
            set_catalog(original)


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------

class TestModelCatalogTool:
    def test_list_action(self, catalog):
        set_catalog(catalog)
        from tools.model_catalog_tool import _handle_model_catalog
        result = json.loads(_handle_model_catalog({"action": "list"}))
        assert result["total"] == 2
        set_catalog(None)

    def test_find_action(self, catalog):
        set_catalog(catalog)
        from tools.model_catalog_tool import _handle_model_catalog
        result = json.loads(_handle_model_catalog({
            "action": "find",
            "tags": {"family": "qwen"},
        }))
        assert result["total"] == 1
        set_catalog(None)

    def test_find_no_match(self, catalog):
        set_catalog(catalog)
        from tools.model_catalog_tool import _handle_model_catalog
        result = json.loads(_handle_model_catalog({
            "action": "find",
            "tags": {"family": "mistral"},
        }))
        assert result["total"] == 0
        set_catalog(None)

    def test_status_action(self, catalog):
        set_catalog(catalog)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        from tools.model_catalog_tool import _handle_model_catalog
        with patch("agent.model_catalog.requests.get", return_value=mock_resp):
            result = json.loads(_handle_model_catalog({
                "action": "status",
                "model_id": "test-qwen",
            }))
        assert result["total"] == 1
        set_catalog(None)

    def test_unknown_action(self, catalog):
        set_catalog(catalog)
        from tools.model_catalog_tool import _handle_model_catalog
        result = json.loads(_handle_model_catalog({"action": "invalid"}))
        assert "error" in result
        set_catalog(None)

    def test_no_catalog_configured(self):
        set_catalog(None)
        from tools.model_catalog_tool import _handle_model_catalog
        result = json.loads(_handle_model_catalog({"action": "list"}))
        assert "error" in result

    def test_check_fn_gate(self, catalog):
        from tools.model_catalog_tool import _check_catalog_available
        set_catalog(None)
        assert not _check_catalog_available()
        set_catalog(catalog)
        assert _check_catalog_available()
        set_catalog(None)
