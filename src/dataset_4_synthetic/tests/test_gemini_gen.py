"""Phase 4: Gemini generator tests. Mock google.genai.Client."""
import sys
from pathlib import Path
import unittest.mock as mock
import pytest

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))

from schemas.workflow_schema import Workflow
from data import gemini_gen


@mock.patch("data.gemini_gen.genai.Client")
def test_generate_workflow_returns_pydantic_workflow(mock_client_class):
    mock_response = mock.Mock()
    mock_response.text = '{"steps": [{"step_id": 1, "tool": "scale_service", "params": {"replicas": 5}}]}'
    mock_client = mock.Mock()
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    result = gemini_gen.generate_workflow("Fix the latency", "scale_service, restart_pod")

    assert isinstance(result, Workflow)
    assert len(result.steps) == 1
    assert result.steps[0].tool == "scale_service"
    assert result.steps[0].params.model_dump(exclude_none=True) == {"replicas": 5}


@mock.patch("data.gemini_gen.genai.Client")
def test_generate_workflow_calls_client_with_correct_config(mock_client_class):
    mock_response = mock.Mock()
    mock_response.text = '{"steps": []}'
    mock_client = mock.Mock()
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    gemini_gen.generate_workflow("Restart pod", "restart_pod")

    call_kwargs = mock_client.models.generate_content.call_args[1]
    assert "config" in call_kwargs
    config = call_kwargs["config"]
    assert config.get("response_mime_type") == "application/json"
    assert "response_schema" in config or "response_json_schema" in config
