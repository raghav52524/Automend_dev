"""Phase 1: Pydantic schema validation tests (TDD)."""
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))

from schemas.workflow_schema import Step, Workflow, FormatBMessage


class TestStep:
    """Tests for Step model."""

    def test_valid_step_parses(self):
        data = {"step_id": 1, "tool": "scale_service", "params": {"deployment": "fraud-model", "replicas": 5}}
        step = Step.model_validate(data)
        assert step.step_id == 1
        assert step.tool == "scale_service"
        assert step.params.model_dump(exclude_none=True) == {"deployment": "fraud-model", "replicas": 5}

    def test_invalid_step_missing_field_fails(self):
        with pytest.raises(ValidationError):
            Step.model_validate({"step_id": 1, "tool": "scale_service"})  # missing params

    def test_invalid_step_wrong_type_fails(self):
        with pytest.raises(ValidationError):
            Step.model_validate({"step_id": "one", "tool": "scale_service", "params": {}})

    def test_invalid_step_extra_allowed_but_required_enforced(self):
        data = {"step_id": 1, "tool": "restart_pod", "params": {}}
        step = Step.model_validate(data)
        assert step.params.model_dump(exclude_none=True) == {}


class TestWorkflow:
    """Tests for Workflow model."""

    def test_valid_workflow_parses(self):
        data = {
            "steps": [
                {"step_id": 1, "tool": "scale_service", "params": {"deployment": "fraud-model", "replicas": 5}}
            ]
        }
        wf = Workflow.model_validate(data)
        assert len(wf.steps) == 1
        assert wf.steps[0].tool == "scale_service"

    def test_workflow_empty_steps_parses(self):
        wf = Workflow.model_validate({"steps": []})
        assert wf.steps == []

    def test_invalid_workflow_missing_steps_fails(self):
        with pytest.raises(ValidationError):
            Workflow.model_validate({})

    def test_invalid_workflow_steps_not_list_fails(self):
        with pytest.raises(ValidationError):
            Workflow.model_validate({"steps": "not-a-list"})


class TestFormatBMessage:
    """Tests for FormatBMessage model."""

    def test_valid_message_parses(self):
        data = {"role": "user", "content": "Fix the latency on the fraud model."}
        msg = FormatBMessage.model_validate(data)
        assert msg.role == "user"
        assert msg.content == "Fix the latency on the fraud model."

    def test_system_and_assistant_roles_parse(self):
        for role in ("system", "assistant"):
            msg = FormatBMessage.model_validate({"role": role, "content": "some content"})
            assert msg.role == role
            assert msg.content == "some content"

    def test_invalid_message_missing_role_fails(self):
        with pytest.raises(ValidationError):
            FormatBMessage.model_validate({"content": "only content"})

    def test_invalid_message_missing_content_fails(self):
        with pytest.raises(ValidationError):
            FormatBMessage.model_validate({"role": "user"})
