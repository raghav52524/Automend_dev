"""Pydantic models for Dataset 4 workflow JSON (Format B)."""
from typing import Optional

from pydantic import BaseModel, ConfigDict


class StepParams(BaseModel):
    """Known params for workflow steps (closed schema for Gemini API)."""

    model_config = ConfigDict(extra="forbid")
    deployment: Optional[str] = None
    replicas: Optional[int] = None
    pod: Optional[str] = None


class Step(BaseModel):
    """Single step in a workflow."""

    step_id: int
    tool: str
    params: StepParams


class Workflow(BaseModel):
    """Workflow containing a list of steps."""

    steps: list[Step]


class FormatBMessage(BaseModel):
    """ChatML message (role + content)."""

    role: str
    content: str
