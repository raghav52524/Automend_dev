"""Gemini 2.5 Pro API calls for synthetic workflow generation."""
import os

from dotenv import load_dotenv

load_dotenv()

from google import genai

from schemas.workflow_schema import Workflow
from data.pipeline_logger import get_logger

logger = get_logger(__name__)
GEMINI_MODEL = "gemini-2.5-flash"

# Set GEMINI_API_KEY in env, or replace the default below with your key (do not commit real keys).
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", "dummy-replace-with-real-key")


# Hand-built JSON schema for Gemini (API rejects Pydantic-generated additionalProperties).
WORKFLOW_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer"},
                    "tool": {"type": "string"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "deployment": {"type": "string"},
                            "replicas": {"type": "integer"},
                            "pod": {"type": "string"},
                        },
                    },
                },
                "required": ["step_id", "tool", "params"],
            },
        },
    },
    "required": ["steps"],
}


def generate_workflow(user_intent: str, available_tools: list[str] | str) -> Workflow:
    """Call Gemini to generate a structured workflow JSON and return a validated Workflow.
    available_tools: list of tool names (e.g. from config/available_tools.json) or a single string."""
    logger.info("Generating workflow for intent: %s", user_intent[:50] + "..." if len(user_intent) > 50 else user_intent)
    if isinstance(available_tools, list):
        tools_str = ", ".join(available_tools)
    else:
        tools_str = available_tools
    client = genai.Client(api_key=GOOGLE_API_KEY)
    prompt = (
        f"Given the user intent and available tools, output a JSON workflow (steps with step_id, tool, params). "
        f"User intent: {user_intent}. Available tools: {tools_str}. "
        "Output only valid JSON with a 'steps' array; each step has step_id (int), tool (string), params (object with optional deployment, replicas, pod)."
    )
    config = {
        "response_mime_type": "application/json",
        "response_json_schema": WORKFLOW_JSON_SCHEMA,
    }
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    workflow = Workflow.model_validate_json(response.text)
    logger.info("Workflow generated: %d steps", len(workflow.steps))
    return workflow
