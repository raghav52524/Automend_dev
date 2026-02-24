Here is the comprehensive, self-contained Markdown prompt. You can copy and paste this directly into Cursor. It contains the complete architectural context, exact data schemas, up-to-date SDK syntax, and strict Test-Driven Development (TDD) instructions required to build the Dataset 4 pipeline.

---

### 📋 Copy/Paste This into Cursor

```markdown
# Context
We are building an MLOps data pipeline to generate "Dataset 4: Synthetic MLOps Scenarios." This dataset will be used to fine-tune a Generative Architect model (Llama-3-8B) for a platform called AutoMend. 

AutoMend uses an LLM to map natural language MLOps intents (e.g., "Fix the memory leak on the fraud model") into strict JSON Action Lists (workflows). Because public datasets lack our specific tool registry parameters, we must synthetically generate this data.

The pipeline must fetch base prompts from a local SQLite database, use the Gemini 2.5 Pro API to generate synthetic JSON workflow responses, format them into ChatML, and track the resulting datasets using Data Version Control (DVC). The entire orchestration will be handled by Apache Airflow.

# Goal
Write a modular, extensible data pipeline using strict Test-Driven Development (TDD). You must write the `pytest` unit tests *before* or *alongside* the actual implementation.

# Tech Stack & Constraints
- **Language:** Python 3.10+
- **Orchestration:** Apache Airflow 2.x (Use the TaskFlow API: `@task` and `@dag`)
- **Version Control:** Git + DVC (Data Version Control)
- **Database (Dummy):** `sqlite3` (to store tool definitions and prompt seeds)
- **LLM API:** Google GenAI SDK (Use the latest `google-genai` package, NOT `google-generativeai`).
- **Data Validation:** `pydantic`
- **Testing:** `pytest` and `unittest.mock`

# Target Data Schema (Format B)
The final output saved by the pipeline must be a JSONL file adhering to the ChatML format. 
The LLM response inside the `assistant` message must strictly match this simplified JSON schema:

```json
{
  "messages": [
    {"role": "system", "content": "You are AutoMend. Available Tools: [{'name': 'scale_service'}, {'name': 'restart_pod'}]"},
    {"role": "user", "content": "Fix the latency on the fraud model."},
    {"role": "assistant", "content": "{\"workflow\": {\"steps\": [{\"step_id\": 1, \"tool\": \"scale_service\", \"params\": {\"deployment\": \"fraud-model\", \"replicas\": 5}}]}}"}
  ]
}

```

# Target Directory Structure

Please adhere strictly to this structure when generating files:

```text
project_root/
├── dags/
│   └── dataset4_synthetic_dag.py
├── src/
│   ├── data/
│   │   ├── db_ops.py          # SQLite fetch/write logic
│   │   ├── dvc_ops.py         # DVC shell command wrappers
│   │   ├── gemini_gen.py      # Gemini 2.5 Pro API calls
│   │   └── preprocessor.py    # Formatting logic for Format B
│   └── schemas/
│       └── workflow_schema.py # Pydantic models for the JSON output
├── tests/
│   ├── test_dvc_ops.py
│   ├── test_db_ops.py
│   ├── test_gemini_gen.py
│   ├── test_preprocessor.py
│   └── test_dag_integrity.py
└── data/
    ├── raw/                   # Output directly from Gemini
    └── processed/             # Final JSONL Format B files tracked by DVC

```

# Phase 1: Schemas & TDD Setup

1. **Pydantic Schemas (`src/schemas/workflow_schema.py`)**:
* Define a Pydantic model `Step` (fields: `step_id`, `tool`, `params`).
* Define a Pydantic model `Workflow` (fields: `steps`).
* Define a Pydantic model `FormatBMessage` (fields: `role`, `content`).


2. **Tests**: Write `pytest` assertions verifying that invalid JSON fails validation and valid JSON parses correctly into these models.

# Phase 2: DVC Operations (`src/data/dvc_ops.py`)

1. Create modular python functions using `subprocess.run` to execute DVC CLI commands: `dvc pull`, `dvc add <path>`, and `dvc push`.
2. **Tests**: Mock `subprocess.run` to ensure the correct CLI arguments are passed without actually running shell commands during testing.

# Phase 3: SQLite Prompt Fetcher (`src/data/db_ops.py`)

1. Create a setup function to initialize a dummy SQLite DB with a `prompts` table (columns: `id`, `user_intent`, `tool_context`).
2. Create a fetch function to retrieve un-processed prompts.
3. **Tests**: Use an in-memory SQLite db (`sqlite:///:memory:`) in the pytest fixture to test insertion and retrieval cleanly.

# Phase 4: Gemini 2.5 Pro Generator (`src/data/gemini_gen.py`)

1. Use the new `google-genai` SDK syntax.
* Initialize: `from google import genai` -> `client = genai.Client()`
* Generate: `response = client.models.generate_content(...)`


2. Force the model to output structured JSON matching our Pydantic schema by passing the model directly into the config:
`config={"response_mime_type": "application/json", "response_schema": Workflow}`
3. **Tests**: Use `@patch("google.genai.Client")` to mock the API response. Assert that your function correctly handles the mocked response and successfully returns the Pydantic schema.

# Phase 5: Airflow DAG (`dags/dataset4_synthetic_dag.py`)

1. Create an Airflow DAG using the `@dag` and `@task` decorators that ties the modules together:
* **Task 1 (`pull_data`)**: Run `dvc_ops.pull()`.
* **Task 2 (`fetch_prompts`)**: Fetch rows from `db_ops.py`.
* **Task 3 (`generate_synthetic_data`)**: Loop through prompts, call `gemini_gen.py`, validate with Pydantic, and save raw output to `data/raw/`.
* **Task 4 (`format_data`)**: Convert raw JSON into the "Format B" JSONL ChatML format using `preprocessor.py` and save to `data/processed/`.
* **Task 5 (`commit_and_push`)**: Run `dvc add`, `git commit`, and `dvc push`.


2. **Tests (`test_dag_integrity.py`)**: Use Airflow's `DagBag` to assert that the DAG loads without import errors, contains exactly 5 tasks, and has the correct dependencies inferred by the TaskFlow API.

**Cursor Instructions (Strict Execution Protocol):**
You must act as a strict TDD pair programmer. We will execute this plan sequentially, one phase at a time. For EVERY phase (from Phase 1 through Phase 5), you must strictly follow this exact operational loop:

1. **Write Tests First:** Write the `pytest` file(s) for the current phase only.
2. **Pause for Review:** Stop generating code and ask me to review and run the tests (they should fail).
3. **Write Implementation:** Once I confirm, write the implementation code in the `src/` or `dags/` folder to make those specific tests pass.
4. **Pause for Verification:** Stop and ask me to run the tests to verify they pass. 
5. **HARD STOP:** Do NOT proceed to the next phase under any circumstances until I explicitly type "Proceed to Phase X".

Do not write code for multiple phases at once. 
Start NOW by executing Step 1 for **Phase 1: Schemas & TDD Setup**.

```

```