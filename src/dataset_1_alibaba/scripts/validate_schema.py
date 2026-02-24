"""
Validate Format A output - check sequence structure and label validity
"""
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_processed_dir, LOGS_DIR
    PROCESSED_DIR = get_ds1_processed_dir()
    LOG_DIR = LOGS_DIR
except ImportError:
    PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"
    LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "validate.log", mode="a")
    ]
)
log = logging.getLogger(__name__)

INPUT_PATH = PROCESSED_DIR / "format_a_sequences.json"


def validate_schema(path=INPUT_PATH):
    log.info(f"Validating Format A schema: {path}")
    errors = []

    with open(path, "r") as f:
        sequences = json.load(f)

    if len(sequences) == 0:
        errors.append("No sequences found")

    for i, seq in enumerate(sequences):
        if "sequence_ids" not in seq:
            errors.append(f"Row {i}: missing sequence_ids")
        if "label" not in seq:
            errors.append(f"Row {i}: missing label")
        if not isinstance(seq.get("sequence_ids"), list):
            errors.append(f"Row {i}: sequence_ids is not a list")
        if len(seq.get("sequence_ids", [])) == 0:
            errors.append(f"Row {i}: sequence_ids is empty")
        if seq.get("label") not in [0, 1, 2, 3, 4]:
            errors.append(f"Row {i}: invalid label {seq.get('label')}")
        for token in seq.get("sequence_ids", []):
            if not isinstance(token, int):
                errors.append(f"Row {i}: non-integer token {token}")

    if errors:
        for e in errors:
            log.error(f"  {e}")
        return False
    else:
        log.info(f"Validation PASSED — {len(sequences)} sequences all valid")
        return True


if __name__ == "__main__":
    passed = validate_schema()
    print("PASSED" if passed else "FAILED")