"""Pytest configuration for DS1 Alibaba tests."""
import sys
from pathlib import Path

# Must be done before any DS1 imports
DS1_ROOT = Path(__file__).resolve().parent.parent
DS1_SCRIPTS = DS1_ROOT / "scripts"

# Add scripts directory and DS1_ROOT to path
sys.path.insert(0, str(DS1_SCRIPTS))
sys.path.insert(0, str(DS1_ROOT))
