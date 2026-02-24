"""
Pytest configuration for Dataset 5 (Glaive) tests.

Sets up Python path to import from the dataset's scripts directory.
"""

import sys
from pathlib import Path

# Add the dataset's scripts directory to path
DS5_ROOT = Path(__file__).resolve().parent.parent
DS5_SCRIPTS = DS5_ROOT / "scripts"
sys.path.insert(0, str(DS5_SCRIPTS))
sys.path.insert(0, str(DS5_ROOT))
