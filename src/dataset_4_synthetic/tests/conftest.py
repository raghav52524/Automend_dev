"""
Pytest configuration for Dataset 4 (Synthetic) tests.

Sets up Python path to import from the dataset's src directory.
"""

import sys
from pathlib import Path

# Add the dataset's src directory to path for imports like "from src.data import ..."
DS4_ROOT = Path(__file__).resolve().parent.parent
DS4_SRC = DS4_ROOT / "src"
sys.path.insert(0, str(DS4_SRC.parent))  # Add parent so "src.data" resolves correctly
sys.path.insert(0, str(DS4_ROOT))
