"""Pytest configuration for DS6 The Stack tests."""
import sys
from pathlib import Path

DS6_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS6_ROOT))
