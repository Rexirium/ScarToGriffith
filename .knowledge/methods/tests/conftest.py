"""Make the repo's scripts/ directory importable for the page tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
