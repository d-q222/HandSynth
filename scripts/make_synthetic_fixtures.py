"""Regenerate fixtures/landmarks/synthetic/*.json."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "tests"))
from synthetic_hand import write_all  # noqa: E402

print(write_all(root / "fixtures/landmarks/synthetic"), "fixtures written")
