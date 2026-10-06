import json
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "landmarks"


def load_fixtures(key):
    """All fixture JSONs (recursive) whose expected[key] is not null -> list of (name, data)."""
    out = []
    for p in sorted(FIXTURE_ROOT.rglob("*.json")):
        d = json.loads(p.read_text())
        if (d.get("expected") or {}).get(key) is not None:
            out.append(pytest.param(d, id=p.name))
    return out
