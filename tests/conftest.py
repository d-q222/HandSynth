import json
import re
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "landmarks"


# PRD C8: a finger fixture's expected tuple is the tag in its filename (left_TFFFF_001.json).
# The filename wins over the JSON's expected.fingers, so fixing a misread recording is a rename.
_TAGGED = re.compile(r"^(left|right)_([TF]{5})_\d+\.json$", re.IGNORECASE)


def load_fixtures(key, dirs=None):
    """Fixture JSONs (recursive; only under `dirs` if given) with an expected `key` -> pytest params (id = filename)."""
    out = []
    roots = [FIXTURE_ROOT / d for d in dirs] if dirs else [FIXTURE_ROOT]
    for p in sorted(q for r in roots for q in r.rglob("*.json")):
        d = json.loads(p.read_text())
        expected = dict(d.get("expected") or {})
        m = _TAGGED.match(p.name)
        if key == "fingers" and m:
            expected["fingers"] = [c == "T" for c in m.group(2).upper()]
        d["expected"] = expected
        if expected.get(key) is not None:
            out.append(pytest.param(d, id=p.name))
    return out
