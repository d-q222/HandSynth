"""Synthetic MediaPipe-like hands in the app's landmark convention.

Coordinates: height-normalized, aspect-corrected image coords of the mirrored
frame (y down, x scaled by width/height, z toward camera is negative).
Palm faces the camera, fingers up. Run as a script to write the fixtures:

    .venv/bin/python tests/synthetic_hand.py
"""
import itertools
import json
import math
import random
import zlib
from pathlib import Path

HAND_SIZE = 0.18  # wrist -> middle MCP at scale 1

# Local hand frame in units of hand size: wrist at origin, +x toward the pinky
# side of a Right hand's palm view mirrored (thumb at -x for Right), y down.
# name: (mcp_xy, (proximal, middle, distal) lengths)
_FINGERS = {
    "index": ((-0.33, -0.95), (0.50, 0.30, 0.25)),
    "middle": ((0.0, -1.0), (0.55, 0.35, 0.27)),
    "ring": ((0.30, -0.95), (0.50, 0.33, 0.26)),
    "pinky": ((0.55, -0.85), (0.40, 0.25, 0.22)),
}
# cumulative flexion (deg) toward the camera at MCP, PIP, DIP
_CURL = (80.0, 175.0, 235.0)
_THUMB_CMC = (-0.25, -0.20)
_THUMB_OUT = [(-0.54, -0.55, 0.0), (-0.715, -0.85, -0.02), (-0.865, -1.11, -0.03)]
_THUMB_IN = [(-0.38, -0.40, -0.05), (-0.15, -0.52, -0.12), (0.08, -0.56, -0.16)]


def _finger(mcp, lens, curled, jitter):
    cum = _CURL if curled else (0.0, 0.0, 0.0)
    pts, (x, y), z = [], mcp, 0.0
    pts.append((x, y, z))
    for ln, a in zip(lens, cum):
        a = math.radians(a + jitter)
        y -= ln * math.cos(a)
        z -= ln * math.sin(a)
        pts.append((x, y, z))
    return pts


def make_hand(state, label="Right", roll_deg=0.0, scale=1.0, center=(0.89, 0.5),
              xy_noise=0.0, z_noise=0.0, seed=None):
    rng = random.Random(seed)
    sign = 1.0 if label == "Right" else -1.0  # Right: thumb toward smaller x
    thumb, *four = state
    loc = [(0.0, 0.0, 0.0), (*_THUMB_CMC, 0.0)]
    loc += _THUMB_IN if not thumb else _THUMB_OUT
    for f, up in zip(_FINGERS.values(), four):
        loc += _finger(*f, curled=not up, jitter=0.0)
    r = math.radians(roll_deg)
    c, s = math.cos(r), math.sin(r)
    out = []
    for x, y, z in loc:
        x *= sign
        x, y = x * c - y * s, x * s + y * c
        out.append([x, y, z])
    # center on the wrist/middle-MCP midpoint
    mx, my = (out[0][0] + out[9][0]) / 2, (out[0][1] + out[9][1]) / 2
    k = HAND_SIZE * scale
    res = []
    for x, y, z in out:
        res.append((center[0] + (x - mx) * k + rng.gauss(0, xy_noise),
                    center[1] + (y - my) * k + rng.gauss(0, xy_noise),
                    z * k + rng.gauss(0, z_noise)))
    return res


def state_str(state):
    return "".join("T" if s else "F" for s in state)


def write_fixture(path, label, lm, fingers, roll):
    data = {"label": label, "landmarks": [[round(v, 5) for v in p] for p in lm],
            "score": 0.95, "source": "synthetic",
            "expected": {"fingers": list(fingers) if fingers is not None else None,
                         "roll_deg": roll}}
    Path(path).write_text(json.dumps(data))


def write_all(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    n = 0
    variations = [(0.0, 1.0, 0.0), (-30.0, 0.8, 0.002), (30.0, 1.3, 0.003)]
    for label in ("Left", "Right"):
        for state in itertools.product((True, False), repeat=5):
            for i, (roll, scale, noise) in enumerate(variations, 1):
                lm = make_hand(state, label, roll, scale, xy_noise=noise,
                               z_noise=noise * 3, seed=zlib.crc32(f"{label}{state}{i}".encode()))
                write_fixture(out / f"{label.lower()}_{state_str(state)}_{i:03d}.json",
                              label, lm, state, roll)
                n += 1
        for r in (45, -45, 0):
            k = 0
            for scale in (1.0, 1.4):
                for state in ((True,) * 5, (False, True, True, False, False)):
                    k += 1
                    lm = make_hand(state, label, r, scale, xy_noise=0.002, seed=r + k)
                    write_fixture(out / f"{label.lower()}_roll{r:+03d}_{k:03d}.json",
                                  label, lm, state, float(r))
                    n += 1
    return n


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    print(write_all(root / "fixtures/landmarks/synthetic"), "fixtures written")
