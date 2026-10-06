"""SPOILER WARNING: stand-in implementations of the hand-written modules. Don't open this before writing fingers.py / features.py yourself.

The app uses these until hand_synth.fingers / hand_synth.features are written.
Landmarks: 21 (x, y, z) tuples, height-normalized aspect-corrected coords of the
mirrored frame (see tracking.py).
"""
import math
from types import SimpleNamespace

# ── finger detection (joint geometry; independent of image rotation) ─────────
FINGER_ANGLE_MIN = 115.0   # deg; interior angle at PIP must be at least this
FINGER_REACH_MIN = 1.08    # |tip - wrist| / |pip - wrist|
THUMB_ANGLE_MIN = 130.0    # deg; interior angle at the thumb IP joint
THUMB_SPREAD_MIN = 0.95    # |thumb tip - pinky MCP| / hand size


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _dist(lm, i, j):
    return _norm(_sub(lm[i], lm[j]))


def _angle(lm, a, b, c):
    """Interior angle (deg) at joint b between segments b->a and b->c, in 3D."""
    u, v = _sub(lm[a], lm[b]), _sub(lm[c], lm[b])
    d = _norm(u) * _norm(v)
    if d == 0:
        return 180.0
    dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / d
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _finger_extended(lm, mcp):
    pip, dip, tip = mcp + 1, mcp + 2, mcp + 3
    reach = _dist(lm, tip, 0) / max(_dist(lm, pip, 0), 1e-9)
    return _angle(lm, mcp, pip, dip) >= FINGER_ANGLE_MIN and reach >= FINGER_REACH_MIN


def _thumb_extended(lm):
    size = max(_dist(lm, 0, 9), 1e-9)
    return (_angle(lm, 2, 3, 4) >= THUMB_ANGLE_MIN
            and _dist(lm, 4, 17) / size >= THUMB_SPREAD_MIN)


def fingers_up(landmarks, label):
    """(thumb, index, middle, ring, pinky); `label` unused (geometry is mirror-symmetric)."""
    lm = landmarks
    return (_thumb_extended(lm),) + tuple(_finger_extended(lm, m) for m in (5, 9, 13, 17))


# ── features ─────────────────────────────────────────────────────────────────
def hand_size(landmarks):
    return math.hypot(landmarks[9][0] - landmarks[0][0], landmarks[9][1] - landmarks[0][1])


def wrist_xy(landmarks):
    return (landmarks[0][0], landmarks[0][1])


def pinch_norm(landmarks):
    d = math.hypot(landmarks[4][0] - landmarks[8][0], landmarks[4][1] - landmarks[8][1])
    return d / max(hand_size(landmarks), 1e-9)


def roll_deg(landmarks):
    dx = landmarks[9][0] - landmarks[0][0]
    dy = landmarks[9][1] - landmarks[0][1]
    return math.degrees(math.atan2(dx, -dy))


def feature_vector(landmarks):
    wx, wy, wz = landmarks[0]
    s = max(hand_size(landmarks), 1e-9)
    return [v / s for p in landmarks for v in ((p[0] - wx), (p[1] - wy), (p[2] - wz))]


# ── probes and resolvers ─────────────────────────────────────────────────────
PROBE_LANDMARKS = [
    (0.8900, 0.5900, 0.0000),
    (0.8450, 0.5540, 0.0000),
    (0.7928, 0.4910, 0.0000),
    (0.7613, 0.4370, -0.0036),
    (0.7343, 0.3902, -0.0054),
    (0.8306, 0.4190, 0.0000),
    (0.8306, 0.3290, 0.0000),
    (0.8306, 0.2750, 0.0000),
    (0.8306, 0.2300, 0.0000),
    (0.8900, 0.4100, 0.0000),
    (0.8900, 0.3110, 0.0000),
    (0.8900, 0.2480, 0.0000),
    (0.8900, 0.1994, 0.0000),
    (0.9440, 0.4190, 0.0000),
    (0.9440, 0.3290, 0.0000),
    (0.9440, 0.2696, 0.0000),
    (0.9440, 0.2228, 0.0000),
    (0.9890, 0.4370, 0.0000),
    (0.9890, 0.3650, 0.0000),
    (0.9890, 0.3200, 0.0000),
    (0.9890, 0.2804, 0.0000),
]

_STANDIN_FEATURES = {"hand_size": hand_size, "wrist_xy": wrist_xy, "pinch_norm": pinch_norm,
                     "roll_deg": roll_deg, "feature_vector": feature_vector}


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


_FEATURE_VALID = {
    "hand_size": lambda r: _is_num(r) and r > 0,
    "wrist_xy": lambda r: len(r) == 2 and all(_is_num(v) for v in r),
    "pinch_norm": _is_num,
    "roll_deg": _is_num,
    "feature_vector": lambda r: len(r) == 63 and all(_is_num(v) for v in r),
}


def _guard(manual, standin):
    """Call `manual`; on any exception fall back to `standin` and record the error."""
    def wrapper(*args, **kwargs):
        try:
            return manual(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - never crash the app from user code
            wrapper.last_error = f"{type(e).__name__}: {e}"
            return standin(*args, **kwargs)
    wrapper.last_error = None
    wrapper.__name__ = getattr(manual, "__name__", "manual")
    return wrapper


def resolve_fingers():
    """-> (fingers_up callable, status). status: "manual" | "stand-in" | "stand-in (fingers.py error: ...)"."""
    from hand_synth import fingers
    try:
        r = fingers.fingers_up(PROBE_LANDMARKS, "Right")
        if not (isinstance(r, tuple) and len(r) == 5):
            raise TypeError(f"expected a 5-tuple, got {r!r}")
    except NotImplementedError:
        return fingers_up, "stand-in"
    except Exception as e:  # noqa: BLE001
        return fingers_up, f"stand-in (fingers.py error: {type(e).__name__}: {e})"
    return _guard(fingers.fingers_up, fingers_up), "manual"


def resolve_features():
    """-> (SimpleNamespace of the 5 feature functions, {name: status}). Each resolved independently."""
    from hand_synth import features
    ns, status = SimpleNamespace(), {}
    for name, standin in _STANDIN_FEATURES.items():
        fn = getattr(features, name, None)
        try:
            if fn is None:
                raise NotImplementedError
            if not _FEATURE_VALID[name](fn(PROBE_LANDMARKS)):
                raise ValueError("implausible return value")
        except NotImplementedError:
            ns.__dict__[name], status[name] = standin, "stand-in"
        except Exception as e:  # noqa: BLE001
            ns.__dict__[name], status[name] = standin, f"stand-in (features.py error: {type(e).__name__}: {e})"
        else:
            ns.__dict__[name], status[name] = _guard(fn, standin), "manual"
    return ns, status
