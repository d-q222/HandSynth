import math

FingerState = tuple[bool, bool, bool, bool, bool]

FINGER_ANGLE_MIN = 115.0   # deg; interior angle at PIP must be at least this
FINGER_REACH_MIN = 1.08    # |tip - wrist| / |pip - wrist|
THUMB_ANGLE_MIN = 130.0    # deg; interior angle at the thumb IP joint
THUMB_SPREAD_MIN = 0.95    # |thumb tip - pinky MCP| / hand size


def _subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _distance(lm, i, j):
    return _norm(_subtract(lm[i], lm[j]))


def _angle(lm, a, b, c):
    """Interior angle (deg) at joint b between segments b->a and b->c, in 3D."""
    u, v = _subtract(lm[a], lm[b]), _subtract(lm[c], lm[b])
    d = _norm(u) * _norm(v)
    if d == 0:
        return 180.0
    dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / d
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _finger_extended(lm, mcp):
    pip, dip, tip = mcp + 1, mcp + 2, mcp + 3
    reach = _distance(lm, tip, 0) / max(_distance(lm, pip, 0), 1e-9)
    return _angle(lm, mcp, pip, dip) >= FINGER_ANGLE_MIN and reach >= FINGER_REACH_MIN


def _thumb_extended(lm):
    size = max(_distance(lm, 0, 9), 1e-9)
    return (_angle(lm, 2, 3, 4) >= THUMB_ANGLE_MIN
            and _distance(lm, 4, 17) / size >= THUMB_SPREAD_MIN)


def fingers_up(landmarks) -> FingerState:
    """Return (thumb, index, middle, ring, pinky). True = extended.

    `landmarks`: 21 (x, y, z) tuples, MediaPipe normalized image coords. Joint geometry only,
    so it reads left and right hands alike.
    """
    lm = landmarks
    return (_thumb_extended(lm),) + tuple(_finger_extended(lm, m) for m in (5, 9, 13, 17))
