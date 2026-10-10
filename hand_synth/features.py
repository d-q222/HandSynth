import math


def hand_size(landmarks) -> float:
    """Distance wrist(0) → middle MCP(9)."""
    return math.hypot(landmarks[9][0] - landmarks[0][0], landmarks[9][1] - landmarks[0][1])


def wrist_xy(landmarks) -> tuple[float, float]:
    """landmarks[0][:2]."""
    return (landmarks[0][0], landmarks[0][1])


def pinch_norm(landmarks) -> float:
    """dist(4, 8) / hand_size, unclamped."""
    d = math.hypot(landmarks[4][0] - landmarks[8][0], landmarks[4][1] - landmarks[8][1])
    return d / max(hand_size(landmarks), 1e-9)


def roll_deg(landmarks) -> float:
    """Signed angle of wrist(0)→middle MCP(9) vs. straight up, in degrees. Upright ≈ 0."""
    dx = landmarks[9][0] - landmarks[0][0]
    dy = landmarks[9][1] - landmarks[0][1]
    return math.degrees(math.atan2(dx, -dy))


def feature_vector(landmarks) -> list[float]:
    """All 21 points translated to wrist origin, scaled by hand_size, flattened (63 floats)."""
    wx, wy, wz = landmarks[0]
    s = max(hand_size(landmarks), 1e-9)
    return [v / s for p in landmarks for v in ((p[0] - wx), (p[1] - wy), (p[2] - wz))]
