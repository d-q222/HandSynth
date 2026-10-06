"""MANUAL — written by Daniel (AI review allowed after tests/test_features.py passes).

Until these raise something other than NotImplementedError, the app runs on
hand_synth/standins.py and shows a "STAND-IN" badge on screen.
"""


def hand_size(landmarks) -> float:
    """Distance wrist(0) → middle MCP(9)."""
    raise NotImplementedError


def wrist_xy(landmarks) -> tuple[float, float]:
    """landmarks[0][:2]."""
    raise NotImplementedError


def pinch_norm(landmarks) -> float:
    """dist(4, 8) / hand_size, unclamped."""
    raise NotImplementedError


def roll_deg(landmarks) -> float:
    """Signed angle of wrist(0)→middle MCP(9) vs. straight up, in degrees. Upright ≈ 0."""
    raise NotImplementedError


def feature_vector(landmarks) -> list[float]:
    """All 21 points translated to wrist origin, scaled by hand_size, flattened (63 floats)."""
    raise NotImplementedError
