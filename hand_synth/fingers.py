"""MANUAL — written by Daniel, no AI assistance (course reconstruct gate).

Claude Code ships only this stub, the fixtures, and tests/test_fingers.py.
Until this raises something other than NotImplementedError, the app runs on
hand_synth/standins.py and shows a "STAND-IN" badge on screen.
"""

FingerState = tuple[bool, bool, bool, bool, bool]


def fingers_up(landmarks, label: str) -> FingerState:
    """Return (thumb, index, middle, ring, pinky). True = extended.

    `landmarks`: 21 (x, y, z) tuples, MediaPipe normalized image coords.
    `label`: "Left" | "Right", as the user experiences it (mirrored view).
    """
    raise NotImplementedError  # Daniel writes this.
