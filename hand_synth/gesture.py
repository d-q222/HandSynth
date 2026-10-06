"""The ML seam: main.py depends only on GestureSource.degree(hand)."""
from typing import Protocol

from .stabilize import Hold

KEEP = "KEEP"


def degree_from_fingers(state, finger_map):
    if state is None:
        return KEEP
    return finger_map.get(state, KEEP)


class GestureSource(Protocol):
    def degree(self, hand): ...


class FingerMapSource:
    """v1: fingers_fn(landmarks, label) -> Hold -> finger_map."""

    def __init__(self, finger_map, fingers_fn, hold_frames):
        self.finger_map = finger_map
        self.fingers_fn = fingers_fn
        self._hold = Hold(hold_frames)
        self._hold_frames = hold_frames

    @property
    def state(self):
        return self._hold.value

    def degree(self, hand):
        self._hold.update(tuple(self.fingers_fn(hand.landmarks, hand.label)))
        return degree_from_fingers(self._hold.value, self.finger_map)

    def reset(self):
        self._hold = Hold(self._hold_frames)


class ClassifierSource:
    """Later: feature_fn -> model.predict -> degree. Label None/0 = silence."""

    def __init__(self, model, feature_fn, hold_frames):
        self.model = model
        self.feature_fn = feature_fn
        self._hold = Hold(hold_frames)
        self._hold_frames = hold_frames

    def degree(self, hand):
        label = self.model.predict([self.feature_fn(hand.landmarks)])[0]
        label = None if label in (None, 0) else int(label)
        self._hold.update(("d", label))
        held = self._hold.value
        return KEEP if held is None else held[1]

    def reset(self):
        self._hold = Hold(self._hold_frames)
