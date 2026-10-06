"""Overlay smoke tests: every mode draws on any frame size without crashing, and the key bar hit-tests."""
import cv2
import numpy as np
import pytest

from hand_synth import config as cfg
from hand_synth.tracking import Hand, HandsFrame
from hand_synth.ui import KeyBar, OverlayState, draw_overlay, numeral
from synthetic_hand import make_hand

OPEN = (True,) * 5


def state(**kw):
    base = dict(key="G", chord="Am7", degree=2, flipped=False, mod="add7", chord_fingers=OPEN, expr_fingers=None,
                roll=12.0, zone="natural", gain=0.5, brightness=0.3, cutoff_hz=1200.0, waveform="triangle",
                chord_hand="Left", expr_hand="Right")
    base.update(kw)
    return OverlayState(**base)


@pytest.mark.parametrize("size", [(1280, 720), (640, 480), (320, 240)])
@pytest.mark.parametrize("mode", cfg.UI_MODES)
def test_overlay_draws_everywhere(size, mode):
    w, h = size
    aspect = w / h
    hands = HandsFrame(
        left=Hand("Left", make_hand(OPEN, "Left", center=(0.02, 0.98)), 0.9),            # hanging off the corner
        right=Hand("Right", make_hand(OPEN, "Right", center=(aspect - 0.02, 0.05)), 0.9),
        aspect=aspect)
    for s in (state(), state(degree=None, chord="", hint=True), state(xrun=True, audio_error="boom " * 40,
              badges=["Stand-in · fingers"], flash="saved 1 fixture(s)", roll=None, expr_fingers=OPEN)):
        frame = np.full((h, w, 3), 90, np.uint8)
        draw_overlay(frame, hands, s, cfg.KEYBAR_HEIGHT, cfg.HUD_HEIGHT, mode)
        if mode != "clean":
            KeyBar(cfg.KEY_NAMES, lambda k: None, cfg.KEYBAR_HEIGHT).draw(frame, s.key)
        assert frame.dtype == np.uint8 and frame.shape == (h, w, 3)
        assert frame.std() > 0                                  # something was drawn


def test_keybar_hit_testing():
    picked = []
    kb = KeyBar(cfg.KEY_NAMES, picked.append, cfg.KEYBAR_HEIGHT)
    kb.mouse(cv2.EVENT_LBUTTONDOWN, 100, 30, 0, None)          # before the first draw: no geometry yet
    assert picked == []
    kb.draw(np.zeros((720, 1280, 3), np.uint8), "C")
    y = 16 + cfg.KEYBAR_HEIGHT // 2
    kb.mouse(cv2.EVENT_LBUTTONDOWN, 20, y, 0, None)
    kb.mouse(cv2.EVENT_LBUTTONDOWN, 1263, y, 0, None)
    kb.mouse(cv2.EVENT_LBUTTONDOWN, 640, 400, 0, None)          # below the bar: ignored
    assert picked == ["C", "B"]
    kb.visible = False
    kb.mouse(cv2.EVENT_LBUTTONDOWN, 20, y, 0, None)
    assert picked == ["C", "B"]


def test_numerals():
    assert [numeral(d, False) for d in range(1, 8)] == ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
    assert numeral(1, True) == "i" and numeral(2, True) == "II" and numeral(7, True) == "vii"
