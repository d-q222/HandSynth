"""Regenerate docs/screenshot.png: the full overlay drawn over synthetic hands (no camera needed)."""
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(root), str(root / "tests")]
from hand_synth import config as cfg, standins  # noqa: E402
from hand_synth.main import Instrument  # noqa: E402
from hand_synth.tracking import Hand, HandsFrame  # noqa: E402
from hand_synth.ui import KeyBar, OverlayState, draw_overlay  # noqa: E402
from synthetic_hand import make_hand  # noqa: E402

left = Hand("Left", make_hand((False, True, False, False, False), "Left", roll_deg=-8, scale=1.1, center=(0.52, 0.56)), 0.97)
right = Hand("Right", make_hand((False, True, True, True, False), "Right", roll_deg=6, scale=1.1, center=(1.30, 0.50)), 0.97)
hands = HandsFrame(left, right)
inst = Instrument(standins.fingers_up, standins, key="G")
for _ in range(12):
    inst.update(hands)

y = np.linspace(0, 1, 720)[:, None, None]
frame = ((np.array([46, 40, 36]) * (1 - y) + np.array([24, 20, 18]) * y) * np.ones((1, 1280, 1))).astype(np.uint8)
s = inst.state
view = OverlayState(
    key=s["key"], chord=inst.chord_name(), degree=s["degree"], flipped=s["flipped"], mod=s["mod"],
    chord_fingers=inst.chord_fingers, expr_fingers=inst.expr_fingers, roll=inst.roll, zone=inst.hold_roll.value,
    gain=inst.ema_gain.value, brightness=inst.ema_bright.value, cutoff_hz=inst.cutoff_hz, waveform=inst.waveform,
    space=inst.ema_space.value, chord_hand=cfg.CHORD_HAND, expr_hand=cfg.EXPR_HAND, fps=30)
draw_overlay(frame, hands, view, cfg.KEYBAR_HEIGHT, cfg.HUD_HEIGHT, "full")
KeyBar(cfg.KEY_NAMES, lambda k: None, cfg.KEYBAR_HEIGHT).draw(frame, s["key"])
out = root / "docs" / "screenshot.png"
cv2.imwrite(str(out), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
print(f"{inst.chord_name()} -> {out.relative_to(root)}")
