"""Keyboard-driven proof of music + synth, no camera.  python -m hand_synth.demo_keys"""
import sys

import cv2
import numpy as np

from . import config as cfg
from . import ui
from .keycodes import arrow
from .main import voicing_options
from .music import chord_name, chord_voicing, midi_to_hz
from .synth.base import SynthParams
from .synth.numpy_synth import NumpySynth

HELP = [
    ("1 – 7", "degree"), ("0", "silence"), ("M", "flip major / minor"),
    ("S", "none / sus2 / sus4"), ("J  K  L", "add7 · add9 · add6"), ("N", "no modifier"),
    ("Left Right", "key"), ("Up Down", "volume"), ("[  ]", "brightness"), ("W", "waveform"), ("Q", "quit"),
]
SIZE = (960, 540)


def draw(img, keybar, key, degree, flipped, mod, wave, gain, cutoff, xrun):
    w, h = SIZE
    img[:] = (20, 18, 17)
    keybar.draw(img, key)
    s = ui.OverlayState(key=key, chord="" if degree is None else chord_name(key, degree, flipped, mod),
                        degree=degree, flipped=flipped, mod=mod, chord_fingers=None, expr_fingers=None,
                        roll=None, zone="natural", gain=gain / cfg.GAIN_RANGE[1], brightness=0.0,
                        cutoff_hz=cutoff, waveform=wave, chord_hand=cfg.CHORD_HAND, expr_hand=cfg.EXPR_HAND,
                        xrun=xrun)
    ui._chord_card(img, s, ui.MARGIN, ui.MARGIN + cfg.KEYBAR_HEIGHT + 12, detailed=True)
    x, y = w // 2 + 20, ui.MARGIN + cfg.KEYBAR_HEIGHT + 12
    ui.panel(img, x - 20, y, w - ui.MARGIN, y + 34 + 24 * len(HELP), r=18)
    ui.draw_text(img, "KEYBOARD", x, y + 16, 12, ui.TEXT_3, "bold")
    for i, (k, what) in enumerate(HELP):
        ui.draw_text(img, k, x, y + 38 + 24 * i, 15, ui.TEXT, "medium")
        ui.draw_text(img, what, x + 90, y + 38 + 24 * i, 15, ui.TEXT_2)
    my = h - ui.MARGIN - 100
    ui.panel(img, ui.MARGIN, my, w // 2, h - ui.MARGIN, r=18)
    mw = w // 2 - ui.MARGIN - 44
    ui._meter(img, ui.MARGIN + 22, my + 18, mw, "VOLUME", s.gain, f"{s.gain * 100:.0f}%")
    lo, hi = cfg.CUTOFF_HZ
    ui._meter(img, ui.MARGIN + 22, my + 60, mw, "BRIGHTNESS", np.log(cutoff / lo) / np.log(hi / lo), f"{cutoff:,.0f} Hz")
    ui.pill(img, f"Wave · {wave.capitalize()}", ui.MARGIN, my - 44, 15, ui.TEXT, ui.MUTED, alpha=0.85)
    if xrun:
        ui.pill(img, "XRUN", w // 2 - 70, my - 44, 15, ui.TEXT, ui.ALERT)


def main():
    synth = NumpySynth(cfg)
    try:
        synth.start()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    key_i = cfg.KEY_NAMES.index(cfg.KEY)
    degree, flipped, mod = 1, False, "none"
    gain, cutoff = 0.5 * cfg.GAIN_RANGE[1], 2000.0
    lo, hi = cfg.REVERB_MIX_RANGE
    reverb = lo + (hi - lo) * cfg.REVERB_START
    wave_i = cfg.WAVEFORMS.index(cfg.WAVEFORM)
    last = None
    voicing = []
    params_dirty = True
    win = cfg.WINDOW_NAME + " - keys"
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    picked = {"key": None}
    keybar = ui.KeyBar(cfg.KEY_NAMES, lambda k: picked.update(key=k), cfg.KEYBAR_HEIGHT)
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, keybar.mouse)
    try:
        while True:
            if picked["key"]:
                key_i, picked["key"] = cfg.KEY_NAMES.index(picked["key"]), None
            key = cfg.KEY_NAMES[key_i]
            state = (key, degree, flipped, mod)
            if state != last:
                voicing = [] if degree is None else chord_voicing(
                    key, degree, flipped, mod, voicing or None, **voicing_options())
                synth.set_chord([midi_to_hz(n) for n in voicing])
                last = state
            if params_dirty:
                synth.set_params(SynthParams(gain, cutoff, cfg.WAVEFORMS[wave_i], reverb=reverb))
                params_dirty = False

            draw(img, keybar, key, degree, flipped, mod, cfg.WAVEFORMS[wave_i], gain, cutoff, synth.recent_xrun())
            cv2.imshow(win, img)

            k = cv2.waitKeyEx(15)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == -1:
                continue
            a = arrow(k)
            c = chr(k) if 0 <= k < 128 else ""
            if c in ("q", "\x1b"):
                break
            elif c in "1234567" and c:
                degree = int(c)
            elif c == "0":
                degree = None
            elif c == "m":
                flipped = not flipped
            elif c == "s":
                mod = {"none": "sus2", "sus2": "sus4"}.get(mod, "none")
            elif c in ("j", "k", "l"):
                m = {"j": "add7", "k": "add9", "l": "add6"}[c]
                mod = "none" if mod == m else m
            elif c == "n":
                mod = "none"
            elif a == "left":
                key_i = (key_i - 1) % len(cfg.KEY_NAMES)
            elif a == "right":
                key_i = (key_i + 1) % len(cfg.KEY_NAMES)
            elif a == "up":
                gain = min(cfg.GAIN_RANGE[1], gain + 0.05); params_dirty = True
            elif a == "down":
                gain = max(cfg.GAIN_RANGE[0], gain - 0.05); params_dirty = True
            elif c == "[":
                cutoff = max(cfg.CUTOFF_HZ[0], cutoff / 1.2); params_dirty = True
            elif c == "]":
                cutoff = min(cfg.CUTOFF_HZ[1], cutoff * 1.2); params_dirty = True
            elif c == "w":
                wave_i = (wave_i + 1) % len(cfg.WAVEFORMS); params_dirty = True
    finally:
        synth.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
