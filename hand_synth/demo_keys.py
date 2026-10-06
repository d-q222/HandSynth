"""Keyboard-driven proof of music + synth, no camera.  python -m hand_synth.demo_keys"""
import sys

import cv2
import numpy as np

from . import config as cfg
from .keycodes import arrow
from .music import chord_freqs, chord_name
from .synth.base import SynthParams
from .synth.numpy_synth import NumpySynth

HELP = [
    "1-7  degree        0  silence",
    "m    flip major/minor",
    "s    cycle none/sus2/sus4",
    "j/k/l  toggle add7/add9/add6 (7,9,6 are degrees)",
    "n    mod none",
    "left/right  key   up/down  gain",
    "[ / ]  cutoff     w  waveform",
    "q / Esc  quit",
]


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
    wave_i = cfg.WAVEFORMS.index(cfg.WAVEFORM)
    last = None
    params_dirty = True
    try:
        while True:
            key = cfg.KEY_NAMES[key_i]
            state = (key, degree, flipped, mod)
            if state != last:
                synth.set_chord([] if degree is None else chord_freqs(key, degree, flipped, mod, cfg.ROOT_OCTAVE))
                last = state
            if params_dirty:
                synth.set_params(SynthParams(gain, cutoff, cfg.WAVEFORMS[wave_i]))
                params_dirty = False

            img = np.zeros((360, 560, 3), np.uint8)
            name = "-" if degree is None else chord_name(key, degree, flipped, mod)
            lines = [f"key {key}   chord {name}", f"degree {degree}   flip {flipped}   mod {mod}",
                     f"wave {cfg.WAVEFORMS[wave_i]}   gain {gain:.2f}   cutoff {cutoff:.0f} Hz"]
            if synth.recent_xrun():
                lines.append("XRUN")
            for i, t in enumerate(lines):
                cv2.putText(img, t, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            for i, t in enumerate(HELP):
                cv2.putText(img, t, (12, 150 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            cv2.imshow(cfg.WINDOW_NAME + " keys", img)

            k = cv2.waitKeyEx(15)
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
