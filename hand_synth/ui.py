"""Overlay drawing and the clickable key bar. Everything draws onto the BGR frame in place.

Text is rendered with a real TrueType font (Pillow) and cached as alpha masks, so the per-frame
cost is a handful of small blends. Panels are rounded, translucent, and blended only over their
own region.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .music import quality
from .tracking import Hand, HandsFrame

# ── Palette (BGR) ────────────────────────────────────────────────────────────
TEXT = (245, 245, 245)
TEXT_2 = (175, 175, 175)
TEXT_3 = (125, 125, 125)
PANEL = (22, 20, 18)
TRACK = (70, 66, 62)
CHORD_COLOR = (212, 234, 94)    # teal: chord hand
EXPR_COLOR = (36, 191, 251)     # amber: expression hand
ALERT = (113, 113, 248)
MUTED = (120, 105, 95)          # slate: stand-in badges
PANEL_ALPHA = 0.62
MARGIN = 16

# MediaPipe's 21-point hand topology
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FINGER_LETTERS = "TIMRP"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]

# ── Fonts ────────────────────────────────────────────────────────────────────
_FONT_FILES = {   # (path, collection index) per weight; first that loads wins
    "regular": [("/System/Library/Fonts/HelveticaNeue.ttc", 0), ("Arial.ttf", 0), ("DejaVuSans.ttf", 0)],
    "medium": [("/System/Library/Fonts/HelveticaNeue.ttc", 10), ("Arial.ttf", 0), ("DejaVuSans.ttf", 0)],
    "bold": [("/System/Library/Fonts/HelveticaNeue.ttc", 1), ("Arial Bold.ttf", 0), ("DejaVuSans-Bold.ttf", 0)],
}


@lru_cache(maxsize=64)
def _font(size: int, weight: str):
    for path, index in _FONT_FILES[weight]:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


@lru_cache(maxsize=1024)
def _sprite(s: str, size: int, weight: str) -> np.ndarray:
    """Alpha mask (float32 0..1, h×w) of `s`; top of the array is the font's ascender line."""
    font = _font(size, weight)
    ascent, descent = font.getmetrics()
    width = max(1, int(np.ceil(font.getlength(s))) + 2)
    img = Image.new("L", (width, ascent + descent), 0)
    ImageDraw.Draw(img).text((0, 0), s, font=font, fill=255, anchor="la")
    return np.asarray(img, dtype=np.float32) / 255.0


def text_size(s: str, size: int, weight: str = "regular") -> tuple[int, int]:
    a = _sprite(s, size, weight)
    return a.shape[1], a.shape[0]


def draw_text(frame, s: str, x: int, y: int, size: int, color=TEXT, weight="regular", anchor="lt") -> int:
    """Draw `s` with its top-left (or "ct" top-center / "rt" top-right) at (x, y). Returns its width."""
    if not s:
        return 0
    a = _sprite(s, size, weight)
    w = a.shape[1]
    if anchor == "ct":
        x -= w // 2
    elif anchor == "rt":
        x -= w
    _blend(frame, a, int(x), int(y), color)
    return w


def _blend(frame, alpha: np.ndarray, x: int, y: int, color, opacity: float = 1.0):
    fh, fw = frame.shape[:2]
    h, w = alpha.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None] * opacity
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1.0 - a) + np.array(color, np.float32) * a).astype(np.uint8)


@lru_cache(maxsize=128)
def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    scale = 4                                       # supersample for smooth corners
    m = np.zeros((h * scale, w * scale), np.uint8)
    rs = min(r, w // 2, h // 2) * scale
    W, H = w * scale, h * scale
    cv2.rectangle(m, (rs, 0), (W - rs, H), 255, -1)
    cv2.rectangle(m, (0, rs), (W, H - rs), 255, -1)
    for cx, cy in ((rs, rs), (W - rs - 1, rs), (rs, H - rs - 1), (W - rs - 1, H - rs - 1)):
        cv2.circle(m, (cx, cy), rs, 255, -1, cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def panel(frame, x0, y0, x1, y1, r=14, color=PANEL, alpha=PANEL_ALPHA):
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    _blend(frame, _rounded_mask(int(x1 - x0), int(y1 - y0), int(r)), int(x0), int(y0), color, alpha)


def pill(frame, s, x, y, size, fg, bg, alpha=0.9, weight="medium", pad=(12, 6), anchor="lt") -> tuple[int, int]:
    """Text in a rounded pill. Returns (width, height)."""
    tw, th = text_size(s, size, weight)
    w, h = tw + 2 * pad[0], th + 2 * pad[1]
    if anchor == "rt":
        x -= w
    elif anchor == "ct":
        x -= w // 2
    panel(frame, x, y, x + w, y + h, r=h // 2, color=bg, alpha=alpha)
    draw_text(frame, s, x + pad[0], y + pad[1], size, fg, weight)
    return w, h


# ── State + widgets ──────────────────────────────────────────────────────────
@dataclass
class OverlayState:
    key: str
    chord: str                          # display name, "" when silent
    degree: int | None
    flipped: bool
    mod: str
    chord_fingers: tuple | None
    expr_fingers: tuple | None
    roll: float | None
    zone: str
    gain: float                         # 0..1 position in GAIN_RANGE
    brightness: float                   # 0..1 position in CUTOFF_HZ
    cutoff_hz: float
    waveform: str
    chord_hand: str
    expr_hand: str
    space: float = 0.0                  # 0..1 position in REVERB_MIX_RANGE
    pinned: frozenset = frozenset()     # controls set by slider (hand no longer drives them)
    waveforms: tuple = ("sine", "triangle", "saw")
    hint: bool = False                  # no hands in view for a while: show how to start
    fps: float = 0.0
    xrun: bool = False
    audio_error: str | None = None
    badges: list[str] = field(default_factory=list)   # e.g. "STAND-IN fingers"
    flash: str | None = None


class KeyBar:
    """12 buttons across the top of the frame. Handles the cv2 mouse callback."""

    def __init__(self, key_names, on_select: Callable[[str], None], height: int):
        self.key_names = list(key_names)
        self.on_select = on_select
        self.height = height
        self.visible = True
        self._x0, self._x1 = MARGIN, MARGIN
        self._hover = None

    def _index_at(self, x, y):
        if not self.visible or self._x1 <= self._x0:
            return None
        if not (MARGIN <= y < MARGIN + self.height and self._x0 <= x < self._x1):
            return None
        return min(len(self.key_names) - 1, int((x - self._x0) * len(self.key_names) / (self._x1 - self._x0)))

    def draw(self, frame, current_key):
        w = frame.shape[1]
        self._x0, self._x1 = MARGIN, w - MARGIN
        panel(frame, self._x0, MARGIN, self._x1, MARGIN + self.height, r=self.height // 2)
        n = len(self.key_names)
        bw = (self._x1 - self._x0 - 8) / n
        by0, by1 = MARGIN + 5, MARGIN + self.height - 5
        for i, name in enumerate(self.key_names):
            bx0 = int(self._x0 + 4 + i * bw) + 2
            bx1 = int(self._x0 + 4 + (i + 1) * bw) - 2
            if name == current_key:
                panel(frame, bx0, by0, bx1, by1, r=(by1 - by0) // 2, color=TEXT, alpha=0.95)
                color, weight = PANEL, "bold"
            else:
                if i == self._hover:
                    panel(frame, bx0, by0, bx1, by1, r=(by1 - by0) // 2, color=TEXT, alpha=0.14)
                color, weight = TEXT, "medium"
            th = text_size(name, 19, weight)[1]
            draw_text(frame, name, (bx0 + bx1) // 2, (by0 + by1) // 2 - th // 2, 19, color, weight, anchor="ct")

    def mouse(self, event, x, y, flags, param):
        i = self._index_at(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover = i
        elif event == cv2.EVENT_LBUTTONDOWN and i is not None:
            self.on_select(self.key_names[i])


class HudControls:
    """Mouse handling for the bottom panel: drag a slider to set (and pin) a control, click its
    HAND/PINNED chip to give it back to the hand, click a waveform chip to switch. Geometry comes
    from the last draw; nothing is clickable while the panel is hidden."""

    def __init__(self, on_slide: Callable[[str, float], None], on_release: Callable[[str], None],
                 on_wave: Callable[[str], None]):
        self.on_slide, self.on_release, self.on_wave = on_slide, on_release, on_wave
        self.clear()
        self._drag = None

    def clear(self):
        self.sliders: dict[str, tuple[int, int, int, int]] = {}   # name -> bar (x0, y0, x1, y1)
        self.chips: dict[str, tuple[int, int, int, int]] = {}     # name -> HAND/PINNED chip rect
        self.waves: dict[str, tuple[int, int, int, int]] = {}     # waveform -> chip rect

    def _slide(self, name, x):
        x0, _, x1, _ = self.sliders[name]
        self.on_slide(name, min(1.0, max(0.0, (x - x0) / max(1, x1 - x0))))

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, r in self.chips.items():
                if _inside(r, x, y):
                    return self.on_release(name)
            for wf, r in self.waves.items():
                if _inside(r, x, y):
                    return self.on_wave(wf)
            for name, (x0, y0, x1, y1) in self.sliders.items():
                if _inside((x0 - 6, y0 - 10, x1 + 6, y1 + 10), x, y):   # generous grab area
                    self._drag = name
                    return self._slide(name, x)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag in self.sliders:
            self._slide(self._drag, x)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None


def _inside(r, x, y):
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]


def numeral(degree: int, flipped: bool) -> str:
    """Roman numeral with case for quality: V, ii, vii°."""
    q = quality(degree, flipped)
    r = ROMAN[degree - 1]
    return r if q == "M" else r.lower() + ("°" if q == "dim" else "")


def draw_overlay(frame, hands: HandsFrame, s: OverlayState, keybar_height: int, hud_height: int, mode="full",
                 controls: "HudControls | None" = None):
    """mode: "full" (everything) | "minimal" (key bar, hands, chord) | "clean" (chord only).
    Audio alerts show in every mode. `controls` receives the bottom panel's clickable geometry."""
    h, w = frame.shape[:2]
    if controls is not None:
        controls.clear()
    if mode != "clean":
        for hand in hands.hands():
            is_chord = hand.label == s.chord_hand
            _draw_hand(frame, hand, h, CHORD_COLOR if is_chord else EXPR_COLOR, "CHORD" if is_chord else "SHAPE")

    top = MARGIN + (keybar_height + 12 if mode != "clean" else 0)
    _chord_card(frame, s, MARGIN, top, detailed=mode == "full")

    alerts = (["AUDIO ERROR"] if s.audio_error else []) + (["XRUN"] if s.xrun else [])
    bx = w - MARGIN
    for b in alerts + (s.badges if mode == "full" else []):
        bw, _ = pill(frame, b, bx, top, 13, TEXT, ALERT if b in alerts else MUTED, alpha=0.85, anchor="rt")
        bx -= bw + 8

    hud_top = h - MARGIN - hud_height
    if s.hint and mode != "clean":
        _hint(frame, w // 2, (top + hud_top) // 2 if mode == "full" else h // 2)
    if mode == "full":
        _hud(frame, s, MARGIN, hud_top, w - MARGIN, h - MARGIN, controls)
    flash_y = (hud_top if mode == "full" else h - MARGIN) - 46
    if s.flash:
        pill(frame, s.flash, w // 2, flash_y, 17, PANEL, TEXT, alpha=0.92, anchor="ct")
    if s.audio_error:
        pill(frame, s.audio_error[:110], MARGIN, flash_y, 14, TEXT, ALERT, alpha=0.85)


def _chord_card(frame, s: OverlayState, x, y, detailed: bool):
    silent = s.degree is None
    name = "Rest" if silent else s.chord
    parts = [f"Key of {s.key}", "silent" if silent else numeral(s.degree, s.flipped)]
    if not silent and s.flipped:
        parts.append("borrowed")
    sub = "  ·  ".join(parts)
    nw, nh = text_size(name, 76, "bold")
    sw, sh = text_size(sub, 17, "medium") if detailed else (0, 0)
    pad = 20
    cw = max(nw, sw, 150) + 2 * pad
    ch = nh + (sh + 6 if detailed else 0) + pad
    panel(frame, x, y, x + cw, y + ch, r=18)
    draw_text(frame, name, x + pad, y + 8, 76, TEXT_3 if silent else TEXT, "bold")
    if detailed:
        draw_text(frame, sub, x + pad, y + nh + 4, 17, TEXT_2, "medium")


def _hud(frame, s: OverlayState, x0, y0, x1, y1, controls=None):
    panel(frame, x0, y0, x1, y1, r=18)
    col = (x1 - x0) / 4
    pad = 22
    cx = [int(x0 + pad + i * col) for i in range(4)]
    yt = y0 + 18
    for i in range(1, 4):                                   # hairline separators
        xs = int(x0 + i * col) - 8
        seg = frame[y0 + 18:y1 - 18, xs:xs + 1]
        seg[:] = (seg * 0.7 + 30).astype(np.uint8)

    # Chord hand
    _label(frame, f"{s.chord_hand.upper()} HAND  ·  CHORD", cx[0], yt, CHORD_COLOR)
    _finger_dots(frame, cx[0], yt + 26, s.chord_fingers, CHORD_COLOR)
    if s.roll is not None:
        draw_text(frame, f"Roll {s.roll:+.0f}°  ·  {'flipped' if s.zone == 'flipped' else 'upright'}",
                  cx[0], yt + 62, 15, TEXT_2)
    else:
        draw_text(frame, "Out of view  ·  chord holds", cx[0], yt + 62, 15, TEXT_3)

    # Expression hand
    _label(frame, f"{s.expr_hand.upper()} HAND  ·  SHAPE", cx[1], yt, EXPR_COLOR)
    _finger_dots(frame, cx[1], yt + 26, s.expr_fingers, EXPR_COLOR)
    if s.expr_fingers is not None:
        draw_text(frame, f"Modifier  ·  {s.mod}", cx[1], yt + 62, 15, TEXT_2)
    else:
        draw_text(frame, "Out of view", cx[1], yt + 62, 15, TEXT_3)

    # Sliders: hand-driven (amber) until dragged, then pinned (white) until the chip is clicked
    mw = int(col - 2 * pad)
    for name, x, y, label, value, readout in (
            ("volume", cx[2], yt, "VOLUME", s.gain, f"{s.gain * 100:.0f}%"),
            ("brightness", cx[2], yt + 44, "BRIGHTNESS", s.brightness, f"{s.cutoff_hz:,.0f} Hz"),
            ("space", cx[3], yt, "SPACE", s.space, f"{s.space * 100:.0f}%")):
        bar, chip = _meter(frame, x, y, mw, label, value, readout, pinned=name in s.pinned)
        if controls is not None:
            controls.sliders[name], controls.chips[name] = bar, chip

    # Waveform chips + help
    _label(frame, "WAVE", cx[3], yt + 44, TEXT_3)
    wx = cx[3] + mw
    for wf in reversed(s.waveforms):
        on = wf == s.waveform
        cw, ch = pill(frame, wf.capitalize(), wx, yt + 40, 12, PANEL if on else TEXT_2, TEXT if on else TRACK,
                      alpha=0.95 if on else 0.8, weight="bold" if on else "medium", pad=(8, 3), anchor="rt")
        if controls is not None:
            controls.waves[wf] = (wx - cw, yt + 40, wx, yt + 40 + ch)
        wx -= cw + 5
    draw_text(frame, "Drag sliders  ·  H view  W wave  R rec  Q quit", cx[3], yt + 70, 11, TEXT_3)
    draw_text(frame, f"{s.fps:.0f} fps", cx[3] + mw, yt + 70, 11, TEXT_3, anchor="rt")


def _hint(frame, cx, cy):
    title, body = "Raise your hands", "Left hand picks the chord  ·  right hand shapes it"
    tw, th = text_size(title, 34, "bold")
    bw, bh = text_size(body, 17, "medium")
    w, h = max(tw, bw) + 64, th + bh + 44
    panel(frame, cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2, r=22)
    draw_text(frame, title, cx, cy - h // 2 + 18, 34, TEXT, "bold", anchor="ct")
    draw_text(frame, body, cx, cy - h // 2 + 26 + th, 17, TEXT_2, "medium", anchor="ct")


def _label(frame, s, x, y, color):
    draw_text(frame, s, x, y, 12, color, "bold")


def _finger_dots(frame, x, y, fingers, color):
    for i, letter in enumerate(FINGER_LETTERS):
        c = (x + 13 + i * 32, y + 13)
        up = fingers is not None and fingers[i]
        if up:
            cv2.circle(frame, c, 13, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, c, 12, TRACK, 1, cv2.LINE_AA)
        th = text_size(letter, 12, "bold")[1]
        draw_text(frame, letter, c[0], c[1] - th // 2, 12, PANEL if up else TEXT_3, "bold", anchor="ct")


def _meter(frame, x, y, width, label, value, readout, pinned=None):
    """Label + readout + bar. With `pinned` given (True/False) it is a slider: knob + HAND/PINNED chip.
    Returns (bar rect, chip rect or None)."""
    value = min(1.0, max(0.0, value))
    _label(frame, label, x, y, TEXT_3)
    chip = None
    if pinned is not None:
        lw = text_size(label, 12, "bold")[0]
        tag = "PINNED" if pinned else "HAND"
        cw, ch = pill(frame, tag, x + lw + 8, y - 2, 9, PANEL if pinned else TEXT_3, TEXT if pinned else TRACK,
                      alpha=0.95 if pinned else 0.7, weight="bold", pad=(6, 2))
        chip = (x + lw + 8, y - 2, x + lw + 8 + cw, y - 2 + ch)
    draw_text(frame, readout, x + width, y - 2, 14, TEXT, "medium", anchor="rt")
    by = y + 22
    color = TEXT if pinned else EXPR_COLOR
    panel(frame, x, by, x + width, by + 8, r=4, color=TRACK, alpha=0.9)
    if value > 0:
        panel(frame, x, by, x + max(8, int(width * value)), by + 8, r=4, color=color, alpha=1.0)
    if pinned is not None:
        kx = x + int(width * value)
        cv2.circle(frame, (kx, by + 4), 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (kx, by + 4), 3, PANEL, -1, cv2.LINE_AA)
    return (x, by, x + width, by + 8), chip


def _draw_hand(frame, hand: Hand, h: int, color, tag: str):
    pts = [(int(x * h), int(y * h)) for x, y, _ in hand.landmarks]   # landmarks are height-normalized
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for i, p in enumerate(pts):
        tip = i in (4, 8, 12, 16, 20)
        cv2.circle(frame, p, 6 if tip else 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 3 if tip else 2, TEXT, -1, cv2.LINE_AA)
    wx, wy = pts[0]
    pill(frame, tag, wx, wy + 14, 12, PANEL, color, alpha=0.95, weight="bold", pad=(9, 4), anchor="ct")
