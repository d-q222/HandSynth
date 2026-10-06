"""Overlay drawing and the clickable key bar. Everything draws onto the BGR frame in place."""
from dataclasses import dataclass, field
from typing import Callable

import cv2

from .tracking import Hand, HandsFrame

FONT = cv2.FONT_HERSHEY_DUPLEX          # Hershey fonts are ASCII-only: no "·", "°", "—"
WHITE, GREY, DARK = (245, 245, 245), (150, 150, 150), (25, 25, 25)
CHORD_COLOR = (255, 200, 60)    # BGR: sky blue, chord hand
EXPR_COLOR = (60, 170, 255)     # BGR: amber, expression hand
ACCENT = (120, 230, 120)
ALERT = (60, 60, 235)

# MediaPipe's 21-point hand topology
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FINGER_LETTERS = "TIMRP"


@dataclass
class OverlayState:
    key: str
    chord: str                          # display name, "-" when silent
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
        self._width = 0
        self._hover = None
        self.visible = True

    def _index_at(self, x, y):
        if self._width <= 0 or not (0 <= y < self.height) or not (0 <= x < self._width):
            return None
        return int(x * len(self.key_names) / self._width)

    def draw(self, frame, current_key):
        self._width = frame.shape[1]
        n = len(self.key_names)
        bw = self._width / n
        _blend_rect(frame, (0, 0), (self._width, self.height), DARK, 0.75)
        for i, name in enumerate(self.key_names):
            x0, x1 = int(i * bw), int((i + 1) * bw)
            if name == current_key:
                cv2.rectangle(frame, (x0 + 3, 4), (x1 - 3, self.height - 4), ACCENT, -1)
                color = DARK
            else:
                if i == self._hover:
                    cv2.rectangle(frame, (x0 + 3, 4), (x1 - 3, self.height - 4), (80, 80, 80), -1)
                cv2.rectangle(frame, (x0 + 3, 4), (x1 - 3, self.height - 4), (90, 90, 90), 1)
                color = WHITE
            _center_text(frame, name, (x0 + x1) // 2, self.height // 2, 0.75, color, 1)

    def mouse(self, event, x, y, flags, param):
        i = self._index_at(x, y) if self.visible else None
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover = i
        elif event == cv2.EVENT_LBUTTONDOWN and i is not None:
            self.on_select(self.key_names[i])


def draw_overlay(frame, hands: HandsFrame, s: OverlayState, keybar_height: int, hud_height: int, mode="full"):
    """mode: "full" (everything) | "minimal" (key bar, hands, chord) | "clean" (chord name only).
    Audio alerts show in every mode."""
    h, w = frame.shape[:2]
    if mode != "clean":
        for hand in hands.hands():
            _draw_hand(frame, hand, h, CHORD_COLOR if hand.label == s.chord_hand else EXPR_COLOR)

    # Big chord name under the key bar
    y = (keybar_height if mode != "clean" else 0) + 70
    _text(frame, s.chord, (24, y), 2.2, WHITE, 3)
    if mode != "full":
        alerts = [a for a in ("AUDIO ERROR" if s.audio_error else None, "XRUN" if s.xrun else None) if a]
        if alerts:
            _text(frame, "  ".join(alerts), (w - 260, y - 30), 0.7, ALERT, 2)
        if s.flash:
            _center_text(frame, s.flash, w // 2, h - 40, 0.8, ACCENT, 2)
        return
    sub = f"key {s.key}"
    if s.degree is not None:
        sub += f"   degree {_roman(s.degree)}" + ("   flipped" if s.flipped else "")
    if s.mod != "none":
        sub += f"   {s.mod}"
    _text(frame, sub, (26, y + 34), 0.7, GREY, 1)

    # Status badges, top-right under the key bar
    bx = w - 16
    badges = list(s.badges)
    if s.xrun:
        badges.insert(0, "XRUN")
    if s.audio_error:
        badges.insert(0, "AUDIO ERROR")
    for b in badges:
        (tw, th), _ = cv2.getTextSize(b, FONT, 0.55, 1)
        x0 = bx - tw - 16
        color = ALERT if b in ("XRUN", "AUDIO ERROR") else (40, 110, 190)
        cv2.rectangle(frame, (x0, keybar_height + 12), (bx, keybar_height + 40), color, -1)
        _text(frame, b, (x0 + 8, keybar_height + 32), 0.55, WHITE, 1, shadow=False)
        bx = x0 - 8

    # Bottom HUD
    top = h - hud_height
    _blend_rect(frame, (0, top), (w, h), DARK, 0.7)
    col = w // 4
    _hand_panel(frame, 20, top + 30, f"{s.chord_hand} hand: chord", CHORD_COLOR, s.chord_fingers,
                f"roll {s.roll:+.0f} deg  {s.zone}" if s.roll is not None else "not in view")
    _hand_panel(frame, 20 + col, top + 30, f"{s.expr_hand} hand: expression", EXPR_COLOR, s.expr_fingers,
                f"mod {s.mod}" if s.expr_fingers is not None else "not in view (mod none)")
    _meter(frame, 20 + 2 * col, top + 30, col - 40, "volume", s.gain, f"{s.gain * 100:.0f}%")
    _meter(frame, 20 + 2 * col, top + 80, col - 40, "brightness", s.brightness, f"{s.cutoff_hz:.0f} Hz")
    x = 20 + 3 * col
    _text(frame, f"wave  {s.waveform}", (x, top + 38), 0.6, WHITE, 1)
    _text(frame, f"{s.fps:4.0f} fps", (x, top + 66), 0.6, GREY, 1)
    _text(frame, "click a key or <- ->", (x, top + 92), 0.45, GREY, 1)
    _text(frame, "w wave  h hide  r rec  q quit", (x, top + 110), 0.45, GREY, 1)

    if s.flash:
        _center_text(frame, s.flash, w // 2, top - 30, 0.8, ACCENT, 2)
    if s.audio_error:
        _text(frame, s.audio_error[:90], (20, top - 14), 0.5, ALERT, 1)


def _draw_hand(frame, hand: Hand, h: int, color):
    pts = [(int(x * h), int(y * h)) for x, y, _ in hand.landmarks]   # landmarks are height-normalized
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 5 if i in (4, 8, 12, 16, 20) else 3, WHITE, -1, cv2.LINE_AA)
    wx, wy = pts[0]
    _text(frame, hand.label, (wx - 20, wy + 30), 0.6, color, 2)


def _hand_panel(frame, x, y, title, color, fingers, detail):
    _text(frame, title, (x, y), 0.55, color, 1)
    for i, letter in enumerate(FINGER_LETTERS):
        cx, cy = x + 14 + i * 34, y + 30
        up = fingers is not None and fingers[i]
        cv2.circle(frame, (cx, cy), 13, color if up else (70, 70, 70), -1 if up else 2, cv2.LINE_AA)
        _center_text(frame, letter, cx, cy, 0.45, DARK if up else GREY, 1, shadow=False)
    _text(frame, detail, (x, y + 68), 0.5, WHITE, 1)


def _meter(frame, x, y, width, label, value, readout):
    value = min(1.0, max(0.0, value))
    _text(frame, label, (x, y), 0.55, GREY, 1)
    _text(frame, readout, (x + width - 90, y), 0.55, WHITE, 1)
    cv2.rectangle(frame, (x, y + 10), (x + width, y + 24), (70, 70, 70), -1)
    cv2.rectangle(frame, (x, y + 10), (x + int(width * value), y + 24), EXPR_COLOR, -1)


def _blend_rect(frame, p0, p1, color, alpha):
    (x0, y0), (x1, y1) = p0, p1
    roi = frame[max(0, y0):y1, max(0, x0):x1]
    if roi.size:
        roi[:] = (roi * (1 - alpha) + [c * alpha for c in color]).astype(roi.dtype)


def _text(frame, s, org, scale, color, thick, shadow=True):
    if shadow:
        cv2.putText(frame, s, (org[0] + 1, org[1] + 2), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _center_text(frame, s, cx, cy, scale, color, thick, shadow=True):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
    _text(frame, s, (cx - tw // 2, cy + th // 2), scale, color, thick, shadow)


def _roman(n: int) -> str:
    return ["I", "II", "III", "IV", "V", "VI", "VII"][n - 1]
