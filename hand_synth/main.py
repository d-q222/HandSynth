"""The loop. Wires every module together: track → detect → stabilize → map → synth.

Run: `python -m hand_synth.main` (or `hand-synth` after `pip install -e .`).
`Instrument` holds all mapping state and is testable without a camera or speakers;
`main()` is only camera, window, keys and audio plumbing around it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from . import config as cfg
from . import keycodes, standins
from .gesture import KEEP, FingerMapSource
from .music import MODS, chord_name, chord_voicing, midi_to_hz
from .stabilize import EMA, Hold, RollZone
from .synth import SynthParams
from .tracking import REPO_ROOT, HandsFrame


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def lerp(rng: tuple[float, float], t: float) -> float:
    return rng[0] + (rng[1] - rng[0]) * t


def logmap(rng: tuple[float, float], t: float) -> float:
    return rng[0] * (rng[1] / rng[0]) ** t


def check_config(c=cfg) -> list[str]:
    """Problems in config.py that would otherwise only surface mid-performance (e.g. a typo in MOD_MAP)."""
    errs = []

    def finger_key(name, k):
        if not (isinstance(k, tuple) and len(k) == 5 and all(isinstance(b, bool) for b in k)):
            errs.append(f"{name} key {k!r} must be a tuple of 5 True/False values (thumb, index, middle, ring, pinky)")

    for k, v in c.FINGER_MAP.items():
        finger_key("FINGER_MAP", k)
        if v is not None and v not in range(1, 8):
            errs.append(f"FINGER_MAP{k}: {v!r} must be a scale degree 1-7 or None (silence)")
    for k, v in c.MOD_MAP.items():
        finger_key("MOD_MAP", k)
        if v not in MODS:
            errs.append(f"MOD_MAP{k}: {v!r} is not one of {sorted(MODS)}")
    for name, value, allowed in (("KEY", c.KEY, c.KEY_NAMES), ("WAVEFORM", c.WAVEFORM, c.WAVEFORMS),
                                 ("UI_MODE", c.UI_MODE, c.UI_MODES), ("CHORD_HAND", c.CHORD_HAND, ("Left", "Right")),
                                 ("HANDEDNESS", c.HANDEDNESS, ("position", "mediapipe")),
                                 ("EXPR_HAND", c.EXPR_HAND, ("Left", "Right"))):
        if value not in allowed:
            errs.append(f"{name} = {value!r} must be one of {list(allowed)}")
    if c.CHORD_HAND == c.EXPR_HAND:
        errs.append("CHORD_HAND and EXPR_HAND must be different hands")
    for name in ("CUTOFF_HZ", "GAIN_RANGE", "REVERB_MIX_RANGE", "VOICING_RANGE"):
        lo, hi = getattr(c, name)
        if not lo < hi:
            errs.append(f"{name} = {(lo, hi)} must be (low, high) with low < high")
    if not c.PINCH_MIN < c.PINCH_MAX:
        errs.append("PINCH_MIN must be < PINCH_MAX")
    if c.WRIST_Y_QUIET == c.WRIST_Y_LOUD or c.WRIST_X_DRY == c.WRIST_X_WET:
        errs.append("WRIST_Y_QUIET/LOUD and WRIST_X_DRY/WET must differ")
    return errs


class Instrument:
    """HandsFrame in → chord + synth params out. One `update` per camera frame."""

    def __init__(self, fingers_fn, feats, key=cfg.KEY):
        self.fingers_fn, self.feats = fingers_fn, feats
        self.gesture = FingerMapSource(cfg.FINGER_MAP, fingers_fn, cfg.HOLD_FRAMES)
        self.hold_mod = Hold(cfg.HOLD_FRAMES)
        self.roll_zone = RollZone(cfg.ROLL_MINOR_DEG, cfg.ROLL_DEADBAND_DEG)
        self.hold_roll = Hold(cfg.HOLD_FRAMES, initial="natural")
        self.ema_gain = EMA(cfg.EMA_ALPHA, cfg.GAIN_START)
        self.ema_bright = EMA(cfg.EMA_ALPHA, cfg.BRIGHTNESS_START)
        self.ema_space = EMA(cfg.EMA_ALPHA, cfg.REVERB_START)
        self.waveform = cfg.WAVEFORM
        self.state = {"key": key, "degree": None, "flipped": False, "mod": "none"}
        self.voicing: list[int] = []  # MIDI notes now sounding (voice-led from the previous chord)
        self.roll = None            # chord-hand roll this frame, None when not in view
        self.chord_fingers = None   # held tuples, for the overlay
        self.expr_fingers = None
        self.expr_missing = 0       # consecutive frames without the expression hand

    def update(self, hands: HandsFrame, key: str | None = None) -> bool:
        """Advance one frame. Returns True when the chord (key/degree/flipped/mod) changed."""
        chord_hand = hands.left if cfg.CHORD_HAND == "Left" else hands.right
        expr_hand = hands.right if cfg.EXPR_HAND == "Right" else hands.left
        new = dict(self.state)
        if key is not None:
            new["key"] = key

        if chord_hand:
            deg = self.gesture.degree(chord_hand)
            if deg != KEEP:
                new["degree"] = deg
            self.chord_fingers = self.gesture.state
            self.roll = self.feats.roll_deg(chord_hand.landmarks)
            new["flipped"] = self.hold_roll.update(self.roll_zone.update(self.roll)) == "flipped"
        else:                       # hand lost mid-chord: chord keeps sounding
            self.roll = self.chord_fingers = None

        if expr_hand:
            self.expr_missing = 0
            self.expr_fingers = self.hold_mod.update(self.fingers_fn(expr_hand.landmarks, expr_hand.label))
            mod = cfg.MOD_MAP.get(self.expr_fingers, KEEP)
            if mod != KEEP:
                new["mod"] = mod
            wx, wy = self.feats.wrist_xy(expr_hand.landmarks)
            xf = wx / hands.aspect if cfg.EXPR_HAND == "Right" else 1.0 - wx / hands.aspect   # 0 = midline side
            self.ema_space.update(clamp01((xf - cfg.WRIST_X_DRY) / (cfg.WRIST_X_WET - cfg.WRIST_X_DRY)))
            self.ema_gain.update(clamp01((wy - cfg.WRIST_Y_QUIET) / (cfg.WRIST_Y_LOUD - cfg.WRIST_Y_QUIET)))
            self.ema_bright.update(clamp01((self.feats.pinch_norm(expr_hand.landmarks) - cfg.PINCH_MIN)
                                           / (cfg.PINCH_MAX - cfg.PINCH_MIN)))
        else:
            # Gone for HOLD_FRAMES frames → mod resets (volume, brightness and space hold). A one-frame
            # tracking dropout changes nothing, so the chord doesn't stutter.
            self.expr_missing += 1
            if self.expr_missing >= cfg.HOLD_FRAMES:
                new["mod"] = "none"
                self.expr_fingers = None
                self.hold_mod = Hold(cfg.HOLD_FRAMES)   # re-entry must re-settle, not replay the old mod

        changed = new != self.state
        self.state = new
        if changed:
            s = new
            self.voicing = [] if s["degree"] is None else chord_voicing(
                s["key"], s["degree"], s["flipped"], s["mod"], cfg.ROOT_OCTAVE, self.voicing or None,
                cfg.VOICE_LEADING, cfg.VOICING_RANGE, cfg.BASS_NOTE)
        return changed

    def freqs(self) -> list[float]:
        return [midi_to_hz(n) for n in self.voicing]

    def chord_name(self) -> str:
        s = self.state
        return "-" if s["degree"] is None else chord_name(s["key"], s["degree"], s["flipped"], s["mod"])

    @property
    def cutoff_hz(self) -> float:
        return logmap(cfg.CUTOFF_HZ, self.ema_bright.value)

    def params(self) -> SynthParams:
        return SynthParams(gain=lerp(cfg.GAIN_RANGE, self.ema_gain.value), cutoff_hz=self.cutoff_hz,
                           waveform=self.waveform, reverb=lerp(cfg.REVERB_MIX_RANGE, self.ema_space.value))

    def cycle_waveform(self):
        self.waveform = cfg.WAVEFORMS[(cfg.WAVEFORMS.index(self.waveform) + 1) % len(cfg.WAVEFORMS)]


def dump_fixture(hands: HandsFrame, fingers_fn, roll_fn) -> list[Path]:
    """Write each visible hand to FIXTURE_DIR/recorded/ as JSON (tests + future ML data).
    The filename tag (left_TFFFF_007.json) is the expected tuple the tests use. It starts as the
    stand-in's reading at record time; if that reading is wrong, rename the file."""
    out_dir = REPO_ROOT / cfg.FIXTURE_DIR / "recorded"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for hand in hands.hands():
        fs = fingers_fn(hand.landmarks, hand.label)
        tag = "".join("T" if f else "F" for f in fs)
        n = 1
        while (path := out_dir / f"{hand.label.lower()}_{tag}_{n:03d}.json").exists():
            n += 1                          # never overwrite (files get renamed after review)
        path.write_text(json.dumps({
            "label": hand.label,
            "landmarks": [list(p) for p in hand.landmarks],
            "score": hand.score,
            "source": "recorded",
            "expected": {"fingers": list(fs), "roll_deg": round(roll_fn(hand.landmarks), 1)},
        }))
        paths.append(path)
    return paths


def fit_width(frame, width: int):
    """Scale a frame to `width` px wide (keeping aspect) so the window fits on a laptop screen
    whatever resolution the camera actually delivers."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def parse_args(argv):
    p = argparse.ArgumentParser(prog="hand-synth", description="Webcam chord instrument.")
    p.add_argument("--camera", default=str(cfg.CAMERA_INDEX),
                   help="camera index, or a video file path (default: config.CAMERA_INDEX)")
    p.add_argument("--key", default=cfg.KEY, choices=cfg.KEY_NAMES, help="startup key")
    p.add_argument("--ui", default=cfg.UI_MODE, choices=cfg.UI_MODES, help="overlay detail (cycle live with h)")
    return p.parse_args(argv)


def open_camera(source: str):
    """Open the camera. On first launch macOS shows a permission prompt and the first open fails
    immediately, so keep retrying for CAMERA_WAIT_S to give the user time to click Allow."""
    deadline = time.monotonic() + (cfg.CAMERA_WAIT_S if source.isdigit() else 0)
    while True:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if source.isdigit():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_SIZE[1])
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            return cap, frame
        cap.release()
        if time.monotonic() >= deadline:
            sys.exit(f"hand-synth: cannot read from camera {source!r} "
                     "(allow camera access for your terminal app in System Settings → Privacy & Security → Camera).")
        print("waiting for camera (allow access if macOS is asking)...")
        time.sleep(1.0)


def main(argv=None):
    from .synth.numpy_synth import NumpySynth
    from .tracking import Tracker
    from .ui import KeyBar, OverlayState, draw_overlay

    args = parse_args(sys.argv[1:] if argv is None else argv)
    if errs := check_config():
        sys.exit("hand-synth: fix hand_synth/config.py:\n  - " + "\n  - ".join(errs))

    fingers_fn, fingers_status = standins.resolve_fingers()
    feats, feature_status = standins.resolve_features()
    badges = []
    if fingers_status != "manual":
        badges.append("Stand-in · fingers")
    if any(v != "manual" for v in feature_status.values()):
        badges.append("Stand-in · features")
    for name, status in [("fingers_up", fingers_status), *feature_status.items()]:
        print(f"  {name:15s} {status}")

    cap, frame = open_camera(args.camera)
    print(f"  camera          {frame.shape[1]}x{frame.shape[0]} -> window width {cfg.CAMERA_SIZE[0]}")
    synth = NumpySynth(cfg)
    try:
        synth.start()
    except RuntimeError as e:
        cap.release()
        sys.exit(f"hand-synth: {e}")
    tracker = Tracker(cfg.MODEL_PATH, cfg.MODEL_URL, cfg.MAX_HANDS, cfg.MIN_DETECTION_CONF, cfg.MIN_TRACKING_CONF,
                      cfg.HANDEDNESS)

    inst = Instrument(fingers_fn, feats, key=args.key)
    pending = {"key": None}           # set by the mouse callback (runs inside waitKey), applied next frame
    keybar = KeyBar(cfg.KEY_NAMES, on_select=lambda k: pending.update(key=k), height=cfg.KEYBAR_HEIGHT)
    cv2.namedWindow(cfg.WINDOW_NAME)
    cv2.setMouseCallback(cfg.WINDOW_NAME, keybar.mouse)

    ui_mode = args.ui
    flash, flash_until = None, 0.0
    last_hands_t = -1e9               # when a hand was last in view (drives the empty-state hint)
    fps, last_t = 0.0, time.monotonic()
    try:
        while True:
            if frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
            frame = fit_width(frame, cfg.CAMERA_SIZE[0])
            if cfg.MIRROR:
                frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)

            key, pending["key"] = pending["key"], None
            if inst.update(hands, key):     # key change re-issues the same degree in the new key
                synth.set_chord(inst.freqs())
            synth.set_params(inst.params())

            now = time.monotonic()
            if hands.hands():
                last_hands_t = now
            fps = 0.9 * fps + 0.1 / max(1e-6, now - last_t)
            last_t = now
            s = inst.state
            view = OverlayState(
                key=s["key"], chord=inst.chord_name(), degree=s["degree"], flipped=s["flipped"], mod=s["mod"],
                chord_fingers=inst.chord_fingers, expr_fingers=inst.expr_fingers,
                roll=inst.roll, zone=inst.hold_roll.value, gain=inst.ema_gain.value,
                brightness=inst.ema_bright.value, cutoff_hz=inst.cutoff_hz, waveform=inst.waveform,
                space=inst.ema_space.value, hint=now - last_hands_t > cfg.HINT_AFTER_S,
                chord_hand=cfg.CHORD_HAND, expr_hand=cfg.EXPR_HAND,
                fps=fps, xrun=synth.recent_xrun(), audio_error=synth.error, badges=badges,
                flash=flash if now < flash_until else None,
            )
            draw_overlay(frame, hands, view, cfg.KEYBAR_HEIGHT, cfg.HUD_HEIGHT, ui_mode)
            keybar.visible = ui_mode != "clean"
            if keybar.visible:
                keybar.draw(frame, s["key"])
            cv2.imshow(cfg.WINDOW_NAME, frame)
            frame = None

            k = cv2.waitKeyEx(1)
            arrow = keycodes.arrow(k)
            if k in (ord("q"), 27) or cv2.getWindowProperty(cfg.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif k == ord("r"):
                paths = dump_fixture(hands, standins.fingers_up, standins.roll_deg)
                flash = f"saved {len(paths)} fixture(s)" if paths else "no hands to record"
                flash_until = now + cfg.FLASH_S
                for p in paths:
                    print(f"saved {p.relative_to(REPO_ROOT)}")
            elif k == ord("w"):
                inst.cycle_waveform()
            elif k == ord("h"):
                ui_mode = cfg.UI_MODES[(cfg.UI_MODES.index(ui_mode) + 1) % len(cfg.UI_MODES)]
                flash, flash_until = f"view: {ui_mode}", now + cfg.FLASH_S
            elif arrow in ("left", "right"):
                step = 1 if arrow == "right" else -1
                pending["key"] = cfg.KEY_NAMES[(cfg.KEY_NAMES.index(s["key"]) + step) % len(cfg.KEY_NAMES)]
    except KeyboardInterrupt:
        pass
    finally:
        synth.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
