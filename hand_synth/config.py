"""ALL tunables. Data only. No logic, no imports.

Redesigning the instrument = editing this file. FINGER_MAP and MOD_MAP are
provisional and expected to change after the first play session.
"""

# ── Key ──────────────────────────────────────────────────────────────────────
KEY = "C"                       # runtime-changeable via UI; this is the startup default
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ── Hands ────────────────────────────────────────────────────────────────────
CHORD_HAND = "Left"             # which handedness label plays chords
EXPR_HAND = "Right"
MIRROR = True                   # flip frame so it behaves like a mirror (and labels match real hands)

# finger tuple order: (thumb, index, middle, ring, pinky). True = extended.
FINGER_MAP = {                  # CHORD HAND → scale degree (None = silence)
    (True, True, True, True, True):      1,     # open hand = tonic
    (False, False, False, False, False): None,  # fist = silence
    (False, True, False, False, False):  2,
    (False, True, True, False, False):   3,
    (False, True, True, True, False):    4,
    (False, True, True, True, True):     5,
    (True, False, False, False, False):  6,
    (True, True, False, False, False):   7,
    # anything else: keep previous chord
}
MOD_MAP = {                     # EXPRESSION HAND → modification
    (True, True, True, True, True):      "none",   # open / relaxed = plain chord
    (False, True, False, False, False):  "sus2",
    (False, True, True, False, False):   "sus4",
    (False, True, True, True, False):    "add7",
    (False, True, True, True, True):     "add9",
    (True, False, False, False, False):  "add6",
    # anything else: keep previous modification. EXPECTED TO CHANGE after first play session.
}
ROOT_OCTAVE = 3

# ── Stabilization ────────────────────────────────────────────────────────────
ROLL_MINOR_DEG = 35.0           # chord hand rolled more than this from upright → flip quality
ROLL_DEADBAND_DEG = 10.0        # enter flipped above 35+10, return to natural below 35-10
HOLD_FRAMES = 5                 # discrete hysteresis
EMA_ALPHA = 0.3                 # continuous smoothing

# ── Expression controls ──────────────────────────────────────────────────────
PINCH_MIN, PINCH_MAX = 0.15, 0.9    # normalized pinch range → 0..1
CUTOFF_HZ = (300.0, 6000.0)         # brightness range, log-mapped
GAIN_RANGE = (0.0, 0.8)             # wrist-y (top of frame = loud)
GAIN_START = 0.5                    # 0..1 position in GAIN_RANGE before the expression hand appears
BRIGHTNESS_START = 0.5              # 0..1 position in CUTOFF_HZ before the expression hand appears

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
BLOCK_SIZE = 256                # raise to 512 if the overlay shows XRUN persistently
WAVEFORM = "triangle"           # "sine" | "triangle" | "saw"   (cycle live with 'w')
WAVEFORMS = ["sine", "triangle", "saw"]
ATTACK_S, RELEASE_S = 0.02, 0.08
GAIN_RAMP_S = 0.02              # master-gain smoothing inside the audio thread
CUTOFF_RAMP_S = 0.03            # cutoff smoothing inside the audio thread
STOP_FADE_S = 0.15              # fade-out on quit before the stream closes
AUDIO_DEVICE = None             # None = system default output; or an index/name from sounddevice.query_devices()

# ── Camera / tracking ────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_SIZE = (1280, 720)       # requested capture size (camera may pick the nearest)
MAX_HANDS = 2
MIN_DETECTION_CONF = 0.6
MIN_TRACKING_CONF = 0.5
MODEL_PATH = "models/hand_landmarker.task"     # relative to the repo root
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
FIXTURE_DIR = "fixtures/landmarks"             # where 'r' dumps landmark JSON

# ── UI ───────────────────────────────────────────────────────────────────────
WINDOW_NAME = "hand synth"
KEYBAR_HEIGHT = 48              # px, 12 key buttons across the top of the frame
PANEL_WIDTH = 300               # px, info panel on the right side of the frame
