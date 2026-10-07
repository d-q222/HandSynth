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
HANDEDNESS = "position"         # "position": screen-left hand = Left (robust) | "mediapipe": trust the classifier

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
    (True, True, False, False, False):   "7",      # "L" shape: flat 7th — dominant 7 on major chords (D7 in D)
    # anything else: keep previous modification. EXPECTED TO CHANGE after first play session.
}
ROOT_OCTAVE = 3
VOICING_STYLE = "balanced"      # "balanced": root alone in the bass, tones spaced like the overtone series,
                                #   no m2/m9 clashes, low-interval limits respected | "close": tight voice-led cluster
VOICE_LEADING = True            # prefer voicings that move as few semitones as possible from the last chord
UPPER_RANGE = (48, 76)          # balanced: MIDI window for the notes above the bass (C3..E5)
REGISTER_CENTER = 58            # balanced: where the upper voices like to sit (A#3), under a singing voice
VOICING_RANGE = (45, 69)        # close: MIDI window the voiced chord tones stay in (A2..A4)
BASS_NOTE = True                # close: add the chord root an octave below (balanced always has it)

# ── Stabilization ────────────────────────────────────────────────────────────
ROLL_MINOR_DEG = 35.0           # chord hand rolled more than this from upright → flip quality
ROLL_DEADBAND_DEG = 10.0        # enter flipped above 35+10, return to natural below 35-10
HOLD_FRAMES = 5                 # discrete hysteresis
EMA_ALPHA = 0.3                 # continuous smoothing

# ── Expression controls ──────────────────────────────────────────────────────
PINCH_MIN, PINCH_MAX = 0.15, 0.9    # normalized pinch range → 0..1
CUTOFF_HZ = (300.0, 6000.0)         # brightness range, log-mapped
VOLUME_MAX_GAIN = 1.0               # linear gain at the top of the volume range
VOLUME_RANGE_DB = 36.0              # volume is a dB fader (equal hand moves = equal loudness steps):
                                    #   top = 0 dB, bottom = -36 dB, fading to silence in the last few %
WRIST_Y_QUIET, WRIST_Y_LOUD = 0.9, 0.3     # expression-hand wrist y (0 top..1 bottom) at the bottom / top of the volume range
GAIN_START = 0.75                   # 0..1 volume position before the expression hand appears (≈ -9 dB)
REVERB_MIX_RANGE = (0.0, 1.0)       # wet mix, mapped from expression-hand wrist x (inward = dry, outward = wash)
WRIST_X_DRY, WRIST_X_WET = 0.55, 0.92   # expression-hand wrist x, fraction of frame width from the chord hand's side
REVERB_START = 0.35                 # 0..1 position in REVERB_MIX_RANGE before the expression hand appears
BRIGHTNESS_START = 0.5              # 0..1 position in CUTOFF_HZ before the expression hand appears

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
BLOCK_SIZE = 256                # raise to 512 if the overlay shows XRUN persistently
WAVEFORM = "triangle"           # "sine" | "triangle" | "saw"   (cycle live with 'w')
WAVEFORMS = ["sine", "triangle", "saw"]
ATTACK_S, RELEASE_S = 0.02, 0.08
DETUNE_CENTS = 7.0              # unison: centre oscillator + a pair detuned ±this many cents (0 = off)
UNISON_SPREAD = 0.5             # stereo width of the detuned pair: 0 = mono, 1 = hard left/right
REVERB_ROOM = 0.92              # reverb tail length: comb feedback, 0..0.97 (0.92 ≈ 2.5 s tail)
REVERB_DAMP = 0.3               # high-frequency damping inside the tail, 0..1 (lower = brighter, airier)
REVERB_PREDELAY_MS = 25.0       # gap before the reverb starts, so it separates from the dry pad
GAIN_RAMP_S = 0.02              # master-gain smoothing inside the audio thread
CUTOFF_RAMP_S = 0.03            # cutoff smoothing inside the audio thread
STOP_FADE_S = 0.15              # fade-out on quit before the stream closes
OUTPUT_MAKEUP_DB = 4.0          # output level boost; a soft limiter (rarely engaged) keeps peaks below full scale
AUDIO_DEVICE = None             # None = system default output; or an index/name from sounddevice.query_devices()

# ── Camera / tracking ────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_SIZE = (1280, 720)       # requested capture size; frames are scaled to this width for display
CAMERA_WAIT_S = 30.0            # keep retrying the camera this long (first-run permission prompt)
MAX_HANDS = 2
MIN_DETECTION_CONF = 0.6
MIN_TRACKING_CONF = 0.5
MODEL_PATH = "models/hand_landmarker.task"     # relative to the repo root
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
FIXTURE_DIR = "fixtures/landmarks"             # where 'r' dumps landmark JSON

# ── UI ───────────────────────────────────────────────────────────────────────
WINDOW_NAME = "Hand Synth"
KEYBAR_HEIGHT = 44              # px, 12 key buttons across the top of the frame
UI_MODE = "full"                # "full" | "minimal" | "clean"   (cycle live with 'h')
UI_MODES = ["full", "minimal", "clean"]
HUD_HEIGHT = 112                # px, translucent info strip along the bottom of the frame
HINT_AFTER_S = 1.5              # show "raise your hands" after this long with no hands in view
FLASH_S = 1.5                   # how long transient messages ("saved fixture") stay on screen
