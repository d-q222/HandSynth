"""Chord theory: key + degree + flip + mod -> MIDI, Hz, name. Pure functions."""
from . import config

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
DIATONIC_QUALITY = ["M", "m", "m", "M", "M", "m", "dim"]
TRIAD = {"M": [0, 4, 7], "m": [0, 3, 7], "dim": [0, 3, 6]}
FLIP = {"M": "m", "m": "M", "dim": "m"}                  # chord-hand rotation
SEVENTH = {1: 11, 2: 10, 3: 10, 4: 11, 5: 10, 6: 10, 7: 10}
MODS = {
    "none": lambda iv, deg: iv,
    "sus2": lambda iv, deg: [0, 2, 7],                    # replaces the third
    "sus4": lambda iv, deg: [0, 5, 7],
    "add7": lambda iv, deg: iv + [SEVENTH[deg]],
    "add9": lambda iv, deg: iv + [14],
    "add6": lambda iv, deg: iv + [9],
}
_TRIAD_SUFFIX = {"M": "", "m": "m", "dim": "dim"}
_ADD7_SUFFIX = {("M", 11): "maj7", ("M", 10): "7", ("m", 10): "m7",
                ("m", 11): "mMaj7", ("dim", 10): "m7b5", ("dim", 11): "dimMaj7"}
_ADD6_SUFFIX = {"M": "6", "m": "m6", "dim": "dim7"}


def key_to_semitone(key):
    try:
        return config.KEY_NAMES.index(key)
    except ValueError:
        raise ValueError(f"bad key: {key!r}") from None


def scale(key):
    root = key_to_semitone(key)
    return [(root + s) % 12 for s in MAJOR_SCALE]


def _check(degree, mod):
    if degree not in range(1, 8):
        raise ValueError(f"bad degree: {degree!r}")
    if mod not in MODS:
        raise ValueError(f"bad mod: {mod!r}")


def quality(degree, flipped):
    if degree not in range(1, 8):
        raise ValueError(f"bad degree: {degree!r}")
    q = DIATONIC_QUALITY[degree - 1]
    return FLIP[q] if flipped else q


def chord_midi(key, degree, flipped=False, mod="none", octave=3):
    _check(degree, mod)
    root = 12 * (octave + 1) + scale(key)[degree - 1]
    iv = MODS[mod](list(TRIAD[quality(degree, flipped)]), degree)
    return [root + i for i in iv]


def midi_to_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def chord_freqs(key, degree, flipped=False, mod="none", octave=3):
    return [midi_to_hz(n) for n in chord_midi(key, degree, flipped, mod, octave)]


def chord_name(key, degree, flipped=False, mod="none"):
    _check(degree, mod)
    q = quality(degree, flipped)
    if mod == "none":
        suffix = _TRIAD_SUFFIX[q]
    elif mod in ("sus2", "sus4"):
        suffix = mod
    elif mod == "add7":
        suffix = _ADD7_SUFFIX[(q, SEVENTH[degree])]
    elif mod == "add9":
        suffix = _TRIAD_SUFFIX[q] + "add9"
    else:
        suffix = _ADD6_SUFFIX[q]
    return config.KEY_NAMES[scale(key)[degree - 1]] + suffix
