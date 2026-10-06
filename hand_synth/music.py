"""Chord theory: key + degree + flip + mod -> MIDI, Hz, name. Pure functions."""
from itertools import product

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


def _placements(pc, low, high):
    return [n for n in range(low, high + 1) if n % 12 == pc]


def voice_lead(notes, prev, low, high):
    """Re-voice `notes` to move as little as possible from `prev`, keeping each pitch
    class once, within [low, high]. Returns sorted MIDI notes.

    Cost = sum of each new note's distance to the nearest prev note plus each prev
    note's distance to the nearest new note; ties go to the more compact, then lower
    voicing. With no prev: the chord as given, shifted by octaves into range (root
    stays lowest). A range narrower than an octave is widened upward to 12 semitones
    so every pitch class fits; a chord that cannot be shifted into range falls back
    to its most compact in-range placement.
    """
    notes = sorted(notes)
    if not notes:
        return []
    if high - low < 11:
        high = low + 11
    if not prev:
        shift = 0
        while notes[0] + shift < low:
            shift += 12
        while notes[-1] + shift > high and notes[0] + shift - 12 >= low:
            shift -= 12
        if notes[0] + shift >= low and notes[-1] + shift <= high:
            return [n + shift for n in notes]
    pcs = [n % 12 for n in notes]
    options = [_placements(pc, low, high) for pc in pcs]
    prev = list(prev) if prev else []
    best_key, best = None, None
    for combo in product(*options):
        voicing = sorted(combo)
        if prev:
            cost = (sum(min(abs(n - p) for p in prev) for n in voicing)
                    + sum(min(abs(n - p) for n in voicing) for p in prev))
        else:
            cost = 0
        key = (cost, voicing[-1] - voicing[0], sum(voicing))
        if best_key is None or key < best_key:
            best_key, best = key, voicing
    return best


def with_bass(notes, root_midi):
    """Add the chord root an octave below `root_midi`, sorted, no duplicates."""
    return sorted(set(notes) | {root_midi - 12})


def chord_voicing(key, degree, flipped, mod, octave, prev, voice_leading, voicing_range, bass):
    """chord_midi -> optional voice_lead within voicing_range -> optional bass.

    `prev` is the previous chord_voicing result (bass included, lowest note stripped
    when bass=True). The bass is root-12, dropped further by octaves if needed to sit
    below the voiced notes. Returns sorted MIDI notes.
    """
    notes = chord_midi(key, degree, flipped, mod, octave)
    root = notes[0]
    if voice_leading:
        upper_prev = sorted(prev)[1:] if (prev and bass) else prev
        notes = voice_lead(notes, upper_prev, voicing_range[0], voicing_range[1])
    if not bass:
        return sorted(notes)
    b = root - 12
    while b >= min(notes):
        b -= 12
    return sorted(set(notes) | {b})
