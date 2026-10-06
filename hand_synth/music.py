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


# ---- balanced voicing ----

# Low Interval Limits: interval in semitones -> lowest acceptable MIDI note for the
# lower note of that interval (Berklee-style). Octaves and anything wider than a
# major tenth have no limit.
_LIL = {1: 52, 2: 51, 3: 48, 4: 46, 5: 46, 6: 46, 7: 34, 8: 41, 9: 41, 10: 41,
        11: 41, 13: 40, 14: 39, 15: 36, 16: 34}

# Soft-score weights (lower score = better). Hard-rule breaches cost _HARD each so
# a chord with no legal voicing still gets its least-bad one instead of crashing.
_W_SHAPE = 0.5      # per semitone an interval is wider than the one below it
_W_STEP = 3.0       # per exposed whole step between adjacent upper voices
_W_REGISTER = 0.3   # per semitone the upper-voice mean sits from register_center
_W_MOVE = 1.0       # per semitone of voice-leading movement (nearest-note, both ways)
_W_OMIT5 = 1.5      # dropping the perfect fifth of a 4-note chord
_W_FAR_LIL = 2.0    # non-adjacent pair, within a tenth, below its low interval limit
_W_BASS_GAP = 0.8   # per semitone the bass->lowest-upper gap is short of an octave
_W_SPAN = 0.05      # per semitone of upper-voice span (space belongs below, not on top)
_HARD = 1000.0


def _upper_sets(intervals):
    """Candidate pitch-class sets for 3-4 upper voices: every non-root pc, the perfect
    fifth optional in 4-note chords, the root optionally doubled (always, in triads).
    -> [(pcs, omitted5)]"""
    pcs = sorted({i % 12 for i in intervals} - {0})
    out = []
    for drop5 in ((False, True) if (7 in pcs and len(pcs) >= 3) else (False,)):
        core = [p for p in pcs if not (drop5 and p == 7)]
        for s in (core, core + [0]):
            if 3 <= len(s) <= 4:
                out.append((s, drop5))
    return out


def _voicing_score(upper, bass, ninth_over, prev, center, added=frozenset()):
    """(hard breaches, soft score) for sorted `upper` voices over `bass`. `added` = pitch classes
    of added notes (6th/7th/9th), which must sit above every root in the upper voices."""
    notes = [bass] + upper
    hard, soft = 0, 0.0
    n = len(notes)
    for i in range(n):
        lo = notes[i]
        for j in range(i + 1, n):
            d = notes[j] - lo
            if d % 12 == 1:
                hard += 1                                   # m2 / m9 anywhere
            elif lo < _LIL.get(d, 0):
                if j == i + 1:
                    hard += 1                               # adjacent voices: LIL is hard
                else:
                    soft += _W_FAR_LIL
    gaps = [b - a for a, b in zip(notes, notes[1:])]
    if not 7 <= gaps[0] <= 24:
        hard += 1
    soft += _W_BASS_GAP * max(0, 12 - gaps[0])              # overtone series: octave first
    for g in gaps[1:]:
        if g > 12:
            hard += 1
        if g == 2:
            soft += _W_STEP
    for a, b in zip(gaps, gaps[1:]):
        if b > a:
            soft += _W_SHAPE * (b - a)                      # wide low, narrow high
    if ninth_over is not None:
        ninth = [u for u in upper if u % 12 == ninth_over[0]]
        thirds = [u for u in upper if u % 12 in ninth_over[1]]
        if ninth and thirds and ninth[0] < max(thirds):
            hard += 1
    if added:
        roots_up = [u for u in upper if u % 12 == bass % 12]
        if roots_up and any(u % 12 in added and u < roots_up[-1] for u in upper):
            hard += 1                                       # added notes go above the root
    soft += _W_REGISTER * abs(sum(upper) / len(upper) - center)
    soft += _W_SPAN * (upper[-1] - upper[0])
    if prev:
        soft += _W_MOVE * (sum(min(abs(u - p) for p in prev) for u in upper)
                           + sum(min(abs(u - p) for u in upper) for p in prev))
    return hard, soft


def balanced_voicing(intervals, root_midi, prev_upper, *, bass_midi, upper_range=(48, 76),
                     register_center=58, voice_leading=True):
    """Root alone in the bass (bass_midi), chord tones above it, balanced. Returns
    sorted MIDI, bass first.

    Brute force over every placement of each candidate upper pitch-class set within
    upper_range. Hard rules: no minor 2nd/9th between any two voices, low interval
    limits on adjacent voices, 9th above every 3rd, added notes (6/7/9) above every
    upper-voice root, adjacent upper voices within an
    octave, bass to lowest upper voice a 5th..two octaves. Soft score: wide-low /
    narrow-high spacing, exposed whole steps, distance from register_center, movement
    from prev_upper (when voice_leading), omitted 5th.
    """
    root_pc = root_midi % 12
    lo, hi = upper_range
    if hi - lo < 11:
        hi = lo + 11
    pcs_rel = {i % 12 for i in intervals}
    thirds = {p for p in (3, 4) if p in pcs_rel}
    ninth_over = (((2 + root_pc) % 12), {(t + root_pc) % 12 for t in thirds}) \
        if (2 in pcs_rel and thirds) else None
    prev = list(prev_upper) if (voice_leading and prev_upper) else None
    added = frozenset((i + root_pc) % 12 for i in intervals if i >= 9)   # 6th, 7th, 9th (from mods)
    best_key, best = None, None
    for rel, omitted5 in _upper_sets(intervals):
        options = [_placements((p + root_pc) % 12, lo, hi) for p in rel]
        base = _W_OMIT5 if omitted5 else 0.0
        for combo in product(*options):
            upper = sorted(combo)
            hard, soft = _voicing_score(upper, bass_midi, ninth_over, prev, register_center, added)
            key = (hard * _HARD + soft + base, upper)
            if best_key is None or key < best_key:
                best_key, best = key, upper
    return [bass_midi] + best


def chord_voicing(key, degree, flipped, mod, prev, *, style="balanced", octave=3,
                  voice_leading=True, voicing_range=(45, 69), bass=True,
                  upper_range=(48, 76), register_center=58):
    """Voice a chord. `prev` is the previous chord_voicing result (or None).

    style "balanced": the root alone in the bass at MIDI 12*octave + root pc (C2..B2
    for octave 3), upper voices from balanced_voicing (prev's bass stripped). With
    bass=False only the upper voices are returned.
    style "close": chord_midi -> optional voice_lead within voicing_range -> optional
    bass (root-12, dropped further by octaves to sit below the voiced notes).
    Returns sorted MIDI notes.
    """
    notes = chord_midi(key, degree, flipped, mod, octave)
    root = notes[0]
    upper_prev = sorted(prev)[1:] if (prev and bass) else prev
    if style == "balanced":
        out = balanced_voicing([n - root for n in notes], root, upper_prev, bass_midi=root - 12,
                               upper_range=upper_range, register_center=register_center,
                               voice_leading=voice_leading)
        return out if bass else out[1:]
    if style != "close":
        raise ValueError(f"bad style: {style!r}")
    if voice_leading:
        notes = voice_lead(notes, upper_prev, voicing_range[0], voicing_range[1])
    if not bass:
        return sorted(notes)
    b = root - 12
    while b >= min(notes):
        b -= 12
    return sorted(set(notes) | {b})
