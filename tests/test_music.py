import math
import random
import statistics
import time

import pytest

from hand_synth import config, music


def test_examples():
    assert music.chord_midi("C", 1) == [48, 52, 55]
    assert music.chord_midi("C", 1, flipped=True) == [48, 51, 55]
    assert music.chord_name("G", 2, False, "add7") == "Am7"
    assert music.chord_name("C", 1, False, "sus4") == "Csus4"
    assert music.chord_midi("B", 1)[0] == 59


@pytest.mark.parametrize("key", config.KEY_NAMES)
def test_roots_all_keys(key):
    k = config.KEY_NAMES.index(key)
    for d in range(1, 8):
        root = music.chord_midi(key, d)[0]
        assert (root - 12 * 4) % 12 == (k + music.MAJOR_SCALE[d - 1]) % 12
        assert 12 * 4 <= root < 12 * 5 + 12
    assert music.scale("G") == [7, 9, 11, 0, 2, 4, 6]


@pytest.mark.parametrize("deg,expect", [
    (1, [0, 4, 7]), (2, [0, 3, 7]), (3, [0, 3, 7]), (4, [0, 4, 7]),
    (5, [0, 4, 7]), (6, [0, 3, 7]), (7, [0, 3, 6])])
def test_diatonic_triads(deg, expect):
    m = music.chord_midi("C", deg)
    assert [n - m[0] for n in m] == expect


@pytest.mark.parametrize("deg,expect", [
    (1, [0, 3, 7]), (2, [0, 4, 7]), (3, [0, 4, 7]), (4, [0, 3, 7]),
    (5, [0, 3, 7]), (6, [0, 4, 7]), (7, [0, 3, 7])])
def test_flip_every_degree(deg, expect):
    m = music.chord_midi("C", deg, flipped=True)
    assert [n - m[0] for n in m] == expect


@pytest.mark.parametrize("deg,mod,iv,name", [
    (1, "none", [0, 4, 7], "C"),
    (1, "sus2", [0, 2, 7], "Csus2"),
    (1, "sus4", [0, 5, 7], "Csus4"),
    (1, "add7", [0, 4, 7, 11], "Cmaj7"),
    (1, "add9", [0, 4, 7, 14], "Cadd9"),
    (1, "add6", [0, 4, 7, 9], "C6"),
    (5, "none", [0, 4, 7], "G"),
    (5, "sus2", [0, 2, 7], "Gsus2"),
    (5, "sus4", [0, 5, 7], "Gsus4"),
    (5, "add7", [0, 4, 7, 10], "G7"),
    (5, "add9", [0, 4, 7, 14], "Gadd9"),
    (5, "add6", [0, 4, 7, 9], "G6"),
])
def test_mods_degrees_1_and_5(deg, mod, iv, name):
    m = music.chord_midi("C", deg, mod=mod)
    assert [n - m[0] for n in m] == iv
    assert music.chord_name("C", deg, False, mod) == name


def test_names():
    n = music.chord_name
    assert n("C", 2) == "Dm"
    assert n("C", 7) == "Bdim"
    assert n("C", 1, True) == "Cm"
    assert n("C", 1, True, "add7") == "CmMaj7"
    assert n("C", 2, False, "add7") == "Dm7"
    assert n("C", 7, False, "add7") == "Bm7b5"
    assert n("C", 3, False, "add9") == "Emadd9"
    assert n("C", 7, False, "add9") == "Bdimadd9"
    assert n("C", 2, False, "add6") == "Dm6"
    assert n("C", 7, False, "add6") == "Bdim7"
    assert n("C", 2, True, "add7") == "D7"
    assert n("C", 2, True, "sus4") == "Dsus4"
    assert n("D", 4) == "G"
    assert n("A", 1) == "A"
    assert n("F#", 4) == "B"
    assert n("A#", 1) == "A#"


def test_freqs():
    assert music.midi_to_hz(69) == pytest.approx(440.0)
    assert music.midi_to_hz(57) == pytest.approx(220.0)
    f = music.chord_freqs("C", 1)
    assert f == pytest.approx([130.8128, 164.8138, 195.9977], abs=1e-3)
    assert music.chord_freqs("C", 1, octave=4)[0] == pytest.approx(261.6256, abs=1e-3)


def test_errors():
    with pytest.raises(ValueError):
        music.chord_midi("H", 1)
    with pytest.raises(ValueError):
        music.chord_midi("C", 0)
    with pytest.raises(ValueError):
        music.chord_midi("C", 8)
    with pytest.raises(ValueError):
        music.chord_midi("C", 1, mod="add13")
    with pytest.raises(ValueError):
        music.chord_name("C", 1, False, "nope")


# ---- voice leading ----

RANGE = (45, 69)


def _close(key, d, flipped, mod, prev, voice_leading=True, bass=False):
    return music.chord_voicing(key, d, flipped, mod, prev, style="close", octave=3,
                               voice_leading=voice_leading, voicing_range=RANGE, bass=bass)


def _walk(key, degs, **kw):
    prev, out = None, []
    for d in degs:
        prev = _close(key, d, False, "none", prev, bass=kw.get("bass", False))
        out.append(prev)
    return out


def test_cold_start_close_position():
    assert _close("C", 1, False, "none", None) == [48, 52, 55]
    assert music.voice_lead([59, 63, 66], None, 45, 69) == [59, 63, 66]
    assert music.voice_lead([48, 52, 55], [], 60, 72) == [60, 64, 67]


def test_c_f_g_c_smooth_and_in_range():
    v = _walk("C", [1, 4, 5, 1])
    for a, b in zip(v, v[1:]):
        assert all(min(abs(n - p) for p in a) <= 5 for n in b)
        assert all(min(abs(n - p) for n in b) <= 5 for p in a)
    for chord in v:
        assert all(RANGE[0] <= n <= RANGE[1] for n in chord)
    assert 60 in v[1] or 48 in v[1]          # C kept in F


def test_common_tones_kept():
    c = _close("C", 1, False, "none", None)
    f = _close("C", 4, False, "none", c)
    am = _close("C", 6, False, "none", c)
    assert set(c) & set(f) == {48}
    assert set(c) & set(am) == {48, 52}


def test_pitch_classes_preserved_all():
    rng = random.Random(1)
    for key in config.KEY_NAMES:
        for d in range(1, 8):
            for mod in music.MODS:
                base = music.chord_midi(key, d, False, mod, 3)
                prev = sorted(rng.sample(range(45, 70), rng.randint(3, 5)))
                out = music.voice_lead(base, prev, *RANGE)
                assert sorted(n % 12 for n in out) == sorted(n % 12 for n in base)
                assert out == sorted(out) and all(RANGE[0] <= n <= RANGE[1] for n in out)
                assert out == music.voice_lead(base, prev, *RANGE)
                cold = music.voice_lead(base, None, *RANGE)
                assert sorted(n % 12 for n in cold) == sorted(n % 12 for n in base)
                assert all(RANGE[0] <= n <= RANGE[1] for n in cold)


def test_passthrough_when_off():
    for key in ("C", "F#", "B"):
        for d in range(1, 8):
            for mod in music.MODS:
                assert (_close(key, d, True, mod, [50, 55], voice_leading=False)
                        == music.chord_midi(key, d, True, mod, 3))


def test_bass():
    assert music.with_bass([48, 52, 55], 48) == [36, 48, 52, 55]
    for key in config.KEY_NAMES:
        for d in range(1, 8):
            for vl in (False, True):
                out = _close(key, d, False, "add9", None, voice_leading=vl, bass=True)
                root = music.chord_midi(key, d, False, "add9", 3)[0]
                assert (root - out[0]) % 12 == 0 and out[0] <= root - 12
                assert out[0] < out[1]
    # prev with bass is stripped before voice leading
    a = _close("C", 1, False, "none", None, bass=True)
    b = _close("C", 4, False, "none", a, bass=True)
    assert b[0] == 41 and b[1:] == _close("C", 4, False, "none", a[1:])


def test_narrow_range_widened_not_crash():
    out = music.voice_lead([48, 52, 55, 62], [50], 50, 55)
    assert sorted(n % 12 for n in out) == [0, 2, 4, 7]


# ---- balanced voicing ----

# Low Interval Limits (independent copy of the research table): interval -> lowest
# acceptable MIDI note for the lower note. Octave and > major 10th: no limit.
LIL = {1: 52, 2: 51, 3: 48, 4: 46, 5: 46, 6: 46, 7: 34, 8: 41, 9: 41, 10: 41, 11: 41,
       13: 40, 14: 39, 15: 36, 16: 34}
UPPER = (48, 76)
ALL_CHORDS = [(k, d, f, m) for k in config.KEY_NAMES for d in range(1, 8)
              for f in (False, True) for m in music.MODS]


def _bal(key, d, f, m, prev=None, **kw):
    return music.chord_voicing(key, d, f, m, prev, style="balanced", **kw)


def _check_balanced(out, key, d, f, m):
    base = music.chord_midi(key, d, f, m, 3)
    root, rel = base[0], [n - base[0] for n in base]
    pcs = {n % 12 for n in base}
    upper = out[1:]
    assert out == sorted(out) and len(set(out)) == len(out)
    assert out[0] == root - 12                                  # root alone in the bass
    assert out[0] < UPPER[0] and all(UPPER[0] <= n <= UPPER[1] for n in upper)
    assert 3 <= len(upper) <= 4
    assert {n % 12 for n in out} <= pcs
    optional = {(root + 7) % 12} if (7 in rel and len(rel) >= 4) else set()
    assert pcs - optional <= {n % 12 for n in out}
    for i, lo in enumerate(out):
        for hi in out[i + 1:]:
            assert (hi - lo) % 12 != 1, (out, lo, hi)          # no m2 / m9 anywhere
    for lo, hi in zip(out, out[1:]):
        assert lo >= LIL.get(hi - lo, 0), (out, lo, hi)        # low interval limits
    if 14 in rel and ({3, 4} & set(rel)):
        ninth = [n for n in upper if n % 12 == (root + 2) % 12]
        thirds = [n for n in upper if (n - root) % 12 in (3, 4)]
        assert min(ninth) > max(thirds)


def test_balanced_rules_all_chords_cold():
    for key, d, f, m in ALL_CHORDS:
        out = _bal(key, d, f, m)
        _check_balanced(out, key, d, f, m)
        assert out == _bal(key, d, f, m)                        # deterministic
        if m == "none":                                          # triads keep R, 3, 5
            assert {n % 12 for n in out} == {n % 12 for n in music.chord_midi(key, d, f, m)}


def test_balanced_rules_hold_while_voice_leading():
    rng = random.Random(7)
    for key in config.KEY_NAMES:
        prev = None
        for _ in range(60):
            d, f, m = rng.randint(1, 7), rng.random() < 0.3, rng.choice(list(music.MODS))
            prev = _bal(key, d, f, m, prev)
            _check_balanced(prev, key, d, f, m)


def test_balanced_examples():
    assert _bal("C", 1, False, "none") == [36, 55, 60, 64]      # C2 G3 C4 E4
    assert _bal("C", 1, False, "add7") == [36, 55, 59, 64]      # C2 G3 B3 E4: no C over the B
    assert _bal("C", 1, False, "add9") == [36, 52, 62, 67]      # C2 E3 D4 G4: 9th over the 3rd
    assert _bal("C", 1, False, "none", bass=False) == _bal("C", 1, False, "none")[1:]


def _nearest_moves(a, b):
    return ([min(abs(n - p) for p in a) for n in b] + [min(abs(n - p) for n in b) for p in a])


@pytest.mark.parametrize("key", ["C", "G"])
def test_balanced_progression_smooth_and_holds_common_tones(key):
    degs = [1, 4, 5, 6, 2, 5, 1]
    prev, seq = None, []
    for d in degs:
        prev = _bal(key, d, False, "none", prev)
        seq.append(prev)
    moves = []
    for (da, a), (db, b) in zip(zip(degs, seq), zip(degs[1:], seq[1:])):
        ua, ub = a[1:], b[1:]
        mv = _nearest_moves(ua, ub)
        moves.append(max(mv))
        assert max(mv) <= 5
        common = {n % 12 for n in ua} & {n % 12 for n in music.chord_midi(key, db)}
        if common:
            assert set(ua) & set(ub), (a, b)                     # a common tone is held
    assert sum(moves) / len(moves) <= 3


def test_balanced_without_voice_leading_ignores_prev():
    for key, d, f, m in ALL_CHORDS[::7]:
        assert _bal(key, d, f, m, [30, 70, 71, 72], voice_leading=False) == _bal(key, d, f, m)


def test_balanced_voicing_never_crashes_on_impossible_constraints():
    out = music.balanced_voicing([0, 4, 7, 11, 14], 48, None, bass_midi=36, upper_range=(60, 62))
    assert out[0] == 36 and len(out) >= 3


def _roughness(notes, harmonics=(1, 3, 5, 7, 9, 11)):
    """Plomp-Levelt / Sethares roughness of triangle-like tones (odd partials, 1/n^2)."""
    parts = [(music.midi_to_hz(m) * h, 1.0 / h ** 2) for m in notes for h in harmonics]
    r = 0.0
    for i, (f1, a1) in enumerate(parts):
        for f2, a2 in parts[i + 1:]:
            s = 0.24 / (0.0207 * min(f1, f2) + 18.96)
            df = abs(f2 - f1)
            r += a1 * a2 * (math.exp(-3.5 * s * df) - math.exp(-5.75 * s * df))
    return r


def test_balanced_is_less_rough_than_close():
    bal = close = 0.0
    worst = 0.0
    for key, d, f, m in ALL_CHORDS:
        b = _roughness(_bal(key, d, f, m))
        c = _roughness(music.chord_voicing(key, d, f, m, None, style="close"))
        bal, close, worst = bal + b, close + c, max(worst, b / c)
    assert bal < 0.5 * close                                     # >= 50% less rough on average (58% now)
    assert worst < 1.05                                          # never rougher (max ratio 1.00 now)


def test_balanced_fast():
    prev = _bal("C", 2, False, "add9")
    times = []
    for m in ("add7", "add9", "add6"):
        for _ in range(50):
            t = time.perf_counter()
            _bal("C", 1, False, m, prev)
            times.append(time.perf_counter() - t)
    assert statistics.median(times) < 0.002
