import random

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


def _walk(key, degs, **kw):
    prev, out = None, []
    for d in degs:
        prev = music.chord_voicing(key, d, False, "none", 3, prev, True, RANGE, kw.get("bass", False))
        out.append(prev)
    return out


def test_cold_start_close_position():
    assert music.chord_voicing("C", 1, False, "none", 3, None, True, RANGE, False) == [48, 52, 55]
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
    c = music.chord_voicing("C", 1, False, "none", 3, None, True, RANGE, False)
    f = music.chord_voicing("C", 4, False, "none", 3, c, True, RANGE, False)
    am = music.chord_voicing("C", 6, False, "none", 3, c, True, RANGE, False)
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
                assert (music.chord_voicing(key, d, True, mod, 3, [50, 55], False, RANGE, False)
                        == music.chord_midi(key, d, True, mod, 3))


def test_bass():
    assert music.with_bass([48, 52, 55], 48) == [36, 48, 52, 55]
    for key in config.KEY_NAMES:
        for d in range(1, 8):
            for vl in (False, True):
                out = music.chord_voicing(key, d, False, "add9", 3, None, vl, RANGE, True)
                root = music.chord_midi(key, d, False, "add9", 3)[0]
                assert (root - out[0]) % 12 == 0 and out[0] <= root - 12
                assert out[0] < out[1]
    # prev with bass is stripped before voice leading
    a = music.chord_voicing("C", 1, False, "none", 3, None, True, RANGE, True)
    b = music.chord_voicing("C", 4, False, "none", 3, a, True, RANGE, True)
    assert b[0] == 41 and b[1:] == music.chord_voicing("C", 4, False, "none", 3, a[1:], True, RANGE, False)


def test_narrow_range_widened_not_crash():
    out = music.voice_lead([48, 52, 55, 62], [50], 50, 55)
    assert sorted(n % 12 for n in out) == [0, 2, 4, 7]
