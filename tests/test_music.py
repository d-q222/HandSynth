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
