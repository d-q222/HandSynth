import pytest

from hand_synth.stabilize import EMA, Hold, RollZone


def test_hold_needs_n_frames():
    h = Hold(3, initial="a")
    assert h.update("b") == "a"
    assert h.update("b") == "a"
    assert h.update("b") == "b"
    assert h.value == "b"


def test_hold_glitch_ignored():
    h = Hold(3, initial="a")
    for v in ["a", "a", "b", "a", "a"]:
        assert h.update(v) == "a"
    # alternating candidates never promote
    for v in ["b", "c", "b", "c", "b", "c"]:
        assert h.update(v) == "a"


def test_hold_starts_none_and_immediate():
    h = Hold(3)
    assert h.update("x") is None
    h.update("x")
    assert h.update("x") == "x"
    assert Hold(1).update(5) == 5
    assert Hold(0).update(5) == 5


def test_hold_none_value_and_tuples():
    h = Hold(2, initial=(1,))
    h.update(None)
    assert h.update(None) is None
    assert h.update((1,)) is None


def test_rollzone_thresholds():
    z = RollZone(35, 10)
    assert z.zone == "natural"
    assert z.update(44) == "natural"
    assert z.update(46) == "flipped"
    assert z.update(-30) == "flipped"      # inside deadband stays
    assert z.update(-26) == "flipped"
    assert z.update(24) == "natural"
    assert z.update(-40) == "natural"
    assert z.update(-50) == "flipped"


def test_rollzone_no_chatter():
    z = RollZone(35, 10)
    out = [z.update(35 + (5 if i % 2 else -5)) for i in range(40)]
    assert set(out) == {"natural"}
    z.update(60)
    out = [z.update(35 + (5 if i % 2 else -5)) for i in range(40)]
    assert set(out) == {"flipped"}


def test_ema():
    e = EMA(0.3)
    assert e.update(1.0) == pytest.approx(0.3)
    for _ in range(60):
        e.update(1.0)
    assert e.value == pytest.approx(1.0, abs=1e-6)
    assert EMA(0.5, initial=2.0).update(2.0) == 2.0
