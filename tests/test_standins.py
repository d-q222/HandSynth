"""Stand-ins must pass every fixture and feature check; resolvers must pick correctly."""
import pytest

import features_checks as fc
from conftest import load_fixtures
from hand_synth import features, fingers, standins


@pytest.mark.parametrize("fx", load_fixtures("fingers"))
def test_standin_fingers(fx):
    lm = [tuple(p) for p in fx["landmarks"]]
    assert standins.fingers_up(lm, fx["label"]) == tuple(fx["expected"]["fingers"])


def test_standin_roll():
    fc.check_roll_upright_and_rotated(standins, False)


def test_standin_pinch():
    fc.check_pinch_scale_invariant(standins, False)


def test_standin_size_wrist():
    fc.check_hand_size_and_wrist(standins, False)


def test_standin_feature_vector():
    fc.check_feature_vector(standins, False)


@pytest.mark.parametrize("fx", fc.ROLL_FIXTURES)
def test_standin_roll_fixtures(fx):
    fc.check_roll_fixture(standins, fx, False)


def test_probe_is_open_right_hand():
    assert len(standins.PROBE_LANDMARKS) == 21
    assert standins.fingers_up(standins.PROBE_LANDMARKS, "Right") == (True,) * 5


# ── resolvers ────────────────────────────────────────────────────────────────
def test_resolve_fingers_follows_manual_module_state(monkeypatch):
    fn, status = standins.resolve_fingers()
    if status != "manual":  # real stub (or owner's code erroring): stand-in in use
        assert status.startswith("stand-in")
    monkeypatch.setattr(fingers, "fingers_up", lambda lm, label: (True,) * 5)
    fn, status = standins.resolve_fingers()
    assert status == "manual" and fn(None, "Left") == (True,) * 5


def test_resolve_fingers_stub(monkeypatch):
    def stub(lm, label):
        raise NotImplementedError
    monkeypatch.setattr(fingers, "fingers_up", stub)
    fn, status = standins.resolve_fingers()
    assert status == "stand-in" and fn is standins.fingers_up


def test_resolve_fingers_broken_then_runtime_fallback(monkeypatch):
    def boom(lm, label):
        raise KeyError("oops")
    monkeypatch.setattr(fingers, "fingers_up", boom)
    fn, status = standins.resolve_fingers()
    assert status.startswith("stand-in (fingers.py error: KeyError")
    # probe passes but a later call fails -> wrapper falls back
    calls = []

    def flaky(lm, label):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("bad frame")
        return (False,) * 5
    monkeypatch.setattr(fingers, "fingers_up", flaky)
    fn, status = standins.resolve_fingers()
    assert status == "manual" and fn.last_error is None
    assert fn(standins.PROBE_LANDMARKS, "Right") == (True,) * 5
    assert fn.last_error.startswith("ValueError")


def test_resolve_fingers_bad_return(monkeypatch):
    monkeypatch.setattr(fingers, "fingers_up", lambda lm, label: None)
    assert standins.resolve_fingers()[1].startswith("stand-in (fingers.py error: TypeError")


def test_resolve_features_independent(monkeypatch):
    def stub(*a):
        raise NotImplementedError
    monkeypatch.setattr(features, "hand_size", lambda lm: 0.5)
    monkeypatch.setattr(features, "wrist_xy", stub)
    monkeypatch.setattr(features, "pinch_norm", lambda lm: "x")  # implausible
    monkeypatch.setattr(features, "roll_deg", lambda lm: 1 / 0)
    monkeypatch.setattr(features, "feature_vector", stub)
    ns, st = standins.resolve_features()
    assert st["hand_size"] == "manual" and ns.hand_size(None) == 0.5
    assert st["wrist_xy"] == "stand-in" and ns.wrist_xy is standins.wrist_xy
    assert st["pinch_norm"].startswith("stand-in (features.py error")
    assert "ZeroDivisionError" in st["roll_deg"]
    assert st["feature_vector"] == "stand-in"
    assert set(st) == {"hand_size", "wrist_xy", "pinch_norm", "roll_deg", "feature_vector"}


def test_resolve_features_runtime_fallback(monkeypatch):
    n = []

    def flaky(lm):
        n.append(1)
        if len(n) > 1:
            raise RuntimeError("x")
        return 0.0
    monkeypatch.setattr(features, "roll_deg", flaky)
    ns, st = standins.resolve_features()
    assert st["roll_deg"] == "manual"
    up = fc.make_hand(fc.OPEN)
    assert ns.roll_deg(up) == pytest.approx(0, abs=1)
    assert ns.roll_deg.last_error.startswith("RuntimeError")


def test_manual_bad_values_at_play_time_fall_back(monkeypatch):
    """A manual function that passes the startup probe but later returns a bad value must not
    reach the live loop (None → stand-in), and numpy/list results are normalized."""
    import numpy as np
    calls = []

    def sometimes_none(lm, label):
        calls.append(1)
        return None if len(calls) > 1 else [np.True_, True, 1, 0, False]
    monkeypatch.setattr(fingers, "fingers_up", sometimes_none)
    fn, status = standins.resolve_fingers()
    assert status == "manual"
    assert fn(standins.PROBE_LANDMARKS, "Right") == (True,) * 5     # stand-in answered
    assert fn.last_error is not None

    monkeypatch.setattr(fingers, "fingers_up", lambda lm, label: [np.True_, True, 1, 0, False])
    fn, _ = standins.resolve_fingers()
    out = fn(standins.PROBE_LANDMARKS, "Right")
    assert out == (True, True, True, False, False) and isinstance(out, tuple) and hash(out)

    monkeypatch.setattr(features, "roll_deg", lambda lm: np.float32(12.5))
    ns, status = standins.resolve_features()
    assert status["roll_deg"] == "manual" and ns.roll_deg(standins.PROBE_LANDMARKS) == 12.5
