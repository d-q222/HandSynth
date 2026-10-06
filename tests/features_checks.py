"""Shared feature checks, run against hand_synth.features and the stand-ins."""
import math

import pytest

from conftest import load_fixtures
from synthetic_hand import make_hand

OPEN = (True,) * 5


def rotated(lm, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    wx, wy, _ = lm[0]
    return [(wx + (x - wx) * c - (y - wy) * s, wy + (x - wx) * s + (y - wy) * c, z) for x, y, z in lm]


def run_checks(f, skip_on_stub):
    """f: object with the five feature functions."""
    def call(name, *a):
        try:
            return getattr(f, name)(*a)
        except NotImplementedError:
            if skip_on_stub:
                pytest.skip(f"{name}() not written yet")
            raise
    up = make_hand(OPEN)
    return up, call


def check_roll_upright_and_rotated(f, skip_on_stub):
    up, call = run_checks(f, skip_on_stub)
    assert call("roll_deg", up) == pytest.approx(0, abs=1)
    for deg in (90, -90, 45, -45):
        assert call("roll_deg", rotated(up, deg)) == pytest.approx(deg, abs=1)


def check_pinch_scale_invariant(f, skip_on_stub):
    up, call = run_checks(f, skip_on_stub)
    big = [(2 * x, 2 * y, 2 * z) for x, y, z in up]
    assert call("pinch_norm", big) == pytest.approx(call("pinch_norm", up))
    assert call("pinch_norm", up) > 0


def check_hand_size_and_wrist(f, skip_on_stub):
    up, call = run_checks(f, skip_on_stub)
    assert call("hand_size", up) == pytest.approx(0.18, rel=1e-6)
    assert call("hand_size", rotated(up, 70)) == pytest.approx(0.18, rel=1e-6)
    assert tuple(call("wrist_xy", up)) == pytest.approx(tuple(up[0][:2]))


def check_feature_vector(f, skip_on_stub):
    up, call = run_checks(f, skip_on_stub)
    v = call("feature_vector", up)
    assert len(v) == 63
    assert v[:3] == pytest.approx([0, 0, 0])
    shifted = [(x + 0.3, y - 0.1, z) for x, y, z in up]
    scaled = [(2 * x, 2 * y, 2 * z) for x, y, z in up]
    assert call("feature_vector", shifted) == pytest.approx(v)
    assert call("feature_vector", scaled) == pytest.approx(v)
    # hand size becomes 1: middle MCP at distance 1 from the wrist origin
    assert math.hypot(v[27], v[28]) == pytest.approx(1.0)


def check_roll_fixture(f, fx, skip_on_stub):
    _, call = run_checks(f, skip_on_stub)
    lm = [tuple(p) for p in fx["landmarks"]]
    assert call("roll_deg", lm) == pytest.approx(fx["expected"]["roll_deg"], abs=5)


ROLL_FIXTURES = load_fixtures("roll_deg")
