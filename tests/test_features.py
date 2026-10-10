"""Gate for hand_synth.features."""
import math

import pytest

from conftest import load_fixtures
from hand_synth.features import feature_vector, hand_size, pinch_norm, roll_deg, wrist_xy
from synthetic_hand import make_hand

UP = make_hand((True,) * 5)


def rotated(lm, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    wx, wy, _ = lm[0]
    return [(wx + (x - wx) * c - (y - wy) * s, wy + (x - wx) * s + (y - wy) * c, z) for x, y, z in lm]


def test_roll_deg():
    assert roll_deg(UP) == pytest.approx(0, abs=1)
    for deg in (90, -90, 45, -45):
        assert roll_deg(rotated(UP, deg)) == pytest.approx(deg, abs=1)


def test_pinch_norm_scale_invariant():
    big = [(2 * x, 2 * y, 2 * z) for x, y, z in UP]
    assert pinch_norm(big) == pytest.approx(pinch_norm(UP))
    assert pinch_norm(UP) > 0


def test_hand_size_and_wrist_xy():
    assert hand_size(UP) == pytest.approx(0.18, rel=1e-6)
    assert hand_size(rotated(UP, 70)) == pytest.approx(0.18, rel=1e-6)
    assert tuple(wrist_xy(UP)) == pytest.approx(tuple(UP[0][:2]))


def test_feature_vector():
    v = feature_vector(UP)
    assert len(v) == 63
    assert v[:3] == pytest.approx([0, 0, 0])
    shifted = [(x + 0.3, y - 0.1, z) for x, y, z in UP]
    scaled = [(2 * x, 2 * y, 2 * z) for x, y, z in UP]
    assert feature_vector(shifted) == pytest.approx(v)
    assert feature_vector(scaled) == pytest.approx(v)
    # hand size becomes 1: middle MCP at distance 1 from the wrist origin
    assert math.hypot(v[27], v[28]) == pytest.approx(1.0)


@pytest.mark.parametrize("fx", load_fixtures("roll_deg"))
def test_roll_fixtures(fx, request):
    lm = [tuple(p) for p in fx["landmarks"]]
    if roll_deg(lm) != pytest.approx(fx["expected"]["roll_deg"], abs=5):
        pytest.fail(request.node.callspec.id, pytrace=False)
