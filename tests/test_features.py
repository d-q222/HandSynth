"""Gate for hand_synth.features (skips until the functions are written)."""
import pytest

import features_checks as fc
from hand_synth import features


def test_roll_deg():
    fc.check_roll_upright_and_rotated(features, True)


def test_pinch_norm_scale_invariant():
    fc.check_pinch_scale_invariant(features, True)


def test_hand_size_and_wrist_xy():
    fc.check_hand_size_and_wrist(features, True)


def test_feature_vector():
    fc.check_feature_vector(features, True)


@pytest.mark.parametrize("fx", fc.ROLL_FIXTURES)
def test_roll_fixtures(fx, request):
    try:
        fc.check_roll_fixture(features, fx, True)
    except AssertionError:
        pytest.fail(request.node.callspec.id, pytrace=False)
