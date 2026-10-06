"""The reconstruct gate: hand_synth.fingers.fingers_up must match every fixture."""
import pytest

from conftest import load_fixtures
from hand_synth import fingers


@pytest.mark.parametrize("fx", load_fixtures("fingers"))
def test_fingers_up(fx, request):
    lm = [tuple(p) for p in fx["landmarks"]]
    try:
        got = fingers.fingers_up(lm, fx["label"])
    except NotImplementedError:
        pytest.skip("fingers_up() not written yet")
    if tuple(got) != tuple(fx["expected"]["fingers"]):
        pytest.fail(request.node.callspec.id, pytrace=False)
