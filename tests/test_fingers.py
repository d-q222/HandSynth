"""hand_synth.fingers.fingers_up must match every fixture."""
import pytest

from conftest import load_fixtures
from hand_synth.fingers import fingers_up


@pytest.mark.parametrize("fx", load_fixtures("fingers"))
def test_fingers_up(fx, request):
    lm = [tuple(p) for p in fx["landmarks"]]
    if tuple(fingers_up(lm)) != tuple(fx["expected"]["fingers"]):
        pytest.fail(request.node.callspec.id, pytrace=False)
