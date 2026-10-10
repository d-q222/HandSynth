from types import SimpleNamespace

from hand_synth.gesture import (KEEP, ClassifierSource, FingerMapSource,
                                degree_from_fingers)

FM = {(True,) * 5: 1, (False,) * 5: None, (False, True, False, False, False): 2}
OPEN, FIST, IDX, ODD = (True,) * 5, (False,) * 5, (False, True, False, False, False), (True, False, True, False, True)


def hand(tag=None):
    return SimpleNamespace(landmarks=tag)


def test_degree_from_fingers():
    assert degree_from_fingers(OPEN, FM) == 1
    assert degree_from_fingers(FIST, FM) is None
    assert degree_from_fingers(ODD, FM) == KEEP
    assert degree_from_fingers(None, FM) == KEEP


def make_src(seq, n=3):
    it = iter(seq)
    seen = []

    def fn(lm):
        seen.append(lm)
        return next(it)
    return FingerMapSource(FM, fn, n), seen


def test_finger_source_hold_and_state():
    src, seen = make_src([OPEN, OPEN, OPEN, IDX, IDX, IDX])
    assert src.state is None
    assert src.degree(hand("lm")) == KEEP
    assert seen == ["lm"]
    src.degree(hand())
    assert src.degree(hand()) == 1
    assert src.state == OPEN
    src.degree(hand()); src.degree(hand())
    assert src.state == OPEN
    assert src.degree(hand()) == 2
    assert src.state == IDX


def test_finger_source_glitch_and_unmapped():
    src, _ = make_src([OPEN] * 3 + [FIST, OPEN, OPEN] + [ODD] * 3 + [FIST] * 3)
    for _ in range(3):
        src.degree(hand())
    assert src.degree(hand()) == 1          # glitch ignored
    src.degree(hand()); src.degree(hand())
    for _ in range(3):
        d = src.degree(hand())
    assert d == KEEP and src.state == ODD   # unmapped -> KEEP
    for _ in range(3):
        d = src.degree(hand())
    assert d is None                        # fist = silence


def test_finger_source_reset():
    src, _ = make_src([OPEN] * 3 + [OPEN] * 3)
    for _ in range(3):
        src.degree(hand())
    assert src.state == OPEN
    src.reset()
    assert src.state is None
    assert src.degree(hand()) == KEEP


class FakeModel:
    def __init__(self, labels):
        self.labels, self.calls = iter(labels), []

    def predict(self, X):
        self.calls.append(X)
        return [next(self.labels)]


def test_classifier_source():
    m = FakeModel([3, 3, 3, 5, 3, 3, 0, 0, 0, None, None, None])
    src = ClassifierSource(m, lambda lm: [1.0, 2.0], 3)
    out = [src.degree(hand("x")) for _ in range(12)]
    assert m.calls[0] == [[1.0, 2.0]]
    assert out[:3] == [KEEP, KEEP, 3]
    assert out[3:6] == [3, 3, 3]            # glitch 5 ignored
    assert out[6:9] == [3, 3, None]         # 0 -> silence
    assert out[9:] == [None, None, None]
    src.reset()
    m.labels = iter([2])
    assert src.degree(hand()) == KEEP
