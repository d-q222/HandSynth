"""PRD C8 manual acceptance checklist, automated with synthetic hands (stand-in geometry)."""
import pytest

from hand_synth import config as cfg
from hand_synth import standins
from hand_synth.main import Instrument
from hand_synth.music import chord_freqs
from hand_synth.tracking import Hand, HandsFrame, _assign
from synthetic_hand import make_hand

OPEN = (True, True, True, True, True)
FIST = (False, False, False, False, False)
INDEX = (False, True, False, False, False)
IM = (False, True, True, False, False)
IMR = (False, True, True, True, False)


def left(state, roll=0.0):
    return Hand("Left", make_hand(state, "Left", roll_deg=roll, center=(0.5, 0.5)), 0.99)


def right(state, y=0.5, scale=1.0):
    return Hand("Right", make_hand(state, "Right", center=(1.3, y), scale=scale), 0.99)


def play(inst, frames, lh=None, rh=None, key=None):
    changed = False
    for i in range(frames):
        changed |= inst.update(HandsFrame(left=lh, right=rh), key if i == 0 else None)
    return changed


@pytest.fixture
def inst():
    return Instrument(standins.fingers_up, standins, key="G")


def names(inst):
    return inst.chord_name()


def test_checklist(inst):
    n = cfg.HOLD_FRAMES + 1
    play(inst, n, lh=left(OPEN))
    assert names(inst) == "G"
    play(inst, n, lh=left(INDEX))
    assert names(inst) == "Am"
    play(inst, n, lh=left(INDEX, roll=50))
    assert names(inst) == "A"                      # flipped
    play(inst, n, lh=left(INDEX, roll=0))
    assert names(inst) == "Am"
    play(inst, n, lh=left(INDEX), rh=right(IM))
    assert names(inst) == "Asus4"
    play(inst, n, lh=left(INDEX), rh=right(IMR))
    assert names(inst) == "Am7"
    assert inst.freqs() == chord_freqs("G", 2, False, "add7", cfg.ROOT_OCTAVE)
    play(inst, n, lh=left(FIST), rh=right(OPEN))
    assert inst.freqs() == [] and names(inst) == "-"
    play(inst, n, lh=left(OPEN), key="D")
    assert names(inst) == "D"


def test_single_frame_glitch_does_not_change_chord(inst):
    play(inst, cfg.HOLD_FRAMES + 1, lh=left(OPEN))
    assert not play(inst, 1, lh=left(INDEX))
    assert names(inst) == "G"
    assert not play(inst, 3, lh=left(OPEN))


def test_hand_lost_keeps_chord_and_expr_loss_resets_mod(inst):
    n = cfg.HOLD_FRAMES + 1
    play(inst, n, lh=left(OPEN), rh=right(IM))
    assert names(inst) == "Gsus4"
    play(inst, n)                                   # both hands gone
    assert names(inst) == "G"                       # chord holds, mod resets
    assert inst.roll is None and inst.expr_fingers is None


def test_unmapped_state_keeps_previous_chord(inst):
    n = cfg.HOLD_FRAMES + 1
    play(inst, n, lh=left(OPEN))
    play(inst, n, lh=left((False, False, False, False, True)))   # pinky only: unmapped
    assert names(inst) == "G"


def test_key_change_retargets_sounding_chord(inst):
    play(inst, cfg.HOLD_FRAMES + 1, lh=left(INDEX))
    assert play(inst, 1, lh=left(INDEX), key="C")
    assert names(inst) == "Dm"


def test_expression_height_and_pinch(inst):
    for _ in range(30):
        inst.update(HandsFrame(left=None, right=right(OPEN, y=0.3)))
    high_gain = inst.params().gain
    for _ in range(30):
        inst.update(HandsFrame(left=None, right=right(OPEN, y=0.85)))
    assert inst.params().gain < high_gain           # lower hand = quieter
    bright = inst.params().cutoff_hz
    for _ in range(30):
        inst.update(HandsFrame(left=None, right=right(FIST, y=0.85)))   # curled index tip meets folded thumb → small pinch
    assert inst.params().cutoff_hz < bright
    lo, hi = cfg.CUTOFF_HZ
    assert lo <= inst.params().cutoff_hz <= hi


def test_controls_hold_when_expression_hand_leaves(inst):
    for _ in range(30):
        inst.update(HandsFrame(left=None, right=right(OPEN, y=0.3)))
    p = inst.params()
    play(inst, 10)
    assert inst.params() == p


def test_waveform_cycles(inst):
    seen = {inst.waveform}
    for _ in range(len(cfg.WAVEFORMS)):
        inst.cycle_waveform()
        seen.add(inst.waveform)
    assert seen == set(cfg.WAVEFORMS)


def test_tracker_assign_breaks_duplicate_labels():
    a = Hand("Right", make_hand(OPEN, "Left", center=(0.4, 0.5)), 0.9)
    b = Hand("Right", make_hand(OPEN, "Right", center=(1.3, 0.5)), 0.9)
    hf = _assign([b, a], 16 / 9)
    assert hf.left is a and hf.right is b
