"""PRD C8 manual acceptance checklist, automated with synthetic hands (stand-in geometry)."""
import pytest

from hand_synth import config as cfg
from hand_synth import standins
from hand_synth.main import Instrument
from hand_synth.music import chord_midi
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
    assert {n % 12 for n in inst.voicing} == {n % 12 for n in chord_midi("G", 2, False, "add7")}
    assert len(inst.freqs()) == len(inst.voicing)
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


def test_dump_fixture_never_overwrites(tmp_path, monkeypatch):
    import json
    import hand_synth.main as m
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    hands = HandsFrame(left=left(OPEN), right=None)
    first = m.dump_fixture(hands, standins.fingers_up, standins.roll_deg)[0]
    second = m.dump_fixture(hands, standins.fingers_up, standins.roll_deg)[0]
    assert first.name == "left_TTTTT_001.json" and second.name == "left_TTTTT_002.json"
    second_bytes = second.read_bytes()
    first.rename(first.with_name("left_FTTTT_001.json"))        # reviewed + renamed
    third = m.dump_fixture(hands, standins.fingers_up, standins.roll_deg)[0]
    assert third != second and second.read_bytes() == second_bytes   # freed slot reused, nothing clobbered
    d = json.loads(third.read_text())
    assert d["label"] == "Left" and len(d["landmarks"]) == 21 and d["source"] == "recorded"


def test_fixture_filename_tag_is_the_truth(tmp_path, monkeypatch):
    import json
    import conftest
    (tmp_path / "left_FTTTT_001.json").write_text(json.dumps(
        {"label": "Left", "landmarks": [[0, 0, 0]] * 21, "expected": {"fingers": [True] * 5}}))
    monkeypatch.setattr(conftest, "FIXTURE_ROOT", tmp_path)
    [param] = conftest.load_fixtures("fingers")
    assert param.values[0]["expected"]["fingers"] == [False, True, True, True, True]


def test_progression_is_voice_led(inst):
    """G → C → D → G in G: upper voices move a few semitones, never a block leap."""
    n = cfg.HOLD_FRAMES + 1
    prev = None
    for state in (OPEN, IMR, (False, True, True, True, True), OPEN):     # I, IV, V, I
        play(inst, n, lh=left(state))
        upper = inst.voicing[1:] if cfg.BASS_NOTE else inst.voicing
        lo, hi = cfg.VOICING_RANGE
        assert all(lo <= m <= hi for m in upper)
        if prev:
            assert max(min(abs(a - b) for b in prev) for a in upper) <= 5
        prev = upper


def test_shipped_config_is_valid():
    from hand_synth.main import check_config
    assert check_config() == []


def test_config_typos_are_caught():
    from types import SimpleNamespace
    from hand_synth.main import check_config
    bad = SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    bad.MOD_MAP = {**cfg.MOD_MAP, (True, False, False, False, True): "add11"}
    bad.FINGER_MAP = {**cfg.FINGER_MAP, (True, True): 3, OPEN: 8}
    bad.EXPR_HAND = "Left"
    errs = check_config(bad)
    assert any("add11" in e for e in errs)
    assert any("(True, True)" in e for e in errs)
    assert any("8" in e and "FINGER_MAP" in e for e in errs)
    assert any("different hands" in e for e in errs)


@pytest.mark.parametrize("expr", ["Right", "Left"])
def test_space_grows_outward_for_either_expression_hand(monkeypatch, expr):
    monkeypatch.setattr(cfg, "EXPR_HAND", expr)
    monkeypatch.setattr(cfg, "CHORD_HAND", "Left" if expr == "Right" else "Right")
    aspect = 16 / 9

    def space_at(frac_from_midline):          # 0 = frame centre, 1 = the hand's own outer edge
        x = aspect * (0.5 + 0.5 * frac_from_midline if expr == "Right" else 0.5 - 0.5 * frac_from_midline)
        inst = Instrument(standins.fingers_up, standins, key="C")
        hand = Hand(expr, make_hand(OPEN, expr, center=(x, 0.5)), 0.9)
        for _ in range(40):
            inst.update(HandsFrame(left=hand if expr == "Left" else None, right=hand if expr == "Right" else None))
        return inst.params().reverb
    assert space_at(0.85) > space_at(0.15)


def test_fit_width_keeps_aspect():
    import numpy as np
    from hand_synth.main import fit_width
    assert fit_width(np.zeros((1080, 1920, 3), np.uint8), 1280).shape == (720, 1280, 3)
    f = np.zeros((720, 1280, 3), np.uint8)
    assert fit_width(f, 1280) is f
