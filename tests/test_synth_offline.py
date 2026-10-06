import numpy as np

from hand_synth import config as cfg
from hand_synth.synth.base import SynthParams
from hand_synth.synth.numpy_synth import NumpySynth

B = cfg.BLOCK_SIZE
C = [130.81, 164.81, 196.0]       # C3 E3 G3
AM = [110.0, 130.81, 164.81]      # A2 C3 E3 (shares C, E)
CSUS4 = [130.81, 174.61, 196.0]   # shares C, G


def mk(**p):
    s = NumpySynth(cfg)
    s.set_params(SynthParams(**{"gain": 0.6, "cutoff_hz": 3000.0, "waveform": "triangle", **p}))
    return s


def render(s, n, fn=None, at=None):
    a = s.render_offline(at) if at else np.zeros(0, np.float32)
    if fn:
        fn()
    return np.concatenate([a, s.render_offline(n - (at or 0))])


def steady_jump(x):
    return np.abs(np.diff(x[10 * B:19 * B])).max()


def no_spike(x, at, factor=1.6):
    around = np.abs(np.diff(x[(at - 1) * B:(at + 6) * B])).max()
    assert around <= steady_jump(x) * factor + 1e-3, (around, steady_jump(x))


def sane(x):
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0


def test_chord_switches_click_free():
    for new in (AM, CSUS4, C + [246.94]):
        s = mk()
        s.set_chord(C)
        x = render(s, 50, lambda: s.set_chord(new), at=20)
        sane(x)
        assert np.abs(x[40 * B:]).max() > 0.05
        no_spike(x, 20)


def test_silence_after_empty_chord():
    s = mk()
    s.set_chord(C)
    x = render(s, 50, lambda: s.set_chord([]), at=20)
    sane(x)
    tail = x[(20 + int(cfg.RELEASE_S * cfg.SAMPLE_RATE / B) + 3) * B:]
    assert np.abs(tail).max() < 1e-4
    assert s.active_voice_count() == 0
    no_spike(x, 20)


def test_waveform_switch_click_free():
    s = mk()
    s.set_chord(C)
    x = render(s, 50, lambda: s.set_params(SynthParams(0.6, 3000.0, "saw")), at=20)
    sane(x)
    # saw has genuine steps, so only compare with a loose bound against its own steady state
    assert np.abs(np.diff(x[19 * B:22 * B])).max() < 3 * np.abs(np.diff(x[40 * B:49 * B])).max()
    s = mk()
    s.set_chord(C)
    x = render(s, 50, lambda: s.set_params(SynthParams(0.6, 3000.0, "sine")), at=20)
    no_spike(x, 20)


def test_gain_change_click_free():
    s = mk()
    s.set_chord(C)
    x = render(s, 50, lambda: s.set_params(SynthParams(0.0, 3000.0, "triangle")), at=20)
    sane(x)
    no_spike(x, 20)
    assert np.abs(x[-5 * B:]).max() < 0.01


def test_cutoff_reduces_highs():
    def hf(cut):
        s = mk(cutoff_hz=cut, waveform="saw")
        s.set_chord(C)
        x = s.render_offline(60)[20 * B:]
        sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        f = np.fft.rfftfreq(len(x), 1 / cfg.SAMPLE_RATE)
        return sp[f > 3000].sum()
    assert hf(400.0) < 0.2 * hf(6000.0)


def test_released_voice_requested_again_keeps_phase():
    s = mk(waveform="sine")
    s.set_chord([220.0])
    s.render_offline(30)
    s.set_chord([])
    s.render_offline(1)
    v = next(iter(s._voices.values()))
    assert v.releasing and 0 < v.level < 1
    expected_phase = (v.phase + 2 * np.pi * 220.0 / cfg.SAMPLE_RATE * B) % (2 * np.pi)
    lvl = v.level
    s.set_chord([220.0])
    x = s.render_offline(1)
    assert s.active_voice_count() == 1
    v2 = next(iter(s._voices.values()))
    assert v2 is v and not v2.releasing and v2.level > lvl
    assert abs(v2.phase - expected_phase) < 1e-6
    sane(x)


def test_no_level_step_when_note_count_changes():
    s = mk(waveform="sine")
    s.set_chord(C)
    x = render(s, 50, lambda: s.set_chord(C + [246.94]), at=20)
    sane(x)
    no_spike(x, 20)


def test_import_without_audio_device():
    s = NumpySynth(cfg)
    s.stop()
    s.stop()
    assert s.error is None and not s.recent_xrun()
