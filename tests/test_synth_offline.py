import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

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


# ---- reverb / detune --------------------------------------------------------


def cfg_with(**kw):
    return SimpleNamespace(**{**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}, **kw})


def test_reverb_tail_after_release_and_decays():
    rel = 20 + int(cfg.RELEASE_S * cfg.SAMPLE_RATE / B) + 3
    for rv, ring in ((0.6, True), (0.0, False)):
        s = mk(reverb=rv)
        s.set_chord(C)
        x = render(s, 20 + 1800, lambda: s.set_chord([]), at=20)     # ~10 s: room for a 2.5 s tail
        sane(x)
        tail = x[rel * B:(rel + 40) * B]
        if ring:
            assert np.abs(tail).max() > 1e-3
            assert np.abs(x[-20 * B:]).max() == 0.0          # true silence within ~10 s
        else:
            assert np.abs(tail).max() < 1e-4


def test_reverb_change_click_free():
    for new in (0.0, 0.6):
        s = mk(reverb=0.3 if new == 0.6 else 0.6)
        s.set_chord(C)
        x = render(s, 50, lambda: s.set_params(SynthParams(0.6, 3000.0, "triangle", new)), at=20)
        sane(x)
        no_spike(x, 20)


def test_no_clip_loud_chord_with_reverb():
    s = mk(gain=0.8, reverb=0.6)
    s.set_chord(C + [246.94, 293.66])
    x = s.render_offline(400)
    assert np.abs(x).max() < 0.999, np.abs(x).max()


def _spectrum(cents):
    s = NumpySynth(cfg_with(DETUNE_CENTS=cents, OUTPUT_MAKEUP_DB=0.0))     # oscillators only, limiter idle
    s.set_params(SynthParams(0.8, 8000.0, "sine"))
    s.set_chord([440.0])
    x = s.render_offline(400)[100 * B:].astype(np.float64)
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)), len(x) * 8))
    return sp, np.fft.rfftfreq(len(x) * 8, 1 / cfg.SAMPLE_RATE)


def test_detune_spectrum():
    cents = 40.0
    lo, hi = 440 * 2 ** (-cents / 1200), 440 * 2 ** (cents / 1200)
    sp, f = _spectrum(cents)
    m = sp.max()

    def band(a, b):
        return sp[(f > a) & (f < b)].max()
    assert band(439.5, 440.5) > 0.99 * m                     # centre oscillator dominates
    assert band(lo - 0.5, lo + 0.5) > 0.2 * m and band(hi - 0.5, hi + 0.5) > 0.2 * m
    sp0, f0 = _spectrum(0.0)
    assert abs(f0[sp0.argmax()] - 440.0) < 0.3
    assert sp0[(f0 > 443.0) & (f0 < 2000)].max() < 0.01 * sp0.max()
    assert sp0[(f0 < 437.0) & (f0 > 20)].max() < 0.01 * sp0.max()


def test_detune_zero_is_single_oscillator():
    s = NumpySynth(cfg_with(DETUNE_CENTS=0.0))
    s.set_chord(C)
    s.render_offline(3)
    assert s._ratios == ()


def test_block_render_time():
    s = mk(reverb=0.6)
    s.set_chord(C + [246.94, 293.66])
    s.render_offline(50)
    ts = []
    for _ in range(300):
        t = time.perf_counter()
        s._render_block()
        ts.append(time.perf_counter() - t)
    med = float(np.median(ts)) * 1e3
    print(f"median ms/block: {med:.3f}")
    assert med < 1.5, med


def test_unison_never_cancels_the_fundamental():
    """A detuned pair alone beats to silence once per beat period; the centre oscillator prevents it."""
    s = NumpySynth(cfg_with(DETUNE_CENTS=7.0))
    s.set_params(SynthParams(0.8, 8000.0, "sine"))
    s.set_chord([65.41])                                      # C2 bass note: slowest, most audible beat
    x = np.abs(s.render_offline(int(3.0 * cfg.SAMPLE_RATE / B))[20 * B:].astype(np.float64))
    period = int(cfg.SAMPLE_RATE / 65.41) + 1
    env = np.array([x[i:i + period].max() for i in range(0, len(x) - period, period)])
    assert env.min() > 0.45 * env.max()                       # worst dip ~ -5 dB, never a drop-out


@pytest.mark.parametrize("block", [64, 1024, 2048])
def test_reverb_identical_for_any_block_size(block):
    """Reverb is processed in chunks <= the shortest delay, so its impulse response doesn't depend on
    BLOCK_SIZE (a delay clamped to the block size would turn it into a metallic flutter echo)."""
    def impulse_response(bs):
        s = NumpySynth(cfg_with(BLOCK_SIZE=bs))
        x = np.zeros(8192)
        x[0] = 1.0
        return np.concatenate([s._reverb_wet(x[i:i + bs]) for i in range(0, len(x), bs)])
    ref, got = impulse_response(256), impulse_response(block)
    assert np.abs(ref).max() > 0.01
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_stereo_reverb_is_wide_and_long():
    """SPACE must be audible: the wet signal differs between ears (decorrelated) and rings ~2 s+."""
    s = NumpySynth(cfg)
    x = np.zeros(int(4 * cfg.SAMPLE_RATE))
    x[0] = 1.0
    ir = np.concatenate([s._reverb_wet(x[i:i + B]) for i in range(0, len(x), B)])
    left, right = ir[:, 0], ir[:, 1]
    assert abs(np.corrcoef(left, right)[0, 1]) < 0.5                  # two ears hear different rooms
    e = np.cumsum((left ** 2 + right ** 2)[::-1])[::-1]
    t60 = np.argmax(10 * np.log10(e / e[0] + 1e-30) < -60) / cfg.SAMPLE_RATE
    assert t60 > 1.8                                                   # a real room, not a short smear


def test_space_changes_the_sound_a_lot():
    """Full space vs none: the release tail is tens of dB louder, so the control is clearly audible."""
    def tail_db(mix):
        s = mk(reverb=mix)
        s._wet_s = s._wet_prev = mix
        s.set_chord(C)
        x = render(s, 200, lambda: s.set_chord([]), at=150)
        seg = x[(150 + 40) * B:(150 + 80) * B]                          # ~0.2-0.45 s after release
        return 20 * np.log10(np.sqrt(np.mean(seg.astype(float) ** 2)) + 1e-12)
    assert tail_db(1.0) - tail_db(0.0) > 30


def test_stereo_output_shape_and_mid_compat():
    s = mk(reverb=0.0)
    s.set_chord(C)
    st = s.render_offline(10, stereo=True)
    assert st.shape == (10 * B, 2) and st.dtype == np.float32
    assert not np.allclose(st[:, 0], st[:, 1])                         # unison pair is panned


def test_soft_limiter_is_transparent_below_knee_and_never_clips():
    from hand_synth.synth.numpy_synth import _LIMIT_KNEE, _soft_limit
    x = np.linspace(-3, 3, 10001)
    y = _soft_limit(x)
    quiet = np.abs(x) <= _LIMIT_KNEE
    np.testing.assert_array_equal(y[quiet], x[quiet])
    assert np.abs(y).max() < 1.0 and np.all(np.diff(y) > 0)                # monotonic, below full scale


# ---- following the system output --------------------------------------------


class FakeSD:
    """Stands in for sounddevice: one default output whose name the test can switch."""
    def __init__(self, name):
        self.name, self.inits, self.streams = name, 0, []

    def query_devices(self, device=None, kind=None):
        return {"name": self.name, "max_output_channels": 2}

    def OutputStream(self, **kw):
        self.streams.append(SimpleNamespace(start=lambda: None, stop=lambda: None, close=lambda: None))
        return self.streams[-1]

    def _terminate(self):
        pass

    def _initialize(self):
        self.inits += 1


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    return cond()


def test_follows_system_output_switch(monkeypatch):
    import hand_synth.synth.numpy_synth as ns
    sd, dev = FakeSD("Speakers"), {"id": 1}
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(ns, "default_output_id", lambda: dev["id"])
    s = NumpySynth(cfg_with(AUDIO_FOLLOW_POLL_S=0.01, STOP_FADE_S=0.0))
    s.start()
    assert s.output_name == "Speakers" and len(sd.streams) == 1
    sd.name, dev["id"] = "AirPods", 2
    assert wait_for(lambda: s.output_name == "AirPods")
    assert sd.inits == 1 and len(sd.streams) == 2 and s.error is None
    s.stop()
    assert s._follower is None and s._stream is None


def test_pinned_device_is_not_followed(monkeypatch):
    import hand_synth.synth.numpy_synth as ns
    sd = FakeSD("Interface")
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(ns, "default_output_id", lambda: 1)
    s = NumpySynth(cfg_with(AUDIO_DEVICE="Interface", STOP_FADE_S=0.0))
    s.start()
    assert s._follower is None
    s.stop()
