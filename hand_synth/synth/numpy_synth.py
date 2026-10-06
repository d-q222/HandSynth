"""Additive numpy synth: ADSR unison voices -> smoothed-normalised mix -> 2-pole LPF -> reverb -> gain.

Threading: one SynthTarget behind one Lock. The audio callback copies it, releases
the lock, then renders. Voices are owned by the render side only.
"""
import math
import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .base import SynthParams

TWO_PI = 2.0 * math.pi


@dataclass
class SynthTarget:
    freqs: tuple = ()
    params: SynthParams = SynthParams()
    fading: bool = False        # stop() sets this: fade master to 0


class _Voice:
    __slots__ = ("freq", "phase", "ph_u", "level", "releasing")

    def __init__(self, freq):
        self.freq = freq
        self.phase = 0.0        # centre oscillator phase (the only oscillator when DETUNE_CENTS = 0)
        self.ph_u = [1.3, 3.4]  # detuned pair phases, started apart so they don't peak together
        self.level = 0.0
        self.releasing = False


def _shape(name, ph):
    if name == "sine":
        return np.sin(ph)
    if name == "saw":
        return np.mod(ph, TWO_PI) / math.pi - 1.0
    return (2.0 / math.pi) * np.arcsin(np.sin(ph))      # triangle (default)


_COMB_DELAYS = (1116, 1188, 1277, 1356)     # Freeverb tunings @44.1k, scaled to the sample rate
_ALLPASS_DELAYS = (556, 441)
_COMB_SUM_GAIN = 0.25
# Unison mix: centre + detuned pair. A pair alone beats to silence once per beat period (the held
# bass note would drop out every 1-2 s); a strong centre keeps the fundamental steady (±~4 dB shimmer).
_CENTRE_W, _SIDE_W = 0.6, 0.2
_FLUSH = 1e-9


class _Ring:
    """Delay line of length d: read the n <= d samples written d samples ago, then overwrite them."""
    __slots__ = ("buf", "d", "w")

    def __init__(self, d):
        self.buf = np.zeros(d)
        self.d = d
        self.w = 0

    def read(self, n):
        a = self.buf[self.w:self.w + n]
        if len(a) == n:
            return a.copy()
        return np.concatenate((a, self.buf[:n - len(a)]))

    def write(self, x):
        n = len(x)
        k = min(n, self.d - self.w)
        self.buf[self.w:self.w + k] = x[:k]
        if k < n:
            self.buf[:n - k] = x[k:]
        self.w = (self.w + n) % self.d

    def clear(self):
        self.buf[:] = 0.0


class NumpySynth:
    def __init__(self, cfg):
        self.sr = float(cfg.SAMPLE_RATE)
        self.block = int(cfg.BLOCK_SIZE)
        self._cfg = cfg
        self._lock = threading.Lock()
        self._target = SynthTarget(params=SynthParams(waveform=cfg.WAVEFORM))
        self._stream = None
        self.error = None
        self.xrun_count = 0
        self._last_xrun = -1e9

        # render-side state (callback thread only)
        self._voices = {}
        self._wave = cfg.WAVEFORM
        self._n = np.arange(self.block, dtype=np.float64)
        self._att = 1.0 / (max(cfg.ATTACK_S, 1e-4) * self.sr)    # level per sample
        self._rel = 1.0 / (max(cfg.RELEASE_S, 1e-4) * self.sr)
        self._dt = self.block / self.sr
        self._gain_k = 1.0 - math.exp(-self._dt / max(cfg.GAIN_RAMP_S, 1e-4))
        self._cut_k = 1.0 - math.exp(-self._dt / max(cfg.CUTOFF_RAMP_S, 1e-4))
        self._stop_rate = 1.0 / (max(cfg.STOP_FADE_S, 1e-4) * self.sr)
        self._norm_k = 1.0 - math.exp(-self._dt / max(cfg.RELEASE_S, 1e-4))
        self._norm = 1.0        # smoothed count of held (non-releasing) voices
        p = self._target.params
        self._gain_s = p.gain
        self._gain_prev = p.gain
        self._fc = p.cutoff_hz
        self._zi = [np.zeros(1), np.zeros(1)]
        self._fade = 1.0
        self._ratios = ()
        c = float(getattr(cfg, "DETUNE_CENTS", 0.0))
        if c > 0:
            self._ratios = (2.0 ** (-c / 1200.0), 2.0 ** (c / 1200.0))
        # reverb: tuned Freeverb delays, processed in chunks no longer than the shortest delay so
        # each chunk only reads samples written by earlier chunks (works for any BLOCK_SIZE)
        sc = self.sr / 44100.0
        self._combs = [_Ring(max(16, int(round(d * sc)))) for d in _COMB_DELAYS]
        self._aps = [_Ring(max(16, int(round(d * sc)))) for d in _ALLPASS_DELAYS]
        self._rv_chunk = min(r.d for r in self._combs + self._aps)
        self._fb = min(max(float(getattr(cfg, "REVERB_ROOM", 0.84)), 0.0), 0.95)
        self._damp = min(max(float(getattr(cfg, "REVERB_DAMP", 0.4)), 0.0), 0.99)
        self._comb_zi = [np.zeros(1) for _ in self._combs]
        self._wet_s = p.reverb
        self._wet_prev = p.reverb

    # ---- public API -------------------------------------------------------
    def set_chord(self, freqs):
        with self._lock:
            self._target.freqs = tuple(float(f) for f in freqs)

    def set_params(self, p):
        with self._lock:
            self._target.params = p

    def start(self):
        if self._stream is not None:
            return
        import sounddevice as sd
        try:
            self._fade = 1.0
            with self._lock:
                self._target.fading = False
            self._stream = sd.OutputStream(
                samplerate=self.sr, blocksize=self.block, channels=1,
                dtype="float32", callback=self._cb, device=self._cfg.AUDIO_DEVICE)
            self._stream.start()
        except Exception as e:  # PortAudioError, ValueError for bad device, ...
            self._stream = None
            try:
                devices = str(sd.query_devices())
            except Exception as e2:
                devices = f"(could not list devices: {e2})"
            msg = " ".join(str(e).split())
            raise RuntimeError(f"Audio output failed: {msg}\n{devices}") from e

    def stop(self):
        s, self._stream = self._stream, None
        if s is None:
            return
        with self._lock:
            self._target.fading = True
        deadline = time.monotonic() + self._cfg.STOP_FADE_S + 0.5
        while self._fade > 0.0 and time.monotonic() < deadline:   # wait for the fade itself, not a guess
            time.sleep(0.005)
        time.sleep(2 * self._dt)                                 # let the last silent block play out
        try:
            s.stop()
            s.close()
        except Exception:
            pass

    def render_offline(self, n_blocks):
        out = np.empty(n_blocks * self.block, dtype=np.float32)
        for i in range(n_blocks):
            out[i * self.block:(i + 1) * self.block] = self._render_block()
        return out

    def active_voice_count(self):
        return len(self._voices)

    def recent_xrun(self):
        return time.monotonic() - self._last_xrun < 1.0

    # ---- audio thread -----------------------------------------------------
    def _cb(self, outdata, frames, time_info, status):
        try:
            if status and getattr(status, "output_underflow", False):
                self.xrun_count += 1
                self._last_xrun = time.monotonic()
            outdata[:, 0] = self._render_block()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            outdata.fill(0)

    def _sync_voices(self, freqs):
        wanted = {round(f * 100): f for f in freqs if f > 0}
        for k, v in self._voices.items():
            v.releasing = k not in wanted      # returning freq = re-attack, phase kept
        for k, f in wanted.items():
            if k not in self._voices:
                self._voices[k] = _Voice(f)

    def _reverb_chunk(self, x):
        """4 parallel damped combs -> 2 series allpasses (Freeverb), vectorised; len(x) <= every delay."""
        d, m = self._damp, len(x)
        acc = None
        for i, c in enumerate(self._combs):
            o = c.read(m)
            f, self._comb_zi[i] = lfilter([1.0 - d], [1.0, -d], o, zi=self._comb_zi[i])
            c.write(x + f * self._fb)
            acc = o if acc is None else acc + o
        acc *= _COMB_SUM_GAIN
        for a in self._aps:
            o = a.read(m)
            a.write(acc + 0.5 * o)
            acc = o - acc
        return acc

    def _reverb(self, x, wet_target, ramp):
        self._wet_s += (min(max(wet_target, 0.0), 1.0) - self._wet_s) * self._gain_k
        wet = self._wet_prev + (self._wet_s - self._wet_prev) * ramp
        self._wet_prev = self._wet_s
        m = self._rv_chunk
        acc = np.concatenate([self._reverb_chunk(x[i:i + m]) for i in range(0, len(x), m)])
        if np.abs(x).max() < _FLUSH and all(np.abs(r.buf).max() < _FLUSH for r in self._combs + self._aps):
            # tail gone: true silence, no denormals
            for r in self._combs + self._aps:
                r.clear()
            for z in self._comb_zi:
                z[:] = 0.0
            acc = np.zeros_like(acc)
        return x * (1.0 - 0.5 * wet) + acc * wet

    def _render_block(self):
        with self._lock:
            freqs, p, fading = self._target.freqs, self._target.params, self._target.fading
        n = self.block
        k = self._n + 1.0
        was_silent = not self._voices
        self._sync_voices(freqs)
        held = sum(1 for v in self._voices.values() if not v.releasing)
        norm_prev = self._norm
        if held:
            self._norm = float(held) if was_silent else self._norm + (held - self._norm) * self._norm_k
            if was_silent:
                norm_prev = self._norm

        mix = np.zeros(n)
        levels = np.zeros(n)
        new_wave = p.waveform if p.waveform in ("sine", "triangle", "saw") else self._wave
        xfade = new_wave != self._wave
        ramp = k / n
        dead = []
        for key, v in self._voices.items():
            rate = -self._rel if v.releasing else self._att
            env = v.level + rate * k
            np.clip(env, 0.0, 1.0, out=env)
            w0 = TWO_PI * v.freq / self.sr
            ph = v.phase + w0 * self._n
            sig = _shape(self._wave, ph)
            if xfade:
                sig = sig * (1.0 - ramp) + _shape(new_wave, ph) * ramp
            v.phase = (v.phase + w0 * n) % TWO_PI
            if self._ratios:
                sig = sig * _CENTRE_W
                for j, r in enumerate(self._ratios):
                    ph = v.ph_u[j] + (w0 * r) * self._n
                    s_ = _shape(self._wave, ph)
                    if xfade:
                        s_ = s_ * (1.0 - ramp) + _shape(new_wave, ph) * ramp
                    sig += _SIDE_W * s_
                    v.ph_u[j] = (v.ph_u[j] + w0 * r * n) % TWO_PI
            mix += env * sig
            levels += env
            v.level = float(env[-1])
            if v.releasing and v.level <= 0.0:
                dead.append(key)
        for key in dead:
            del self._voices[key]
        self._wave = new_wave

        # Normalise by the larger of the summed envelopes (equal-sum crossfades between chords)
        # and the smoothed held-voice count (so attacks from / releases to silence keep their
        # full envelope shape instead of sitting at full level until the sum drops below 1).
        norm = norm_prev + (self._norm - norm_prev) * ramp
        mix /= np.maximum(np.maximum(1.0, norm), levels)

        # cutoff: smooth toward target, one-pole x2 (12 dB/oct)
        self._fc += (p.cutoff_hz - self._fc) * self._cut_k
        fc = min(max(self._fc, 20.0), 0.45 * self.sr)
        a = 1.0 - math.exp(-TWO_PI * fc / self.sr)
        den = [1.0, -(1.0 - a)]
        y, self._zi[0] = lfilter([a], den, mix, zi=self._zi[0])
        y, self._zi[1] = lfilter([a], den, y, zi=self._zi[1])
        if not self._voices:                            # flush filter state toward exact zero (no denormals)
            for z in self._zi:
                if abs(z[0]) < _FLUSH:
                    z[:] = 0.0

        y = self._reverb(y, p.reverb, ramp)

        # gain: smoothed target, linear ramp across the block
        self._gain_s += (p.gain - self._gain_s) * self._gain_k
        g = self._gain_prev + (self._gain_s - self._gain_prev) * ramp
        self._gain_prev = self._gain_s
        y = y * g

        if fading or self._fade < 1.0:                 # stop fade (linear)
            target = 0.0 if fading else 1.0
            step = self._stop_rate * (1 if target > self._fade else -1)
            f = np.clip(self._fade + step * k, 0.0, 1.0)
            self._fade = float(f[-1])
            y = y * f
        return np.clip(y, -1.0, 1.0).astype(np.float32)
