"""Additive numpy synth: ADSR voices -> smoothed-normalised mix -> 2-pole LPF -> gain.

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
    __slots__ = ("freq", "phase", "level", "releasing")

    def __init__(self, freq):
        self.freq = freq
        self.phase = 0.0
        self.level = 0.0
        self.releasing = False


def _shape(name, ph):
    if name == "sine":
        return np.sin(ph)
    if name == "saw":
        return np.mod(ph, TWO_PI) / math.pi - 1.0
    return (2.0 / math.pi) * np.arcsin(np.sin(ph))      # triangle (default)


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
            ph = v.phase + (TWO_PI * v.freq / self.sr) * self._n
            sig = _shape(self._wave, ph)
            if xfade:
                sig = sig * (1.0 - ramp) + _shape(new_wave, ph) * ramp
            mix += env * sig
            levels += env
            v.phase = (v.phase + (TWO_PI * v.freq / self.sr) * n) % TWO_PI
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
