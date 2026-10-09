"""Additive numpy synth, stereo: ADSR unison voices (detuned pair panned L/R) -> smoothed-normalised
mix -> 2-pole LPF -> pre-delayed stereo Freeverb -> gain.

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
from .coreaudio import default_output_id

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
_STEREO_SPREAD = 23                         # Freeverb: right-channel delays are this much longer
_COMB_SUM_GAIN = 0.25
_WET_GAIN = 0.4                             # full space ≈ +1 dB vs dry on a sustained pad (space ≠ volume)
# Unison mix: centre + detuned pair. A pair alone beats to silence once per beat period (the held
# bass note would drop out every 1-2 s); a strong centre keeps the fundamental steady (±~4 dB shimmer).
_CENTRE_W, _SIDE_W = 0.6, 0.2
_FLUSH = 1e-9
_LIMIT_KNEE = 0.7                           # soft limiter: linear below this, tanh curve above it


def _soft_limit(y):
    """Leave |y| <= knee untouched; above it, bend smoothly toward 1.0 (slope-continuous, never clips)."""
    a = np.abs(y)
    over = a > _LIMIT_KNEE
    if over.any():
        k = _LIMIT_KNEE
        y = np.where(over, np.sign(y) * (k + (1.0 - k) * np.tanh((a - k) / (1.0 - k))), y)
    return y


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


class _Freeverb:
    """One channel of Freeverb: 4 parallel damped combs -> 2 series allpasses. Vectorised: `process`
    works in chunks no longer than the shortest delay, so each chunk only reads earlier samples."""

    def __init__(self, sr, spread, feedback, damp):
        sc = sr / 44100.0
        self.combs = [_Ring(max(16, int(round((d + spread) * sc)))) for d in _COMB_DELAYS]
        self.aps = [_Ring(max(16, int(round((d + spread) * sc)))) for d in _ALLPASS_DELAYS]
        self.zi = [np.zeros(1) for _ in self.combs]
        self.fb, self.damp = feedback, damp
        self.shortest = min(r.d for r in self.combs + self.aps)

    def _chunk(self, x):
        d, m = self.damp, len(x)
        acc = None
        for i, c in enumerate(self.combs):
            o = c.read(m)
            f, self.zi[i] = lfilter([1.0 - d], [1.0, -d], o, zi=self.zi[i])
            c.write(x + f * self.fb)
            acc = o if acc is None else acc + o
        acc *= _COMB_SUM_GAIN
        for a in self.aps:
            o = a.read(m)
            a.write(acc + 0.5 * o)
            acc = o - acc
        return acc

    def process(self, x, chunk):
        return np.concatenate([self._chunk(x[i:i + chunk]) for i in range(0, len(x), chunk)])

    def quiet(self):
        return all(np.abs(r.buf).max() < _FLUSH for r in self.combs + self.aps)

    def clear(self):
        for r in self.combs + self.aps:
            r.clear()
        for z in self.zi:
            z[:] = 0.0


class NumpySynth:
    def __init__(self, cfg):
        self.sr = float(cfg.SAMPLE_RATE)
        self.block = int(cfg.BLOCK_SIZE)
        self._cfg = cfg
        self._lock = threading.Lock()
        self._target = SynthTarget(params=SynthParams(waveform=cfg.WAVEFORM))
        self._stream = None
        self._follower = None           # thread that moves the stream when the macOS output changes
        self._follow_stop = None
        self.output_name = None
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
        self._zi = [np.zeros((1, 2)), np.zeros((1, 2))]      # 2-pole LPF state, per channel
        self._fade = 1.0
        self.out_rms = 0.0      # level of the last rendered block (diagnostics)
        self._makeup = 10 ** (float(getattr(cfg, "OUTPUT_MAKEUP_DB", 0.0)) / 20.0)
        self._ratios = ()
        c = float(getattr(cfg, "DETUNE_CENTS", 0.0))
        if c > 0:
            self._ratios = (2.0 ** (-c / 1200.0), 2.0 ** (c / 1200.0))
        # detuned pair panning: lower oscillator leans left, upper leans right (mid stays unchanged)
        sp = min(max(float(getattr(cfg, "UNISON_SPREAD", 0.5)), 0.0), 1.0)
        self._pan = ((1.0 + sp, 1.0 - sp), (1.0 - sp, 1.0 + sp))   # (L, R) weights for (low, high)
        # reverb: pre-delay, then one Freeverb bank per ear (right delays offset = stereo width)
        fb = min(max(float(getattr(cfg, "REVERB_ROOM", 0.9)), 0.0), 0.97)
        damp = min(max(float(getattr(cfg, "REVERB_DAMP", 0.3)), 0.0), 0.99)
        self._banks = [_Freeverb(self.sr, spread, fb, damp) for spread in (0, _STEREO_SPREAD)]
        pd = int(round(float(getattr(cfg, "REVERB_PREDELAY_MS", 0.0)) * self.sr / 1000.0))
        self._predelay = _Ring(pd) if pd > 0 else None
        self._rv_chunk = min([b.shortest for b in self._banks] + ([pd] if pd > 0 else []))
        self.channels = 2                                       # set from the device in start()
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
        self._open()
        if self._cfg.AUDIO_DEVICE is None and default_output_id() is not None:
            self._follow_stop = threading.Event()
            self._follower = threading.Thread(target=self._follow_system_output, name="audio-follow", daemon=True)
            self._follower.start()

    def stop(self):
        if self._follower is not None:
            self._follow_stop.set()
            self._follower.join()
            self._follower = None
        self._close()

    def _follow_system_output(self):
        """Watcher thread: when the macOS output changes, move the stream there (fade out, reopen, fade in)."""
        import sounddevice as sd
        poll = float(getattr(self._cfg, "AUDIO_FOLLOW_POLL_S", 0.5))
        current = default_output_id()
        while not self._follow_stop.wait(poll):
            dev = default_output_id()
            if dev is None or (dev == current and self._stream is not None):
                continue
            current = dev
            self._close()
            sd._terminate()             # PortAudio only sees new devices and the new default after a re-init
            sd._initialize()
            try:
                self._open(fade_in=True)
                self.error = None
                print(f"audio output -> {self.output_name}", flush=True)
            except RuntimeError as e:   # retried on the next poll, since _stream stays None
                self.error = str(e).splitlines()[0]

    def _open(self, fade_in=False):
        import sounddevice as sd
        try:
            self._fade = 0.0 if fade_in else 1.0
            with self._lock:
                self._target.fading = False
            dev = sd.query_devices(self._cfg.AUDIO_DEVICE, "output")
            self.channels = 2 if dev["max_output_channels"] >= 2 else 1
            self._stream = sd.OutputStream(
                samplerate=self.sr, blocksize=self.block, channels=self.channels,
                dtype="float32", callback=self._cb, device=self._cfg.AUDIO_DEVICE)
            self._stream.start()
            self.output_name = dev["name"]
        except Exception as e:  # PortAudioError, ValueError for bad device, ...
            self._stream = None
            try:
                devices = str(sd.query_devices())
            except Exception as e2:
                devices = f"(could not list devices: {e2})"
            msg = " ".join(str(e).split())
            raise RuntimeError(f"Audio output failed: {msg}\n{devices}") from e

    def _close(self):
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

    def render_offline(self, n_blocks, stereo=False):
        """Same render code as the device, no device. Mono (mid = (L+R)/2) unless stereo=True."""
        out = np.concatenate([self._render_block() for _ in range(n_blocks)])
        return out if stereo else out.mean(axis=1, dtype=np.float32)

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
            y = self._render_block()
            if self.channels == 2:
                outdata[:] = y
            else:
                outdata[:, 0] = y.mean(axis=1)
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

    def _reverb_wet(self, x):
        """Mono input -> (n, 2) wet signal: pre-delay, then the left and right Freeverb banks."""
        m = self._rv_chunk
        if self._predelay is not None:
            pd, src = self._predelay, x
            x = np.empty_like(src)
            for i in range(0, len(src), m):
                seg = src[i:i + m]
                x[i:i + len(seg)] = pd.read(len(seg))
                pd.write(seg)
        wet = np.stack([b.process(x, m) for b in self._banks], axis=1) * _WET_GAIN
        if np.abs(x).max() < _FLUSH and all(b.quiet() for b in self._banks) and (
                self._predelay is None or np.abs(self._predelay.buf).max() < _FLUSH):
            for b in self._banks:                               # tail gone: true silence, no denormals
                b.clear()
            if self._predelay is not None:
                self._predelay.clear()
            wet[:] = 0.0
        return wet

    def _reverb(self, dry, wet_target, ramp):
        """dry (n, 2) -> dry*(1 - wet/2) + reverb*wet, wet mix smoothed and ramped across the block."""
        self._wet_s += (min(max(wet_target, 0.0), 1.0) - self._wet_s) * self._gain_k
        wet = (self._wet_prev + (self._wet_s - self._wet_prev) * ramp)[:, None]
        self._wet_prev = self._wet_s
        return dry * (1.0 - 0.5 * wet) + self._reverb_wet(dry.mean(axis=1)) * wet

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

        mix = np.zeros((n, 2))
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
                sig = np.repeat((sig * _CENTRE_W)[:, None], 2, axis=1)
                for j, r in enumerate(self._ratios):
                    ph = v.ph_u[j] + (w0 * r) * self._n
                    s_ = _shape(self._wave, ph)
                    if xfade:
                        s_ = s_ * (1.0 - ramp) + _shape(new_wave, ph) * ramp
                    sig += _SIDE_W * s_[:, None] * self._pan[j]
                    v.ph_u[j] = (v.ph_u[j] + w0 * r * n) % TWO_PI
            else:
                sig = sig[:, None]
            mix += env[:, None] * sig
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
        mix /= np.maximum(np.maximum(1.0, norm), levels)[:, None]

        # cutoff: smooth toward target, one-pole x2 (12 dB/oct)
        self._fc += (p.cutoff_hz - self._fc) * self._cut_k
        fc = min(max(self._fc, 20.0), 0.45 * self.sr)
        a = 1.0 - math.exp(-TWO_PI * fc / self.sr)
        den = [1.0, -(1.0 - a)]
        y, self._zi[0] = lfilter([a], den, mix, axis=0, zi=self._zi[0])
        y, self._zi[1] = lfilter([a], den, y, axis=0, zi=self._zi[1])
        if not self._voices:                            # flush filter state toward exact zero (no denormals)
            for z in self._zi:
                if np.abs(z).max() < _FLUSH:
                    z[:] = 0.0

        y = self._reverb(y, p.reverb, ramp)

        # gain: smoothed target, linear ramp across the block
        self._gain_s += (p.gain - self._gain_s) * self._gain_k
        g = self._gain_prev + (self._gain_s - self._gain_prev) * ramp
        self._gain_prev = self._gain_s
        y = y * g[:, None]

        if fading or self._fade < 1.0:                 # stop fade (linear)
            target = 0.0 if fading else 1.0
            step = self._stop_rate * (1 if target > self._fade else -1)
            f = np.clip(self._fade + step * k, 0.0, 1.0)
            self._fade = float(f[-1])
            y = y * f[:, None]
        y = np.clip(_soft_limit(y * self._makeup), -1.0, 1.0).astype(np.float32)
        self.out_rms = float(np.sqrt(np.mean(y * y)))
        return y
