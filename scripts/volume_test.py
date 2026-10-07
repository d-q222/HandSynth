"""Audio check, no camera: the same synth engine plays one chord at LOUD, MEDIUM and QUIET (15 dB steps).
If these three don't sound clearly different, the problem is the audio output, not hand tracking.
Run: .venv/bin/python scripts/volume_test.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hand_synth import config as cfg  # noqa: E402
from hand_synth.main import voicing_options, volume_gain  # noqa: E402
from hand_synth.music import chord_voicing, midi_to_hz  # noqa: E402
from hand_synth.synth import SynthParams  # noqa: E402
from hand_synth.synth.numpy_synth import NumpySynth  # noqa: E402

import sounddevice as sd  # noqa: E402

print("Output device:", sd.query_devices(cfg.AUDIO_DEVICE, "output")["name"])
synth = NumpySynth(cfg)
synth.start()
chord = [midi_to_hz(n) for n in chord_voicing("C", 1, False, "none", None, **voicing_options())]
try:
    for label, pos in (("LOUD   (slider 100%)", 1.0), ("MEDIUM (slider 58%, -15 dB)", 1 - 15 / cfg.VOLUME_RANGE_DB),
                       ("QUIET  (slider 17%, -30 dB)", 1 - 30 / cfg.VOLUME_RANGE_DB)):
        synth.set_params(SynthParams(gain=volume_gain(pos), cutoff_hz=2500.0, waveform="triangle", reverb=0.2))
        synth.set_chord(chord)
        print(f"  {label} ...", flush=True)
        time.sleep(2.5)
        print(f"      measured output {20 * __import__('math').log10(synth.out_rms + 1e-9):.1f} dBFS")
        synth.set_chord([])
        time.sleep(0.8)
finally:
    synth.stop()
print("Done. Were the three clearly different in loudness?")
