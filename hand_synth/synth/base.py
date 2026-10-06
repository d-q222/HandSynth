"""The synth interface. Everything above synth/ talks only to this.

A pyo or SuperCollider backend implements these four methods and nothing else changes.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SynthParams:
    gain: float = 0.5          # 0..1 master gain (already mapped through config.GAIN_RANGE)
    cutoff_hz: float = 2000.0  # low-pass cutoff
    waveform: str = "triangle"  # "sine" | "triangle" | "saw"


class Synth(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...                      # fade out then close, no pop
    def set_chord(self, freqs: list[float]) -> None: ...   # [] = silence
    def set_params(self, p: SynthParams) -> None: ...
