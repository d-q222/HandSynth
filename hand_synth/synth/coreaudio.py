"""The macOS system output device, read straight from CoreAudio (ctypes, no extra deps).

PortAudio snapshots the device list when it initialises, so it never notices the user switching
output (AirPods connecting, picking another speaker). This is the cheap probe the synth polls to
notice; it returns None anywhere CoreAudio is unavailable, which turns following off.
"""
import ctypes
import sys

_CA_PATH = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
_SYSTEM_OBJECT = 1                      # kAudioObjectSystemObject


class _Address(ctypes.Structure):      # AudioObjectPropertyAddress
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


def _fourcc(s: str) -> int:
    return int.from_bytes(s.encode("ascii"), "big")


# kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain
_DEFAULT_OUTPUT = _Address(_fourcc("dOut"), _fourcc("glob"), 0)
_ca = None


def default_output_id() -> int | None:
    """CoreAudio ID of the current system output device, or None (not macOS / query failed)."""
    global _ca
    if sys.platform != "darwin":
        return None
    try:
        if _ca is None:
            _ca = ctypes.CDLL(_CA_PATH)
        dev, size = ctypes.c_uint32(0), ctypes.c_uint32(4)
        status = _ca.AudioObjectGetPropertyData(_SYSTEM_OBJECT, ctypes.byref(_DEFAULT_OUTPUT), 0, None,
                                                ctypes.byref(size), ctypes.byref(dev))
    except OSError:
        return None
    return dev.value if status == 0 and dev.value else None
