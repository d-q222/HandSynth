# Hand Synth — setup + throwaway demo (before the real build)

Companion to [`HAND_SYNTH_PRD.md`](./HAND_SYNTH_PRD.md). The PRD's Part E tells Claude Code to confirm this is done before scaffolding anything.

Goal: prove every layer works on your Mac in the smallest possible form. Three scripts, each under 30 lines. Nothing here is kept; the real project starts from the PRD in a fresh folder.

Time: ~60–90 min. Stop when script 3 makes a sound that follows your finger.

## 0. Setup (10 min)

```bash
mkdir -p ~/code/hand_synth_demo && cd ~/code/hand_synth_demo
python3 --version            # want 3.11 or 3.12; mediapipe wheels lag on the newest Python
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install mediapipe opencv-python numpy sounddevice
python -c "import mediapipe, cv2, sounddevice; print('ok')"
python -c "import sounddevice as sd; print(sd.query_devices())"   # confirm an output device exists
```

Mac gotchas:
- First camera use triggers a macOS permission prompt for whatever launched Python (Terminal, iTerm, VS Code). Run from Terminal the first time; if the window is black, check System Settings → Privacy & Security → Camera.
- If `pip install mediapipe` fails on your Python version, the fix is a 3.11 venv, not fighting the wheel.
- `cv2.imshow` only paints when `cv2.waitKey` is called. Every loop needs it.

## 1. `demo_cam.py` — webcam + landmarks + one printed coordinate (20 min)

Proves: capture, MediaPipe, mirror flip, handedness labels, landmark indexing.

```python
import cv2, mediapipe as mp

hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)                               # mirror, so labels match your real hands
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
            tip = lm.landmark[8]                              # index fingertip
            print(hd.classification[0].label, f"index tip x={tip.x:.2f} y={tip.y:.2f}")
    cv2.imshow("demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release(); cv2.destroyAllWindows()
```

Checks before moving on:
- Hold up only your left hand (your chord hand). Console must say `Left`. If it says `Left`, the mirror/handedness assumption is wrong on your setup; note it, the PRD's `MIRROR` flag handles it.
- Move the index finger up. `y` must go *down* toward 0. Image y grows downward; this is the fact `fingers_up()` depends on.
- Note: `mp.solutions` is the legacy API. It is the shortest path for a demo. The real project can use either; the wrapper hides it.

## 2. `demo_chord.py` — play one chord from math, no camera (20 min)

Proves: MIDI→Hz, chord intervals, phase-accumulator oscillators, envelope, no clicks. This is section A4 "chords from an oscillator" in runnable form.

```python
import numpy as np, sounddevice as sd

SR = 44100
def hz(midi): return 440 * 2 ** ((midi - 69) / 12)

root = 48                                   # C3
chord = [root + i for i in (0, 4, 7)]       # major triad; try (0,3,7) for minor, add 11 for maj7
dur = 2.0
t = np.arange(int(SR * dur)) / SR

sig = sum(np.sin(2 * np.pi * hz(n) * t) for n in chord) / len(chord)
env = np.ones_like(t)
fade = int(0.05 * SR)                       # 50 ms fade in/out: remove this line's effect and you will hear the click
env[:fade] = np.linspace(0, 1, fade); env[-fade:] = np.linspace(1, 0, fade)

sd.play((sig * env * 0.5).astype(np.float32), SR); sd.wait()
```

Experiments (5 min each, worth doing):
- Swap intervals to `(0,3,7)` and `(0,3,6)`; hear major → minor → diminished.
- Change `root` to `55` (G3). Same intervals, new key. That is the whole "one KEY value" idea.
- Set `fade = 1`. Hear the click. Now you know why the PRD insists on envelopes.
- Replace `np.sin(x)` with `np.sign(np.sin(x))` (square) or `2/np.pi*np.arcsin(np.sin(x))` (triangle). Pick the one you'd want to sing over.

## 3. `demo_theremin.py` — finger height → pitch, continuous (30 min)

Proves: the two-thread model. Camera loop writes a shared value; audio callback reads it and keeps phase continuous across buffers. This is the exact structure of `numpy_synth.py`, minus chords and envelopes.

```python
import cv2, mediapipe as mp, numpy as np, sounddevice as sd, threading

SR, BLOCK = 44100, 256
target_hz, gain = 220.0, 0.0
lock = threading.Lock()
phase = 0.0
cur_hz = 220.0

def callback(out, frames, time, status):
    global phase, cur_hz
    with lock: th, g = target_hz, gain
    cur_hz += (th - cur_hz) * 0.1                       # glide toward target, avoids zipper noise
    ph = phase + 2 * np.pi * cur_hz / SR * np.arange(frames)
    out[:, 0] = (0.3 * g * np.sin(ph)).astype(np.float32)
    phase = ph[-1] % (2 * np.pi)                        # phase persists across callbacks: no click

stream = sd.OutputStream(samplerate=SR, blocksize=BLOCK, channels=1, callback=callback); stream.start()
hands = mp.solutions.hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with lock:
        if res.multi_hand_landmarks:
            y = res.multi_hand_landmarks[0].landmark[8].y     # 0 top .. 1 bottom
            target_hz, gain = 220 * 2 ** (2 * (1 - y)), 1.0     # two octaves of range
        else:
            gain = 0.0
    cv2.imshow("theremin", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

stream.stop(); stream.close(); cap.release(); cv2.destroyAllWindows()
```

What to notice:
- Raise your finger, pitch rises; hand leaves frame, silence. Latency should feel near-instant.
- Comment out the `phase = ph[-1] ...` line. Buzzing/clicks at 172 Hz. That is what "never reset phase" in the PRD means.
- Remove the lock. It will probably still work. It is there because the real synth swaps a whole list of voices, not one float.

## 4. Done when

- All three scripts run from a clean `source .venv/bin/activate`.
- You can say, without looking, why the mirror flip comes before MediaPipe, why fingertip y decreases when the finger goes up, why the fade exists, and why phase carries across callbacks.
- Then: new folder, hand Claude Code the PRD, stop at step 3 of D2 and write `fingers_up()`.

## What this demo deliberately does not do

No `fingers_up()` (that stays cold for the gate), no config, no chords-from-gestures, no filter, no UI. Resist adding them here. The point of a throwaway is that you throw it away.
