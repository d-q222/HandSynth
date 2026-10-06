# Hand Synth

A webcam instrument. Hold your hands up to the camera: the **left hand picks the chord**, the
**right hand shapes it**. Chords stay in the chosen key, switch without clicks, and sit low enough
to sing over. The full spec is in [`HAND_SYNTH_PRD.md`](./HAND_SYNTH_PRD.md).

## Quick start (macOS, Apple Silicon)

```bash
uv venv -p 3.12 .venv            # or: python3.12 -m venv .venv
uv pip install -p .venv/bin/python -e ".[dev]"     # or: .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m hand_synth.main
```

- Use Python **3.11 or 3.12**. `mediapipe` is pinned to `0.10.21`, because 1.0.x crashes on macOS in its
  Metal helper even with the CPU delegate.
- The hand model (`models/hand_landmarker.task`, ~7.5 MB) downloads automatically on first run.
- **Run it from Terminal the first time.** macOS asks for camera permission for the app that
  launched Python. If you get `cannot read from camera`, allow it in System Settings → Privacy &
  Security → Camera, then rerun. The app keeps retrying for `CAMERA_WAIT_S` (15 s), so you can click
  Allow on the first-run prompt without restarting.
- Sound only, no camera: `.venv/bin/python -m hand_synth.demo_keys`. Keys `1`–`7` pick a degree, `0` is silence,
  `m` flips major/minor, `s` cycles sus2/sus4, `j`/`k`/`l` toggle add7/add9/add6, `←`/`→` change key,
  `↑`/`↓` change gain, `[`/`]` change cutoff, `w` changes the waveform.

Options:
- `--key G` sets the startup key.
- `--ui minimal` sets the starting overlay mode.
- `--camera 1` picks another camera, and `--camera clip.mp4` plays a recorded video instead.

## How to play

| Hand | Gesture | Effect |
|---|---|---|
| **Left** (chords) | open hand | I (tonic) |
| | index | ii |
| | index + middle | iii |
| | index + middle + ring | IV |
| | four fingers (no thumb) | V |
| | thumb | vi |
| | thumb + index | vii° |
| | fist | silence (fades out) |
| | roll the hand > ~45° (back < ~25°) | flip major ↔ minor (dim → minor) |
| **Right** (expression) | open | plain chord |
| | index | sus2 |
| | index + middle | sus4 |
| | index + middle + ring | add7 |
| | four fingers | add9 |
| | thumb | add6 |
| | raise / lower the hand | louder / quieter |
| | open / close the thumb–index pinch | brighter / darker |

Click a key along the top bar to change the key; `←`/`→` do the same.
Other keys:
- `h` hides the clutter, cycling through three overlay modes:
  - **full**: everything.
  - **minimal**: the key bar, the hand skeletons and the chord name.
  - **clean**: only the chord name over the video.

  Set the startup mode with `UI_MODE` in config or `--ui`.
- `w` cycles the waveform (sine / triangle / saw).
- `r` records the current landmarks as a test fixture.
- `q` or `Esc` quits, with a fade-out.

Other behaviour:
- **Unknown gestures** keep the current chord, so moving between two shapes never dips to silence.
- **Losing the left hand** keeps the chord sounding; make a fist to stop it.
- **When the right hand leaves**, the modification resets to none, but volume and brightness hold.
- **Every gesture mapping lives in [`hand_synth/config.py`](hand_synth/config.py)** (`FINGER_MAP`, `MOD_MAP`), along with every other tunable. To redesign the instrument, edit that file.

## The hand-written parts: stand-ins until they exist

`fingers.py::fingers_up()` and `features.py` are written by hand, with no AI (see PRD D1–D3).
Until they are written, the app runs on temporary stand-ins in `hand_synth/standins.py` and shows a
**STAND-IN** badge in the corner.

- `standins.py` opens with a spoiler warning, so don't read it before writing your own. Its finger
  detector deliberately uses a different method (3D joint angles) from the one the PRD suggests.
- Each function switches over **automatically** once yours stops raising `NotImplementedError`,
  one function at a time. On startup the console prints `manual` or `stand-in` for each one.
- If your function raises an error during play, that one call falls back to the stand-in
  instead of crashing the instrument.
- Your gate tests:
  ```bash
  .venv/bin/python -m pytest tests/test_fingers.py tests/test_features.py
  ```
  These skip while the stubs are unwritten. On a failure they report only the fixture name.

## Fixtures

The fixtures live under `fixtures/landmarks/`, one JSON file per hand:
`{"label", "landmarks": [[x,y,z]×21], "score", "source", "expected": {"fingers", "roll_deg"}}`.

| Folder | Contents |
|---|---|
| `synthetic/` | Generated hands: all 32 finger states × both hands × roll/scale/noise variations, plus roll poses. Regenerate with `python scripts/make_synthetic_fixtures.py`. |
| `real/` | Landmarks from MediaPipe's public sample images, labels checked by eye. |
| `recorded/` | What `r` saves from your own camera. The filename and `expected.fingers` come from the stand-in's reading at that moment, so **check each one and rename it if the stand-in misread your hand.** |

Recording ~15 poses per hand plus a few rolled poses is the best test of your `fingers_up()`. It's also the starting dataset for the later classifier (`gesture.ClassifierSource`).

## Coordinates

`tracking.py` returns aspect-corrected landmarks:
- `y` is MediaPipe's normalized y (0 at the top, 1 at the bottom).
- `x` and `z` are multiplied by width/height, so `x` runs 0..1.78 on a 16:9 camera.

This makes angles and distances real: a hand rolled 45° reads as 45°. On raw normalized
coordinates a 16:9 frame squashes that to ~29°, and the 35° flip threshold would need ~60° of wrist rotation.

## Layout

```
hand_synth/
  config.py        all tunables (data only)
  main.py          camera loop; Instrument = per-frame mapping (testable)
  tracking.py      MediaPipe HandLandmarker wrapper -> HandsFrame
  fingers.py       MANUAL  fingers_up()           (stub until written)
  features.py      MANUAL  roll/pinch/wrist/vector (stub until written)
  standins.py      temporary stand-ins for the two above
  stabilize.py     Hold, RollZone, EMA
  gesture.py       finger tuple -> degree; ClassifierSource seam
  music.py         key/scale/chord math -> MIDI, Hz, names
  synth/base.py    Synth interface (start/stop/set_chord/set_params)
  synth/numpy_synth.py  sounddevice backend
  ui.py            overlay + clickable key bar
  demo_keys.py     keyboard-only synth demo
tests/             pytest; no camera or speakers needed
```

To swap the audio backend (pyo, SuperCollider), implement the four methods in `synth/base.py` and
change the one `NumpySynth(cfg)` line in `main.py` (PRD C6).

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite covers:
- music theory for all 12 keys × 7 degrees × mods and flips;
- the stabilizers;
- offline synth rendering, asserting no clicks on chord, waveform and gain changes and true silence after release;
- the stand-ins against every fixture;
- the PRD's acceptance checklist, run as `tests/test_instrument.py` on synthetic hands.

## Acceptance checklist (live, PRD C8)

- [ ] Hold up only your **right** hand: the overlay labels it `Right`. If the labels are swapped, change `MIRROR` or the `CHORD_HAND`/`EXPR_HAND` settings in config.
- [ ] Select key **G**. Left open hand plays G; left index alone plays Am.
- [ ] Roll the left hand ~45° to get A (flipped); back upright gives Am.
- [ ] Right index + middle gives Asus4; right index + middle + ring gives Am7.
- [ ] Closing the right pinch darkens the sound; lowering the right hand makes it quieter.
- [ ] A left fist gives silence, with a fade.
- [ ] Click **D**: an open hand now plays D major.
- [ ] Hold a chord and sing over it for 30 seconds without hearing a click.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot read from camera` | Grant camera permission to your terminal app (see Quick start), or try `--camera 1`. |
| `Audio output failed` | The message lists your devices. Set `AUDIO_DEVICE` in config to one of them. |
| Red **XRUN** badge | The audio buffer is underrunning. Raise `BLOCK_SIZE` to 512. |
| Chords flicker | Raise `HOLD_FRAMES`. If they feel sluggish, lower it. |
| Major/minor flips too easily or not at all | Tune `ROLL_MINOR_DEG` and `ROLL_DEADBAND_DEG`. The HUD shows the live roll angle. |
| Volume range feels off | Tune `WRIST_Y_QUIET` / `WRIST_Y_LOUD`, the wrist heights for silent and full. |
