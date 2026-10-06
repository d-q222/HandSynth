"""MediaPipe wrapper → HandsFrame.

Landmark coordinates are *aspect-corrected*: y is MediaPipe's normalized y (0 top .. 1 bottom),
x and z are MediaPipe's normalized x/z multiplied by width/height. Distances and angles are
then isotropic, so a hand rolled 45° measures 45° on a 16:9 camera (raw normalized coords
would report ~29°). x spans 0..width/height (≈1.78 for 16:9).

The frame passed to `process` must already be mirrored when config.MIRROR is set; MediaPipe's
handedness assumes a selfie (mirrored) image, so labels then match the user's real hands.
"""
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

REPO_ROOT = Path(__file__).resolve().parent.parent

Landmarks = list[tuple[float, float, float]]


@dataclass
class Hand:
    label: str                  # "Left" | "Right" (as the user experiences it)
    landmarks: Landmarks        # 21 aspect-corrected points, see module docstring
    score: float


@dataclass
class HandsFrame:
    left: Hand | None
    right: Hand | None
    aspect: float = 16 / 9      # width / height of the processed frame

    def hands(self) -> list[Hand]:
        return [h for h in (self.left, self.right) if h is not None]


def ensure_model(model_path: str, url: str) -> Path:
    """Return the model path, downloading it on first run."""
    path = Path(model_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


class Tracker:
    def __init__(self, model_path: str, model_url: str, max_hands=2, min_conf=0.6, min_tracking_conf=0.5,
                 handedness="position"):
        self.handedness = handedness
        self._last_single = None        # label of the last lone hand (midline dead band)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path, model_url))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, frame_bgr) -> HandsFrame:
        h, w = frame_bgr.shape[:2]
        aspect = w / h
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)   # VIDEO mode needs strictly increasing ms
        self._last_ts = ts
        res = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        found = []
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            pts = [(p.x * aspect, p.y, p.z * aspect) for p in lms]
            found.append(Hand(label=handed[0].category_name, landmarks=pts, score=handed[0].score))
        hf = _assign(found, aspect, self.handedness, self._last_single)
        if len(found) == 1:
            self._last_single = found[0].label
        return hf

    def close(self):
        self._landmarker.close()


MIDLINE_DEADBAND = 0.06    # fraction of frame width around the centre where a lone hand keeps its label


def _assign(found: list[Hand], aspect: float, handedness: str = "position", last_single=None) -> HandsFrame:
    """Place hands into left/right slots.

    "position" (default): in a mirrored view the user's left hand is on image-left, so two hands are
    labelled by wrist x and a single hand by which half of the frame it is in (keeping `last_single`
    inside a small dead band around the midline, so a lone hand can't flicker roles). Independent of
    MediaPipe's handedness classifier, whose labels depend on mirroring and hand orientation.
    "mediapipe": trust MediaPipe's labels; if both hands get the same label, break the tie by wrist x.
    """
    if handedness == "position":
        if len(found) == 2:
            a, b = sorted(found, key=lambda hd: hd.landmarks[0][0])
            a.label, b.label = "Left", "Right"
        elif len(found) == 1:
            x = found[0].landmarks[0][0] / aspect - 0.5
            if last_single and abs(x) < MIDLINE_DEADBAND:
                found[0].label = last_single
            else:
                found[0].label = "Left" if x < 0 else "Right"
    elif len(found) == 2 and found[0].label == found[1].label:
        a, b = sorted(found, key=lambda hd: hd.landmarks[0][0])
        a.label, b.label = "Left", "Right"
    left = next((hd for hd in found if hd.label == "Left"), None)
    right = next((hd for hd in found if hd.label == "Right"), None)
    return HandsFrame(left=left, right=right, aspect=aspect)
