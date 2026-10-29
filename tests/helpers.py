"""Test building blocks: fakes for the neural networks, synthetic images and videos, a small test dataset type.

Unit and integration tests never touch the network or model weights, so `pytest` runs offline in seconds.
"""

from pathlib import Path

import cv2
import numpy as np

from mbfs_crawler.datasets import DatasetType
from mbfs_crawler.models import Detection, Detector, Encoder

FIXTURES = Path(__file__).parent / "fixtures"


class FakeDetector(Detector):
    """Finds bright rectangles (pixels > 200) in an image. Track ids follow left-to-right order (ids 1, 2, ...)."""

    registry_name = "fake"

    def __init__(self, cls_name: str = "person", conf: float = 0.9, untracked_frames: int = 0):
        self.cls_name = cls_name
        self.conf = conf
        self.untracked_frames = untracked_frames  # the first N track() calls of a video return track_id -1
        self.calls = 0
        self._tracked_calls = 0

    @property
    def class_names(self) -> list[str]:
        return [self.cls_name]

    def _find(self, image: np.ndarray) -> list[Detection]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours((gray > 200).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted(cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 16)
        return [Detection((float(x), float(y), float(x + w), float(y + h)), self.conf, self.cls_name)
                for x, y, w, h in boxes]

    def detect(self, images, conf=None):
        self.calls += len(images)
        return [[d for d in self._find(img) if conf is None or d.conf >= conf] for img in images]

    def track(self, image, persist):
        self.calls += 1
        if not persist:
            self._tracked_calls = 0
        self._tracked_calls += 1
        untracked = self._tracked_calls <= self.untracked_frames
        return [Detection(d.xyxy, d.conf, d.cls_name, -1 if untracked else i + 1)
                for i, d in enumerate(self._find(image))]


class FakeEncoder(Encoder):
    """Unit vector of a 16x16 grayscale thumbnail: identical images -> similarity 1, different images -> lower."""

    dim = 256

    def __init__(self):
        self.calls = 0
        self.images = 0

    def encode(self, images):
        self.calls += 1
        self.images += len(images)
        out = []
        for img in images:
            gray = (img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.float32)
            h, w = gray.shape
            gray = gray[h // 5: h - h // 5, w // 5: w - w // 5]  # the object's texture, not the shared edges
            detail = gray - cv2.GaussianBlur(gray, (0, 0), 6)
            v = cv2.resize(detail, (16, 16), interpolation=cv2.INTER_AREA).ravel()
            v -= v.mean()
            norm = np.linalg.norm(v)
            out.append(v / norm if norm else np.full(self.dim, self.dim ** -0.5, np.float32))
        return np.array(out, np.float32).reshape(len(images), self.dim)


# ----------------------------------------------------------------------------
# Synthetic media
# ----------------------------------------------------------------------------

def textured_background(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Mid-grey noise: never "flat" nor "letterboxed" for the quality checks."""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 140, (height, width, 3), dtype=np.uint8)


def draw_object(frame: np.ndarray, x: int, y: int, w: int, h: int, pattern: int = 0) -> None:
    """A bright, sharp rectangle (all pixels > 200, so FakeDetector finds it) with its own coarse texture: the
    same pattern looks the same in every frame, different patterns give clearly different vectors."""
    rng = np.random.default_rng(1000 + pattern)
    blocks = rng.integers(205, 256, (max(1, h // 8), max(1, w // 8), 3), dtype=np.uint8)
    frame[y:y + h, x:x + w] = cv2.resize(blocks, (w, h), interpolation=cv2.INTER_NEAREST)


def make_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def people_video(path: Path, seconds: float = 4.0, fps: float = 10.0, objects: int = 2,
                 empty_from: float | None = None) -> Path:
    """Moving bright 'people' on a textured background; frames after `empty_from` seconds are empty."""
    frames = []
    for i in range(int(seconds * fps)):
        frame = textured_background(320, 240, seed=i)
        if empty_from is None or i / fps < empty_from:
            for k in range(objects):
                draw_object(frame, 20 + k * 150 + i % 5, 40 + i % 3, 40, 120, pattern=k)
        frames.append(frame)
    return make_video(path, frames, fps)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

TEST_TYPE = {   # one dataset type, as it would appear under its name in datasets.yaml
    "description": "people, for tests",
    "detector": {"model": "models/fake.pt", "classes": ["person"]},
    "processor": {"type": "crop", "pad": 0.1, "every": 0.2, "track_gap": 1.0, "max_per_track": 3},
    "quality": {"min_conf": 0.5, "min_height": 50, "min_sharpness": 5},
    "filters": {"duplicate": {"max_similarity": 0.98}, "reference": {"k": 3, "min_similarity": 0.5}},
}


def make_type(name: str = "testppl", **changes) -> DatasetType:
    return DatasetType.from_dict(name, {**TEST_TYPE, **changes}).validate()
