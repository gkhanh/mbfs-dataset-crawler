"""Neural networks and vectors: YOLO detectors, the DINOv2 image encoder, and the vector index."""

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mbfs_crawler.config import DETECTORS
from mbfs_crawler.utils import is_cuda, torch_device

# ==============================================================================================================
# Detector interface: find objects in images, and follow them across video frames.
# ==============================================================================================================

@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]  # box in pixels
    conf: float
    cls_name: str = ""
    track_id: int = -1  # set by track(); -1 for still images


class Detector(ABC):
    registry_name = ""

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Classes the model knows, in ID order."""

    @abstractmethod
    def detect(self, images: list[np.ndarray], conf: float | None = None) -> list[list[Detection]]:
        """Detect objects in several images at once (batched)."""

    @abstractmethod
    def track(self, image: np.ndarray, persist: bool) -> list[Detection]:
        """Detect and track in the next video frame. persist=False starts a new tracker (new video).

        Detections the tracker has not confirmed yet come back with track_id -1.
        """


# ==============================================================================================================
# Any Ultralytics YOLO model (.pt or .onnx): the internal models in models/, or yolov8n.pt (COCO, 80 classes).
#
# `classes` keeps only those classes (filtered inside the model, which is cheaper than filtering afterwards).
# `tile` adds overlapping crops at the model's input size, so objects a few pixels wide (distant birds) are found.
# ==============================================================================================================

@DETECTORS.register("ultralytics")
class UltralyticsDetector(Detector):
    def __init__(self, ctx, model: str, classes: tuple[str, ...] | list[str] = (), imgsz: int = 640,
                 conf: float | None = None, tile: bool = False, tile_overlap: float = 0.2):
        self.ctx = ctx
        self.model_path = resolve_model(model, ctx.paths.home)
        self.classes = tuple(classes)
        self.imgsz = imgsz
        self.conf = conf
        self.tile = tile
        self.tile_overlap = tile_overlap
        self._model = None
        self._class_ids: list[int] | None = None

    @property
    def model(self):
        return self._load()

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            # Known names like yolov8n.pt are downloaded by Ultralytics to this path on first use.
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(self.model_path), task="detect")
            names = model.names
            missing = [c for c in self.classes if c not in names.values()]
            if missing:
                raise ValueError(f"{self.model_path.name} has no class {missing}; its classes are "
                                 f"{list(names.values())}")
            self._class_ids = [i for i, n in names.items() if n in self.classes] if self.classes else None
            self._model = model
        return self._model

    @property
    def class_names(self) -> list[str]:
        names = self.model.names
        return [names[i] for i in sorted(names)]

    @property
    def half(self) -> bool:
        """FP16 roughly doubles speed on NVIDIA GPUs; it is not used on CPU, Apple GPUs or ONNX models."""
        return self.ctx.half and is_cuda(self.ctx.compute_device()) and self.model_path.suffix == ".pt"

    def _options(self, conf: float | None) -> dict:
        self._load()  # the class filter is known once the model is loaded
        options = {"imgsz": self.imgsz, "device": self.ctx.compute_device(), "verbose": False}
        if self.half:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            # Ultralytics 8.4 renamed `half` to `quantize`; support both.
            options.update({"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True})
        if self._class_ids is not None:
            options["classes"] = self._class_ids
        conf = self.conf if conf is None else conf
        if conf is not None:
            options["conf"] = conf
        return options

    def detect(self, images: list[np.ndarray], conf: float | None = None) -> list[list[Detection]]:
        if not images:
            return []
        if self.tile:
            return [self._detect_tiled(image, conf) for image in images]
        results = self.model.predict(list(images), **self._options(conf))
        self.ctx.stats.detector_calls += len(images)
        return [self._detections(result) for result in results]

    def track(self, image: np.ndarray, persist: bool) -> list[Detection]:
        result = self.model.track(image, persist=persist, tracker="bytetrack.yaml", **self._options(None))[0]
        self.ctx.stats.detector_calls += 1
        # Without confirmed tracks, Ultralytics leaves the raw detections in place (no ids): return them with
        # track_id -1. Processors skip them, but they still tell adaptive sampling that something is in view.
        return self._detections(result, tracked=result.boxes.id is not None)

    def _detections(self, result, tracked: bool = False) -> list[Detection]:
        boxes = result.boxes
        names = result.names
        ids = boxes.id.int().tolist() if tracked else [-1] * len(boxes)
        return [Detection(tuple(xyxy), conf, names[int(cls)], tid) for xyxy, conf, cls, tid in
                zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), ids, strict=True)]

    def _detect_tiled(self, image: np.ndarray, conf: float | None) -> list[Detection]:
        """Whole image + overlapping tiles in one batch, then overlapping boxes merged (NMS)."""
        height, width = image.shape[:2]
        tiles = tile_boxes(width, height, self.imgsz, self.tile_overlap)
        # Slices of a frame are not contiguous in memory; Ultralytics' resize can crash on them, so copy.
        crops = [image] + [np.ascontiguousarray(image[y1:y2, x1:x2]) for x1, y1, x2, y2 in tiles]
        results = self.model.predict(crops, **self._options(conf))
        self.ctx.stats.detector_calls += len(crops)
        found = self._detections(results[0])
        for (x1, y1, _, _), result in zip(tiles, results[1:], strict=True):
            found += [Detection((a + x1, b + y1, c + x1, d + y1), det.conf, det.cls_name)
                      for det in self._detections(result) for a, b, c, d in [det.xyxy]]
        return merge_boxes(found)


def resolve_model(model: str, home: Path) -> Path:
    path = Path(model).expanduser()
    return path if path.is_absolute() else home / path


def tile_boxes(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping size x size tiles covering the image; none if the image already fits in one."""
    if width <= size and height <= size:
        return []
    step = max(1, int(size * (1 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= size:
            return [0]
        points = list(range(0, length - size, step))
        return points + [length - size]

    return [(x, y, min(x + size, width), min(y + size, height)) for y in starts(height) for x in starts(width)]


def merge_boxes(detections: list[Detection], iou: float = 0.5) -> list[Detection]:
    """Non-maximum suppression per class: keep the most confident of overlapping boxes."""
    kept: list[Detection] = []
    for name in {d.cls_name for d in detections}:
        group = [d for d in detections if d.cls_name == name]
        boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in (d.xyxy for d in group)]
        keep = cv2.dnn.NMSBoxes(boxes, [d.conf for d in group], 0.0, iou)
        kept += [group[i] for i in np.array(keep).flatten()]
    return sorted(kept, key=lambda d: -d.conf)


# ==============================================================================================================
# DINOv2-small image vectors: the same person in the same pose gives a dot product close to 1.
# ==============================================================================================================

class Encoder(ABC):
    dim: int

    @abstractmethod
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """(len(images), dim) unit vectors, one per BGR image."""


class DinoV2Encoder(Encoder):
    dim = 384
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = torch_device(device)
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            import torch

            warnings.filterwarnings("ignore", message="xFormers is not available")
            # Weights are downloaded once to the torch hub cache. trust_repo avoids an interactive prompt.
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False, trust_repo=True)
            self._model = model.eval().to(self.device)
            self._mean = torch.tensor(self._MEAN, device=self.device).view(1, 3, 1, 1)
            self._std = torch.tensor(self._STD, device=self.device).view(1, 3, 1, 1)
        return self._model

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), np.float32)
        import torch

        model = self._load()
        out = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = np.stack([cv2.cvtColor(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA),
                                               cv2.COLOR_BGR2RGB) for img in images[start:start + self.batch_size]])
                x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float().div(255)
                vectors = model((x - self._mean) / self._std).float().cpu().numpy()
                out.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


# ==============================================================================================================
# Vectors of every image in the dataset, stored in an append-only file (row i = row i of crops.csv).
#
# Saving writes only the new rows. (worker-dataset rewrote the whole embeddings.npy after every source, so the total
# written grew with the square of the dataset size.) Loading is one read; dropping the rows of an interrupted source
# is a file truncate. An old embeddings.npy is converted once, then kept as embeddings.npy.bak.
#
# Similarity search is an exact matrix-vector product, about 1 ms per 10k images: simple and fast enough for
# datasets up to roughly a million images.
# ==============================================================================================================

class VectorIndex:
    def __init__(self, path: Path | None = None, dim: int = 384, legacy: Path | None = None):
        self.path = path
        self.dim = dim
        self.buf = np.zeros((0, dim), np.float32)
        self.n = 0
        self._saved = 0  # rows already in the file
        if path is None:
            return
        if not path.exists() and legacy is not None and legacy.exists():
            self._import_legacy(legacy)
        if path.exists():
            data = np.fromfile(path, dtype=np.float32)
            rows = data.size // dim
            self.buf = data[: rows * dim].reshape(rows, dim)
            self.n = self._saved = rows
            if data.size != rows * dim:  # a half-written last row after a crash
                self._truncate_file(rows)

    @property
    def vectors(self) -> np.ndarray:
        return self.buf[: self.n]

    def add(self, vector: np.ndarray) -> None:
        if self.n == len(self.buf):  # full: double the capacity (amortized O(1) per add)
            self.buf = np.concatenate([self.buf, np.zeros((max(1024, self.n), self.dim), np.float32)])
        self.buf[self.n] = vector
        self.n += 1

    def max_similarity(self, vector: np.ndarray) -> float:
        """Highest similarity (0-1) between `vector` and the stored vectors; 0 if there are none."""
        return float((self.vectors @ vector).max()) if self.n else 0.0

    def knn(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(similarities, row indices) of the k most similar stored vectors, best first."""
        sims = self.vectors @ vector
        if k < len(sims):
            top = np.argpartition(-sims, k)[:k]  # O(n) instead of sorting everything
        else:
            top = np.arange(len(sims))
        top = top[np.argsort(-sims[top])]
        return sims[top], top

    def flush(self) -> None:
        """Append the rows added since the last flush."""
        if self.path is None or self.n <= self._saved:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            f.write(np.ascontiguousarray(self.buf[self._saved: self.n], np.float32).tobytes())
        self._saved = self.n

    def truncate(self, n: int) -> None:
        """Keep only the first `n` rows (used to drop the rows of an unfinished source)."""
        if n >= self.n:
            return
        self.n = n
        if self._saved > n:
            self._truncate_file(n)

    def keep(self, mask: np.ndarray) -> None:
        """Keep only the rows where `mask` is True (rows removed by `clean`); rewrites the file atomically."""
        self.buf = np.ascontiguousarray(self.vectors[mask], np.float32)
        self.n = len(self.buf)
        if self.path is not None:
            tmp = self.path.with_name(self.path.name + ".tmp")
            self.buf.tofile(tmp)
            os.replace(tmp, self.path)
            self._saved = self.n

    def _truncate_file(self, rows: int) -> None:
        with open(self.path, "r+b") as f:
            f.truncate(rows * self.dim * 4)
        self._saved = rows

    def _import_legacy(self, legacy: Path) -> None:
        vectors = np.load(legacy).astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"{legacy} holds vectors of size {vectors.shape[1:]}, expected {self.dim}")
        tmp = self.path.with_name(self.path.name + ".tmp")
        vectors.tofile(tmp)
        os.replace(tmp, self.path)
        os.replace(legacy, legacy.with_name(legacy.name + ".bak"))
