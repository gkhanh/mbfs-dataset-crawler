"""Turning one video or image into samples: quality rules, frame sampling, crops, full frames."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mbfs_crawler.config import PROCESSORS
from mbfs_crawler.models import Detection
from mbfs_crawler.sources import MediaItem
from mbfs_crawler.utils import (
    Box,
    changed_fraction,
    is_broken,
    pad_box,
    progress,
    sharpness,
    thumbnail,
    to_gray,
    video_path,
)

# ==============================================================================================================
# Sample (a candidate image for the dataset) and the Processor interface.
# ==============================================================================================================

@dataclass
class Sample:
    image: np.ndarray
    name: str                  # file name it will be saved under
    source_key: str            # key of the video / image it came from
    frame: int = 0             # frame number in the video (0 for still images)
    time_s: float = 0.0
    track_id: int = -1         # tracker ID (-1 for still images and full frames)
    conf: float = 0.0
    box: Box = (0, 0, 0, 0)    # crop area (with padding) in the original frame
    sharpness: float = 0.0
    det_class: str = ""
    vector: np.ndarray | None = None  # image embedding, computed once and reused by every filter
    max_sim: float = 0.0       # similarity to the closest image already in the dataset


# Hands samples to the filter chain and the store; returns which ones were kept, in order.
Submit = Callable[[list[Sample]], list[bool]]


class Processor(ABC):
    registry_name = ""

    def __init__(self, ctx):
        self.ctx = ctx

    @abstractmethod
    def process_video(self, path: Path, item: MediaItem, submit: Submit) -> bool:
        """Process a whole video. Returns False if it stopped early because nothing new was being found."""

    @abstractmethod
    def process_images(self, images: list[tuple[MediaItem, np.ndarray]]) -> list[list[Sample]]:
        """Samples for each still image, in the same order (images are processed in batches)."""


# ==============================================================================================================
# Quality rules: cheap checks (no model) that run inside the processors, before any embedding is computed.
#
# They run before the "sharpest crop per time window" choice, so a broken crop can never be picked as the best one.
# ==============================================================================================================

@dataclass(frozen=True)
class QualityRules:
    min_conf: float = 0.25        # minimum detector confidence
    min_height: int = 0           # minimum object height in pixels (crop processor)
    min_sharpness: float = 0.0    # Laplacian variance; 0 = no blur check
    reject_broken: bool = True    # flat frames, letterbox bars, strips cut off by the frame edge

    def check_detection(self, conf: float, height: float) -> str | None:
        """Reason to reject a detection before cropping, or None."""
        if conf < self.min_conf:
            return "low_conf"
        if height < self.min_height:
            return "too_small"
        return None

    def check_image(self, gray: np.ndarray, touches_edge: bool) -> tuple[str | None, float]:
        """(reason to reject or None, sharpness). `gray` is computed once by the caller and shared."""
        if self.reject_broken and is_broken(gray, touches_edge):
            return "broken", 0.0
        score = sharpness(gray)
        if score < self.min_sharpness:
            return "blurry", score
        return None, score

    def check_values(self, conf: float, height: float, score: float) -> str | None:
        """Re-check stored values (crops.csv) without reading any image, for `clean`."""
        return self.check_detection(conf, height) or ("blurry" if score < self.min_sharpness else None)


# ==============================================================================================================
# Reading video frames: one frame every `every` seconds, decoding only those frames.
#
# Adaptive sampling: after `IDLE_SECONDS` without any target, each further empty frame doubles the interval up to
# `every_max`; the first frame with a target resets it. Empty stretches of a long CCTV video then cost a fraction of
# the detector calls, while busy scenes (and short detector misses) keep the full rate. every_max = every (or None)
# turns it off.
# ==============================================================================================================

IDLE_SECONDS = 2.0  # back off only after this long without a target


class VideoOpenError(RuntimeError):
    """The video file cannot be opened."""


@dataclass
class Frame:
    index: int        # frame number in the video
    time_s: float
    fps: float
    image: np.ndarray


class FrameSampler:
    def __init__(self, every: float, every_max: float | None = None, stats=None):
        self.every = every
        self.every_max = every_max if every_max and every_max > every else every
        self.stats = stats
        self._step = 1
        self._base = 1
        self._max = 1
        self._time = 0.0                  # time of the last frame handed out
        self._idle_since: float | None = None

    def feedback(self, found: bool) -> None:
        """Tell the sampler whether the last frame had a target."""
        if found:
            self._step, self._idle_since = self._base, None
            return
        if self._idle_since is None:
            self._idle_since = self._time
        if self._time - self._idle_since >= IDLE_SECONDS:
            self._step = min(self._step * 2, self._max)

    def frames(self, path: Path, desc: str | None = None) -> Iterator[Frame]:
        cap = cv2.VideoCapture(video_path(path))
        if not cap.isOpened():
            raise VideoOpenError(f"cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._base = self._step = max(1, round(fps * self.every))
        self._max = max(self._base, round(fps * self.every_max))
        self._idle_since = None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        index, next_index, analysed = -1, 0, 0
        try:
            with progress(total=total, desc=desc or path.stem, unit="frame", leave=False) as bar:
                while cap.grab():  # grab() skips the colour conversion of frames that are not analysed
                    index += 1
                    bar.update()
                    if index < next_index:
                        continue
                    ok, image = cap.retrieve()
                    if not ok:
                        break
                    analysed += 1
                    self._time = index / fps
                    yield Frame(index, self._time, fps, image)
                    next_index = index + self._step
        finally:
            cap.release()  # released before anyone deletes the file (Windows locks open files)
            if self.stats is not None and index >= 0:
                self.stats.frames_decoded += index + 1
                self.stats.frames_sampled += analysed
                self.stats.frames_skipped_adaptive += max(0, index // self._base + 1 - analysed)


# ==============================================================================================================
# Crop processor: detect objects (tracked across video frames) and save one crop per object, for labelling what
# is inside it (hands and heads on a person, plates on a vehicle, ...).
#
# Avoiding near-identical crops of the same object:
#   1. Each object is tracked through the video (ByteTrack). Per `track_gap` seconds only its sharpest crop is kept,
#      and at most `max_per_track` crops per object per video.
#   2. The pipeline then compares every crop with everything already in the dataset (duplicate filter).
#   3. `stop_after` consecutive rejected crops stop the video early: a long static camera has nothing new left.
# ==============================================================================================================

@dataclass
class _TrackWindow:
    """The sharpest crop of one tracked object in its current time window."""

    saved: int = 0
    window_start: float = 0.0
    best: Sample | None = None


def cut_object(frame: np.ndarray, det: Detection, name: str, source_key: str, rules: QualityRules, pad: float,
               stats=None, frame_idx: int = 0, time_s: float = 0.0) -> Sample | None:
    """Crop one object with padding. None if it fails a quality rule (confidence, size, broken, blurry)."""
    reason = rules.check_detection(det.conf, det.xyxy[3] - det.xyxy[1])
    if reason is None:
        frame_h, frame_w = frame.shape[:2]
        box = pad_box(det.xyxy, pad, frame_w, frame_h)
        image = frame[box[1]:box[3], box[0]:box[2]].copy()
        if image.size == 0:
            return None
        touches_edge = box[0] == 0 or box[1] == 0 or box[2] == frame_w or box[3] == frame_h
        reason, score = rules.check_image(to_gray(image), touches_edge)  # one gray conversion for both checks
        if reason is None:
            return Sample(image, name, source_key, frame_idx, time_s, det.track_id, det.conf, box, score,
                          det.cls_name)
    if stats is not None:
        stats.rejected[reason] += 1
    return None


@PROCESSORS.register("crop")
class ObjectCropProcessor(Processor):
    def __init__(self, ctx, pad: float = 0.1, every: float = 0.2, every_max: float | None = None,
                 track_gap: float = 2.0, max_per_track: int = 5, stop_after: int = 0):
        super().__init__(ctx)
        self.pad = pad
        self.every = every
        self.every_max = every_max
        self.track_gap = track_gap
        self.max_per_track = max_per_track
        self.stop_after = stop_after

    def _detector(self):
        detector = self.ctx.detector()
        if detector is None:
            raise ValueError("the crop processor needs a detector (datasets.yaml: detector: {model: ...})")
        return detector

    def process_video(self, path: Path, item: MediaItem, submit: Submit) -> bool:
        detector, rules, stats = self._detector(), self.ctx.quality, self.ctx.stats
        sampler = FrameSampler(self.every, self.every_max, stats)
        windows: dict[int, _TrackWindow] = {}   # every object seen in this video
        first_seen: dict[int, int] = {}         # flush order = order in which objects first appeared
        pending: dict[int, _TrackWindow] = {}   # only windows holding a crop: the per-frame loop stays small
        rejected_in_row, persist = 0, False
        with closing(sampler.frames(path, desc=item.key)) as frames:
            for frame in frames:
                detections = detector.track(frame.image, persist=persist)
                persist = True
                sampler.feedback(bool(detections))
                for det in detections:
                    if det.track_id < 0:  # not confirmed by the tracker yet
                        continue
                    window = windows.get(det.track_id)
                    if window is None:
                        window = windows[det.track_id] = _TrackWindow()
                        first_seen[det.track_id] = len(first_seen)
                    if self.max_per_track and window.saved >= self.max_per_track:
                        continue
                    name = f"{item.key}_f{frame.index:07d}_id{det.track_id}.jpg"
                    sample = cut_object(frame.image, det, name, item.key, rules, self.pad, stats,
                                        frame.index, frame.time_s)
                    if sample is not None:
                        _offer(window, sample)
                        pending[det.track_id] = window
                # Windows whose time is up hand over their sharpest crop.
                due = sorted((tid for tid, w in pending.items() if frame.time_s - w.window_start >= self.track_gap),
                             key=first_seen.__getitem__)
                if not due:
                    continue
                flushed = [pending.pop(tid) for tid in due]
                for window, kept in zip(flushed, submit([w.best for w in flushed]), strict=True):
                    window.saved += kept
                    window.best = None
                    rejected_in_row = 0 if kept else rejected_in_row + 1
                if self.stop_after and rejected_in_row >= self.stop_after:
                    break
        leftovers = [pending[tid].best for tid in sorted(pending, key=first_seen.__getitem__)]
        if leftovers:
            submit(leftovers)
        return not (self.stop_after and rejected_in_row >= self.stop_after)

    def process_images(self, images: list[tuple[MediaItem, np.ndarray]]) -> list[list[Sample]]:
        if not images:
            return []
        rules, stats = self.ctx.quality, self.ctx.stats
        found = self._detector().detect([image for _, image in images], conf=rules.min_conf)
        results = []
        for (item, image), detections in zip(images, found, strict=True):
            samples = [cut_object(image, det, f"{item.key}_p{i}.jpg", item.key, rules, self.pad, stats)
                       for i, det in enumerate(detections)]
            results.append([s for s in samples if s is not None])
        return results


def _offer(window: _TrackWindow, sample: Sample) -> None:
    """Keep the sharper crop of this object's current time window."""
    if window.best is None:
        window.window_start = sample.time_s
        window.best = sample
    elif sample.sharpness > window.best.sharpness:
        window.best = sample


# ==============================================================================================================
# Frame processor: keep whole frames, for models that run on the full camera image (e.g. distant birds).
#
#   * Motion check: a frame in which almost no pixel changed noticeably since the last analysed frame is skipped
#     before the detector runs (compared on a 320-px wide copy, so a distant bird moving still counts as a change).
#     Fixed cameras watching an empty sky skip most frames.
#   * Gate (optional, needs a detector): keep frames with at least `min_count` targets.
#   * Background frames: `background_ratio` of the frames with no target at all are kept as negatives.
# ==============================================================================================================

MOTION_WIDTH = 320  # width of the copies compared by the motion check


@PROCESSORS.register("frame")
class FrameProcessor(Processor):
    def __init__(self, ctx, every: float = 1.0, every_max: float | None = None, min_change: float = 0.0,
                 min_count: int = 1, background_ratio: float = 0.0, stop_after: int = 0):
        super().__init__(ctx)
        self.every = every
        self.every_max = every_max
        self.min_change = min_change
        self.min_count = min_count
        self.background_ratio = background_ratio
        self.stop_after = stop_after
        self._empty_frames = 0

    def process_video(self, path: Path, item: MediaItem, submit: Submit) -> bool:
        detector, stats = self.ctx.detector(), self.ctx.stats
        sampler = FrameSampler(self.every, self.every_max, stats)
        last_thumb, rejected_in_row = None, 0
        with closing(sampler.frames(path, desc=item.key)) as frames:
            for frame in frames:
                if self.min_change:
                    thumb = thumbnail(frame.image, MOTION_WIDTH)
                    if last_thumb is not None and changed_fraction(thumb, last_thumb) < self.min_change:
                        stats.frames_skipped_motion += 1
                        sampler.feedback(False)
                        continue
                    last_thumb = thumb
                detections = None
                if detector is not None:
                    detections = detector.detect([frame.image], conf=self.ctx.quality.min_conf)[0]
                sampler.feedback(detections is None or bool(detections))
                sample = self._sample(frame.image, f"{item.key}_f{frame.index:07d}.jpg", item.key, detections,
                                      frame.index, frame.time_s)
                if sample is None:
                    continue
                rejected_in_row = 0 if submit([sample])[0] else rejected_in_row + 1
                if self.stop_after and rejected_in_row >= self.stop_after:
                    return False
        return True

    def process_images(self, images: list[tuple[MediaItem, np.ndarray]]) -> list[list[Sample]]:
        if not images:
            return []
        detector = self.ctx.detector()
        found = (detector.detect([image for _, image in images], conf=self.ctx.quality.min_conf)
                 if detector is not None else [None] * len(images))
        results = []
        for (item, image), detections in zip(images, found, strict=True):
            sample = self._sample(image, f"{item.key}_p0.jpg", item.key, detections)
            results.append([sample] if sample is not None else [])
        return results

    def _sample(self, image: np.ndarray, name: str, source_key: str, detections: list[Detection] | None,
                frame_idx: int = 0, time_s: float = 0.0) -> Sample | None:
        if detections is not None and len(detections) < self.min_count:
            if detections or not self._keep_background():
                return None
        reason, score = self.ctx.quality.check_image(to_gray(image), touches_edge=False)
        if reason:
            self.ctx.stats.rejected[reason] += 1
            return None
        height, width = image.shape[:2]
        detections = detections or []
        best = max(detections, key=lambda d: d.conf, default=None)
        return Sample(image, name, source_key, frame_idx, time_s, conf=best.conf if best else 0.0,
                      box=(0, 0, width, height), sharpness=score, det_class=best.cls_name if best else "")

    def _keep_background(self) -> bool:
        """Keep every n-th frame without targets (deterministic, unlike random sampling)."""
        if not self.background_ratio:
            return False
        self._empty_frames += 1
        return self._empty_frames % max(1, round(1 / self.background_ratio)) == 0
