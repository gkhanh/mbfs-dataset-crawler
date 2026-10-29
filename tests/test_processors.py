"""processors/: frame sampling (adaptive), the crop processor (vs worker-dataset's loop), the frame processor."""

import random
from dataclasses import dataclass

import numpy as np
import pytest
from helpers import FakeDetector, make_type, make_video, people_video, textured_background

import mbfs_crawler.processors as processors_mod
from mbfs_crawler.models import Detection
from mbfs_crawler.processors import (
    IDLE_SECONDS,
    Frame,
    FrameProcessor,
    FrameSampler,
    ObjectCropProcessor,
    QualityRules,
    Sample,
    VideoOpenError,
    cut_object,
)
from mbfs_crawler.sources import MediaItem, MediaType

VIDEO = MediaItem("v.mp4", MediaType.VIDEO, "clip", local=True)


def accept_all(samples):
    return [True] * len(samples)


# ============================================================================
# Frame sampling
# ============================================================================


def test_sampler_reads_one_frame_every_n_seconds(tmp_path):
    path = people_video(tmp_path / "v.mp4", seconds=3, fps=10)
    sampler = FrameSampler(every=0.2)
    frames = list(sampler.frames(path))
    assert [f.index for f in frames] == list(range(0, 30, 2))
    assert frames[1].time_s == pytest.approx(0.2) and frames[0].image.shape == (240, 320, 3)


def test_adaptive_sampling_backs_off_only_after_a_quiet_period_and_resets(tmp_path):
    path = people_video(tmp_path / "v.mp4", seconds=12, fps=10)
    sampler = FrameSampler(every=0.2, every_max=1.0)
    found_from = 9.0  # nothing found until 9 s, then something in every frame
    indices = []
    for frame in sampler.frames(path):
        indices.append(frame.index)
        sampler.feedback(frame.time_s >= found_from)
    gaps = np.diff(indices)
    quiet = [g for i, g in zip(indices, gaps, strict=False) if i / 10 < IDLE_SECONDS]
    assert set(quiet) == {2}                       # full rate during the first quiet seconds
    assert max(gaps) == 10                         # then up to every_max (1 s = 10 frames)
    assert all(g == 2 for i, g in zip(indices, gaps, strict=False) if i / 10 >= found_from)


def test_sampler_counts_decoded_and_skipped_frames(tmp_path):
    from mbfs_crawler.config import RunStats

    path = people_video(tmp_path / "v.mp4", seconds=10, fps=10)
    stats = RunStats()
    sampler = FrameSampler(every=0.2, every_max=1.0, stats=stats)
    for _ in sampler.frames(path):
        sampler.feedback(False)
    assert stats.frames_decoded == 100
    assert stats.frames_sampled + stats.frames_skipped_adaptive == 50 and stats.frames_skipped_adaptive > 20


def test_sampler_without_every_max_never_adapts(tmp_path):
    path = people_video(tmp_path / "v.mp4", seconds=5, fps=10)
    sampler = FrameSampler(every=0.2, every_max=None)
    indices = []
    for frame in sampler.frames(path):
        indices.append(frame.index)
        sampler.feedback(False)
    assert indices == list(range(0, 50, 2))


def test_unreadable_video_raises(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    with pytest.raises(VideoOpenError):
        list(FrameSampler(0.2).frames(tmp_path / "bad.mp4"))


# ============================================================================
# Crop processor
# ============================================================================


def test_cut_object_applies_quality_rules():
    from mbfs_crawler.config import RunStats

    frame = textured_background(400, 400)
    frame[50:250, 100:200] = np.random.default_rng(1).integers(0, 255, (200, 100, 3), dtype=np.uint8)
    rules = QualityRules(min_conf=0.5, min_height=100, min_sharpness=10)
    stats = RunStats()
    ok = cut_object(frame, Detection((100, 50, 200, 250), 0.9, "person", 3), "a.jpg", "src", rules, 0.1, stats, 7, 0.7)
    assert ok is not None and ok.box == (90, 30, 210, 270) and ok.track_id == 3 and ok.det_class == "person"
    assert ok.frame == 7 and ok.sharpness > 10
    assert cut_object(frame, Detection((100, 50, 200, 250), 0.3), "b.jpg", "src", rules, 0.1, stats) is None
    assert cut_object(frame, Detection((100, 50, 200, 120), 0.9), "c.jpg", "src", rules, 0.1, stats) is None
    flat = np.full((400, 400, 3), 128, np.uint8)
    assert cut_object(flat, Detection((100, 50, 200, 250), 0.9), "d.jpg", "src", rules, 0.1, stats) is None
    assert stats.rejected == {"low_conf": 1, "too_small": 1, "broken": 1}


@dataclass
class _Window:
    saved: int = 0
    window_start: float = 0.0
    best: Sample | None = None


def worker_dataset_loop(frames, track_gap, max_per_track, cut, add):
    """worker-dataset's Harvester.process_video, copied as it was: loops over every track ever seen."""
    windows = {}
    for index, time_s, detections in frames:
        for det in detections:
            window = windows.setdefault(det.track_id, _Window())
            if max_per_track and window.saved >= max_per_track:
                continue
            sample = cut(det, index, time_s)
            if sample is not None:
                if window.best is None:
                    window.window_start, window.best = time_s, sample
                elif sample.sharpness > window.best.sharpness:
                    window.best = sample
        for window in windows.values():
            if window.best and time_s - window.window_start >= track_gap:
                window.saved += add(window.best)
                window.best = None
    for window in windows.values():
        if window.best:
            add(window.best)


def scripted_cut(det, index, time_s):
    """Deterministic stand-in for cut_object: sharpness comes from the box, some detections fail quality."""
    if det.conf < 0.2:
        return None
    return Sample(np.zeros((1, 1, 3), np.uint8), f"clip_f{index:07d}_id{det.track_id}.jpg", "clip", index, time_s,
                  det.track_id, det.conf, sharpness=det.xyxy[0])


def decide(sample):
    """Deterministic keep/reject, like a duplicate filter would."""
    return hash(sample.name) % 3 != 0


@pytest.mark.parametrize("seed", range(5))
def test_active_track_bookkeeping_matches_worker_dataset_exactly(monkeypatch, ctx_factory, seed):
    rng = random.Random(seed)
    script = []
    for index in range(0, 600, 2):
        present = rng.sample(range(1, 15), rng.randint(0, 6))
        script.append((index, index / 10, [Detection((rng.random() * 100, 0, 1, 1), rng.random(), "person", tid)
                                           for tid in present]))

    expected = []
    worker_dataset_loop(script, 2.0, 3, scripted_cut,
                        lambda s: expected.append((s.name, decide(s))) or decide(s))

    class ScriptedSampler:
        def __init__(self, *args, **kwargs):
            pass

        def feedback(self, found):
            pass

        def frames(self, path, desc=None):
            for index, time_s, _ in script:
                yield Frame(index, time_s, 10.0, np.zeros((4, 4, 3), np.uint8))

    detections = {index: dets for index, _, dets in script}
    detector = FakeDetector()
    detector.track = lambda image, persist: detections[current[0]]
    current = [0]
    original_frames = ScriptedSampler.frames

    def frames_tracking_index(self, path, desc=None):
        for frame in original_frames(self, path, desc):
            current[0] = frame.index
            yield frame

    ScriptedSampler.frames = frames_tracking_index
    monkeypatch.setattr(processors_mod, "FrameSampler", ScriptedSampler)
    monkeypatch.setattr(processors_mod, "cut_object", lambda frame, det, name, key, rules, pad, stats, index, time_s:
                        scripted_cut(det, index, time_s))
    got = []

    def submit(samples):
        results = [decide(s) for s in samples]
        got.extend((s.name, r) for s, r in zip(samples, results, strict=True))
        return results

    processor = ObjectCropProcessor(ctx_factory(detector=detector), track_gap=2.0, max_per_track=3)
    assert processor.process_video(None, VIDEO, submit) is True
    assert got == expected and len(expected) > 20


def test_crop_processor_keeps_the_sharpest_crop_per_window_and_limits_per_track(tmp_path, ctx_factory):
    path = people_video(tmp_path / "v.mp4", seconds=6, fps=10, objects=2)
    submitted = []
    processor = ObjectCropProcessor(ctx_factory(), every=0.2, track_gap=1.0, max_per_track=2)
    processor.process_video(path, VIDEO, lambda s: submitted.extend(s) or [True] * len(s))
    per_track = {}
    for sample in submitted:
        per_track.setdefault(sample.track_id, []).append(sample)
    assert set(per_track) == {1, 2} and all(len(v) == 2 for v in per_track.values())
    assert all(s.name.startswith("clip_f") and "_id" in s.name for s in submitted)
    assert all(s.image.shape[0] > 120 for s in submitted)  # padded crops


def test_crop_processor_stops_a_video_when_nothing_new_is_found(tmp_path, ctx_factory):
    path = people_video(tmp_path / "v.mp4", seconds=20, fps=10, objects=2)
    calls = []
    processor = ObjectCropProcessor(ctx_factory(), every=0.2, track_gap=0.4, max_per_track=0, stop_after=4)
    finished = processor.process_video(path, VIDEO, lambda s: calls.extend(s) or [False] * len(s))
    assert finished is False and len(calls) <= 6


def test_unconfirmed_detections_are_not_cropped_but_keep_full_rate(tmp_path, ctx_factory):
    path = people_video(tmp_path / "v.mp4", seconds=4, fps=10, objects=1)
    detector = FakeDetector(untracked_frames=100)  # the tracker never confirms anything
    processor = ObjectCropProcessor(ctx_factory(detector=detector), every=0.2, every_max=1.0)
    submitted = []
    processor.process_video(path, VIDEO, lambda s: submitted.extend(s) or accept_all(s))
    assert submitted == [] and detector.calls == 20  # every 0.2 s: the sampler saw "something in view"


def test_crop_processor_on_still_images_in_one_batch(ctx_factory):
    detector = FakeDetector()
    ctx = ctx_factory(detector=detector)
    image = textured_background(320, 240)
    from helpers import draw_object

    draw_object(image, 20, 40, 40, 120)
    draw_object(image, 200, 40, 40, 120, pattern=1)
    items = [(MediaItem("a.jpg", MediaType.IMAGE, "a"), image), (MediaItem("b.jpg", MediaType.IMAGE, "b"),
                                                                 textured_background(320, 240, seed=5))]
    results = ObjectCropProcessor(ctx).process_images(items)
    assert [[s.name for s in r] for r in results] == [["a_p0.jpg", "a_p1.jpg"], []]
    assert detector.calls == 2 and ObjectCropProcessor(ctx).process_images([]) == []


def test_crop_processor_requires_a_detector(ctx_factory):
    ctx = ctx_factory()
    ctx.detector_impl = None
    ctx.dataset_type = make_type(processor={"type": "frame"}, detector=None)
    with pytest.raises(ValueError, match="needs a detector"):
        ObjectCropProcessor(ctx).process_images([(VIDEO, np.zeros((8, 8, 3), np.uint8))])


# ============================================================================
# Frame processor
# ============================================================================


def bird_ctx(ctx_factory, detector=None, **changes):
    dataset_type = make_type("birds", detector={"model": "models/fake.pt", "classes": ["bird"]},
                             processor={"type": "frame"},
                             quality={"min_conf": 0.1, "min_sharpness": 1, "reject_broken": False}, **changes)
    return ctx_factory(dataset_type=dataset_type, detector=detector or FakeDetector(cls_name="bird"))


def sky_video(path, static_seconds=3, moving_seconds=3, fps=10):
    """Static empty sky, then a bird moving across it."""
    sky = textured_background(640, 360, seed=1)
    frames = [sky.copy() for _ in range(static_seconds * fps)]
    for i in range(moving_seconds * fps):
        frame = sky.copy()
        frame[100:112, 50 + i * 15:66 + i * 15] = 255
        frames.append(frame)
    return make_video(path, frames, fps)


def test_motion_check_skips_static_frames_before_the_detector(tmp_path, ctx_factory):
    path = sky_video(tmp_path / "sky.mp4")
    detector = FakeDetector(cls_name="bird")
    ctx = bird_ctx(ctx_factory, detector)
    samples = []
    FrameProcessor(ctx, every=0.5, min_change=0.0005).process_video(
        path, VIDEO, lambda s: samples.extend(s) or accept_all(s))
    assert ctx.stats.frames_skipped_motion >= 4
    assert detector.calls <= ctx.stats.frames_sampled - ctx.stats.frames_skipped_motion
    assert samples and all(s.image.shape == (360, 640, 3) for s in samples)  # full frames, full size


def test_gate_keeps_frames_with_targets_and_some_background(tmp_path, ctx_factory):
    path = sky_video(tmp_path / "sky.mp4", static_seconds=4, moving_seconds=2)
    samples = []
    FrameProcessor(bird_ctx(ctx_factory), every=0.2, background_ratio=0.25).process_video(
        path, VIDEO, lambda s: samples.extend(s) or accept_all(s))
    with_bird = [s for s in samples if s.det_class == "bird"]
    background = [s for s in samples if s.det_class == ""]
    assert len(with_bird) == 10 and len(background) == 5  # every 4th of the 20 empty frames
    assert with_bird[0].name == "clip_f0000040.jpg" and with_bird[0].box == (0, 0, 640, 360)


def test_frames_with_too_few_targets_are_dropped_not_kept_as_background(ctx_factory):
    processor = FrameProcessor(bird_ctx(ctx_factory), min_count=2, background_ratio=1.0)
    one_bird = [Detection((1, 1, 5, 5), 0.9, "bird")]
    assert processor._sample(textured_background(64, 64), "x.jpg", "k", one_bird) is None
    assert processor._sample(textured_background(64, 64), "y.jpg", "k", []).det_class == ""


def test_nothing_is_labelled_by_the_tool(tmp_path, ctx_factory):
    from mbfs_crawler.store import DatasetStore

    assert not hasattr(FrameProcessor(bird_ctx(ctx_factory)), "prelabel")
    store = DatasetStore(tmp_path / "ds")
    assert not (tmp_path / "ds" / "labels").exists()  # collecting only: labelling happens in another tool
    store.close()


def test_frame_processor_without_detector_keeps_every_sampled_frame(tmp_path, ctx_factory):
    path = sky_video(tmp_path / "sky.mp4", static_seconds=1, moving_seconds=1)
    ctx = bird_ctx(ctx_factory)
    ctx.detector_impl = None
    ctx.dataset_type = make_type(processor={"type": "frame"}, detector=None,
                               quality={"min_sharpness": 0, "reject_broken": False})
    samples = []
    FrameProcessor(ctx, every=0.5).process_video(path, VIDEO, lambda s: samples.extend(s) or accept_all(s))
    assert len(samples) == 4
    images = FrameProcessor(ctx).process_images([(VIDEO, textured_background(32, 32))])
    assert len(images[0]) == 1 and images[0][0].name == "clip_p0.jpg"


def test_frame_processor_stops_when_everything_is_rejected(tmp_path, ctx_factory):
    path = sky_video(tmp_path / "sky.mp4", static_seconds=0, moving_seconds=4)
    calls = []
    finished = FrameProcessor(bird_ctx(ctx_factory), every=0.1, stop_after=3).process_video(
        path, VIDEO, lambda s: calls.extend(s) or [False] * len(s))
    assert finished is False and len(calls) == 3
