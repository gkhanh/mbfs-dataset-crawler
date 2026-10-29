"""models.py (YOLO detector with a fake Ultralytics model) and downloads.py."""

import sys
import types

import cv2
import numpy as np
import pytest
from helpers import make_type

import mbfs_crawler.sources as sources_mod
from mbfs_crawler.config import RunContext
from mbfs_crawler.downloads import ImageDownloader, VideoDownloader, scan_cache
from mbfs_crawler.models import Detection, UltralyticsDetector, merge_boxes, resolve_model, tile_boxes
from mbfs_crawler.sources import WIKIMEDIA_HEADERS, MediaItem, MediaType
from mbfs_crawler.utils import url_key


class T:
    """Minimal stand-in for a torch tensor."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values

    def int(self):
        return T([int(v) for v in self.values])


class Boxes:
    def __init__(self, boxes, ids):
        self.xyxy = T([b[:4] for b in boxes])
        self.conf = T([b[4] for b in boxes])
        self.cls = T([b[5] for b in boxes])
        self.id = None if ids is None else T(ids)

    def __len__(self):
        return len(self.conf.values)


class FakeResult:
    def __init__(self, boxes, ids=None):
        self.names = {0: "person", 1: "car", 2: "bird"}
        self.boxes = Boxes(boxes, ids)


class FakeYOLO:
    calls: list = []

    def __init__(self, path, task="detect"):
        self.names = {0: "person", 1: "car", 2: "bird"}
        self.path = path

    def predict(self, images, **options):
        FakeYOLO.calls.append(("predict", len(images), options))
        return [FakeResult([[10, 10, 50, 50, 0.9, 2]]) for _ in images]

    def track(self, image, persist, tracker, **options):
        FakeYOLO.calls.append(("track", persist, options))
        if persist:
            return [FakeResult([[1, 2, 3, 4, 0.8, 0]], ids=[7])]
        return [FakeResult([[1, 2, 3, 4, 0.8, 0]])]  # tracker not confirmed yet: raw boxes, no ids


@pytest.fixture
def yolo(monkeypatch):
    FakeYOLO.calls = []
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", types.SimpleNamespace(DEFAULT_CFG_DICT={"quantize": None}))
    return FakeYOLO


def detector(tmp_path, device="cpu", model="models/m.pt", **options) -> UltralyticsDetector:
    from mbfs_crawler.config import Paths

    ctx = RunContext(Paths(tmp_path), make_type(), device=device)
    return UltralyticsDetector(ctx, model, **options)


def test_class_filter_is_passed_to_the_model(yolo, tmp_path):
    det = detector(tmp_path, classes=["bird", "car"], imgsz=320)
    (found,) = det.detect([np.zeros((64, 64, 3), np.uint8)], conf=0.4)
    kind, count, options = yolo.calls[-1]
    assert options["classes"] == [1, 2] and options["conf"] == 0.4 and options["imgsz"] == 320
    assert found == [Detection((10, 10, 50, 50), 0.9, "bird")]
    assert det.class_names == ["person", "car", "bird"] and det.ctx.stats.detector_calls == 1


def test_unknown_class_names_the_model_classes(yolo, tmp_path):
    with pytest.raises(ValueError, match="has no class"):
        detector(tmp_path, classes=["dragon"]).detect([np.zeros((8, 8, 3), np.uint8)])


@pytest.mark.parametrize("device,model,half", [("0", "m.pt", True), ("cpu", "m.pt", False), ("mps", "m.pt", False),
                                               ("0", "m.onnx", False)])
def test_fp16_only_on_nvidia_with_pytorch_models(yolo, tmp_path, device, model, half):
    det = detector(tmp_path, device=device, model=model)
    det.detect([np.zeros((8, 8, 3), np.uint8)])
    options = yolo.calls[-1][2]
    assert det.half is half and ("quantize" in options) is half and "half" not in options


def test_track_marks_unconfirmed_detections(yolo, tmp_path):
    det = detector(tmp_path)
    first = det.track(np.zeros((8, 8, 3), np.uint8), persist=False)
    later = det.track(np.zeros((8, 8, 3), np.uint8), persist=True)
    assert [d.track_id for d in first] == [-1] and [d.track_id for d in later] == [7]
    assert yolo.calls[0][1] is False and "conf" not in yolo.calls[0][2]  # tracker keeps Ultralytics' default


def test_tiled_detection_runs_whole_image_and_tiles_in_one_batch(yolo, tmp_path):
    det = detector(tmp_path, tile=True, imgsz=640)
    (found,) = det.detect([np.zeros((1080, 1920, 3), np.uint8)])
    kind, count, _ = yolo.calls[-1]
    assert kind == "predict" and count == 1 + len(tile_boxes(1920, 1080, 640, 0.2))
    assert all(d.cls_name == "bird" for d in found)
    assert any(d.xyxy[0] > 600 for d in found)  # boxes from later tiles are shifted into frame coordinates


def test_tiles_cover_the_whole_image():
    width, height, size = 1920, 1080, 640
    tiles = tile_boxes(width, height, size, 0.2)
    covered = np.zeros((height, width), bool)
    for x1, y1, x2, y2 in tiles:
        assert x2 - x1 <= size and y2 - y1 <= size
        covered[y1:y2, x1:x2] = True
    assert covered.all()
    assert tile_boxes(640, 480, 640, 0.2) == []


def test_merge_boxes_keeps_the_best_of_overlapping_boxes_per_class():
    dets = [Detection((0, 0, 10, 10), 0.9, "bird"), Detection((1, 1, 11, 11), 0.5, "bird"),
            Detection((1, 1, 11, 11), 0.6, "car"), Detection((50, 50, 60, 60), 0.3, "bird")]
    merged = merge_boxes(dets)
    assert [(d.cls_name, d.conf) for d in merged] == [("bird", 0.9), ("car", 0.6), ("bird", 0.3)]


def test_model_paths_are_relative_to_home(tmp_path):
    assert resolve_model("models/a.pt", tmp_path) == tmp_path / "models" / "a.pt"
    assert resolve_model(str(tmp_path / "b.onnx"), tmp_path / "x") == tmp_path / "b.onnx"


def image_item(url: str) -> MediaItem:
    return MediaItem(url, MediaType.IMAGE, url_key(url))


def png_bytes() -> bytes:
    return cv2.imencode(".png", np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8))[1].tobytes()


def test_image_is_downloaded_once_and_then_served_from_the_cache(web, ctx_factory):
    (web.root / "bird.png").write_bytes(png_bytes())
    ctx = ctx_factory()
    path = ImageDownloader(ctx).fetch(image_item(web.url("bird.png")))
    assert path.suffix == ".png" and path.parent == ctx.paths.images and ctx.stats.downloaded_bytes > 0
    assert ImageDownloader(ctx).fetch(image_item(web.url("bird.png"))) == path  # a new run: cache, no request
    assert len(web.requests) == 1


def test_web_pages_and_non_images_are_rejected(web, ctx_factory):
    (web.root / "page.html").write_text("<html>not an image</html>", encoding="utf-8")
    (web.root / "fake.jpg").write_bytes(b"\x00" * 100)
    downloader = ImageDownloader(ctx_factory())
    assert downloader.fetch(image_item(web.url("page.html"))) is None
    assert downloader.fetch(image_item(web.url("fake.jpg"))) is None
    assert downloader.fetch(image_item(web.url("missing.jpg"))) is None


def test_extension_comes_from_the_content_type_when_the_link_has_none(web, ctx_factory):
    (web.root / "photo.png").write_bytes(png_bytes())
    item = MediaItem(web.url("photo.png").replace(".png", ""), MediaType.IMAGE, "noext")
    (web.root / "photo").write_bytes(png_bytes())  # served as application/octet-stream
    path = ImageDownloader(ctx_factory()).fetch(item)
    assert path.name == "noext.jpg"


def test_fetch_all_keeps_order_with_parallel_downloads(web, ctx_factory):
    items = []
    for i in range(6):
        (web.root / f"{i}.png").write_bytes(png_bytes())
        items.append(image_item(web.url(f"{i}.png")))
    items.append(image_item(web.url("missing.png")))
    results = list(ImageDownloader(ctx_factory(), workers=4).fetch_all(items))
    assert [item for item, _ in results] == items
    assert [p.stem for _, p in results[:6]] == [i.key for i in items[:6]] and results[6][1] is None


def test_byte_cap_stops_huge_downloads(web, ctx_factory):
    (web.root / "big.png").write_bytes(png_bytes() + b"\x00" * 200_000)
    downloader = ImageDownloader(ctx_factory(), max_bytes=1000)
    assert downloader.fetch(image_item(web.url("big.png"))) is None  # truncated data does not decode
    assert ctx_factory().stats.downloaded_bytes == 0


def test_local_items_are_never_downloaded(ctx_factory, tmp_path):
    item = MediaItem(str(tmp_path / "a.jpg"), MediaType.IMAGE, "a", local=True)
    assert ImageDownloader(ctx_factory()).fetch(item) == tmp_path / "a.jpg"
    assert VideoDownloader(ctx_factory()).fetch(
        MediaItem(str(tmp_path / "v.mp4"), MediaType.VIDEO, "v", local=True)) == tmp_path / "v.mp4"


def test_cache_scan_ignores_partial_downloads(tmp_path):
    for name in ("a.mp4", "b.mp4.part", "c.webm.ytdl", "d.jpg"):
        (tmp_path / name).touch()
    assert set(scan_cache(tmp_path)) == {"a", "d"}
    assert scan_cache(tmp_path / "missing") == {}


# ============================================================================
# Videos (yt-dlp faked)
# ============================================================================


class FakeYoutubeDL:
    attempts: list = []
    fail_first = 0

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=True):
        FakeYoutubeDL.attempts.append(self.options)
        if len(FakeYoutubeDL.attempts) <= FakeYoutubeDL.fail_first:
            raise FakeDownloadError("HTTP Error 403")
        path = self.options["outtmpl"].replace("%(ext)s", "mp4")
        open(path, "wb").write(b"video")
        return {"entries": [None, {"requested_downloads": [{"filepath": path}]}]}


class FakeDownloadError(Exception):
    pass


@pytest.fixture
def fake_ytdlp(monkeypatch):
    FakeYoutubeDL.attempts = []
    FakeYoutubeDL.fail_first = 0
    module = types.SimpleNamespace(YoutubeDL=FakeYoutubeDL, utils=types.SimpleNamespace(
        DownloadError=FakeDownloadError, download_range_func=lambda chapters, ranges: ("ranges", ranges)))
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    return FakeYoutubeDL


def video_item(key="youtube_abc"):
    return MediaItem("https://www.youtube.com/watch?v=abc", MediaType.VIDEO, key, title="t")


def test_video_options_limit_resolution_audio_and_minutes(fake_ytdlp, ctx_factory):
    ctx = ctx_factory()
    options = VideoDownloader(ctx, max_res=720, max_minutes=2).options(video_item())
    assert options["format"] == "bv*/b" and options["playlist_items"] == "1"
    assert options["outtmpl"].endswith("youtube_abc.%(ext)s")
    assert options["download_ranges"] == ("ranges", [(0, 120)])
    assert options["ffmpeg_location"]  # shipped by imageio-ffmpeg
    # ffmpeg writes to the terminal by itself, so it is told to print errors only
    assert options["external_downloader_args"]["ffmpeg"] == ["-loglevel", "error", "-nostats"]


def test_video_falls_back_to_hls_and_is_cached(fake_ytdlp, ctx_factory):
    fake_ytdlp.fail_first = 1
    ctx = ctx_factory()
    ctx.paths.videos.mkdir(parents=True)
    path = VideoDownloader(ctx, max_res=720).fetch(video_item())
    assert path.name == "youtube_abc.mp4" and ctx.stats.downloaded_bytes == 5
    assert [a["format_sort"][1] for a in fake_ytdlp.attempts] == ["vcodec:h264", "proto:m3u8"]
    assert VideoDownloader(ctx).fetch(video_item()) == path and len(fake_ytdlp.attempts) == 2


def test_video_gives_up_after_every_attempt_failed(fake_ytdlp, ctx_factory):
    fake_ytdlp.fail_first = 5
    ctx = ctx_factory()
    ctx.paths.videos.mkdir(parents=True)
    assert VideoDownloader(ctx).fetch(video_item()) is None and len(fake_ytdlp.attempts) == 2


def test_cookies_are_passed_to_ytdlp(fake_ytdlp, ctx_factory):
    ctx = ctx_factory(cookies_from_browser="firefox")
    assert VideoDownloader(ctx).options(video_item())["cookiesfrombrowser"] == ("firefox",)


# ============================================================================
# Direct video files and failures that must never stop a run
# ============================================================================


def test_direct_video_files_download_without_ytdlp(web, ctx_factory, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # yt-dlp is not involved at all
    (web.root / "clip.webm").write_bytes(b"v" * 5000)
    item = MediaItem(web.url("clip.webm"), MediaType.VIDEO, "clip_ab12cd34")
    ctx = ctx_factory()
    path = VideoDownloader(ctx).fetch(item)
    assert path == ctx.paths.videos / "clip_ab12cd34.webm" and path.read_bytes() == b"v" * 5000
    assert list(ctx.paths.videos.glob("*.part")) == [] and ctx.stats.downloaded_bytes == 5000
    assert VideoDownloader(ctx).fetch(item) == path and len(web.requests) == 1  # cached: no second request


class StreamResponse:
    def __init__(self, status=200, headers=None, body=b"video"):
        self.status_code, self.ok, self.headers, self.body = status, status < 400, headers or {}, body
        self.url = "https://upload.wikimedia.org/x.webm"

    def iter_content(self, size):
        yield self.body

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class RecordingSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_wikimedia_files_are_requested_politely_and_retried_on_429(ctx_factory, monkeypatch):
    monkeypatch.setattr(sources_mod.time, "sleep", lambda seconds: None)
    session = RecordingSession(StreamResponse(429, {"Retry-After": "2"}), StreamResponse(429), StreamResponse())
    ctx = ctx_factory()
    ctx.http = lambda: session
    item = MediaItem("https://upload.wikimedia.org/wikipedia/commons/a/ab/Gulls.webm", MediaType.VIDEO, "gulls_1")
    path = VideoDownloader(ctx).fetch(item)
    assert path.read_bytes() == b"video" and len(session.calls) == 3
    assert all(kwargs["headers"] == WIKIMEDIA_HEADERS and kwargs["stream"] for _, kwargs in session.calls)
    other = MediaItem("https://example.org/a.mp4", MediaType.VIDEO, "other_1")  # no special headers elsewhere
    session.responses = [StreamResponse()]
    VideoDownloader(ctx).fetch(other)
    assert session.calls[-1][1]["headers"] is None


def test_a_video_that_stays_rate_limited_is_skipped_with_one_warning(ctx_factory, monkeypatch, log_lines):
    monkeypatch.setattr(sources_mod.time, "sleep", lambda seconds: None)
    session = RecordingSession(*[StreamResponse(429)] * 4)
    ctx = ctx_factory()
    ctx.http = lambda: session
    item = MediaItem("https://upload.wikimedia.org/a/Gulls.webm", MediaType.VIDEO, "gulls_1", title="Gulls.webm")
    assert VideoDownloader(ctx).fetch(item) is None and len(session.calls) == 4
    assert [m for level, m in log_lines if level == "WARNING"] == [
        "! https://upload.wikimedia.org/a/Gulls.webm: still busy after 4 tries (rate limit); "
        "try a smaller --limit later"]


def test_direct_download_over_the_size_cap_is_dropped(ctx_factory, monkeypatch):
    import mbfs_crawler.downloads as downloads_mod

    monkeypatch.setattr(downloads_mod, "MAX_VIDEO_BYTES", 10)
    session = RecordingSession(StreamResponse(body=b"x" * 100))
    ctx = ctx_factory()
    ctx.http = lambda: session
    item = MediaItem("https://example.org/big.mp4", MediaType.VIDEO, "big_1")
    assert VideoDownloader(ctx).fetch(item) is None
    assert list(ctx.paths.videos.glob("*")) == []  # nothing half-written is left behind


def test_an_unexpected_ytdlp_error_is_one_warning_and_the_run_goes_on(fake_ytdlp, ctx_factory, monkeypatch,
                                                                      log_lines):
    """The real crash: archive.org items without a player make yt-dlp raise a plain ValueError."""
    def broken(self, url, download=True):
        raise ValueError("opening play-av tag not found")

    monkeypatch.setattr(fake_ytdlp, "extract_info", broken)
    ctx = ctx_factory()
    ctx.paths.videos.mkdir(parents=True)
    item = MediaItem("https://archive.org/details/x", MediaType.VIDEO, "archiveorg_x", title="Some item")
    assert VideoDownloader(ctx).fetch(item) is None
    warnings = [m for level, m in log_lines if level == "WARNING"]
    assert warnings == ["! cannot download Some item: opening play-av tag not found"]  # once, not once per attempt
    assert sum(1 for level, m in log_lines if level == "DEBUG" and "attempt" in m) == 2  # both tries are in debug


def test_even_a_crash_inside_the_downloader_does_not_escape(ctx_factory, monkeypatch, log_lines):
    downloader = VideoDownloader(ctx_factory())
    monkeypatch.setattr(downloader, "_download", lambda item: (_ for _ in ()).throw(MemoryError("out of memory")))
    assert downloader.fetch(MediaItem("https://x.org/watch?v=1", MediaType.VIDEO, "k_1")) is None
    images = ImageDownloader(ctx_factory())
    monkeypatch.setattr(images, "_download", lambda item: (_ for _ in ()).throw(RuntimeError("weird")))
    assert images.fetch(MediaItem("https://x.org/a.jpg", MediaType.IMAGE, "a_1")) is None
    assert sum(level == "WARNING" for level, _ in log_lines) == 2


def test_the_video_cache_lives_outside_the_project_folder(ctx_factory):
    ctx = ctx_factory()
    assert VideoDownloader(ctx).folder == ctx.paths.home.parent / "downloads" / "videos"
    assert ImageDownloader(ctx).folder == ctx.paths.home.parent / "downloads" / "images"
