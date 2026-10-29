"""utils.py, config.py (paths, registry) and datasets.py (datasets.yaml)."""

import os
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import mbfs_crawler.utils as utils_mod
from mbfs_crawler.config import (
    FILTERS,
    PROCESSORS,
    SOURCES,
    Paths,
    Registry,
    RegistryError,
)
from mbfs_crawler.datasets import ConfigError, DatasetType, load_type, load_types
from mbfs_crawler.utils import (
    changed_fraction,
    is_broken,
    is_image_url,
    local_key,
    log,
    pad_box,
    read_image,
    setup,
    sharpness,
    short_error,
    short_hash,
    slug,
    thumbnail,
    to_gray,
    url_key,
    url_suffix,
    video_key,
    video_path,
    warning_count,
    write_jpeg,
)

# ============================================================================
# Naming
# ============================================================================


def test_slug_keeps_safe_characters_only():
    assert slug("Món ăn / bếp #1") == "M-n-n-b-p-1"
    assert slug("") == "x"
    assert len(slug("a" * 100, limit=10)) == 10


@pytest.mark.parametrize("name", ["con", "CON", "nul", "COM1", "lpt9", "aux", "PRN"])
def test_slug_avoids_windows_reserved_names(name):
    assert slug(name).upper() not in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT9"}
    assert slug(name).startswith(name)


def test_slug_keeps_names_that_only_start_like_reserved_ones():
    assert slug("console") == "console"
    assert slug("com10") == "com10"


@pytest.mark.parametrize("text", ['a<b>c:"d"/e\\f|g?h*i', "tab\tnew\nline", "trailing dots..."])
def test_slug_output_is_a_valid_file_name_everywhere(text):
    name = slug(text)
    assert not set(name) & set('<>:"/\\|?*\t\n') and not name.endswith((".", " "))


def test_local_key_is_stable_and_depends_on_path(tmp_path):
    a, b = tmp_path / "a" / "clip.mp4", tmp_path / "b" / "clip.mp4"
    assert local_key(a) == local_key(a)
    assert local_key(a) != local_key(b)
    assert local_key(a).startswith("clip_")


def test_url_key_is_stable_readable_and_unique():
    url = "https://example.org/photos/Bird%20Flock.JPG?size=2"
    assert url_key(url) == url_key(url)
    assert url_key(url).startswith("Bird-20Flock_")
    assert url_key(url) != url_key(url + "&x=1")
    assert len(short_hash("x")) == 8


def test_video_key_matches_worker_dataset_format():
    assert video_key("Youtube", "EdWbLUVnvQI") == "youtube_EdWbLUVnvQI"
    assert video_key("ArchiveOrg", "some item/1") == "archiveorg_some-item-1"


def test_is_image_url():
    assert is_image_url("https://x.com/photo.JPG?size=2")
    assert not is_image_url("https://www.youtube.com/watch?v=abc")
    assert url_suffix("https://x.com/a/b.webp#frag") == ".webp"


# ============================================================================
# Image helpers
# ============================================================================


def test_pad_box_adds_margin_and_stays_inside_frame():
    assert pad_box((10, 10, 110, 210), 0.1, 1000, 1000) == (0, 0, 120, 230)
    assert pad_box((900, 900, 1000, 1000), 0.5, 1000, 1000) == (850, 850, 1000, 1000)


def test_is_broken_flags_flat_and_letterboxed_images():
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
    assert not is_broken(textured, touches_edge=False)
    assert is_broken(np.full((200, 100, 3), 128, np.uint8), touches_edge=False)
    letterboxed = textured.copy()
    letterboxed[:40] = 0
    assert is_broken(letterboxed, touches_edge=False)


def test_is_broken_flags_thin_strips_only_at_frame_edge():
    rng = np.random.default_rng(0)
    strip = rng.integers(0, 255, (400, 50, 3), dtype=np.uint8)
    assert is_broken(strip, touches_edge=True)
    assert not is_broken(strip, touches_edge=False)


def test_helpers_accept_gray_so_it_is_converted_once():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (120, 80, 3), dtype=np.uint8)
    gray = to_gray(img)
    assert to_gray(gray) is gray
    assert sharpness(gray) == pytest.approx(sharpness(img))
    assert is_broken(gray, False) == is_broken(img, False)


def test_sharpness_is_comparable_across_sizes():
    rng = np.random.default_rng(2)
    img = cv2.resize(rng.integers(0, 255, (64, 64), dtype=np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    big = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    assert sharpness(big) == pytest.approx(sharpness(img), rel=0.3)
    assert sharpness(cv2.GaussianBlur(img, (15, 15), 5)) < sharpness(img) / 3


def test_thumbnail_keeps_aspect_ratio():
    assert thumbnail(np.zeros((1080, 1920, 3), np.uint8)).shape == (180, 320)
    assert thumbnail(np.zeros((1080, 1920, 3), np.uint8), 64).shape == (36, 64)


def test_changed_fraction_notices_a_small_moving_object_but_not_noise():
    rng = np.random.default_rng(4)
    sky = rng.integers(150, 160, (1080, 1920), dtype=np.uint8)
    noisy = np.clip(sky.astype(int) + rng.integers(-5, 6, sky.shape), 0, 255).astype(np.uint8)
    bird = sky.copy()
    bird[500:510, 900:912] = 40  # a 12 x 10 px bird
    assert changed_fraction(thumbnail(sky), thumbnail(noisy)) == 0
    assert changed_fraction(thumbnail(sky), thumbnail(bird)) > 0.00003  # the bird.yaml threshold


def test_read_and_write_work_with_non_ascii_paths(tmp_path):
    folder = tmp_path / "ảnh thử Khánh"
    folder.mkdir()
    image = np.random.default_rng(3).integers(0, 255, (40, 30, 3), dtype=np.uint8)
    write_jpeg(folder / "x.jpg", image)
    loaded = read_image(folder / "x.jpg")
    assert loaded is not None and loaded.shape == image.shape
    assert read_image(folder / "missing.jpg") is None


def test_video_path_is_plain_on_unix(tmp_path):
    if os.name == "nt":
        pytest.skip("Windows uses short paths")
    assert video_path(tmp_path / "ảnh.mp4") == str(tmp_path / "ảnh.mp4")


# ============================================================================
# Device
# ============================================================================


def _fake_torch(monkeypatch, cuda: bool, mps: bool):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda),
                                 backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)))
    monkeypatch.setitem(sys.modules, "torch", fake)


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "0"), (False, True, "mps"), (False, False, "cpu")])
def test_pick_device_prefers_nvidia_then_apple_then_cpu(monkeypatch, cuda, mps, expected):
    _fake_torch(monkeypatch, cuda, mps)
    assert utils_mod.pick_device("auto") == expected


def test_pick_device_keeps_an_explicit_choice(monkeypatch):
    _fake_torch(monkeypatch, True, True)
    assert utils_mod.pick_device("cpu") == "cpu"


def test_device_names():
    assert utils_mod.is_cuda("0") and utils_mod.is_cuda("cuda:1") and not utils_mod.is_cuda("mps")
    assert utils_mod.torch_device("1") == "cuda:1"
    assert utils_mod.torch_device("mps") == "mps"


def test_limit_threads_sets_environment(monkeypatch):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    utils_mod.limit_threads(None)
    assert "OMP_NUM_THREADS" not in os.environ
    utils_mod.limit_threads(2)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    cv2.setNumThreads(-1)


# ============================================================================
# Registry
# ============================================================================


def test_registry_registers_and_creates():
    registry = Registry("thing", "mbfs_crawler.filters")

    @registry.register("dummy")
    class Dummy:
        def __init__(self, value=1):
            self.value = value

    assert registry.create("dummy", value=5).value == 5
    assert Dummy.registry_name == "dummy"
    assert "dummy" in registry.names()


def test_registry_rejects_a_second_class_with_the_same_name():
    registry = Registry("thing", "mbfs_crawler.filters")
    registry.register("same")(type("A", (), {}))
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("same")(type("B", (), {}))


def test_unknown_name_lists_choices_and_suggests_the_closest():
    with pytest.raises(RegistryError, match="Did you mean 'youtube'"):
        SOURCES.get("yotube")


def test_plugins_are_discovered_from_their_packages():
    assert {"youtube", "wikimedia", "openverse", "bing_images", "google_images", "local",
            "url_list", "bilibili", "niconico"} <= set(SOURCES.names())
    assert {"crop", "frame"} <= set(PROCESSORS.names())
    assert {"duplicate", "reference"} <= set(FILTERS.names())


# ============================================================================
# Dataset types (datasets.yaml)
# ============================================================================

REPO_CONFIG = Path(__file__).resolve().parents[1] / "datasets.yaml"


def write_config(tmp_path, text: str) -> Path:
    path = tmp_path / "datasets.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_shipped_datasets_yaml_has_the_three_types_and_they_are_valid():
    types = load_types(REPO_CONFIG)
    assert list(types) == ["birds", "food_factory_workers", "vehicle"]
    for dataset_type in types.values():
        dataset_type.validate()
        assert dataset_type.description and dataset_type.search  # each type says what it searches for
    for name, dataset_type in types.items():  # `search` saves to links/<type>/found.txt; the type reads it back
        assert [Path(p) for p in dataset_type.links] == [REPO_CONFIG.parent / "links" / name / "found.txt"]
        assert Path(dataset_type.links[0]).is_file()  # shipped, so the first `run` finds no missing file
    assert types["birds"].processor_type == "frame" and types["vehicle"].processor_type == "crop"


def test_load_type_gives_one_validated_type_and_suggests_names(tmp_path):
    config = write_config(tmp_path, "birds:\n  search: [a]\n  processor: {type: frame}\n")
    assert load_type("birds", config).search == ("a",)
    with pytest.raises(ConfigError, match="Did you mean 'birds'"):
        load_type("bird", config)


def test_missing_or_broken_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_types(tmp_path / "datasets.yaml")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_types(write_config(tmp_path, "birds: [unclosed\n"))
    with pytest.raises(ConfigError, match="no dataset types"):
        load_types(write_config(tmp_path, "# only comments\n"))


@pytest.mark.parametrize("text,message", [
    ("birds:\n  serch: [a]\n", "unknown setting 'serch'. Did you mean 'search'"),
    ("birds:\n  search: a\n", "list of text"),
    ("birds:\n  filters: [duplicate]\n", "must be a set of settings"),
    ("birds: just text\n", "expected settings"),
    ("bad name!:\n  search: [a]\n", "letters, digits"),
    ("logs:\n  search: [a]\n", "folder the tool uses"),
    ("references:\n  search: [a]\n", "folder the tool uses"),
])
def test_structural_mistakes_name_the_type_and_the_problem(tmp_path, text, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_types(write_config(tmp_path, text))


@pytest.mark.parametrize("settings,message", [
    ({"processor": {"type": "crops"}}, "unknown processor"),
    ({"processor": {"type": "crop", "track_gapp": 1}}, "unknown option"),
    ({"filters": {"duplicate": {"threshold": 0.9}}}, "unknown option"),
    ({"sources": {"wikimedia": {"limt": 3}}}, "unknown option"),
    ({"sources": {"yotube": {}}}, "Did you mean 'youtube'"),
    ({"quality": {"min_conff": 0.5}}, "quality has unknown"),
    ({"download": {"max_resolution": 720}}, "download has unknown"),
    ({"detector": None}, "needs a detector"),
    ({"detector": {"classes": ["person"]}}, "needs a 'model'"),
])
def test_validation_names_the_type_and_the_wrong_setting(settings, message):
    data = {"detector": {"model": "m.pt"}, "processor": {"type": "crop"}, **settings}
    with pytest.raises(ConfigError, match=message) as error:
        DatasetType.from_dict("birds", data).validate()
    assert "type 'birds'" in str(error.value)


def test_frame_type_without_detector_is_valid_and_empty_options_are_defaults():
    dataset_type = DatasetType.from_dict("sky", {"processor": {"type": "frame"}, "filters": {"duplicate": None}})
    assert dataset_type.validate().filters == {"duplicate": {}}


def test_links_are_relative_to_the_file_and_urls_are_kept(tmp_path):
    (tmp_path / "lists").mkdir()
    (tmp_path / "lists" / "a.txt").write_text("https://x.org/v.mp4\n", encoding="utf-8")
    config = write_config(tmp_path, "t:\n  links: [lists/a.txt, https://x.org/b.mp4]\n  processor: {type: frame}\n")
    assert load_types(config)["t"].links == (str(tmp_path / "lists" / "a.txt"), "https://x.org/b.mp4")


def test_types_keep_the_order_of_the_file(tmp_path):
    config = write_config(tmp_path, "zebra:\n  processor: {type: frame}\nalpha:\n  processor: {type: frame}\n")
    assert list(load_types(config)) == ["zebra", "alpha"]


# ============================================================================
# Where results and downloads live (config.Paths)
# ============================================================================


def test_results_and_downloads_default_to_folders_next_to_the_project(monkeypatch, tmp_path):
    monkeypatch.delenv("MBFS_DATA", raising=False)
    monkeypatch.delenv("MBFS_DOWNLOADS", raising=False)
    base = tmp_path.resolve()
    paths = Paths.from_env(base / "mbfs-dataset-crawler")
    assert paths.data_root == base / "dataset"
    assert paths.dataset("bird") == base / "dataset" / "bird"
    assert paths.references("bird") == base / "dataset" / "references" / "bird"
    assert paths.links("bird") == base / "mbfs-dataset-crawler" / "links" / "bird"  # link lists live in the project
    assert paths.logs == base / "dataset" / "logs"
    assert paths.videos == base / "downloads" / "videos" and paths.images == base / "downloads" / "images"
    # the project folder only holds code, models and datasets.yaml
    outputs = [paths.data_root, paths.downloads, paths.logs, paths.videos, paths.images, paths.dataset("x"),
               paths.references("x")]
    assert not any(paths.home in p.parents or p == paths.home for p in outputs)
    assert paths.models == base / "mbfs-dataset-crawler" / "models"


def test_folders_can_be_changed_with_options_or_variables(monkeypatch, tmp_path):
    base = tmp_path.resolve()
    monkeypatch.setenv("MBFS_DATA", str(base / "from-env"))
    monkeypatch.setenv("MBFS_DOWNLOADS", str(base / "cache-env"))
    from_env = Paths.from_env(base / "project")
    assert from_env.data_root == base / "from-env" and from_env.downloads == base / "cache-env"
    option = Paths.from_env(base / "project", base / "chosen", base / "cache")  # an option beats the variable
    assert option.data_root == base / "chosen" and option.downloads == base / "cache"
    assert option.dataset("p") == base / "chosen" / "p"
    assert Paths.from_env(base / "project", "~/x").data_root == Path.home() / "x"


def test_result_dirs_are_the_folders_inside_the_data_root(tmp_path):
    paths = Paths(tmp_path / "project")
    assert paths.result_dirs() == []
    for name in ("bird", "references", "logs"):
        (paths.data_root / name).mkdir(parents=True)
    (paths.data_root / "note.txt").touch()
    assert sorted(p.name for p in paths.result_dirs()) == ["bird", "logs", "references"]


# ============================================================================
# Terminal and log file (utils.setup, --debug)
# ============================================================================


def test_terminal_stays_short_and_the_log_file_keeps_the_details(tmp_path, capsys):
    log_file = tmp_path / "logs" / "run.log"
    setup(log_file, progress=False)
    log.info("shown")
    log.debug("detail")
    log.warning("! careful")
    out = capsys.readouterr().out
    assert "shown" in out and "careful" in out and "detail" not in out
    text = log_file.read_text(encoding="utf-8")
    assert "shown" in text and "careful" in text and "detail" not in text


def test_debug_mode_shows_and_records_everything_with_time_and_thread(tmp_path, capsys):
    log_file = tmp_path / "debug.log"
    setup(log_file, debug=True, progress=False)
    log.debug("GET https://example.org: HTTP 200")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.debug("details", exc_info=True)
    assert "HTTP 200" in capsys.readouterr().out
    text = log_file.read_text(encoding="utf-8")
    assert "DEBUG" in text and "[MainThread]" in text and "HTTP 200" in text
    assert "Traceback" in text and "RuntimeError: boom" in text
    assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", text)


def test_quiet_mode_shows_only_warnings(capsys):
    setup(quiet=True, progress=False)
    log.info("hidden")
    log.warning("! shown")
    out = capsys.readouterr().out
    assert "hidden" not in out and "shown" in out


def test_warnings_are_counted_for_the_end_of_run_summary():
    setup(progress=False)
    log.warning("a")
    log.info("b")
    log.warning("c")
    assert warning_count() == 2
    setup(progress=False)
    assert warning_count() == 0


def test_messages_from_other_threads_are_logged_with_their_thread_name(tmp_path):
    import threading

    log_file = tmp_path / "t.log"
    setup(log_file, debug=True, progress=False)
    thread = threading.Thread(target=lambda: log.debug("from the download thread"), name="download")
    thread.start()
    thread.join()
    assert "[download] from the download thread" in log_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("error,expected", [
    (Exception("ERROR: [generic] Flock.webm?utm_source=commons.wikimedia: Unable to download webpage: "
               "HTTP Error 429: Too many requests (f061ab2) (caused by <HTTPError 429: Too many requests (f061ab2)>)"),
     "Unable to download webpage: HTTP Error 429: Too many requests (f061ab2)"),
    (Exception("ERROR: [generic] https://x.org/a.webm: Unable to download webpage: HTTP Error 404"),
     "Unable to download webpage: HTTP Error 404"),
    (Exception("\x1b[0;31mERROR:\x1b[0m [youtube] abc: Video unavailable\nsecond line"), "Video unavailable"),
    (Exception("404 Client Error: File not found for url: http://127.0.0.1:5000/missing.jpg"),
     "404 Client Error: File not found"),
    (Exception("HTTPSConnectionPool(host='example.invalid', port=443): Max retries exceeded with url: /a.mp4 "
               "(Caused by NameResolutionError(\"<urllib3.connection.HTTPSConnection object>: Failed to resolve\"))"),
     "cannot connect to example.invalid (NameResolutionError)"),
    (ValueError("opening play-av tag not found"), "opening play-av tag not found"),
    (ValueError(""), "ValueError"),
    (None, "unknown error"),
])
def test_short_error_gives_one_readable_line(error, expected):
    assert short_error(error) == expected
    assert len(short_error(Exception("x" * 1000))) == 200


def test_ytdlp_messages_go_to_the_debug_log_not_the_terminal(ctx_factory, log_lines):
    options = ctx_factory().ytdlp_options()
    logger = options["logger"]
    logger.debug("[download] 50%")
    logger.warning("some warning")
    logger.error("\x1b[0;31mERROR:\x1b[0m [generic] x: failed")
    assert [level for level, _ in log_lines] == ["DEBUG", "DEBUG", "DEBUG"]
    assert "\x1b" not in log_lines[2][1] and log_lines[2][1].startswith("yt-dlp: ")
