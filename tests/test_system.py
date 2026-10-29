"""System tests (pytest -m system): the real `mbfs-crawler` program as a subprocess, with the real models.
Network tests (pytest -m network): the live internet sources.

System tests are offline: test videos are built from Ultralytics' bundled sample photos (bus.jpg, zidane.jpg) and
served from a local web server. They need the models in models/ and the DINOv2 weights (downloaded once).
"""

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from helpers import make_video, textured_background

from mbfs_crawler.store import read_csv

REPO = Path(__file__).resolve().parents[1]
PEOPLE = "food_factory_workers"


def sample_photo(name: str) -> np.ndarray:
    from ultralytics.utils import ASSETS

    return cv2.imread(str(ASSETS / name))


def panning_video(path: Path, photo: str, seconds: float = 4, fps: float = 10, size=(640, 480)) -> Path:
    """A camera slowly panning across a sample photo."""
    image = sample_photo(photo)
    scale = size[1] / image.shape[0] * 1.2
    image = cv2.resize(image, None, fx=scale, fy=scale)
    frames = []
    for i in range(int(seconds * fps)):
        x = int((image.shape[1] - size[0]) * i / max(1, seconds * fps - 1))
        frames.append(np.ascontiguousarray(image[: size[1], x: x + size[0]]))
    return make_video(path, frames, fps)


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    for name in ("mbfsn_human_det_v6.onnx", "mbfsn_vehicle_det_v6.onnx"):
        shutil.copy2(REPO / "models" / name, folder / name)
    return folder


def write_config(home: Path, **changes) -> None:
    """The project's own datasets.yaml, with test settings changed or test types added."""
    config = yaml.safe_load((REPO / "datasets.yaml").read_text(encoding="utf-8"))
    for name, settings in changes.items():
        config[name] = {**config.get(name, {}), **settings}
    (home / "datasets.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


@pytest.fixture
def home(tmp_path, models_dir):
    home = tmp_path / "home"
    home.mkdir()
    shutil.copytree(models_dir, home / "models")
    shutil.copytree(REPO / "links", home / "links")
    write_config(home)
    return home


def env() -> dict:
    variables = {k: v for k, v in os.environ.items() if k != "DISPLAY"}  # no screen, like an SSH session
    variables["PYTHONIOENCODING"] = "utf-8"
    return variables


def mbfs(home: Path, *argv, log: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "mbfs_crawler", *argv, "--home", str(home), "--device", "cpu"]
    if log:
        command += ["--log-file", str(log)]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                            env=env(), timeout=900)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.system
def test_people_end_to_end_with_resume_review_and_clean(web, home, tmp_path):
    panning_video(web.root / "people.mp4", "zidane.jpg", seconds=8)
    url = web.url("people.mp4")
    people = yaml.safe_load((home / "datasets.yaml").read_text(encoding="utf-8"))[PEOPLE]
    write_config(home, **{PEOPLE: {"processor": {**people["processor"], "track_gap": 0.5}}})  # a short video

    # A clean run, for comparison.
    clean_home = tmp_path / "clean" / "home"  # its own parent folder, so its results do not mix with `home`'s
    shutil.copytree(home, clean_home)
    mbfs(clean_home, "run", PEOPLE, "-i", url)
    rows = read_csv(clean_home.parent / "dataset" / PEOPLE / "crops.csv")
    expected = sorted(r["file"] for r in rows)
    assert expected and all(r["det_class"] == "person" for r in rows)

    # Killed halfway, then resumed: same result.
    command = [sys.executable, "-m", "mbfs_crawler", "run", PEOPLE, "-i", url, "--home", str(home), "--device", "cpu",
               "--no-progress"]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=env())
    csv_path = home.parent / "dataset" / PEOPLE / "crops.csv"
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline and process.poll() is None:
        if csv_path.exists() and len(read_csv(csv_path)) >= 1:
            process.kill()
            break
        time.sleep(0.05)
    process.wait()
    mbfs(home, "run", PEOPLE, "-i", url)
    dataset = home.parent / "dataset" / PEOPLE
    assert sorted(r["file"] for r in read_csv(dataset / "crops.csv")) == expected
    assert sorted(p.name for p in (dataset / "images").iterdir()) == expected

    report = mbfs(home, "stats", PEOPLE).stdout
    assert f"images: {len(expected)}" in report and "person" in report
    assert not (dataset / "labels").exists()  # nothing is labelled: that is done afterwards, outside the tool

    # Marking one crop as noise rejects it, and every crop that looks like it (the same person), on `clean`.
    mbfs(home, "review", PEOPLE, "--noise", f"images/{expected[0]}")
    mbfs(home, "clean", PEOPLE)
    assert (dataset / "rejected" / "noise" / expected[0]).exists()
    assert expected[0] not in {r["file"] for r in read_csv(dataset / "crops.csv")}


@pytest.mark.system
def test_vehicle_type_crops_the_bus(web, home):
    panning_video(web.root / "street.mp4", "bus.jpg", seconds=4)
    mbfs(home, "run", "vehicle", "-i", web.url("street.mp4"))
    rows = read_csv(home.parent / "dataset" / "vehicle" / "crops.csv")
    assert "bus" in {r["det_class"] for r in rows}


@pytest.mark.system
def test_frame_type_skips_static_frames_and_keeps_full_size(home, tmp_path):
    sky = textured_background(1280, 720, seed=3)
    frames = [sky.copy() for _ in range(40)]
    for i in range(20):
        frame = sky.copy()
        frame[300:310, 100 + i * 40:112 + i * 40] = 30  # a small dark bird crossing the sky
        frames.append(frame)
    video = make_video(tmp_path / "sky.mp4", frames, fps=10)
    write_config(home, sky={  # whole frames without a detector: only the motion check decides
        "processor": {"type": "frame", "every": 0.5, "min_change": 0.00003},
        "quality": {"min_sharpness": 0, "reject_broken": False}, "filters": {"duplicate": {"max_similarity": 0.999}},
    })
    output = mbfs(home, "run", "sky", "-i", str(video)).stdout
    skipped = int(output.split("(adaptive) + ")[1].split(" (no motion)")[0])
    assert skipped >= 5
    images = list((home.parent / "dataset" / "sky" / "images").iterdir())
    assert images and all(cv2.imread(str(p)).shape == (720, 1280, 3) for p in images)


@pytest.mark.system
def test_every_command_runs_headless(home, tmp_path):
    log = tmp_path / "run.log"
    video = panning_video(tmp_path / "v.mp4", "zidane.jpg", seconds=2)
    write_config(home, nothing={"detector": {"model": "models/mbfsn_human_det_v6.onnx"}})  # no searches, no links
    for argv, code in [(["types"], 0), (["sources"], 0), (["models"], 0), (["stats", PEOPLE], 0),
                       (["run", PEOPLE, "-i", str(video)], 0), (["run", "nothing"], 3),
                       (["clean", PEOPLE, "--dry-run"], 0), (["run", "missing"], 2)]:
        result = mbfs(home, *argv, log=log, check=False)
        assert result.returncode == code, (argv, result.stdout, result.stderr)
    text = log.read_text(encoding="utf-8")
    assert "\r" not in text and "saved" in text  # no progress-bar control characters in the log


@pytest.mark.system
def test_installed_wheel_runs_with_a_datasets_yaml(tmp_path):
    source = tmp_path / "src"  # build from a copy, so no build/ folder appears in the project
    source.mkdir()
    shutil.copytree(REPO / "mbfs_crawler", source / "mbfs_crawler", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(REPO / name, source / name)
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(tmp_path / "dist"),
                    str(source)], check=True, capture_output=True, timeout=600)
    (wheel,) = (tmp_path / "dist").glob("mbfs_crawler-*.whl")
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
        entry_points = next(n for n in archive.namelist() if n.endswith("entry_points.txt"))
        assert "mbfs-crawler = mbfs_crawler.cli:main" in archive.read(entry_points).decode()
    shutil.copy2(REPO / "datasets.yaml", tmp_path / "datasets.yaml")
    code = (f"import sys; sys.path.insert(0, {str(site)!r}); import mbfs_crawler; "
            f"assert mbfs_crawler.__file__.startswith({str(site)!r}); from mbfs_crawler.cli import main; "
            f"sys.exit(main(['types', '--home', {str(tmp_path)!r}]))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    for name in ("birds", "food_factory_workers", "vehicle"):
        assert name in result.stdout


# ============================================================================
# Live internet (pytest -m network): sites change, so run these before a release, not on every commit
# ============================================================================


def live_context(tmp_path, name="birds"):
    from mbfs_crawler.config import Paths, RunContext
    from mbfs_crawler.datasets import load_type

    return RunContext(Paths(tmp_path), load_type(name, REPO / "datasets.yaml"), device="cpu")


@pytest.mark.network
@pytest.mark.parametrize("source,query", [("youtube", "birds airport runway"), ("wikimedia", "flock of birds"),
                                          ("internet_archive", "birds airport"), ("openverse", "seagull")])
def test_live_sources_return_results(tmp_path, source, query):
    from mbfs_crawler.pipeline import make_sources

    (found,) = make_sources(live_context(tmp_path), [source], limit=2)
    items = list(found.search(query))
    if source == "openverse" and not items:
        pytest.skip("Openverse did not answer in time (it is often slow); the source handled it")
    assert items and all(item.uri.startswith("https://") and item.key for item in items)


@pytest.mark.network
@pytest.mark.parametrize("source", ["bilibili", "niconico", "bing_images", "google_images"])
def test_optional_live_sources_never_crash(tmp_path, source):
    from mbfs_crawler.pipeline import make_sources

    (found,) = make_sources(live_context(tmp_path), [source], limit=2)
    items = list(found.search("bird"))
    if not items:
        pytest.skip(f"{source} returned nothing from this network (blocked, login wall or no Chrome)")


@pytest.mark.network
def test_live_video_download_takes_only_the_first_seconds(tmp_path):
    from mbfs_crawler.downloads import VideoDownloader
    from mbfs_crawler.sources import WikimediaSource

    ctx = live_context(tmp_path)
    item = next(WikimediaSource(ctx, limit=1, media=["video"]).search("bird flying"))
    path = VideoDownloader(ctx, max_res=360, max_minutes=0.1).fetch(item)
    capture = cv2.VideoCapture(str(path))
    seconds = capture.get(cv2.CAP_PROP_FRAME_COUNT) / (capture.get(cv2.CAP_PROP_FPS) or 30)
    capture.release()
    assert path.exists() and 0 < seconds <= 15


@pytest.mark.network
def test_birds_type_keeps_real_bird_photos_and_labels_nothing(home):
    write_config(home, birds={"search": ["flock of birds flying sky"], "sources": {"wikimedia": {"media": ["image"]}}})
    mbfs(home, "run", "birds", "--limit", "4")
    rows = read_csv(home.parent / "dataset" / "birds" / "crops.csv")
    assert any(r["det_class"] == "bird" for r in rows), "yolov8n found no bird in 4 bird photos"
    assert not (home.parent / "dataset" / "birds" / "labels").exists()
    assert (home / "models" / "yolov8n.pt").exists()  # downloaded once into models/
