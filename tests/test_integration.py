"""Whole commands, in-process, with fake models and a local web server standing in for the internet.

The real yt-dlp, downloaders, processors, filters and store run; only the neural networks are faked.
"""

import shutil
from pathlib import Path

import pytest
import yaml
from helpers import TEST_TYPE, FakeDetector, FakeEncoder, draw_object, people_video, textured_background

from mbfs_crawler.cli import main
from mbfs_crawler.config import SOURCES, RunContext
from mbfs_crawler.pipeline import Pipeline
from mbfs_crawler.sources import Source, image_item
from mbfs_crawler.store import read_csv
from mbfs_crawler.utils import write_jpeg


@SOURCES.register("test_web")
class WebSource(Source):
    """Test only: the images of the local web server."""

    urls: list[str] = []

    def search(self, query):
        for url in WebSource.urls[: self.limit]:
            yield image_item(url, "test_web", "cc-by-4.0")


@pytest.fixture
def fakes(monkeypatch):
    detector, encoder = FakeDetector(), FakeEncoder()
    monkeypatch.setattr(RunContext, "detector", lambda self: detector)
    monkeypatch.setattr(RunContext, "encoder", lambda self: encoder)
    return detector, encoder


REPO_CONFIG = Path(__file__).resolve().parents[1] / "datasets.yaml"
CONFIG = {
    "testppl": TEST_TYPE,                                                            # used with -i only
    "webppl": {**TEST_TYPE, "search": ["workers"], "sources": {"test_web": {"limit": 5}}},  # searches the test server
}


def make_home(home):
    """A project folder with its datasets.yaml."""
    home.mkdir(parents=True)
    (home / "datasets.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    return home


@pytest.fixture
def home(tmp_path):
    return make_home(tmp_path / "home")


def mbfs(home, *argv) -> int:
    return main([*argv, "--home", str(home), "--no-progress"])


def person_image(path, seed=0, objects=2):
    image = textured_background(320, 240, seed)
    for k in range(objects):
        draw_object(image, 20 + k * 150, 40, 40, 120, pattern=seed * 10 + k)  # unique texture per object
    write_jpeg(path, image)
    return path


def dataset(home, name="testppl"):
    return home.parent / "dataset" / name  # results live next to the project folder, not inside it


def test_run_collects_from_links_lists_and_folders_then_resumes(web, home, fakes, tmp_path, capsys):
    people_video(web.root / "people.mp4", seconds=5, objects=2)
    person_image(web.root / "a.jpg", seed=1)
    person_image(web.root / "b.jpg", seed=2)
    links = tmp_path / "links.txt"
    links.write_text(f"# test links\n{web.url('a.jpg')}  # license=cc-by-4.0\n{web.url('b.jpg')}\n", encoding="utf-8")
    folder = tmp_path / "local"
    folder.mkdir()
    person_image(folder / "c.jpg", seed=3, objects=1)

    assert mbfs(home, "run", "testppl", "-i", web.url("people.mp4"), "-i", str(links), "-i", str(folder)) == 0
    out = capsys.readouterr().out
    rows = read_csv(dataset(home) / "crops.csv")
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    video_key = next(k for k in by_source if k.startswith("people_"))  # a direct video link: no yt-dlp
    assert len(by_source[video_key]) >= 2 and all(r["track_id"] in ("1", "2") for r in by_source[video_key])
    assert {r["license"] for r in rows if r["source"].startswith("a_")} == {"cc-by-4.0"}
    assert {r["license"] for r in rows if r["source"].startswith("c_")} == {"local"}
    assert all((dataset(home) / "images" / r["file"]).exists() for r in rows)
    assert (home.parent / "downloads" / "videos" / f"{video_key}.mp4").exists()
    assert "saved" in out and "detector calls" in out and "Images: " in out
    assert sorted(p.name for p in home.iterdir()) == ["datasets.yaml"]  # the project folder got no output at all
    assert (home.parent / "downloads").is_dir() and (home.parent / "dataset").is_dir()
    done = (dataset(home) / "processed.txt").read_text(encoding="utf-8").split()
    assert len(done) == 4 and video_key in done

    requests_before = len(web.requests)
    assert mbfs(home, "run", "testppl", "-i", web.url("people.mp4"), "-i", str(links), "-i", str(folder)) == 0
    assert read_csv(dataset(home) / "crops.csv") == rows
    assert "already processed" in capsys.readouterr().out
    assert not any(r.endswith(("a.jpg", "b.jpg")) for r in web.requests[requests_before:])  # nothing re-downloaded


def test_interrupted_run_resumes_to_the_same_result(monkeypatch, web, home, fakes, tmp_path):
    for i in range(4):
        person_image(web.root / f"{i}.jpg", seed=i)
    links = tmp_path / "links.txt"
    links.write_text("\n".join(web.url(f"{i}.jpg") for i in range(4)), encoding="utf-8")

    clean_home = tmp_path / "clean" / "home"  # own parent folder: its results stay separate
    make_home(clean_home)
    assert mbfs(clean_home, "run", "testppl", "-i", str(links)) == 0
    expected = sorted(r["file"] for r in read_csv(dataset(clean_home) / "crops.csv"))

    original = Pipeline.submit
    calls = []

    def interrupted(self, samples, item):
        calls.append(item.key)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return original(self, samples, item)

    monkeypatch.setattr(Pipeline, "submit", interrupted)
    assert mbfs(home, "run", "testppl", "-i", str(links)) == 130
    monkeypatch.setattr(Pipeline, "submit", original)
    assert mbfs(home, "run", "testppl", "-i", str(links)) == 0
    assert sorted(r["file"] for r in read_csv(dataset(home) / "crops.csv")) == expected
    assert sorted(p.name for p in (dataset(home) / "images").iterdir()) == expected


def test_search_writes_a_link_list_that_run_uses_with_licenses(web, home, fakes, tmp_path, capsys):
    WebSource.urls = [web.url(f"{i}.jpg") for i in range(3)]
    for i in range(3):
        person_image(web.root / f"{i}.jpg", seed=10 + i)
    links = tmp_path / "found.txt"
    assert mbfs(home, "search", "webppl", "--limit", "2", "-o", str(links)) == 0
    assert "2 found: 0 videos, 2 images" in capsys.readouterr().out
    lines = [x for x in links.read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
    assert len(lines) == 2 and all("license=cc-by-4.0" in x for x in lines)
    assert mbfs(home, "run", "testppl", "-i", str(links)) == 0
    rows = read_csv(dataset(home) / "crops.csv")
    assert rows and {r["license"] for r in rows} == {"cc-by-4.0"}


def test_search_saves_to_links_folder_of_the_type_by_default(web, home, fakes, capsys):
    WebSource.urls = [web.url(f"{i}.jpg") for i in range(3)]
    found = home / "links" / "webppl" / "found.txt"
    assert mbfs(home, "search", "webppl", "--limit", "2") == 0
    lines = [x for x in found.read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
    assert len(lines) == 2 and "2 new link(s) saved to" in capsys.readouterr().out
    assert sorted(p.name for p in home.iterdir()) == ["datasets.yaml", "links"]      # nothing in the project root
    assert mbfs(home, "search", "webppl", "--limit", "3") == 0                       # same two + one new one
    assert "1 new link(s) saved" in capsys.readouterr().out
    assert len([x for x in found.read_text(encoding="utf-8").splitlines() if not x.startswith("#")]) == 3


def test_run_without_inputs_uses_the_searches_and_sources_of_the_type(web, home, fakes):
    WebSource.urls = [web.url("q.jpg")]
    person_image(web.root / "q.jpg", seed=20)
    assert mbfs(home, "run", "webppl") == 0
    assert {r["source_name"] for r in read_csv(dataset(home, "webppl") / "crops.csv")} == {"test_web"}


def test_review_then_clean_removes_similar_noise(home, fakes, tmp_path, capsys):
    folder = tmp_path / "local"
    folder.mkdir()
    for i in range(3):
        person_image(folder / f"{i}.jpg", seed=30 + i, objects=1)
    assert mbfs(home, "run", "testppl", "-i", str(folder)) == 0
    files = sorted(r["file"] for r in read_csv(dataset(home) / "crops.csv"))
    assert len(files) == 3

    assert mbfs(home, "review", "testppl", "--noise", f"images/{files[0]}") == 0
    assert mbfs(home, "clean", "testppl", "--dry-run") == 0
    assert files[0] in capsys.readouterr().out
    assert (dataset(home) / "images" / files[0]).exists()
    assert mbfs(home, "clean", "testppl") == 0
    assert (dataset(home) / "rejected" / "noise" / files[0]).exists()
    assert sorted(r["file"] for r in read_csv(dataset(home) / "crops.csv")) == files[1:]

    capsys.readouterr()
    assert mbfs(home, "stats", "testppl") == 0
    report = capsys.readouterr().out
    assert "images: 2" in report and "noise 1" in report and "local 2" in report


@pytest.mark.parametrize("name", list(yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))))
def test_every_type_in_the_project_datasets_yaml_runs(name, fakes, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    shutil.copy(REPO_CONFIG, home / "datasets.yaml")
    video = people_video(tmp_path / "clip.mp4", seconds=3, objects=1)
    assert mbfs(home, "run", name, "-i", str(video)) == 0
    assert (home.parent / "dataset" / name / "processed.txt").read_text(encoding="utf-8").split()


def test_options_move_results_and_downloads_anywhere(web, home, fakes, tmp_path):
    person_image(web.root / "a.jpg", seed=70)
    where = tmp_path / "elsewhere"
    assert mbfs(home, "run", "testppl", "-i", web.url("a.jpg"), "--data-dir", str(where / "results"),
                "--downloads-dir", str(where / "raw")) == 0
    assert (where / "results" / "testppl" / "crops.csv").exists()
    assert any((where / "raw" / "images").iterdir())
    assert not (home.parent / "dataset").exists() and not (home.parent / "downloads").exists()
    assert sorted(p.name for p in home.iterdir()) == ["datasets.yaml"]


def test_bad_links_never_stop_the_run_and_are_summarised(web, home, fakes, tmp_path, capsys):
    person_image(web.root / "ok.jpg", seed=80)
    (web.root / "page.jpg").write_text("<html>this is a web page, not an image</html>", encoding="utf-8")
    (web.root / "fake.mp4").write_bytes(b"this is not a video")
    people_video(web.root / "good.mp4", seconds=4, objects=1)
    links = tmp_path / "links.txt"
    links.write_text("\n".join(web.url(name) for name in
                               ("missing.jpg", "page.jpg", "fake.mp4", "missing.webm", "good.mp4", "ok.jpg")),
                     encoding="utf-8")
    assert mbfs(home, "run", "testppl", "-i", str(links)) == 0
    out = capsys.readouterr().out
    sources = {r["source"].split("_")[0] for r in read_csv(dataset(home) / "crops.csv")}
    assert {"ok", "good"} <= sources                                  # the good ones were still collected
    assert out.count("\n! ") >= 4 and "warnings: " in out             # each failure is one short line + a summary
    assert "Traceback" not in out and "\nError:" not in out         # no crash, no traceback
    assert "404 Client Error: File not found\n" in out or "404 Client Error: File not found" in out
    assert "for url:" not in out                                       # the link is not repeated in every message


def test_debug_run_records_the_whole_run_in_a_log_file(web, home, fakes, tmp_path):
    person_image(web.root / "a.jpg", seed=90)
    assert mbfs(home, "run", "testppl", "-i", web.url("a.jpg"), "-i", web.url("gone.jpg"), "--debug") == 0
    (log_file,) = (tmp_path / "dataset" / "logs").glob("run-*.log")
    text = log_file.read_text(encoding="utf-8")
    assert "command: run testppl" in text and "DEBUG" in text
    assert "GET http://127.0.0.1" in text and "HTTP 404" in text      # every request, with its answer
