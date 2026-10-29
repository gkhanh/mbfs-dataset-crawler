"""pipeline.py, cli.py and setup_check.py."""

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import yaml
from helpers import TEST_TYPE, FakeEncoder, make_type, textured_background

from mbfs_crawler import cli, setup_check
from mbfs_crawler.cli import main
from mbfs_crawler.config import Paths
from mbfs_crawler.pipeline import Pipeline, clean_dataset, collect, make_sources, prefetch
from mbfs_crawler.processors import Sample
from mbfs_crawler.sources import MediaItem, MediaType
from mbfs_crawler.store import read_csv
from mbfs_crawler.utils import write_jpeg

# ============================================================================
# Prefetch
# ============================================================================


def test_prefetch_keeps_order_and_overlaps_work():
    started = []

    def fetch(x):
        started.append(x)
        time.sleep(0.01)
        return x * 10

    assert list(prefetch(range(5), fetch)) == [(i, i * 10) for i in range(5)]
    assert started == list(range(5))


def test_prefetch_stays_at_most_two_items_ahead():
    fetched = []
    gen = prefetch(range(10), lambda x: fetched.append(x) or x)
    next(gen)
    time.sleep(0.2)
    assert len(fetched) <= 3  # the one being processed + one in the queue + one waiting to be queued
    assert [x for x, _ in gen] == list(range(1, 10))


def test_prefetch_reraises_errors_in_the_caller():
    def items():
        yield 1
        raise RuntimeError("source broke")

    gen = prefetch(items(), lambda x: x)
    assert next(gen) == (1, 1)
    with pytest.raises(RuntimeError, match="source broke"):
        next(gen)


def test_prefetch_thread_is_a_daemon():
    gen = prefetch(range(3), lambda x: x)
    next(gen)
    assert all(t.daemon for t in threading.enumerate() if t is not threading.main_thread())


# ============================================================================
# Pipeline
# ============================================================================


def video(key, title="", duration=None) -> MediaItem:
    return MediaItem(f"https://x/{key}", MediaType.VIDEO, key, title=title, duration=duration)


def test_items_are_screened_before_any_download(ctx_factory):
    dataset_type = make_type(exclude_words=["cartoon"], download={"max_duration": 600})
    pipeline = Pipeline(ctx_factory(dataset_type=dataset_type))
    pipeline.store.mark_done("done")
    images = []
    items = [video("a"), video("a"), video("done"), video("b", "Funny CARTOON birds"), video("c", duration=900),
             MediaItem("https://x/p.jpg", MediaType.IMAGE, "p"), video("d", duration=60)]
    assert [i.key for i in pipeline._videos_to_fetch(items, images)] == ["a", "d"]
    assert [i.key for i in images] == ["p"]
    stats = pipeline.ctx.stats
    assert stats.sources_skipped == 1 and stats.excluded == 2
    pipeline.store.close()


def test_long_videos_are_kept_when_only_their_start_is_downloaded(ctx_factory):
    dataset_type = make_type(download={"max_duration": 600, "max_minutes": 5})
    pipeline = Pipeline(ctx_factory(dataset_type=dataset_type))
    assert pipeline._wanted(video("long", duration=3600))
    pipeline.store.close()


def test_force_reprocesses_done_sources(ctx_factory):
    pipeline = Pipeline(ctx_factory(), force=True)
    pipeline.store.mark_done("a")
    assert pipeline._wanted(video("a"))
    pipeline.store.close()


def test_submit_encodes_a_whole_batch_once_and_stores_or_rejects(ctx_factory):
    encoder = FakeEncoder()
    ctx = ctx_factory(encoder=encoder)
    pipeline = Pipeline(ctx)
    same = textured_background(24, 48, seed=1)
    samples = [Sample(same, "a_p0.jpg", "a"), Sample(same.copy(), "a_p1.jpg", "a"),
               Sample(textured_background(24, 48, seed=2), "a_p2.jpg", "a")]
    kept = pipeline.submit(samples, video("a"))
    assert kept == [True, False, True] and encoder.calls == 1 and encoder.images == 3
    assert ctx.stats.saved == 2 and ctx.stats.rejected == {"duplicate": 1}
    pipeline.store.close()
    pipeline.rejected.close()
    assert [r["reason"] for r in read_csv(pipeline.rejected.csv_path)] == ["duplicate"]
    assert pipeline.submit([], video("a")) == []


def test_vectors_are_skipped_entirely_without_filters_that_need_them(ctx_factory):
    encoder = FakeEncoder()
    pipeline = Pipeline(ctx_factory(dataset_type=make_type(filters={}), encoder=encoder))
    assert pipeline.submit([Sample(textured_background(8, 8), "a_p0.jpg", "a")], video("a")) == [True]
    assert encoder.calls == 0
    pipeline.store.close()


def test_collect_yields_inputs_then_queries_and_survives_a_broken_source(ctx_factory, tmp_path):
    write_jpeg(tmp_path / "local.jpg", textured_background(8, 8))

    class Broken:
        registry_name = "broken"

        def search(self, query):
            raise RuntimeError("site changed")
            yield

    class Echo:
        registry_name = "echo"

        def search(self, query):
            yield MediaItem(f"https://x/{query}.jpg", MediaType.IMAGE, query)

    items = list(collect(ctx_factory(), ["q1", "q2"], [str(tmp_path / "local.jpg")], [Broken(), Echo()]))
    assert [i.key for i in items][1:] == ["q1", "q2"] and items[0].local


def test_make_sources_applies_type_options_and_the_limit(ctx_factory):
    ctx = ctx_factory(dataset_type=make_type(sources={"wikimedia": {"limit": 5, "media": ["video"]}}))
    (source,) = make_sources(ctx, ["wikimedia"], limit=2)
    assert source.limit == 2 and source.media == ("video",)


def test_clean_moves_noise_and_low_quality_using_stored_values(ctx_factory):
    encoder = FakeEncoder()
    ctx = ctx_factory(encoder=encoder)
    pipeline = Pipeline(ctx)
    item = video("a")
    images = [textured_background(24, 48, seed=s) for s in (1, 2, 3)]
    samples = [Sample(img, f"a_p{i}.jpg", "a", conf=0.9, box=(0, 0, 24, 200), sharpness=50.0)
               for i, img in enumerate(images)]
    samples[2].conf = 0.1  # fails the quality rule min_conf 0.5
    pipeline.submit(samples, item)
    pipeline.store.mark_done("a")
    pipeline.store.close()
    pipeline.rejected.close()
    refs = ctx.paths.references(ctx.dataset_type.name) / "noise"
    refs.mkdir(parents=True)
    write_jpeg(refs / "n.jpg", images[0])  # the user marked a_p0 as noise

    assert clean_dataset(ctx, dry_run=True) == {"a_p0.jpg": "noise", "a_p2.jpg": "low_conf"}
    assert (ctx.dataset_dir / "images" / "a_p0.jpg").exists()
    images_before = encoder.images
    reasons = clean_dataset(ctx)
    assert reasons == {"a_p0.jpg": "noise", "a_p2.jpg": "low_conf"}
    assert (ctx.dataset_dir / "rejected" / "noise" / "a_p0.jpg").exists()
    assert [r["file"] for r in read_csv(ctx.dataset_dir / "crops.csv")] == ["a_p1.jpg"]
    assert encoder.images == images_before  # stored vectors reused: nothing re-encoded


# ============================================================================
# CLI
# ============================================================================


@pytest.mark.parametrize("given,expected", [
    ([], (["default q"], ["default.txt"])),     # no -i: the type's searches and links
    (["x.mp4"], ([], ["x.mp4"])),               # -i: only those inputs
])
def test_plan_work(given, expected):
    dataset_type = SimpleNamespace(search=("default q",), links=("default.txt",))
    assert cli.plan_work(SimpleNamespace(input=given), dataset_type) == (list(expected[0]), list(expected[1]))


def test_save_links_appends_only_new_links_and_keeps_hand_edits(tmp_path):
    path = tmp_path / "links" / "birds" / "found.txt"          # the folders are created
    items = [MediaItem("https://x/a", MediaType.VIDEO, "a", license="unknown", source_name="youtube"),
             MediaItem("https://x/a", MediaType.VIDEO, "a", license="unknown", source_name="youtube"),  # found twice
             MediaItem("https://x/b.jpg", MediaType.IMAGE, "b", license="cc-by-4.0", source_name="wikimedia")]
    assert cli.save_links(items, path, "birds") == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["# mbfs-crawler search birds", "https://x/a  # license=unknown source=youtube",
                     "https://x/b.jpg  # license=cc-by-4.0 source=wikimedia"]
    with path.open("a", encoding="utf-8") as out:
        out.write("https://mine/own.mp4\n")                    # added by hand
    path.write_text(path.read_text(encoding="utf-8").replace("https://x/b.jpg  # license=cc-by-4.0 source=wikimedia\n",
                                                             ""), encoding="utf-8")  # and one line deleted by hand
    items.append(MediaItem("https://x/c", MediaType.VIDEO, "c", source_name="youtube"))
    assert cli.save_links(items, path, "birds") == 2             # only b (deleted) and c (new) are added again
    text = path.read_text(encoding="utf-8")
    assert text.count("# mbfs-crawler search") == 1 and "https://mine/own.mp4" in text and text.count("x/a") == 1


def test_save_links_to_a_header_only_file_adds_no_second_header(tmp_path):
    path = tmp_path / "found.txt"
    path.write_text("# mbfs-crawler search birds\n", encoding="utf-8")
    assert cli.save_links([MediaItem("https://x/a", MediaType.VIDEO, "a")], path, "birds") == 1
    assert path.read_text(encoding="utf-8").count("#") == 2      # the one header + the license note of the link


def test_expand_paths_handles_files_folders_globs_and_lists(tmp_path):
    for name in ("a_id1.jpg", "b_id2.jpg", "c_id3.jpg"):
        write_jpeg(tmp_path / name, textured_background(4, 4))
    (tmp_path / "sub").mkdir()
    write_jpeg(tmp_path / "sub" / "d.jpg", textured_background(4, 4))
    (tmp_path / "list.txt").write_text("# picked\nc_id3.jpg\n", encoding="utf-8")
    found = cli.expand_paths(["*_id1.jpg", "sub", "list.txt", "missing.jpg"], [tmp_path])
    assert [p.name for p in found] == ["a_id1.jpg", "d.jpg", "c_id3.jpg"]


CONFIG = {
    "people": TEST_TYPE,                                                          # no searches, no links
    "searching": {**TEST_TYPE, "search": ["x"], "sources": {"nope": {}}},        # an unknown source
    "sky": {"processor": {"type": "frame", "track_gap": 1}},                      # not an option of frame
}


@pytest.fixture
def project(tmp_path) -> Path:
    """A project folder at tmp_path/project with a small datasets.yaml; results go to tmp_path/dataset."""
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "datasets.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    return folder


def run(argv, tmp_path):
    return cli.main([*argv, "--home", str(tmp_path / "project"), "--no-progress"])


def test_cli_exit_codes(project, tmp_path, capsys):
    assert run(["types"], tmp_path) == 0
    out = capsys.readouterr().out
    assert "people" in out and "✗ sky" in out and str(project / "datasets.yaml") in out
    assert run(["sources"], tmp_path) == 0 and "wikimedia" in capsys.readouterr().out
    assert run(["run", "peple"], tmp_path) == 2                               # unknown type
    assert "Did you mean 'people'?" in capsys.readouterr().out
    assert run(["run", "people"], tmp_path) == 3                              # no searches, no links, no -i
    assert run(["search", "people"], tmp_path) == 3
    assert run(["run", "searching"], tmp_path) == 2
    assert run(["run", "sky", "-i", "x"], tmp_path) == 2
    assert run(["clean", "people"], tmp_path) == 3
    assert run(["stats", "people"], tmp_path) == 0 and "No dataset yet" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--no-such-option"])
    assert exc.value.code == 2


def test_a_missing_datasets_yaml_is_a_clear_error(tmp_path, capsys):
    assert run(["types"], tmp_path) == 2
    assert run(["run", "birds"], tmp_path) == 2
    assert "datasets.yaml not found. Run from the project folder" in capsys.readouterr().out


def test_cli_writes_a_log_file(project, tmp_path):
    log_file = tmp_path / "run.log"
    assert cli.main(["run", "people", "--home", str(project), "--log-file", str(log_file)]) == 3
    assert "Nothing to do" in log_file.read_text(encoding="utf-8")


def test_light_commands_never_import_torch_and_never_read_stdin(project):
    home = str(project)
    code = ("import sys; from mbfs_crawler.cli import main; "
            f"codes = [main(['types', '--home', {home!r}]), "
            f"main(['stats', 'people', '--home', {home!r}]), "
            f"main(['run', 'people', '--home', {home!r}])]; "
            "print('TORCH' if 'torch' in sys.modules else 'NO-TORCH', codes)")
    result = subprocess.run([sys.executable, "-c", code], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=120)
    assert "NO-TORCH [0, 0, 3]" in result.stdout, result.stderr


def test_review_adds_examples_and_requires_files(project, tmp_path, capsys):
    write_jpeg(tmp_path / "x.jpg", textured_background(8, 8))
    assert run(["review", "people", "--noise", str(tmp_path / "x.jpg")], tmp_path) == 0
    assert (tmp_path / "dataset" / "references" / "people" / "noise" / "x.jpg").exists()
    assert run(["review", "people"], tmp_path) == 2


def test_interrupt_returns_130_with_a_resume_hint(monkeypatch, project, tmp_path):
    def stop(args):
        raise KeyboardInterrupt

    monkeypatch.setitem(cli.COMMANDS, "stats", stop)
    assert run(["stats", "people"], tmp_path) == 130



def test_checks_report_problems_with_a_fix(tmp_path):
    assert setup_check._python().ok
    assert setup_check._package("json", True).ok
    missing = setup_check._package("no_such_package_xyz", True)
    assert not missing.ok and missing.fix == "pip install -e ."
    assert setup_check._package("selenium_missing_xyz", False).required is False
    assert setup_check._ffmpeg().ok
    assert setup_check._writable(tmp_path / "out").ok
    models = setup_check._models(Paths(tmp_path))
    assert len(models) == 1 and not models[0].ok and "MBFS_HOME" in models[0].fix


def test_broken_model_file_is_reported(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "broken.onnx").write_bytes(b"not a model")
    (check,) = setup_check._models(Paths(tmp_path))
    assert not check.ok and check.name == "model broken.onnx"


def test_network_checks_are_optional(monkeypatch):
    def head(url, **kwargs):
        if "bing" in url:
            raise requests.ConnectionError("blocked")

    monkeypatch.setattr(requests, "head", head)
    checks = setup_check._network()
    failed = [c for c in checks if not c.ok]
    assert len(checks) == len(setup_check.HOSTS) and [c.name for c in failed] == ["internet: bing_images"]
    assert not failed[0].required


def test_format_counts_only_required_failures():
    checks = [setup_check.Check("a", True, "fine"), setup_check.Check("b", False, "x", "do y", required=False)]
    text = setup_check.format_checks(checks)
    assert "✓ a: fine" in text and "! b: x\n    → do y" in text and "All required checks passed" in text
    text = setup_check.format_checks([setup_check.Check("c", False, "", "fix it")])
    assert "✗ c" in text and "1 required check(s) failed" in text


def test_check_setup_command_offline(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(setup_check, "_device", lambda device: setup_check.Check("compute device", True, "cpu"))
    monkeypatch.setattr(setup_check, "_dinov2_cache", lambda: setup_check.Check("DINOv2", True, required=False))
    assert main(["check-setup", "--offline", "--home", str(tmp_path / "project")]) == 1  # no models in this folder
    out = capsys.readouterr().out
    assert "✗ models" in out and "internet" not in out


# ============================================================================
# --debug and crash reports
# ============================================================================


def test_debug_mode_writes_a_log_file_in_the_logs_folder(tmp_path, capsys):
    assert run(["sources", "--debug"], tmp_path) == 0
    (log_file,) = (tmp_path / "dataset" / "logs").glob("sources-*.log")  # next to the project, not inside it
    text = log_file.read_text(encoding="utf-8")
    assert "mbfs-crawler 0.1.0, Python" in text and "command: sources --debug" in text and "DEBUG" in text
    assert f"Debug mode: everything is written to {log_file}" in capsys.readouterr().out
    assert not (tmp_path / "project").exists()


def test_without_debug_no_log_file_is_created(tmp_path):
    assert run(["sources"], tmp_path) == 0
    assert not (tmp_path / "dataset").exists()


def test_an_explicit_log_file_is_used_instead(tmp_path):
    log_file = tmp_path / "mine.log"
    assert run(["sources", "--debug", "--log-file", str(log_file)], tmp_path) == 0
    assert "command: sources" in log_file.read_text(encoding="utf-8")
    assert not (tmp_path / "dataset" / "logs").exists()


def test_an_unexpected_error_prints_one_line_and_writes_a_crash_report(monkeypatch, tmp_path, capsys):
    def boom(args):
        raise RuntimeError("disk full")

    monkeypatch.setitem(cli.COMMANDS, "stats", boom)
    assert run(["stats", "people"], tmp_path) == 1          # even without --debug
    out = capsys.readouterr().out
    assert "Error: disk full" in out and "Traceback" not in out
    (report,) = (tmp_path / "dataset" / "logs").glob("crash-*.log")
    assert str(report) in out
    text = report.read_text(encoding="utf-8")
    assert "Traceback" in text and "RuntimeError: disk full" in text and "command: mbfs-crawler stats people" in text


def test_crash_report_failure_is_not_fatal(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(cli.COMMANDS, "stats", lambda args: (_ for _ in ()).throw(RuntimeError("x")))
    (tmp_path / "dataset").write_text("a file where the logs folder should be", encoding="utf-8")
    assert run(["stats", "people"], tmp_path) == 1
    assert "Error: x" in capsys.readouterr().out


def test_options_choose_where_results_and_downloads_go(tmp_path):
    paths = cli._paths(cli.build_parser().parse_args(
        ["stats", "x", "--home", str(tmp_path / "p"), "--data-dir", str(tmp_path / "d"),
         "--downloads-dir", str(tmp_path / "dl")]))
    assert paths.data_root == (tmp_path / "d").resolve() and paths.downloads == (tmp_path / "dl").resolve()


# ============================================================================
# A broken item never stops the run
# ============================================================================


def test_a_video_that_fails_to_process_is_skipped_and_the_next_one_goes_on(ctx_factory, log_lines):
    pipeline = Pipeline(ctx_factory())
    seen = []

    def process_video(path, item, submit):
        seen.append(item.key)
        if item.key == "bad":
            raise RuntimeError("corrupt frame")
        return True

    pipeline.processor.process_video = process_video
    pipeline._process_video(video("bad"), Path("bad.mp4"))
    pipeline._process_video(video("good"), Path("good.mp4"))
    assert seen == ["bad", "good"] and pipeline.store.is_done("good") and not pipeline.store.is_done("bad")
    warnings = [m for level, m in log_lines if level == "WARNING"]
    assert warnings == ["! bad: processing failed (corrupt frame); skipped, it is retried next run"]
    assert any(level == "DEBUG" and m == "details" for level, m in log_lines)  # the traceback is in the debug log
    pipeline.store.close()
    pipeline.rejected.close()


def test_a_batch_of_images_that_fails_to_process_is_skipped(ctx_factory, tmp_path, log_lines):
    write_jpeg(tmp_path / "a.jpg", textured_background(32, 32))
    items = [MediaItem(str(tmp_path / "a.jpg"), MediaType.IMAGE, "a_1", local=True)]
    pipeline = Pipeline(ctx_factory())
    pipeline.processor.process_images = lambda loaded: (_ for _ in ()).throw(RuntimeError("bad batch"))
    pipeline._process_images(items)
    assert any("1 images could not be processed (bad batch)" in m for level, m in log_lines if level == "WARNING")
    assert not pipeline.store.is_done("a_1")
    pipeline.store.close()
    pipeline.rejected.close()
