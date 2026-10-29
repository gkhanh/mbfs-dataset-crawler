"""models.py (vector index), filters.py and store.py."""

import csv

import numpy as np
import pytest
from helpers import FakeEncoder, textured_background

from mbfs_crawler.filters import (
    KEEP,
    DuplicateFilter,
    Filter,
    FilterChain,
    ReferenceFilter,
    ReferenceSet,
    Verdict,
    vote,
)
from mbfs_crawler.models import VectorIndex
from mbfs_crawler.processors import QualityRules, Sample
from mbfs_crawler.sources import MediaItem, MediaType
from mbfs_crawler.store import CSV_FIELDS, DatasetStore, RejectedStore, dataset_report, read_csv
from mbfs_crawler.utils import write_jpeg


def unit(*values) -> np.ndarray:
    v = np.array(values, np.float32)
    return v / np.linalg.norm(v)


def sample(name: str, source: str = "src", seed: int = 0, vector=None, **kwargs) -> Sample:
    return Sample(textured_background(16, 24, seed), name, source, vector=vector, **kwargs)


ITEM = MediaItem("https://x.org/v", MediaType.VIDEO, "src", source_name="youtube", license="cc-by-4.0")

# ============================================================================
# Vector index
# ============================================================================


def test_index_finds_the_most_similar_vectors():
    index = VectorIndex(dim=3)
    assert index.max_similarity(unit(1, 0, 0)) == 0.0
    for v in ([1, 0, 0], [0, 1, 0], [1, 1, 0]):
        index.add(unit(*v))
    assert index.max_similarity(unit(1, 0, 0)) == pytest.approx(1.0)
    sims, rows = index.knn(unit(1, 0.1, 0), k=2)
    assert list(rows) == [0, 2] and sims[0] > sims[1]
    assert list(index.knn(unit(0, 1, 0), k=10)[1]) == [1, 2, 0]


def test_index_grows_past_its_initial_capacity():
    index = VectorIndex(dim=4)
    for i in range(3000):
        index.add(unit(1, i, 0, 0))
    assert index.n == 3000 and index.vectors.shape == (3000, 4)


def test_flush_appends_only_new_rows(tmp_path):
    path = tmp_path / "embeddings.f32"
    index = VectorIndex(path, dim=4)
    index.add(unit(1, 0, 0, 0))
    index.flush()
    assert path.stat().st_size == 16
    index.add(unit(0, 1, 0, 0))
    index.add(unit(0, 0, 1, 0))
    index.flush()
    index.flush()  # nothing new: nothing written
    assert path.stat().st_size == 48
    reopened = VectorIndex(path, dim=4)
    assert reopened.n == 3 and np.allclose(reopened.vectors, index.vectors)


def test_truncate_drops_rows_on_disk_too(tmp_path):
    path = tmp_path / "embeddings.f32"
    index = VectorIndex(path, dim=2)
    for i in range(5):
        index.add(unit(1, i))
    index.flush()
    index.truncate(2)
    assert index.n == 2 and path.stat().st_size == 2 * 2 * 4 and VectorIndex(path, dim=2).n == 2
    index.truncate(10)  # never grows
    assert index.n == 2


def test_half_written_row_is_dropped_on_open(tmp_path):
    path = tmp_path / "embeddings.f32"
    np.array([1, 0, 0, 1, 0.5], np.float32).tofile(path)  # 2 full rows of 2 + half a row
    assert VectorIndex(path, dim=2).n == 2 and path.stat().st_size == 16


def test_keep_rewrites_the_file(tmp_path):
    path = tmp_path / "embeddings.f32"
    index = VectorIndex(path, dim=2)
    for i in range(4):
        index.add(unit(1, i))
    index.flush()
    index.keep(np.array([True, False, True, False]))
    assert index.n == 2 and np.allclose(VectorIndex(path, dim=2).vectors, index.vectors)


def test_legacy_npy_is_converted_once(tmp_path):
    legacy = tmp_path / "embeddings.npy"
    np.save(legacy, np.eye(3, dtype=np.float32))
    index = VectorIndex(tmp_path / "embeddings.f32", dim=3, legacy=legacy)
    assert index.n == 3 and not legacy.exists() and (tmp_path / "embeddings.npy.bak").exists()
    wrong = tmp_path / "other.npy"
    np.save(wrong, np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="expected 3"):
        VectorIndex(tmp_path / "other.f32", dim=3, legacy=wrong)


# ============================================================================
# Filters
# ============================================================================


class Recorder(Filter):
    def __init__(self, ctx, verdict=KEEP):
        super().__init__(ctx)
        self.verdict = verdict
        self.calls = 0

    def check(self, s):
        self.calls += 1
        return self.verdict


def test_chain_stops_at_the_first_rejection(ctx_factory):
    ctx = ctx_factory()
    reject, expensive = Recorder(ctx, Verdict(False, "x")), Recorder(ctx)
    assert FilterChain([reject, expensive]).check(sample("a")).reason == "x"
    assert expensive.calls == 0
    assert FilterChain([Recorder(ctx), Recorder(ctx)]).check(sample("a")).keep
    assert not FilterChain([Recorder(ctx)]).needs_vector


def test_chain_is_built_from_the_dataset_type_in_order(ctx_factory):
    chain = FilterChain.from_type(ctx_factory())
    assert [type(f) for f in chain.filters] == [DuplicateFilter, ReferenceFilter] and chain.needs_vector


def test_duplicate_filter_uses_the_dataset_index(ctx_factory):
    ctx = ctx_factory()
    ctx.store = type("S", (), {"index": VectorIndex(dim=2)})()
    ctx.store.index.add(unit(1, 0))
    dup = DuplicateFilter(ctx, max_similarity=0.9)
    same, other = sample("a", vector=unit(1, 0.01)), sample("b", vector=unit(0, 1))
    assert dup.check(same) == Verdict(False, "duplicate", pytest.approx(same.max_sim))
    assert dup.check(other).keep and other.max_sim == pytest.approx(0.0)


@pytest.mark.parametrize("vector,keep", [
    (unit(1, 0, 0), True),     # closest to good examples
    (unit(0, 1, 0), False),    # closest to noise examples
    (unit(0, 0, 1), True),     # nothing similar enough: kept
])
def test_vote_weighs_good_and_noise_examples(vector, keep):
    refs = np.stack([unit(1, 0, 0), unit(1, 0.2, 0), unit(0, 1, 0), unit(0.1, 1, 0)])
    labels = np.array([1, 1, -1, -1], np.float32)
    verdict = vote(refs, labels, vector, k=3, min_similarity=0.5)
    assert verdict.keep is keep and (keep or verdict.reason == "noise")


def test_reference_filter_without_examples_never_loads_the_encoder(ctx_factory):
    ctx = ctx_factory()
    ctx.encoder = lambda: pytest.fail("the encoder must not be loaded without reference examples")
    assert ReferenceFilter(ctx).check(sample("a", vector=unit(1, 0))) == KEEP


def test_reference_set_encodes_only_new_or_changed_files(tmp_path):
    encoder = FakeEncoder()
    refs = ReferenceSet(tmp_path / "refs", lambda: encoder)
    files = []
    for i in range(3):
        files.append(tmp_path / f"r{i}.jpg")
        write_jpeg(files[-1], textured_background(32, 32, i))
    assert refs.add("good", files[:2]) == 2 and refs.add("noise", files[2:]) == 1
    vectors, labels = refs.load()
    assert vectors.shape == (3, FakeEncoder.dim) and sorted(labels) == [-1, 1, 1] and encoder.images == 3
    refs.load()
    assert encoder.images == 3  # all cached
    write_jpeg(tmp_path / "refs" / "good" / "r0.jpg", textured_background(32, 32, 9))
    refs.load()
    assert encoder.images == 4  # only the changed file
    with pytest.raises(ValueError):
        refs.add("maybe", files)


def test_quality_rules():
    rules = QualityRules(min_conf=0.5, min_height=100, min_sharpness=10)
    assert rules.check_detection(0.4, 200) == "low_conf" and rules.check_detection(0.9, 50) == "too_small"
    assert rules.check_detection(0.9, 200) is None
    assert rules.check_image(np.full((50, 50), 128, np.uint8), False) == ("broken", 0.0)
    reason, score = rules.check_image(textured_background(50, 50)[:, :, 0], False)
    assert reason is None and score > 10
    assert rules.check_values(0.9, 200, 5) == "blurry" and rules.check_values(0.9, 200, 50) is None


# ============================================================================
# Dataset store
# ============================================================================


def test_store_saves_images_rows_vectors_and_processed_sources(tmp_path):
    store = DatasetStore(tmp_path / "ds", dim=2)
    store.add(sample("src_f0000001_id1.jpg", vector=unit(1, 0), conf=0.9, box=(1, 2, 3, 4), det_class="person"),
              ITEM)
    store.mark_done("src")
    store.close()
    rows = read_csv(tmp_path / "ds" / "crops.csv")
    assert rows[0]["file"] == "src_f0000001_id1.jpg" and rows[0]["license"] == "cc-by-4.0"
    assert rows[0]["source_name"] == "youtube" and rows[0]["det_class"] == "person" and rows[0]["x2"] == "3"
    assert (tmp_path / "ds" / "images" / "src_f0000001_id1.jpg").exists()
    assert sorted(p.name for p in (tmp_path / "ds").iterdir()) == ["crops.csv", "embeddings.f32", "images",
                                                                    "processed.txt"]  # no labels: collecting only
    reopened = DatasetStore(tmp_path / "ds", dim=2)
    assert reopened.is_done("src") and reopened.count == 1 and reopened.index.n == 1


def test_interrupted_source_is_rolled_back_on_the_next_run(tmp_path):
    store = DatasetStore(tmp_path / "ds", dim=2)
    store.add(sample("done_p0.jpg", "done", vector=unit(1, 0)), ITEM)
    store.mark_done("done")
    store.add(sample("half_f0000001_id1.jpg", "half", vector=unit(0, 1)), ITEM)  # killed before mark_done
    store.close()
    reopened = DatasetStore(tmp_path / "ds", dim=2)
    assert [r["file"] for r in read_csv(reopened.csv_path)] == ["done_p0.jpg"]
    assert not (tmp_path / "ds" / "images" / "half_f0000001_id1.jpg").exists()
    assert reopened.index.n == 1 and reopened.count == 1


def test_worker_dataset_folder_is_upgraded_in_place(tmp_path):
    folder = tmp_path / "ds"
    (folder / "images").mkdir(parents=True)
    old_fields = CSV_FIELDS[:13]
    with open(folder / "crops.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, old_fields)
        writer.writeheader()
        writer.writerow({k: "1" for k in old_fields} | {"file": "a_p0.jpg", "source": "a"})
    (folder / "processed.txt").write_text("a\n", encoding="utf-8")
    np.save(folder / "embeddings.npy", np.ones((1, 2), np.float32))
    store = DatasetStore(folder, dim=2)
    store.close()
    rows = read_csv(folder / "crops.csv")
    assert list(rows[0]) == CSV_FIELDS and rows[0]["file"] == "a_p0.jpg" and rows[0]["license"] == ""
    assert store.index.n == 1 and (folder / "embeddings.f32").exists()


def test_remove_keeps_rows_and_vectors_aligned(tmp_path):
    store = DatasetStore(tmp_path / "ds", dim=2)
    for i in range(4):
        store.add(sample(f"s_p{i}.jpg", "s", vector=unit(1, i)), ITEM)
    store.mark_done("s")
    assert store.remove({"s_p1.jpg", "s_p3.jpg"}) == 2 and store.remove({"nothing"}) == 0
    store.add(sample("t_p0.jpg", "t", vector=unit(0, 1)), ITEM)
    store.mark_done("t")
    store.close()
    reopened = DatasetStore(tmp_path / "ds", dim=2)
    assert [r["file"] for r in read_csv(reopened.csv_path)] == ["s_p0.jpg", "s_p2.jpg", "t_p0.jpg"]
    assert np.allclose(reopened.index.vectors, [unit(1, 0), unit(1, 2), unit(0, 1)])


def test_missing_vectors_are_filled_in(tmp_path):
    store = DatasetStore(tmp_path / "ds", dim=FakeEncoder.dim)
    for i in range(3):
        store.add(sample(f"s_p{i}.jpg", "s", seed=i), ITEM)  # saved while duplicate checks were off
    store.mark_done("s")
    encoder = FakeEncoder()
    store.ensure_vectors(encoder.encode)
    assert store.index.n == 3 and encoder.images == 3
    store.ensure_vectors(encoder.encode)
    assert encoder.images == 3


def test_rejected_store_keeps_images_only_for_reviewable_reasons(tmp_path):
    rejected = RejectedStore(tmp_path / "ds", done=set())
    rejected.add(sample("a.jpg"), Verdict(False, "noise", -0.7), ITEM)
    rejected.add(sample("b.jpg"), Verdict(False, "duplicate", 0.99), ITEM)
    rejected.close()
    assert (tmp_path / "ds" / "rejected" / "noise" / "a.jpg").exists()
    assert not (tmp_path / "ds" / "rejected" / "duplicate").exists()
    assert [r["reason"] for r in read_csv(rejected.csv_path)] == ["noise", "duplicate"]
    again = RejectedStore(tmp_path / "ds", done={"other"})  # "src" never finished: its rows are dropped
    again.close()
    assert read_csv(rejected.csv_path) == []


def test_report_summarises_the_dataset(tmp_path):
    folder = tmp_path / "ds"
    assert "No dataset yet" in dataset_report(folder)
    store = DatasetStore(folder, dim=2)
    store.add(sample("s_p0.jpg", vector=unit(1, 0), det_class="person"), ITEM)
    store.add(sample("s_p1.jpg", vector=unit(0, 1), det_class="person"), ITEM)
    store.mark_done("src")
    store.close()
    rejected = RejectedStore(folder, store.done)
    rejected.add(sample("x.jpg"), Verdict(False, "duplicate"), ITEM)
    rejected.close()
    report = dataset_report(folder)
    assert "images: 2" in report and "sources processed: 1" in report and "labelled" not in report
    assert "youtube 2" in report and "cc 2" in report and "person 2" in report and "duplicate 1" in report
