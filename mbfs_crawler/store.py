"""The dataset folder: saved samples and their origin, rejected samples, and the stats report."""

import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np

from mbfs_crawler.config import IMAGE_EXTS
from mbfs_crawler.filters import Verdict
from mbfs_crawler.models import VectorIndex
from mbfs_crawler.processors import Sample
from mbfs_crawler.sources import MediaItem, license_category
from mbfs_crawler.utils import read_image, write_jpeg

# ==============================================================================================================
# The dataset folder of one dataset type, kept consistent across interrupted and repeated runs.
#
#     <out>/images/                 saved samples (crops or frames): the result
#     <out>/crops.csv               where every image came from (+ license, detector class, scores)
#     <out>/processed.txt           sources fully processed; skipped on later runs (unless --force)
#     <out>/embeddings.f32          image vectors, row i = row i of crops.csv (duplicate / reference checks)
#
# Same file formats as worker-dataset (images/, crops.csv, processed.txt). Labelling happens outside this tool.
# ==============================================================================================================

# x1..y2 is the saved area (with padding) in the original frame, in pixels.
CSV_FIELDS = ["file", "source", "origin", "frame", "time_s", "track_id", "conf", "x1", "y1", "x2", "y2",
              "sharpness", "max_sim", "det_class", "source_name", "license"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Rewrite a csv atomically (a crash never leaves a half-written file)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


class DatasetStore:
    def __init__(self, out_dir: Path, dim: int = 384):
        self.out_dir = out_dir
        self.images_dir = out_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._done_path = out_dir / "processed.txt"
        self.done: set[str] = (set(self._done_path.read_text(encoding="utf-8").split())
                               if self._done_path.exists() else set())
        self.index = VectorIndex(out_dir / "embeddings.f32", dim, legacy=out_dir / "embeddings.npy")
        self.csv_path = out_dir / "crops.csv"
        self.count = self._drop_unfinished_sources()
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, CSV_FIELDS, restval="")

    def _drop_unfinished_sources(self) -> int:
        """A source interrupted halfway (Ctrl+C) is redone from the start, so remove what it left behind.

        Also upgrades an older crops.csv header. Returns the number of rows kept.
        """
        header_ok = False
        if self.csv_path.exists():
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                header_ok = next(csv.reader(f), None) == CSV_FIELDS
        rows = read_csv(self.csv_path)
        kept = [row for row in rows if row["source"] in self.done]
        for row in rows:
            if row["source"] not in self.done:
                (self.images_dir / row["file"]).unlink(missing_ok=True)
        if len(kept) < len(rows) or not header_ok:
            write_csv(self.csv_path, CSV_FIELDS, kept)
        self.index.truncate(len(kept))  # finished sources always come first, so vector rows still line up
        return len(kept)

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str) -> None:
        """Record a fully processed source. Writes only what is new (csv rows were flushed as they came)."""
        self._csv_file.flush()
        self.index.flush()
        if key not in self.done:
            self.done.add(key)
            with open(self._done_path, "a", encoding="utf-8") as f:
                f.write(key + "\n")

    def add(self, sample: Sample, item: MediaItem) -> None:
        write_jpeg(self.images_dir / sample.name, sample.image)
        if sample.vector is not None:
            self.index.add(sample.vector)
        x1, y1, x2, y2 = sample.box
        self._csv.writerow({
            "file": sample.name, "source": sample.source_key, "origin": item.origin, "frame": sample.frame,
            "time_s": f"{sample.time_s:.2f}", "track_id": sample.track_id, "conf": f"{sample.conf:.3f}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "sharpness": f"{sample.sharpness:.1f}",
            "max_sim": f"{sample.max_sim:.3f}", "det_class": sample.det_class, "source_name": item.source_name,
            "license": item.license,
        })
        self._csv_file.flush()  # keep crops.csv in step with images/ even if the run is killed
        self.count += 1

    def ensure_vectors(self, encode) -> None:
        """Encode images that have no vector yet (e.g. saved while duplicate checks were off)."""
        if self.index.n >= self.count:
            return
        rows = read_csv(self.csv_path)[self.index.n:]
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            images = [read_image(self.images_dir / row["file"]) for row in batch]
            images = [img if img is not None else np.zeros((8, 8, 3), np.uint8) for img in images]
            for vector in encode(images):
                self.index.add(vector)
        self.index.flush()

    def remove(self, files: set[str]) -> int:
        """Drop rows (and their vectors) for these image names. Image files are moved by the caller."""
        self._csv_file.flush()
        rows = read_csv(self.csv_path)
        keep = np.array([row["file"] not in files for row in rows], bool)
        if keep.all():
            return 0
        self._csv_file.close()
        write_csv(self.csv_path, CSV_FIELDS, [row for row, k in zip(rows, keep, strict=True) if k])
        if self.index.n == len(rows):
            self.index.keep(keep)
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, CSV_FIELDS, restval="")
        self.count = int(keep.sum())
        return int((~keep).sum())

    def close(self) -> None:
        self._csv_file.close()
        self.index.flush()


# ==============================================================================================================
# Samples the filter chain rejected, logged in rejected.csv for review.
#
# Images are kept only for reasons worth reviewing (by default "noise", decided by your reference examples), never
# for duplicates: saving thousands of near-copies would cost disk for nothing.
# ==============================================================================================================

REJECTED_FIELDS = ["file", "reason", "score", "source", "origin"]


class RejectedStore:
    def __init__(self, out_dir: Path, done: set[str], save_images_for: tuple[str, ...] = ("noise",)):
        self.dir = out_dir / "rejected"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_images_for = save_images_for
        self.csv_path = self.dir / "rejected.csv"
        header_ok = False
        if self.csv_path.exists():
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                header_ok = next(csv.reader(f), None) == REJECTED_FIELDS
        rows = read_csv(self.csv_path) if header_ok else []
        kept = [row for row in rows if row.get("source") in done]  # same rollback as the dataset store
        if not header_ok or len(kept) < len(rows):  # new file, or one left without a header by a killed run
            write_csv(self.csv_path, REJECTED_FIELDS, kept)
        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._file, REJECTED_FIELDS)

    def add(self, sample: Sample, verdict: Verdict, item: MediaItem) -> None:
        if verdict.reason in self.save_images_for:
            folder = self.dir / verdict.reason
            folder.mkdir(exist_ok=True)
            write_jpeg(folder / sample.name, sample.image)
        self.add_row(sample.name, verdict.reason, verdict.score, sample.source_key, item.origin)

    def add_row(self, file: str, reason: str, score: float, source: str, origin: str) -> None:
        self._csv.writerow({"file": file, "reason": reason, "score": f"{score:.3f}", "source": source,
                            "origin": origin})
        self._file.flush()  # a killed run must not leave half a row behind

    def close(self) -> None:
        self._file.close()


# ==============================================================================================================
# Text summary of a dataset folder (mbfs-crawler stats). Reads csv files only: instant, no model.
# ==============================================================================================================

def dataset_report(dataset_dir: Path) -> str:
    if not (dataset_dir / "crops.csv").exists():
        return f"No dataset yet in {dataset_dir} (run `mbfs-crawler run` first)."
    rows = read_csv(dataset_dir / "crops.csv")
    done = dataset_dir / "processed.txt"
    sources_done = len(done.read_text(encoding="utf-8").split()) if done.exists() else 0
    images_dir = dataset_dir / "images"
    image_names = ({p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
                   if images_dir.is_dir() else set())
    rejected = Counter(row["reason"] for row in read_csv(dataset_dir / "rejected" / "rejected.csv"))

    lines = [f"Dataset: {dataset_dir}",
             f"  images: {len(image_names)}   sources processed: {sources_done}"]
    lines += _counter_lines("by source", Counter(row.get("source_name") or "?" for row in rows))
    lines += _counter_lines("by license", Counter(license_category(row.get("license") or "unknown") for row in rows))
    lines += _counter_lines("by detected class", Counter(row.get("det_class") or "-" for row in rows))
    lines += _counter_lines("rejected", rejected)
    return "\n".join(lines)


def _counter_lines(title: str, counter: Counter) -> list[str]:
    if not counter:
        return []
    return [f"  {title}: " + ", ".join(f"{name} {count}" for name, count in counter.most_common())]
