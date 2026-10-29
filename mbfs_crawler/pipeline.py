"""One run: collect items -> skip what is already done -> download (one ahead) -> process -> filter -> store.

Work is avoided before it is done: known keys are skipped before any download, the same item found by two queries
or sources is handled once, titles are screened before downloading, and the next video downloads while the current
one is processed (the only concurrency in the tool).
"""

import os
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from mbfs_crawler.config import PROCESSORS, SOURCES
from mbfs_crawler.downloads import ImageDownloader, VideoDownloader
from mbfs_crawler.filters import FilterChain, ReferenceSet, vote
from mbfs_crawler.processors import Sample, VideoOpenError
from mbfs_crawler.sources import MediaItem, MediaType, resolve_inputs
from mbfs_crawler.store import DatasetStore, RejectedStore, read_csv
from mbfs_crawler.utils import log, progress, read_image, short_error

T = TypeVar("T")
R = TypeVar("R")
_DONE = object()


def prefetch(items: Iterable[T], fetch: Callable[[T], R]) -> Iterator[tuple[T, R]]:
    """Fetch the next item in a background thread while the caller processes the current one.

    The queue holds one finished item, so at most two downloads wait at any time (disk use stays bounded). The thread
    is a daemon, so Ctrl+C never waits for a download to finish.
    """
    results: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            for item in items:
                results.put((item, fetch(item), None))
        except BaseException as err:  # noqa: BLE001 - re-raised in the caller's thread
            results.put((None, None, err))
            return
        results.put(_DONE)

    threading.Thread(target=worker, name="download", daemon=True).start()
    while True:
        entry = results.get()
        if entry is _DONE:
            return
        item, result, error = entry
        if error is not None:
            raise error
        yield item, result


def make_sources(ctx, names: Iterable[str], limit: int | None = None) -> list:
    """Create the sources up front (with their options from datasets.yaml), so a bad option fails before any work."""
    sources = []
    for name in names:
        options = dict(ctx.dataset_type.sources.get(name, {}))
        if limit is not None:
            options["limit"] = limit
        sources.append(SOURCES.create(name, ctx=ctx, **options))
    return sources


def collect(ctx, queries: Iterable[str], inputs: Iterable[str], sources: Iterable) -> Iterator[MediaItem]:
    """Inputs (-i) first, then every query on every source. Sources run lazily, as items are consumed."""
    yield from resolve_inputs(ctx, inputs, exclude_dirs=ctx.paths.result_dirs())
    for query in queries:
        for source in sources:
            log.info(f"🔎 {source.registry_name}: {query}")
            try:
                yield from source.search(query)
            except Exception as err:  # noqa: BLE001 - one broken source must not stop the run
                log.warning(f"! {source.registry_name} failed for '{query}': {short_error(err)}")
                log.debug("details", exc_info=True)


class Pipeline:
    def __init__(self, ctx, force: bool = False, delete_videos: bool = False):
        self.ctx = ctx
        dataset_type = ctx.dataset_type
        self.processor = PROCESSORS.create(dataset_type.processor_type, ctx=ctx, **dataset_type.processor_options)
        self.store = DatasetStore(ctx.dataset_dir, dim=ctx.encoder().dim)  # the encoder model is not loaded yet
        ctx.store = self.store
        self.rejected = RejectedStore(ctx.dataset_dir, self.store.done)
        self.chain = FilterChain.from_type(ctx)
        download = dataset_type.download
        self.videos = VideoDownloader(ctx, max_res=download.get("max_res", 1080),
                                      max_minutes=download.get("max_minutes"))
        self.images = ImageDownloader(ctx)
        self.max_duration = download.get("max_duration")
        self.exclude_words = [w.lower() for w in dataset_type.exclude_words]
        self.force = force
        self.delete_videos = delete_videos
        self.seen: set[str] = set()
        self.filtered = 0  # samples rejected by the filter chain
        # Vectors are computed only if a filter needs them, or to keep an existing index in step with crops.csv.
        self.use_vectors = self.chain.needs_vector or self.store.index.n > 0
        if self.use_vectors:
            self.store.ensure_vectors(lambda images: ctx.encoder().encode(images))

    def run(self, items: Iterable[MediaItem]):
        images: list[MediaItem] = []
        try:
            for item, path in prefetch(self._videos_to_fetch(items, images), self.videos.fetch):
                if path is not None:
                    self._process_video(item, path)
            self._process_images(images)
        finally:
            self.store.close()
            self.rejected.close()
        return self.ctx.stats

    def _videos_to_fetch(self, items: Iterable[MediaItem], images: list[MediaItem]) -> Iterator[MediaItem]:
        """Runs in the download thread: screens items and sets images aside for batch processing at the end."""
        for item in items:
            if not self._wanted(item):
                continue
            if item.media_type is MediaType.IMAGE:
                images.append(item)
            else:
                yield item

    def _wanted(self, item: MediaItem) -> bool:
        stats = self.ctx.stats
        if item.key in self.seen:
            return False
        self.seen.add(item.key)
        if self.store.is_done(item.key) and not self.force:
            stats.sources_skipped += 1
            if item.media_type is MediaType.VIDEO:
                log.info(f"- {item.key}: already processed, skipping (use --force to redo)")
            return False
        title = item.title.lower()
        if title and any(word in title for word in self.exclude_words):
            stats.excluded += 1
            log.debug(f"- {item.key}: title '{item.title}' matches exclude_words")
            return False
        if self.max_duration and item.duration and item.duration > self.max_duration and not self.videos.max_minutes:
            stats.excluded += 1
            log.info(f"- {item.key}: {item.duration / 60:.0f} min is longer than max_duration, skipped")
            return False
        return True

    def submit(self, samples: list[Sample], item: MediaItem) -> list[bool]:
        """Filter and store samples in order; vectors for the whole list are computed in one batch first."""
        if not samples:
            return []
        if self.use_vectors:
            missing = [s for s in samples if s.vector is None]
            if missing:
                for sample, vector in zip(missing, self.ctx.encoder().encode([s.image for s in missing]), strict=True):
                    sample.vector = vector
        kept = []
        for sample in samples:
            verdict = self.chain.check(sample)
            if verdict.keep:
                self.store.add(sample, item)
                self.ctx.stats.saved += 1
            else:
                self.rejected.add(sample, verdict, item)
                self.ctx.stats.rejected[verdict.reason] += 1
                self.filtered += 1
            kept.append(verdict.keep)
        return kept

    def _process_video(self, item: MediaItem, path: Path) -> None:
        stats = self.ctx.stats
        saved, filtered = stats.saved, self.filtered
        try:
            finished = self.processor.process_video(path, item, lambda samples: self.submit(samples, item))
        except VideoOpenError as err:
            log.warning(f"! {err}")
            return
        except Exception as err:  # noqa: BLE001 - one broken video must never stop the run
            log.warning(f"! {item.key}: processing failed ({short_error(err)}); skipped, it is retried next run")
            log.debug("details", exc_info=True)
            return
        self.store.mark_done(item.key)
        stats.sources_done += 1
        stats.sources_stopped_early += not finished
        note = "" if finished else " (stopped early: nothing new)"
        log.info(f"✓ {item.key}: saved {stats.saved - saved}, rejected {self.filtered - filtered}{note}")
        if self.delete_videos and not item.local:
            path.unlink(missing_ok=True)

    def _process_images(self, items: list[MediaItem]) -> None:
        if not items:
            return
        stats = self.ctx.stats
        saved, filtered, unreadable = stats.saved, self.filtered, 0
        with progress(total=len(items), desc="images", unit="img") as bar:
            for start in range(0, len(items), self.ctx.batch):
                chunk = items[start:start + self.ctx.batch]
                loaded = []
                for item, path in self.images.fetch_all(chunk):
                    image = read_image(path) if path else None
                    if image is None:
                        unreadable += 1
                        continue
                    loaded.append((item, image))
                try:
                    for (item, _), samples in zip(loaded, self.processor.process_images(loaded), strict=True):
                        self.submit(samples, item)
                        self.store.mark_done(item.key)
                        stats.sources_done += 1
                except Exception as err:  # noqa: BLE001 - one broken batch must never stop the run
                    log.warning(f"! {len(loaded)} images could not be processed ({short_error(err)}); skipped, "
                                "they are retried next run")
                    log.debug("details", exc_info=True)
                bar.update(len(chunk))
        log.info(f"✓ {len(items)} images: saved {stats.saved - saved}, rejected {self.filtered - filtered}, "
                 f"{unreadable} could not be downloaded or read")


def clean_dataset(ctx, dry_run: bool = False) -> dict[str, str]:
    """Re-apply the quality rules and the reference filter to images already in the dataset.

    Uses only stored values (crops.csv) and stored vectors, so no image is decoded and no detector runs.
    Rejected images move to rejected/<reason>/. Returns {file: reason}.
    """
    dataset_type = ctx.dataset_type
    store = DatasetStore(ctx.dataset_dir, dim=ctx.encoder().dim)
    ctx.store = store
    rejected = RejectedStore(ctx.dataset_dir, store.done)
    try:
        rows = read_csv(store.csv_path)
        refs = None
        if "reference" in dataset_type.filters:
            refs = ReferenceSet(ctx.paths.references(dataset_type.name), ctx.encoder).load()
            if not len(refs[0]):
                refs = None
            elif store.index.n < len(rows):
                store.ensure_vectors(lambda images: ctx.encoder().encode(images))
        options = dataset_type.filters.get("reference", {})
        rules = ctx.quality
        check_detection = dataset_type.processor_type == "crop"  # whole frames may be background (no detection)
        reasons: dict[str, str] = {}
        for i, row in enumerate(rows):
            conf, sharp = float(row.get("conf") or 0), float(row.get("sharpness") or 0)
            height = float(row.get("y2") or 0) - float(row.get("y1") or 0)
            reason = (rules.check_values(conf, height, sharp) if check_detection
                      else ("blurry" if sharp < rules.min_sharpness else None))
            if reason is None and refs is not None:
                verdict = vote(refs[0], refs[1], store.index.vectors[i], options.get("k", 7),
                               options.get("min_similarity", 0.55))
                reason = None if verdict.keep else verdict.reason
            if reason:
                reasons[row["file"]] = reason
                if not dry_run:
                    _move_rejected(store, rejected, row, reason)
        if reasons and not dry_run:
            store.remove(set(reasons))
        return reasons
    finally:
        store.close()
        rejected.close()


def _move_rejected(store: DatasetStore, rejected: RejectedStore, row: dict, reason: str) -> None:
    folder = rejected.dir / reason
    folder.mkdir(parents=True, exist_ok=True)
    image = store.images_dir / row["file"]
    if image.exists():
        os.replace(image, folder / row["file"])
    rejected.add_row(row["file"], reason, 0.0, row["source"], row.get("origin", ""))
