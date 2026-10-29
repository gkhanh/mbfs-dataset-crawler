"""Shared foundations: paths and constants, plugin registries, and the state of one run (RunContext)."""

from __future__ import annotations

import difflib
import importlib
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from mbfs_crawler.datasets import DatasetType


# ==============================================================================================================
# Paths and constants shared by the whole tool. Three places (see Paths below), the last two OUTSIDE the project:
#   the PROJECT folder (MBFS_HOME, default: the current folder)         code, models/, datasets.yaml
#   the DATA ROOT (--data-dir / MBFS_DATA, default: ../dataset)         <type>/ results, references/, logs/
#   the DOWNLOADS folder (--downloads-dir / MBFS_DOWNLOADS, ../downloads)  shared cache of videos and images
# ==============================================================================================================


VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".wmv", ".flv",
                        ".ogv"})
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")


# Folder names inside the data root that belong to the tool, so a dataset type cannot be called by them.
RESERVED_NAMES = ("references", "logs")


@dataclass(frozen=True)
class Paths:
    """Where things live. THIS CLASS decides every location the tool reads or writes.

    home         the project folder: code, models/, datasets.yaml, links/<type>/ (default: $MBFS_HOME, else the
                 current folder)
    data_root    what you keep (default: $MBFS_DATA, else a folder called "dataset" NEXT TO the project folder)
    downloads    raw downloads (default: $MBFS_DOWNLOADS, else a folder called "downloads" NEXT TO the project folder)

        <data_root>/<type>/         the result: images/, crops.csv, ...   (one folder per dataset type)
        <data_root>/references/     good / noise examples for the retrieval filter, one folder per type
        <data_root>/logs/           debug logs and crash reports
        <downloads>/videos/ images/ cache shared by every type: nothing is ever downloaded twice

    The project folder therefore contains only code.
    """

    home: Path
    data: Path | None = None            # None: the default described above
    downloads_dir: Path | None = None

    @classmethod
    def from_env(cls, home: str | Path | None = None, data: str | Path | None = None,
                 downloads: str | Path | None = None) -> Paths:
        def folder(value, variable):
            value = value or os.environ.get(variable)
            return Path(value).expanduser().resolve() if value else None

        home_dir = Path(home or os.environ.get("MBFS_HOME") or Path.cwd()).expanduser().resolve()
        return cls(home_dir, folder(data, "MBFS_DATA"), folder(downloads, "MBFS_DOWNLOADS"))

    @property
    def data_root(self) -> Path:
        return self.data or self.home.parent / "dataset"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def config(self) -> Path:
        """datasets.yaml: the dataset types (what to collect)."""
        return self.home / "datasets.yaml"

    def links(self, name: str) -> Path:
        """links/<type>/: the link lists of one type (`search` saves the links it finds here, as found.txt)."""
        return self.home / "links" / name

    @property
    def downloads(self) -> Path:
        return self.downloads_dir or self.home.parent / "downloads"

    @property
    def videos(self) -> Path:
        return self.downloads / "videos"

    @property
    def images(self) -> Path:
        return self.downloads / "images"

    @property
    def logs(self) -> Path:
        return self.data_root / "logs"

    def dataset(self, name: str) -> Path:
        return self.data_root / name

    def references(self, name: str) -> Path:
        return self.data_root / "references" / name

    def result_dirs(self) -> list[Path]:
        """Folders the tool writes results into (a folder given with -i is never scanned for its own output).
        The download cache is not among them: downloaded videos may be processed again on purpose."""
        return [p for p in self.data_root.iterdir() if p.is_dir()] if self.data_root.is_dir() else []


# ==============================================================================================================
# Plugin registries. A new source, processor, filter or detector is one class with a decorator:
#
#     @SOURCES.register("my_site")
#     class MySiteSource(Source):
#         ...
#
# The module holding a registry's plugins is imported the first time the registry is used, so adding a class with
# the decorator to sources.py (or processors.py, filters.py, models.py) is enough. Plugins must import heavy
# libraries (torch, ultralytics, yt_dlp, selenium) inside methods, never at the top, so light commands start instantly.
# ==============================================================================================================

T = TypeVar("T")


class RegistryError(ValueError):
    """Unknown or duplicate plugin name."""


class Registry(Generic[T]):
    def __init__(self, kind: str, module: str):
        self.kind = kind
        self.module = module
        self._classes: dict[str, type[T]] = {}
        self._loaded = False

    def register(self, name: str):
        def decorator(cls: type[T]) -> type[T]:
            if self._classes.get(name, cls) is not cls:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            self._classes[name] = cls
            cls.registry_name = name
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        self._load()
        try:
            return self._classes[name]
        except KeyError:
            close = difflib.get_close_matches(name, self._classes, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise RegistryError(f"unknown {self.kind} '{name}'. Available: {', '.join(self.names())}.{hint}") from None

    def create(self, name: str, **params) -> T:
        return self.get(name)(**params)

    def names(self) -> list[str]:
        self._load()
        return sorted(self._classes)

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        importlib.import_module(self.module)


SOURCES: Registry = Registry("source", "mbfs_crawler.sources")
PROCESSORS: Registry = Registry("processor", "mbfs_crawler.processors")
FILTERS: Registry = Registry("filter", "mbfs_crawler.filters")
DETECTORS: Registry = Registry("detector", "mbfs_crawler.models")


# ==============================================================================================================
# State shared by one run: paths, dataset type, device, counters, and models that are created only when first needed.
# ==============================================================================================================

@dataclass
class RunStats:
    """Counters printed at the end of a run, so the work avoided is visible."""

    started: float = field(default_factory=time.monotonic)
    sources_done: int = 0
    sources_skipped: int = 0
    sources_stopped_early: int = 0
    excluded: int = 0
    frames_decoded: int = 0
    frames_sampled: int = 0
    frames_skipped_adaptive: int = 0
    frames_skipped_motion: int = 0
    detector_calls: int = 0
    saved: int = 0
    rejected: Counter = field(default_factory=Counter)
    downloaded_bytes: int = 0

    def summary(self) -> str:
        minutes, seconds = divmod(time.monotonic() - self.started, 60)
        rejected = ", ".join(f"{n} {reason}" for reason, n in self.rejected.most_common()) or "0"
        return (f"sources: {self.sources_done} done, {self.sources_skipped} skipped, "
                f"{self.sources_stopped_early} stopped early (nothing new), {self.excluded} excluded by title\n"
                f"frames: {self.frames_decoded} decoded, {self.frames_sampled} sampled, skipped "
                f"{self.frames_skipped_adaptive} (adaptive) + {self.frames_skipped_motion} (no motion)\n"
                f"detector calls: {self.detector_calls} · saved: {self.saved} · rejected: {rejected}\n"
                f"downloaded: {self.downloaded_bytes / 1e6:.1f} MB · time: {int(minutes)}m {seconds:.0f}s")


@dataclass
class RunContext:
    paths: Paths
    dataset_type: DatasetType          # the block of datasets.yaml being collected
    device: str = "auto"               # resolved on first use, so light commands never import torch
    half: bool = True
    batch: int = 16
    cookies_from_browser: str | None = None
    stats: RunStats = field(default_factory=RunStats)
    store: Any = None                  # the DatasetStore of this run (set by the pipeline)
    detector_impl: Any = None          # created on first use; tests inject fakes here
    encoder_impl: Any = None
    _local: threading.local = field(default_factory=threading.local, repr=False)
    _device: str | None = field(default=None, repr=False)

    def compute_device(self) -> str:
        if self._device is None:
            from mbfs_crawler.utils import pick_device

            self._device = pick_device(self.device)
        return self._device

    @property
    def dataset_dir(self) -> Path:
        return self.paths.dataset(self.dataset_type.name)

    @property
    def quality(self):
        from mbfs_crawler.processors import QualityRules

        return QualityRules(**self.dataset_type.quality)

    def detector(self):
        """The dataset type's detector, loaded once per run. None when the type has no detector."""
        if self.detector_impl is None and self.dataset_type.detector is not None:
            self.detector_impl = DETECTORS.create(self.dataset_type.detector_type, ctx=self,
                                                  **self.dataset_type.detector_options)
        return self.detector_impl

    def encoder(self):
        """Image encoder for duplicate and reference checks (DINOv2-small), loaded once per run."""
        if self.encoder_impl is None:
            from mbfs_crawler.models import DinoV2Encoder

            self.encoder_impl = DinoV2Encoder(self.compute_device())
        return self.encoder_impl

    def http(self):
        """One keep-alive HTTP session per thread (requests sessions are not thread-safe)."""
        session = getattr(self._local, "session", None)
        if session is None:
            import requests

            session = self._local.session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
        return session

    def ytdlp_options(self) -> dict:
        from mbfs_crawler.utils import YtDlpLogger

        # All yt-dlp output goes to our log (debug level): nothing is printed over the progress bars.
        options: dict = {"logger": YtDlpLogger(), "quiet": False, "no_warnings": False}
        if self.cookies_from_browser:
            options["cookiesfrombrowser"] = (self.cookies_from_browser,)
        return options
