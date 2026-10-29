"""Downloads with a shared cache in downloads/, so nothing is downloaded twice (by any dataset type)."""

import mimetypes
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from mbfs_crawler.config import IMAGE_EXTS, VIDEO_EXTS
from mbfs_crawler.sources import WIKIMEDIA_HEADERS, MediaItem, http_get
from mbfs_crawler.utils import log, short_error, url_suffix

# ==============================================================================================================
# Downloaders with a shared cache in downloads/, so nothing is downloaded twice (by any dataset type).
# ==============================================================================================================

_PARTIAL = (".part", ".ytdl", ".temp", ".tmp")


def scan_cache(folder: Path) -> dict[str, Path]:
    """{file stem: path} for finished files, built with one directory listing instead of a glob per lookup."""
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in folder.iterdir() if p.is_file() and not p.name.endswith(_PARTIAL)}


# ==============================================================================================================
# Image downloads: several at once (they mostly wait on the network), checked to really be images.
# ==============================================================================================================

MAX_IMAGE_BYTES = 50_000_000


class ImageDownloader:
    def __init__(self, ctx, folder: Path | None = None, workers: int = 8, max_bytes: int = MAX_IMAGE_BYTES):
        self.ctx = ctx
        self.folder = folder or ctx.paths.images
        self.workers = workers
        self.max_bytes = max_bytes
        self._cache: dict[str, Path] | None = None

    def fetch(self, item: MediaItem) -> Path | None:
        """Path of the image file, downloading it if needed. None if the link is not a readable image."""
        if item.local:
            return Path(item.uri)
        if self._cache is None:
            self._cache = scan_cache(self.folder)
        cached = self._cache.get(item.key)
        if cached:
            return cached
        try:
            path = self._download(item)
        except Exception as err:  # noqa: BLE001 - one bad link must never stop the run
            log.warning(f"! cannot download {item.uri}: {short_error(err)}")
            log.debug("details", exc_info=True)
            return None
        if path:
            self._cache[item.key] = path
        return path

    def fetch_all(self, items: Sequence[MediaItem]) -> Iterator[tuple[MediaItem, Path | None]]:
        """Download in parallel, results in the original order."""
        if self._cache is None:
            self._cache = scan_cache(self.folder)
        if self.workers <= 1 or len(items) <= 1:
            yield from ((item, self.fetch(item)) for item in items)
            return
        with ThreadPoolExecutor(self.workers) as pool:
            yield from zip(items, pool.map(self.fetch, items), strict=True)

    def _download(self, item: MediaItem) -> Path | None:
        import requests

        try:
            with self.ctx.http().get(item.uri, timeout=30, stream=True) as resp:
                log.debug(f"GET {item.uri}: HTTP {resp.status_code}")
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if content_type.startswith("text/"):  # a web page, not an image
                    log.warning(f"! {item.uri} is a web page, not an image or a video")
                    return None
                data = bytearray()
                for chunk in resp.iter_content(1 << 16):
                    data += chunk
                    if len(data) > self.max_bytes:
                        log.warning(f"! {item.uri} is larger than {self.max_bytes / 1e6:.0f} MB, skipped")
                        return None
        except requests.RequestException as err:
            log.warning(f"! cannot download {item.uri}: {short_error(err)}")
            return None
        # Many sites send images as application/octet-stream, so check the data itself rather than the header.
        if cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8) is None:
            log.warning(f"! {item.uri} is not a readable image")
            return None
        suffix = url_suffix(item.uri)
        if suffix not in IMAGE_EXTS:
            guessed = mimetypes.guess_extension(content_type) if content_type.startswith("image/") else None
            suffix = guessed if guessed in IMAGE_EXTS else ".jpg"
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{item.key}{suffix}"
        path.write_bytes(data)
        self.ctx.stats.downloaded_bytes += len(data)
        return path


# ==============================================================================================================
# Video downloads with yt-dlp: picture only (no sound), no higher than needed, optionally only the first minutes.
# ==============================================================================================================

def ffmpeg_path() -> str | None:
    """ffmpeg shipped with the imageio-ffmpeg package, so nobody has to install it (Windows, macOS, Linux)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall back to an ffmpeg on PATH, if any
        return None


MAX_VIDEO_BYTES = 2_000_000_000  # direct video files larger than this are skipped


class VideoDownloader:
    def __init__(self, ctx, folder: Path | None = None, max_res: int = 1080, max_minutes: float | None = None):
        self.ctx = ctx
        self.folder = folder or ctx.paths.videos
        self.max_res = max_res
        self.max_minutes = max_minutes
        self._cache: dict[str, Path] | None = None

    def fetch(self, item: MediaItem) -> Path | None:
        """Path of the video file, downloading it if needed. None if every attempt failed."""
        if item.local:
            return Path(item.uri)
        if self._cache is None:
            self._cache = scan_cache(self.folder)
        cached = self._cache.get(item.key)
        if cached:
            return cached
        log.info(f"↓ {item.title or item.uri}")
        started = time.monotonic()
        try:
            path = self._download_direct(item) if url_suffix(item.uri) in VIDEO_EXTS else self._download(item)
        except Exception as err:  # noqa: BLE001 - one bad link must never stop the run
            log.warning(f"! cannot download {item.title or item.uri}: {short_error(err)}")
            log.debug("details", exc_info=True)
            return None
        if path:
            self._cache[item.key] = path
            size = path.stat().st_size
            self.ctx.stats.downloaded_bytes += size
            log.debug(f"downloaded {path.name}: {size / 1e6:.1f} MB in {time.monotonic() - started:.0f} s")
        return path

    def options(self, item: MediaItem) -> dict:
        import yt_dlp

        options = {
            **self.ctx.ytdlp_options(),
            "noprogress": True,       # yt-dlp's own progress bar would collide with ours
            "outtmpl": str(self.folder / f"{item.key}.%(ext)s"),  # named by key, so the cache finds it again
            "format": "bv*/b",        # picture only when available: no audio to download
            # ffmpeg (used for time ranges and HLS) prints to the terminal by itself: only real errors, no stats
            "external_downloader_args": {"ffmpeg_i": ["-nostdin"], "ffmpeg": ["-loglevel", "error", "-nostats"]},
            "playlist_items": "1",    # a link that turns out to be a playlist: only its first video
        }
        ffmpeg = ffmpeg_path()
        if ffmpeg:
            options["ffmpeg_location"] = ffmpeg
        if self.max_minutes:
            options["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, self.max_minutes * 60)])
        return options

    def _download(self, item: MediaItem) -> Path | None:
        import yt_dlp

        options = self.options(item)
        # Up to max_res, preferring H.264 (OpenCV reads it everywhere). YouTube sometimes answers 403 for direct
        # links; its HLS streams usually still work, so they are the second attempt.
        preferences = [
            [f"res:{self.max_res}", "vcodec:h264", "ext:mp4"],
            [f"res:{self.max_res}", "proto:m3u8", "vcodec:h264"],
        ]
        error = None
        for attempt, preference in enumerate(preferences, start=1):
            try:
                with yt_dlp.YoutubeDL({**options, "format_sort": preference}) as ydl:
                    info = ydl.extract_info(item.uri, download=True)
                if info and info.get("entries"):
                    info = next(e for e in info["entries"] if e)
                return Path(info["requested_downloads"][0]["filepath"])
            except Exception as err:  # noqa: BLE001 - extractors raise many kinds of errors (also on odd pages)
                error = err
                log.debug(f"download attempt {attempt} failed for {item.uri}: {err}")
        hint = " (try --cookies-from-browser firefox)" if "youtu" in item.uri else ""
        log.warning(f"! cannot download {item.title or item.uri}: {short_error(error)}{hint}")
        return None

    def _download_direct(self, item: MediaItem) -> Path | None:
        """A link straight to a video file (.mp4, .webm, ...): plain HTTP, no yt-dlp in between.

        Wikimedia asks scripts to identify themselves and answers "429 too many requests" to anonymous clients, so
        its files are requested with a descriptive User-Agent; http_get waits and retries on 429.
        """
        headers = WIKIMEDIA_HEADERS if "wikimedia.org" in item.uri else None
        resp = http_get(self.ctx, item.uri, retries=4, timeout=60, stream=True, headers=headers)
        if resp is None:
            return None
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{item.key}{url_suffix(item.uri)}"
        partial = path.with_name(path.name + ".part")  # invisible to the cache until complete
        size = 0
        with resp, open(partial, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    break
                f.write(chunk)
        if size > MAX_VIDEO_BYTES:
            partial.unlink(missing_ok=True)
            log.warning(f"! {item.uri} is larger than {MAX_VIDEO_BYTES / 1e9:.0f} GB, skipped")
            return None
        os.replace(partial, path)
        return path
