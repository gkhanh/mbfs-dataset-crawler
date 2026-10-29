"""Small helpers: terminal output, safe file names, image checks and file I/O, compute device."""

import hashlib
import logging
import os
import re
import sys
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mbfs_crawler.config import IMAGE_EXTS

# ==============================================================================================================
# Terminal output: plain messages on stdout, an optional log file, and progress bars only on a real terminal.
# ==============================================================================================================

log = logging.getLogger("mbfs_crawler")
_progress = True
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _TerminalHandler(logging.Handler):
    """Prints through tqdm, so a message (also from the download thread) never breaks a progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:  # noqa: BLE001 - logging must never crash the program
            self.handleError(record)


class _WarningCounter(logging.Filter):
    count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            _WarningCounter.count += 1
        return True


def setup(log_file: str | Path | None = None, quiet: bool = False, debug: bool = False,
          progress: bool = True) -> None:
    """Terminal: short messages (all details with debug=True). Log file: timestamps, thread names, tracebacks."""
    global _progress
    for stream in (sys.stdout, sys.stderr):
        try:  # Windows consoles default to cp1252, which cannot print Vietnamese text or ✓
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    log.handlers.clear()
    log.filters.clear()
    log.propagate = False
    log.setLevel(logging.DEBUG)  # each handler decides what it shows
    log.addFilter(_WarningCounter())
    _WarningCounter.count = 0
    console = _TerminalHandler()
    console.setLevel(logging.DEBUG if debug else logging.WARNING if quiet else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"))
        log.addHandler(handler)
    # Progress bars are noise in a log file or a pipe, so they are shown only on an interactive terminal.
    _progress = progress and sys.stderr.isatty()


def warning_count() -> int:
    """Warnings logged since setup(), for the end-of-run summary."""
    return _WarningCounter.count


def short_error(err: BaseException | None) -> str:
    """One readable line from an exception. yt-dlp errors look like
    "ERROR: [generic] some-id: Unable to download webpage: HTTP Error 429: ... (caused by <...>)"."""
    text = _ANSI.sub("", str(err or "")).strip()
    if not text:
        return type(err).__name__ if err else "unknown error"
    text = text.splitlines()[0]
    connection = re.search(r"host='([^']+)'.*?Caused by (\w+)", text)  # requests: "HTTPSConnectionPool(host=...)"
    if connection and "Max retries exceeded" in text:
        return f"cannot connect to {connection.group(1)} ({connection.group(2)})"
    text = re.sub(r"^ERROR:\s*", "", text)
    text = re.sub(r"^\[[\w:.-]+\]\s+\S+?:\s+", "", text)  # "[generic] <id or link>: "
    text = re.sub(r"\s*\(caused by .*\)$", "", text)
    text = re.sub(r"\s+for url:\s+\S+$", "", text)  # requests repeats the link we already name in the message
    return text[:200]


class YtDlpLogger:
    """yt-dlp's own messages go into our log at debug level (shown with --debug, kept in the debug log file) instead
    of being printed over the progress bar. Failures are reported once, as a short warning, by our own code."""

    def debug(self, msg: str) -> None:
        log.debug("yt-dlp: %s", _ANSI.sub("", msg))

    info = warning = error = debug


def progress_enabled() -> bool:
    return _progress


def progress(iterable=None, **kwargs) -> tqdm:
    return tqdm(iterable, disable=not _progress, **kwargs)


# ==============================================================================================================
# Safe, stable names for files and sources, valid on Windows, macOS and Linux.
# ==============================================================================================================

# Windows refuses these as file names, with or without an extension.
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                               *(f"LPT{i}" for i in range(1, 10))})


def slug(text: str, limit: int = 60) -> str:
    """Any text -> a safe file-name part (letters, digits, '_' and '-' only)."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-")[:limit] or "x"
    return f"{name}_" if name.upper() in _WINDOWS_RESERVED else name


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:length]


def local_key(path: Path) -> str:
    """Stable key for a file on this machine: readable file name + 6 characters of the full path's hash."""
    return f"{slug(path.stem, 40)}_{short_hash(str(path.resolve()), 6)}"


def url_key(url: str) -> str:
    """Stable key for a downloaded link: readable file name + 8 characters of the link's hash."""
    return f"{slug(Path(urllib.parse.urlparse(url).path).stem, 40)}_{short_hash(url)}"


def video_key(extractor: str, video_id: str) -> str:
    """Key of an online video, e.g. "youtube_EdWbLUVnvQI". The same video gets the same key from every query."""
    return f"{slug(extractor).lower()}_{slug(video_id)}"


def url_suffix(url: str) -> str:
    return Path(urllib.parse.urlparse(url).path).suffix.lower()


def is_image_url(url: str) -> bool:
    """A link that ends in an image extension (https://.../photo.jpg)."""
    return url_suffix(url) in IMAGE_EXTS


# ==============================================================================================================
# Small image helpers. Functions accept BGR or grayscale, so callers convert to gray once and reuse it.
# ==============================================================================================================

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


# ----------------------------------------------------------------------------
# File access that also works with non-ASCII paths on Windows (cv2.imread/imwrite do not)
# ----------------------------------------------------------------------------

def read_image(path: Path | str) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_jpeg(path: Path | str, image: np.ndarray, quality: int = 95) -> None:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"cannot encode image for {path}")
    data.tofile(str(path))


def video_path(path: Path) -> str:
    """Path string OpenCV can open. On Windows, non-ASCII paths are swapped for their short (8.3) form."""
    text = str(path)
    if os.name != "nt" or text.isascii():
        return text
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    if ctypes.windll.kernel32.GetShortPathNameW(text, buffer, len(buffer)):
        return buffer.value
    return text


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pad_box(xyxy: tuple[float, float, float, float], pad: float, width: int, height: int) -> Box:
    """Grow a box by `pad` (fraction of its size) on every side, without leaving the frame."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    return (max(0, int(x1 - w * pad)), max(0, int(y1 - h * pad)),
            min(width, int(x2 + w * pad)), min(height, int(y2 + h * pad)))


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian, measured at a fixed height so large and small images compare fairly."""
    gray = to_gray(img)
    gray = cv2.resize(gray, (max(1, round(gray.shape[1] * 256 / gray.shape[0])), 256))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_broken(img: np.ndarray, touches_edge: bool) -> bool:
    """Crops a detector returns that are useless for labelling."""
    gray = to_gray(img)
    aspect = gray.shape[1] / gray.shape[0]
    if touches_edge and not 0.25 <= aspect <= 2.5:  # a thin strip cut off by the frame edge
        return True
    if gray.std() < 20:  # black or flat frame (scene transitions)
        return True
    # black bars from letterboxed or split-screen video
    return max((gray.mean(axis=1) < 12).mean(), (gray.mean(axis=0) < 12).mean()) > 0.15


def thumbnail(img: np.ndarray, width: int = 320) -> np.ndarray:
    """Small grayscale copy used to compare frames cheaply (motion check)."""
    gray = to_gray(img)
    height = max(1, round(gray.shape[0] * width / gray.shape[1]))
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def changed_fraction(a: np.ndarray, b: np.ndarray, level: int = 20) -> float:
    """Fraction of pixels that differ by more than `level` grey levels (camera noise stays below it)."""
    return float((cv2.absdiff(a, b) > level).mean())


# ==============================================================================================================
# Choosing the compute device and limiting CPU threads.
# ==============================================================================================================

def pick_device(device: str = "auto") -> str:
    """Ultralytics device: "0" (NVIDIA GPU), "mps" (Apple GPU) or "cpu". Anything other than "auto" is kept."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_cuda(device: str) -> bool:
    return device.isdigit() or device.startswith("cuda")


def torch_device(device: str) -> str:
    """Ultralytics writes GPUs as "0"; plain torch needs "cuda:0"."""
    return f"cuda:{device}" if device.isdigit() else device


def limit_threads(threads: int | None) -> None:
    """Cap CPU threads of torch, OpenCV and onnxruntime (call before they are imported)."""
    if not threads:
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    import cv2

    cv2.setNumThreads(threads)
