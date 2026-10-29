"""Where media comes from: search sources (no API key), links, local files, and licenses.

A new source is one class here with @SOURCES.register("name") and a search(query) method."""

import html
import json
import re
import time
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from mbfs_crawler.config import IMAGE_EXTS, SOURCES, USER_AGENT, VIDEO_EXTS
from mbfs_crawler.utils import (
    is_image_url,
    local_key,
    log,
    short_error,
    short_hash,
    slug,
    url_key,
    url_suffix,
    video_key,
)

# ==============================================================================================================
# What a source returns (MediaItem), the Source base class, licenses, and a polite HTTP helper.
# ==============================================================================================================

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class MediaItem:
    """One image or video to process, found by a search or given as an input."""

    uri: str                  # link, or a file path when local=True
    media_type: MediaType
    key: str                  # stable id; names the saved images and marks the item as processed
    title: str = ""
    source_name: str = ""
    license: str = "unknown"
    license_url: str = ""
    author: str = ""
    page_url: str = ""
    duration: float | None = None  # seconds, when the source knows it
    local: bool = False

    @property
    def origin(self) -> str:
        """Where the item came from, as written in crops.csv."""
        return self.page_url or self.uri


class Source(ABC):
    """Turns a search query into media items. Register subclasses with @SOURCES.register("name")."""

    registry_name = ""

    def __init__(self, ctx, limit: int = 20):
        self.ctx = ctx
        self.limit = limit

    @abstractmethod
    def search(self, query: str) -> Iterator[MediaItem]:
        """Yield at most `self.limit` items for `query`. Network errors are logged, never raised."""


# ============================================================================
# Licenses
# ============================================================================

LICENSE_CATEGORIES = ("pd", "cc", "cc-nc", "local", "unknown")


def normalize_license(text: str | None) -> str:
    """"CC BY-SA 4.0", "by-sa", "https://creativecommons.org/licenses/by/4.0/" -> "cc-by-sa-4.0", "cc-by-4.0", ..."""
    t = re.sub(r"[\s_]+", "-", (text or "").strip().lower())
    if not t:
        return "unknown"
    if "publicdomain/zero" in t or t.startswith("cc0") or t in {"cc-zero", "cc-cc0"}:
        return "cc0"
    if "publicdomain" in t or t in {"pd", "pdm"} or t.startswith(("public-domain", "pd-")):
        return "pd"
    match = re.search(r"creativecommons\.org/licenses/([a-z-]+)/([\d.]+)", t)
    if match:
        return f"cc-{match.group(1)}-{match.group(2)}"
    if re.fullmatch(r"by(-nc)?(-sa|-nd)?(-[\d.]+)?", t):
        return f"cc-{t}"
    return t


def license_category(license: str) -> str:
    """pd (public domain, CC0), cc (commercial use allowed), cc-nc (non-commercial or no-derivatives), local,
    unknown."""
    if license == "local":
        return "local"
    if license in {"pd", "cc0"}:
        return "pd"
    if license.startswith("cc-"):
        return "cc-nc" if "-nc" in license or "-nd" in license else "cc"
    return "unknown"


# ============================================================================
# HTTP
# ============================================================================

def http_get(ctx, url: str, params: dict | None = None, retries: int = 3, timeout: float = 20, **kwargs):
    """GET with backoff on 429/5xx. Returns the response, or None (with a warning) if it keeps failing."""
    import requests

    for attempt in range(retries):
        try:
            resp = ctx.http().get(url, params=params, timeout=timeout, **kwargs)
        except requests.Timeout:
            log.debug(f"GET {url}: timeout (try {attempt + 1}/{retries})")
            if attempt + 1 < retries:
                continue
            log.warning(f"! {url}: no answer after {retries} tries of {timeout:.0f} s; skipped")
            return None
        except requests.RequestException as err:
            log.warning(f"! {url}: {short_error(err)}")
            return None
        log.debug(f"GET {resp.url}: HTTP {resp.status_code}")
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = _retry_after(resp.headers.get("Retry-After"), default=2 ** (attempt + 1))
            log.debug(f"{url}: busy (HTTP {resp.status_code}), waiting {wait:.0f} s")
            resp.close()
            time.sleep(wait)
            continue
        if resp.ok:
            return resp
        log.warning(f"! {url}: HTTP {resp.status_code}")
        return None
    log.warning(f"! {url}: still busy after {retries} tries (rate limit); try a smaller --limit later")
    return None


def _retry_after(value: str | None, default: float) -> float:
    """Seconds to wait from a Retry-After header (a number, or an HTTP date we do not parse), at most 30."""
    try:
        return min(float(value), 30.0) if value else default
    except ValueError:
        return default


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


# ==============================================================================================================
# Links: single videos, playlists and channels on any site yt-dlp supports, plus direct image links.
# ==============================================================================================================

@SOURCES.register("url_list")
class UrlListSource(Source):
    """A link: video, playlist or channel on ~1800 sites (yt-dlp), or a direct image link."""

    def __init__(self, ctx, limit: int = 0):
        super().__init__(ctx, limit)

    def search(self, query: str) -> Iterator[MediaItem]:
        yield from expand_url(self.ctx, query)


def image_item(url: str, source_name: str = "url_list", license: str = "", **extra) -> MediaItem:
    return MediaItem(url, MediaType.IMAGE, url_key(url), source_name=source_name,
                     license=normalize_license(license), **extra)


def expand_url(ctx, url: str, source_name: str = "url_list", license: str = "") -> Iterator[MediaItem]:
    if is_image_url(url):
        yield image_item(url, source_name, license)
        return
    if url_suffix(url) in VIDEO_EXTS:  # a video file itself: nothing to look up, so no extra request to the site
        yield MediaItem(url, MediaType.VIDEO, url_key(url), title=Path(urllib.parse.urlparse(url).path).name,
                        source_name=source_name, license=normalize_license(license), page_url=url)
        return
    yield from ytdlp_entries(ctx, url, source_name, license, image_fallback=True)


def ytdlp_entries(ctx, url: str, source_name: str, license: str = "",
                  image_fallback: bool = False) -> Iterator[MediaItem]:
    """List the videos behind a link or a yt-dlp search ("ytsearch10:...") without downloading anything."""
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({**ctx.ytdlp_options(), "extract_flat": "in_playlist"}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as err:  # noqa: BLE001 - extractors raise many kinds of errors; one link must not stop the run
        if image_fallback:  # not a video; maybe an image link without an extension (the downloader checks)
            log.debug(f"{url}: no video found ({short_error(err)}); trying it as an image")
            yield image_item(url, source_name, license)
        else:
            log.warning(f"! cannot read {url}: {short_error(err)}")
        return
    if not info:
        return
    entries = [e for e in info.get("entries") or [] if e] if "entries" in info else [info]
    for entry in entries:
        if f".{entry.get('ext')}" in IMAGE_EXTS:  # yt-dlp treats a bare image link as a one-frame video
            yield image_item(url if entry is info else entry["url"], source_name, license)
            continue
        extractor = entry.get("extractor_key") or entry.get("ie_key") or "web"
        target = entry.get("webpage_url") or entry.get("url")
        yield MediaItem(target, MediaType.VIDEO, video_key(extractor, entry["id"]), title=entry.get("title") or "",
                        source_name=source_name, license=normalize_license(license or entry.get("license")),
                        page_url=target, duration=entry.get("duration"))


# ==============================================================================================================
# Inputs given with -i: files, folders, .txt lists (one link, path or list per line) and links.
#
# In a .txt list, lines starting with # are ignored, and a link may carry its license after " #":
#     https://example.org/bird.jpg  # license=cc-by-4.0
# ==============================================================================================================

# "http://...", "https://...", "ytsearch10:..." all look like <scheme>:...  (at least two letters, so "C:\" is a path)
_URL_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]+:", re.IGNORECASE)
_LICENSE_NOTE = re.compile(r"license=(\S+)")


@SOURCES.register("local")
class LocalSource(Source):
    """Files and folders on this machine (videos and images), and .txt lists of links or paths."""

    def __init__(self, ctx, limit: int = 0):
        super().__init__(ctx, limit)

    def search(self, query: str) -> Iterator[MediaItem]:
        yield from resolve_inputs(self.ctx, [query], exclude_dirs=self.ctx.paths.result_dirs())


def resolve_inputs(ctx, items: Iterable[str], exclude_dirs: Iterable[Path] = ()) -> Iterator[MediaItem]:
    """Turn -i values into media items, lazily (links are only expanded when reached).

    Folders are scanned recursively, skipping hidden files and `exclude_dirs` (the tool's own output).
    """
    excluded = tuple(Path(p).resolve() for p in exclude_dirs)
    for item in items:
        yield from _resolve(ctx, item, excluded)


def _split_note(line: str) -> tuple[str, str]:
    """"https://x/bird.jpg  # license=cc-by-4.0" -> ("https://x/bird.jpg", " license=cc-by-4.0")."""
    match = re.match(r"^(.*?)\s+#(.*)$", line)
    return (match.group(1), match.group(2)) if match else (line, "")


def _resolve(ctx, item: str, excluded: tuple[Path, ...]) -> Iterator[MediaItem]:
    text, note = item.strip(), ""
    if not Path(text).expanduser().exists():
        text, note = _split_note(text)
    path = Path(text).expanduser()
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            hidden = any(part.startswith(".") for part in file.relative_to(path).parts)
            if file.is_file() and not hidden and not _inside(file, excluded):
                media = local_item(file)
                if media:
                    yield media
    elif path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".txt":
            lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
            for line in lines:
                if line and not line.startswith("#"):
                    yield from _resolve(ctx, line, excluded)
        elif suffix in VIDEO_EXTS | IMAGE_EXTS:
            yield local_item(path)
        else:
            log.warning(f"! skipping {text}: not a video, image or .txt list")
    elif _URL_PATTERN.match(text):
        match = _LICENSE_NOTE.search(note)
        yield from expand_url(ctx, text, license=match.group(1) if match else "")
    else:
        log.warning(f"! skipping {text}: no such file, folder or link")


def _inside(file: Path, folders: tuple[Path, ...]) -> bool:
    resolved = file.resolve()
    return any(resolved.is_relative_to(folder) for folder in folders)


def local_item(path: Path) -> MediaItem | None:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        media_type = MediaType.VIDEO
    elif suffix in IMAGE_EXTS:
        media_type = MediaType.IMAGE
    else:
        return None
    return MediaItem(str(path), media_type, local_key(path), title=path.name, source_name="local",
                     license="local", local=True)


# ==============================================================================================================
# Video search through yt-dlp search prefixes. No API key is needed.
# ==============================================================================================================

class YtDlpSearchSource(Source):
    """Base class: `prefix` is the yt-dlp search prefix, e.g. "ytsearch" gives "ytsearch10:<query>"."""

    prefix = ""

    def __init__(self, ctx, limit: int = 10):
        super().__init__(ctx, limit)

    def search(self, query: str) -> Iterator[MediaItem]:
        if self.limit > 0:
            yield from ytdlp_entries(self.ctx, f"{self.prefix}{self.limit}:{query}", self.registry_name)


@SOURCES.register("bilibili")
class BilibiliSource(YtDlpSearchSource):
    """Bilibili video search (yt-dlp)."""

    prefix = "bilisearch"


@SOURCES.register("niconico")
class NiconicoSource(YtDlpSearchSource):
    """Niconico video search (yt-dlp)."""

    prefix = "nicosearch"


# ==============================================================================================================
# YouTube video search. Licenses are unknown (copyright risk): the `license` column in crops.csv records that.
# ==============================================================================================================

@SOURCES.register("youtube")
class YouTubeSource(YtDlpSearchSource):
    """YouTube video search (yt-dlp). If YouTube blocks downloads, use --cookies-from-browser."""

    prefix = "ytsearch"


# ==============================================================================================================
# Wikimedia Commons: freely licensed images and videos (CC / public domain), no WIKIMEDIA_API key.
# ==============================================================================================================

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks WIKIMEDIA_API clients to identify themselves.
WIKIMEDIA_HEADERS = {"User-Agent": "mbfs-crawler/0.1 (dataset collection tool; python-requests)"}


@SOURCES.register("wikimedia")
class WikimediaSource(Source):
    """Wikimedia Commons images and videos with their licenses."""

    def __init__(self, ctx, limit: int = 30, media: tuple[str, ...] | list[str] = ("image",), width: int = 1920):
        super().__init__(ctx, limit)
        unknown = set(media) - {"image", "video"}
        if unknown:
            raise ValueError(f"wikimedia: media must be image and/or video, not {sorted(unknown)}")
        self.media = tuple(media)
        self.width = width  # images are fetched as thumbnails this wide instead of multi-MB originals

    def search(self, query: str) -> Iterator[MediaItem]:
        for media in self.media:
            yield from self._search(query, media)

    def _search(self, query: str, media: str) -> Iterator[MediaItem]:
        params = {
            "action": "query", "format": "json", "formatversion": 2, "generator": "search",
            "gsrsearch": f"{query} filetype:{'bitmap' if media == 'image' else 'video'}", "gsrnamespace": 6,
            "prop": "imageinfo", "iiprop": "url|mime|extmetadata",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
        }
        if media == "image":
            params["iiurlwidth"] = self.width
        found = 0
        while found < self.limit:
            params["gsrlimit"] = min(50, self.limit - found)
            resp = http_get(self.ctx, WIKIMEDIA_API, params, headers=WIKIMEDIA_HEADERS)
            if resp is None:
                return
            data = resp.json()
            pages = sorted((data.get("query") or {}).get("pages", []), key=lambda p: p.get("index", 0))
            for page in pages:
                item = self._item(page, media)
                if item:
                    found += 1
                    yield item
                    if found >= self.limit:
                        return
            if "continue" not in data or not pages:
                return
            params.update(data["continue"])

    def _item(self, page: dict, media: str) -> MediaItem | None:
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("url")
        if not url or not info.get("mime", "").startswith(media):
            return None
        meta = info.get("extmetadata") or {}
        title = page.get("title", "").removeprefix("File:")
        return MediaItem(
            info.get("thumburl") or url if media == "image" else url,
            MediaType.IMAGE if media == "image" else MediaType.VIDEO,
            f"wikimedia_{slug(title.rsplit('.', 1)[0], 40)}_{short_hash(url, 6)}",
            title=title, source_name="wikimedia",
            license=normalize_license(meta.get("LicenseShortName", {}).get("value")),
            license_url=meta.get("LicenseUrl", {}).get("value", ""),
            author=strip_html(meta.get("Artist", {}).get("value", "")),
            page_url=info.get("descriptionurl", ""),
        )


# ==============================================================================================================
# Openverse: Creative Commons images from many sites (Flickr, museums, ...), anonymous access, no OPENVERSE_API key.
#
# Anonymous requests are rate-limited and return at most 20 results per page, so keep limits modest. Some queries
# are slow on Openverse's side; they are retried once and then skipped with a warning.
# ==============================================================================================================

OPENVERSE_API = "https://api.openverse.org/v1/images/"
OPENVERSE_PAGE_SIZE = 20  # maximum for anonymous requests


@SOURCES.register("openverse")
class OpenverseSource(Source):
    """Openverse Creative Commons images; commercial=true keeps only licenses that allow commercial use."""

    def __init__(self, ctx, limit: int = 40, commercial: bool = False):
        super().__init__(ctx, limit)
        self.commercial = commercial

    def search(self, query: str) -> Iterator[MediaItem]:
        found, page = 0, 1
        while found < self.limit:
            params = {"q": query, "page": page, "page_size": OPENVERSE_PAGE_SIZE, "mature": "false"}
            if self.commercial:
                params["license_type"] = "commercial"
            resp = http_get(self.ctx, OPENVERSE_API, params, retries=2, timeout=30)
            if resp is None:
                return
            data = resp.json()
            results = data.get("results") or []
            for result in results:
                if not result.get("url"):
                    continue
                found += 1
                yield _openverse_item(result)
                if found >= self.limit:
                    return
            if not results or page >= int(data.get("page_count") or page):
                return
            page += 1


def _openverse_item(result: dict) -> MediaItem:
    license, version = result.get("license", ""), result.get("license_version", "")
    full = license if license in ("cc0", "pdm") else f"cc-{license}-{version}".rstrip("-")
    return MediaItem(result["url"], MediaType.IMAGE, f"openverse_{slug(result.get('id', ''), 36)}",
                     title=result.get("title") or "", source_name="openverse", license=normalize_license(full),
                     license_url=result.get("license_url") or "", author=result.get("creator") or "",
                     page_url=result.get("foreign_landing_url") or "")


# ==============================================================================================================
# Bing image search, read from the plain HTML results page (no browser, no API key).
#
# Not enabled in datasets.yaml: Bing may answer automated requests with unrelated images (seen in October
# 2026), which wastes downloads. Use it explicitly with --sources bing_images and check the results with `search`.
# Bing changes its page from time to time; when this source breaks, only parse_results() needs fixing.
# ==============================================================================================================

BING_URL = "https://www.bing.com/images/async"
BING_PAGE_SIZE = 35
_METADATA = re.compile(r'\bm="(\{.*?\})"', re.DOTALL)


@SOURCES.register("bing_images")
class BingImagesSource(Source):
    """Bing image search (safe search on). Unreliable: check results with `search` first. Licenses unknown."""

    def __init__(self, ctx, limit: int = 50):
        super().__init__(ctx, limit)

    def search(self, query: str) -> Iterator[MediaItem]:
        seen: set[str] = set()
        first = 0
        while len(seen) < self.limit and first < self.limit * 4:  # stop if pages keep repeating results
            params = {"q": query, "first": first, "count": BING_PAGE_SIZE, "mmasync": 1, "adlt": "strict"}
            resp = http_get(self.ctx, BING_URL, params, cookies={"SRCHHPGUSR": "ADLT=STRICT"})
            if resp is None:
                return
            results = parse_results(resp.text)
            if not results:
                return
            for result in results:
                url = result.get("murl")
                if not url or url in seen:
                    continue
                seen.add(url)
                yield MediaItem(url, MediaType.IMAGE, url_key(url), title=result.get("t") or "",
                                source_name="bing_images", page_url=result.get("purl") or "")
                if len(seen) >= self.limit:
                    return
            first += BING_PAGE_SIZE


def parse_results(page: str) -> list[dict]:
    """Each result carries its metadata as HTML-escaped JSON in an m="{...}" attribute."""
    results = []
    for raw in _METADATA.findall(page):
        try:
            results.append(json.loads(html.unescape(raw)))
        except json.JSONDecodeError:
            continue
    return results


# ==============================================================================================================
# Google Images through a headless Chrome (needs `pip install -e ".[google]"` and Google Chrome).
#
# Kept from the original google_images_download project: the search filters and the image-link extraction.
# Google changes its page often; when this source breaks, only extract_image_urls() needs fixing.
# ==============================================================================================================

# Filter values -> Google's "tbs" codes.
GOOGLE_FILTERS: dict[str, dict[str, str]] = {
    "color": {c: f"ic:specific,isc:{c}" for c in ("red", "orange", "yellow", "green", "teal", "blue", "purple",
                                                  "pink", "white", "gray", "black", "brown")},
    "color_type": {"full-color": "ic:color", "black-and-white": "ic:gray", "transparent": "ic:trans"},
    "usage_rights": {"labeled-for-reuse-with-modifications": "sur:fmc", "labeled-for-reuse": "sur:fc",
                     "labeled-for-noncommercial-reuse-with-modification": "sur:fm",
                     "labeled-for-noncommercial-reuse": "sur:f"},
    "size": {"large": "isz:l", "medium": "isz:m", "icon": "isz:i", ">400*300": "isz:lt,islt:qsvga",
             ">640*480": "isz:lt,islt:vga", ">800*600": "isz:lt,islt:svga", ">1024*768": "isz:lt,islt:xga",
             ">2MP": "isz:lt,islt:2mp", ">4MP": "isz:lt,islt:4mp", ">6MP": "isz:lt,islt:6mp",
             ">8MP": "isz:lt,islt:8mp", ">10MP": "isz:lt,islt:10mp", ">12MP": "isz:lt,islt:12mp"},
    "type": {"face": "itp:face", "photo": "itp:photo", "clipart": "itp:clipart", "line-drawing": "itp:lineart",
             "animated": "itp:animated"},
    "time": {"past-24-hours": "qdr:d", "past-7-days": "qdr:w", "past-month": "qdr:m", "past-year": "qdr:y"},
    "aspect_ratio": {"tall": "iar:t", "square": "iar:s", "wide": "iar:w", "panoramic": "iar:xw"},
    "format": {"jpg": "ift:jpg", "gif": "ift:gif", "png": "ift:png", "bmp": "ift:bmp", "webp": "ift:webp"},
}

_PATTERNS = (
    re.compile(r'\["(https://[^"]+\.(?:jpg|jpeg|png|webp|gif)[^"]*?)"'),
    re.compile(r'"ou":"(https://[^"]+\.(?:jpg|jpeg|png|webp|gif)[^"]*?)"'),
)
_ESCAPED = re.compile(r"\\u([0-9a-fA-F]{4})")


@SOURCES.register("google_images")
class GoogleImagesSource(Source):
    """Google Images (headless Chrome) with filters: size, type, color, usage_rights, time, aspect_ratio, ..."""

    def __init__(self, ctx, limit: int = 50, safe_search: bool = True, site: str | None = None,
                 color: str | None = None, color_type: str | None = None, usage_rights: str | None = None,
                 size: str | None = None, type: str | None = None, time: str | None = None,
                 aspect_ratio: str | None = None, format: str | None = None):
        super().__init__(ctx, limit)
        self.safe_search = safe_search
        self.site = site
        self.filters = {"color": color, "color_type": color_type, "usage_rights": usage_rights, "size": size,
                        "type": type, "time": time, "aspect_ratio": aspect_ratio, "format": format}
        for name, value in self.filters.items():
            if value is not None and value not in GOOGLE_FILTERS[name]:
                raise ValueError(f"google_images: {name} must be one of {sorted(GOOGLE_FILTERS[name])}")

    def build_url(self, query: str) -> str:
        params = {"q": query, "tbm": "isch"}
        tbs = ",".join(GOOGLE_FILTERS[name][value] for name, value in self.filters.items() if value is not None)
        if tbs:
            params["tbs"] = tbs
        if self.site:
            params["as_sitesearch"] = self.site
        if self.safe_search:
            params["safe"] = "active"
        return "https://www.google.com/search?" + urllib.parse.urlencode(params)

    def search(self, query: str) -> Iterator[MediaItem]:
        if self.limit <= 0:
            return
        for url in extract_image_urls(self._page_source(self.build_url(query)), self.limit):
            yield MediaItem(url, MediaType.IMAGE, url_key(url), source_name="google_images")

    def _page_source(self, url: str) -> str:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.common.keys import Keys
        except ImportError:
            log.warning('! google_images needs Selenium: pip install -e ".[google]"')
            return ""
        options = Options()
        for argument in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--window-size=1024,768",
                         f"user-agent={USER_AGENT}"):
            options.add_argument(argument)
        try:
            browser = webdriver.Chrome(options=options)  # Selenium Manager finds Chrome and its driver on every OS
        except Exception as err:  # noqa: BLE001 - any failure here means "no usable Chrome"
            log.warning(f"! google_images: cannot start headless Chrome ({short_error(err)}); "
                        "install Google Chrome or leave this source out")
            log.debug("details", exc_info=True)
            return ""
        try:
            browser.get(url)
            time.sleep(2)
            body = browser.find_element(By.TAG_NAME, "body")
            for _ in range(min(40, 5 + self.limit // 10)):  # scroll only as far as the limit needs
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.3)
            return browser.page_source
        except Exception as err:  # noqa: BLE001
            log.warning(f"! google_images: {short_error(err)}")
            log.debug("details", exc_info=True)
            return ""
        finally:
            browser.quit()


def extract_image_urls(page: str, limit: int) -> list[str]:
    """Full-size image links in a Google Images results page, in page order, without duplicates."""
    urls: list[str] = []
    seen: set[str] = set()
    for pattern in _PATTERNS:
        for match in pattern.finditer(page):
            url = _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), match.group(1))
            host = urllib.parse.urlparse(url).netloc
            if url in seen or host.endswith(("gstatic.com", "google.com")):  # Google's own thumbnails and assets
                continue
            seen.add(url)
            urls.append(url)
            if len(urls) >= limit:
                return urls
    return urls
