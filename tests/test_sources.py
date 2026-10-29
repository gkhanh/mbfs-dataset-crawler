"""sources/: parsers against saved real responses (no network), local inputs, licenses, retries."""

import json
import sys
import types

import pytest
import requests
from helpers import FIXTURES

import mbfs_crawler.sources as sources_mod
from mbfs_crawler.sources import (
    BingImagesSource,
    GoogleImagesSource,
    LocalSource,
    MediaType,
    OpenverseSource,
    WikimediaSource,
    YouTubeSource,
    extract_image_urls,
    http_get,
    license_category,
    normalize_license,
    parse_results,
    resolve_inputs,
)


class FakeResponse:
    def __init__(self, data=None, text="", status=200, headers=None):
        self._data, self.text, self.status_code, self.headers = data, text, status, headers or {}
        self.ok = status < 400
        self.url = "https://x"

    def close(self):
        pass

    def json(self):
        return self._data


def serve(monkeypatch, module, *responses):
    """Make `module.http_get` return these responses in order, recording the params it was called with."""
    calls = []
    queue = list(responses)

    def fake_get(ctx, url, params=None, **kwargs):
        calls.append(dict(params or {}))
        return queue.pop(0) if queue else None

    monkeypatch.setattr(module, "http_get", fake_get)
    return calls


# ============================================================================
# Licenses
# ============================================================================


@pytest.mark.parametrize("text,expected", [
    ("CC BY-SA 4.0", "cc-by-sa-4.0"), ("CC BY 2.0", "cc-by-2.0"), ("by-nc-sa", "cc-by-nc-sa"),
    ("https://creativecommons.org/licenses/by/4.0/", "cc-by-4.0"), ("CC0", "cc0"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "cc0"), ("Public domain", "pd"),
    ("http://creativecommons.org/publicdomain/mark/1.0/", "pd"), ("pdm", "pd"), ("", "unknown"), (None, "unknown"),
])
def test_normalize_license(text, expected):
    assert normalize_license(text) == expected


@pytest.mark.parametrize("license,category", [
    ("cc-by-4.0", "cc"), ("cc-by-sa-3.0", "cc"), ("cc-by-nc-2.0", "cc-nc"), ("cc-by-nd-4.0", "cc-nc"),
    ("cc0", "pd"), ("pd", "pd"), ("local", "local"), ("unknown", "unknown"), ("standard youtube license", "unknown"),
])
def test_license_category(license, category):
    assert license_category(license) == category


# ============================================================================
# Search sources (saved real responses)
# ============================================================================


def test_wikimedia_parses_items_licenses_and_thumbnails(monkeypatch, ctx_factory):
    data = json.loads((FIXTURES / "wikimedia.json").read_text(encoding="utf-8"))
    calls = serve(monkeypatch, sys.modules["mbfs_crawler.sources"], FakeResponse(data))
    items = list(WikimediaSource(ctx_factory(), limit=3).search("birds flying over airport"))
    assert len(items) == 3
    assert all(item.media_type is MediaType.IMAGE and item.source_name == "wikimedia" for item in items)
    assert all(item.license.startswith("cc-by") and item.uri.startswith("https://") for item in items)
    assert all("/thumb/" in item.uri for item in items)  # 1920-px thumbnail, not the multi-MB original
    assert all(item.key.startswith("wikimedia_") and item.page_url for item in items)
    assert calls[0]["gsrsearch"].endswith("filetype:bitmap") and calls[0]["gsrlimit"] == 3


def test_wikimedia_follows_continuation_until_the_limit(monkeypatch, ctx_factory):
    data = json.loads((FIXTURES / "wikimedia.json").read_text(encoding="utf-8"))
    page2 = json.loads(json.dumps(data))
    for page in page2["query"]["pages"]:
        page["imageinfo"][0]["url"] += "?v=2"
    page2.pop("continue", None)
    calls = serve(monkeypatch, sys.modules["mbfs_crawler.sources"], FakeResponse(data), FakeResponse(page2))
    items = list(WikimediaSource(ctx_factory(), limit=5).search("birds"))
    assert len(items) == 5 and len({i.key for i in items}) == 5
    assert "gsroffset" in calls[1] or "continue" in calls[1]


def test_wikimedia_rejects_unknown_media(ctx_factory):
    with pytest.raises(ValueError, match="image and/or video"):
        WikimediaSource(ctx_factory(), media=["audio"])


def test_openverse_parses_licenses(monkeypatch, ctx_factory):
    data = json.loads((FIXTURES / "openverse.json").read_text(encoding="utf-8"))
    calls = serve(monkeypatch, sys.modules["mbfs_crawler.sources"], FakeResponse(data))
    items = list(OpenverseSource(ctx_factory(), limit=10, commercial=True).search("cat"))
    assert [i.license for i in items] == ["cc-by-2.0", "cc-by-nc-sa-2.0", "cc-by-2.0"]
    assert [license_category(i.license) for i in items] == ["cc", "cc-nc", "cc"]
    assert calls[0]["license_type"] == "commercial" and len(calls) == 1  # page_count 1: no second request


def test_openverse_stops_quietly_when_the_service_does_not_answer(monkeypatch, ctx_factory):
    serve(monkeypatch, sys.modules["mbfs_crawler.sources"])
    assert list(OpenverseSource(ctx_factory()).search("cat")) == []


BING_PAGE = """<div><a class="iusc" m="{&quot;murl&quot;:&quot;https://a.org/1.jpg&quot;,&quot;t&quot;:&quot;One&quot;,
&quot;purl&quot;:&quot;https://a.org/p1&quot;}" href="#"></a>
<a class="iusc" m="{&quot;murl&quot;:&quot;https://a.org/1.jpg&quot;,&quot;t&quot;:&quot;Dup&quot;}"></a>
<a class="iusc" m="{broken json}"></a>
<a class="iusc" m="{&quot;murl&quot;:&quot;https://b.org/2.png&quot;,&quot;t&quot;:&quot;Two&quot;}"></a></div>"""


def test_bing_parses_metadata_and_skips_broken_entries():
    results = parse_results(BING_PAGE)
    assert [r["murl"] for r in results] == ["https://a.org/1.jpg", "https://a.org/1.jpg", "https://b.org/2.png"]


def test_bing_deduplicates_and_respects_the_limit(monkeypatch, ctx_factory):
    serve(monkeypatch, sys.modules["mbfs_crawler.sources"], FakeResponse(text=BING_PAGE),
          FakeResponse(text=BING_PAGE))
    items = list(BingImagesSource(ctx_factory(), limit=2).search("x"))
    assert [i.uri for i in items] == ["https://a.org/1.jpg", "https://b.org/2.png"]
    assert items[0].title == "One" and items[0].page_url == "https://a.org/p1"


GOOGLE_PAGE = r"""["https://img.example.com/bird.jpg",640,480] ["https://encrypted-tbn0.gstatic.com/x.jpg"]
"ou":"https://img.example.com/bird.jpg" ["https://cdn.example.org/a.png?w=800\u003d1"] ["https://img.example.com/c.webp"]"""


def test_google_extracts_full_size_links_in_order():
    urls = extract_image_urls(GOOGLE_PAGE, limit=10)
    assert urls == ["https://img.example.com/bird.jpg", "https://cdn.example.org/a.png?w=800=1",
                    "https://img.example.com/c.webp"]
    assert extract_image_urls(GOOGLE_PAGE, limit=1) == ["https://img.example.com/bird.jpg"]


def test_google_builds_filters_and_validates_them(ctx_factory):
    source = GoogleImagesSource(ctx_factory(), size="large", type="photo", usage_rights="labeled-for-reuse",
                                site="example.org")
    url = source.build_url("birds sky")
    assert "q=birds+sky" in url and "tbm=isch" in url and "safe=active" in url and "as_sitesearch=example.org" in url
    assert "tbs=sur%3Afc%2Cisz%3Al%2Citp%3Aphoto" in url or "tbs=isz%3Al%2Csur%3Afc" in url or "itp%3Aphoto" in url
    with pytest.raises(ValueError, match="size must be one of"):
        GoogleImagesSource(ctx_factory(), size="huge")


def test_google_without_selenium_is_skipped_with_a_hint(monkeypatch, ctx_factory, caplog):
    monkeypatch.setitem(sys.modules, "selenium", None)
    source = GoogleImagesSource(ctx_factory(), limit=3)
    assert source._page_source("https://www.google.com/") == ""


def test_youtube_search_uses_the_ytsearch_prefix(monkeypatch, ctx_factory):
    seen = []
    monkeypatch.setattr(sys.modules["mbfs_crawler.sources"], "ytdlp_entries",
                        lambda ctx, url, name, **kw: seen.append((url, name)) or iter(()))
    list(YouTubeSource(ctx_factory(), limit=7).search("bird strike"))
    assert seen == [("ytsearch7:bird strike", "youtube")]
    seen.clear()
    list(YouTubeSource(ctx_factory(), limit=0).search("x"))
    assert seen == []


# ============================================================================
# yt-dlp expansion (fake yt-dlp, no network)
# ============================================================================


def fake_ytdlp(monkeypatch, info=None, error=False):
    """Fake yt-dlp. error=True raises its usual DownloadError; an exception instance is raised as it is."""

    class DownloadError(Exception):
        pass

    class YoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            if isinstance(error, Exception):
                raise error
            if error:
                raise DownloadError("unsupported URL")
            return info

    module = types.SimpleNamespace(YoutubeDL=YoutubeDL, utils=types.SimpleNamespace(DownloadError=DownloadError))
    monkeypatch.setitem(sys.modules, "yt_dlp", module)


def test_playlist_becomes_one_item_per_video(monkeypatch, ctx_factory):
    fake_ytdlp(monkeypatch, {"entries": [
        {"id": "AAA", "ie_key": "Youtube", "url": "https://youtu.be/AAA", "title": "One", "duration": 30},
        None,
        {"id": "pic", "ext": "jpg", "url": "https://x.org/pic.jpg"},
    ]})
    items = list(sources_mod.expand_url(ctx_factory(), "https://www.youtube.com/playlist?list=PL"))
    assert [(i.media_type, i.key) for i in items] == [(MediaType.VIDEO, "youtube_AAA"),
                                                      (MediaType.IMAGE, items[1].key)]
    assert items[0].duration == 30 and items[0].title == "One" and items[1].uri == "https://x.org/pic.jpg"


def test_single_video_uses_its_extractor_key(monkeypatch, ctx_factory):
    fake_ytdlp(monkeypatch, {"id": "xyz", "extractor_key": "Vimeo", "webpage_url": "https://vimeo.com/xyz",
                             "license": "CC BY 4.0"})
    (item,) = sources_mod.expand_url(ctx_factory(), "https://vimeo.com/xyz")
    assert item.key == "vimeo_xyz" and item.license == "cc-by-4.0" and item.uri == "https://vimeo.com/xyz"


def test_unreadable_link_falls_back_to_an_image_download(monkeypatch, ctx_factory):
    fake_ytdlp(monkeypatch, error=True)
    (item,) = sources_mod.expand_url(ctx_factory(), "https://cdn.example.org/get?id=5", license="cc0")
    assert item.media_type is MediaType.IMAGE and item.license == "cc0"


def test_failed_search_is_only_a_warning(monkeypatch, ctx_factory):
    fake_ytdlp(monkeypatch, error=True)
    assert list(sources_mod.ytdlp_entries(ctx_factory(), "ytsearch3:x", "youtube")) == []


def test_direct_image_links_skip_ytdlp(monkeypatch, ctx_factory):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # importing it would fail
    (item,) = sources_mod.expand_url(ctx_factory(), "https://x.org/a/bird.PNG")
    assert item.media_type is MediaType.IMAGE and item.key.startswith("bird_")


def test_direct_video_links_need_no_ytdlp_lookup(monkeypatch, ctx_factory):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # importing it would fail: a lookup would be an extra request
    url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Flock_of_gulls.webm?utm_source=commons.wikimedia"
    (item,) = sources_mod.expand_url(ctx_factory(), url, license="cc-by-4.0")
    assert item.media_type is MediaType.VIDEO and item.uri == url and item.license == "cc-by-4.0"
    assert item.title == "Flock_of_gulls.webm" and item.key.startswith("Flock_of_gulls_")
    assert sources_mod.expand_url(ctx_factory(), url).__next__().key == item.key  # the key is stable


def test_a_page_yt_dlp_cannot_parse_is_one_warning_not_a_crash(monkeypatch, ctx_factory, log_lines):
    """archive.org items without a player make yt-dlp raise a plain ValueError, not its DownloadError."""
    fake_ytdlp(monkeypatch, error=ValueError("opening play-av tag not found"))
    assert list(sources_mod.ytdlp_entries(ctx_factory(), "https://archive.org/details/x", "internet_archive")) == []
    warnings = [m for level, m in log_lines if level == "WARNING"]
    assert warnings == ["! cannot read https://archive.org/details/x: opening play-av tag not found"]
    # a link that might be an image falls back to the image downloader (which checks it)
    (item,) = sources_mod.ytdlp_entries(ctx_factory(), "https://x.org/get?id=1", "url_list", image_fallback=True)
    assert item.media_type is MediaType.IMAGE


@pytest.mark.parametrize("value,expected", [("3", 3.0), ("0", 0.0), ("500", 30.0), (None, 4.0),
                                           ("Wed, 21 Oct 2026 07:28:00 GMT", 4.0), ("", 4.0)])
def test_retry_after_accepts_numbers_and_ignores_dates(value, expected):
    assert sources_mod._retry_after(value, default=4.0) == expected


# ============================================================================
# Local inputs
# ============================================================================


def test_resolve_inputs_sorts_files_links_and_txt_lists(monkeypatch, ctx_factory, tmp_path):
    expanded = []
    monkeypatch.setattr(sources_mod, "expand_url",
                        lambda ctx, url, license="": expanded.append((url, license)) or iter(()))
    media = tmp_path / "media"
    (media / "sub").mkdir(parents=True)
    (media / "v.mp4").touch()
    (media / "sub" / "p.jpg").touch()
    (media / ".hidden.jpg").touch()
    (media / "notes.txt").touch()  # a .txt inside a folder is not treated as a link list
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    (out / "images" / "old.jpg").touch()
    links = tmp_path / "links.txt"
    links.write_text("# comment\nhttps://youtu.be/abc\n\nytsearch2:chef\nhttps://x.org/a.jpg  # license=cc-by-4.0\n",
                     encoding="utf-8")

    items = list(resolve_inputs(ctx_factory(), [str(media), str(links), str(out), "does-not-exist"],
                                exclude_dirs=[out]))
    assert [(i.media_type, i.uri) for i in items] == [(MediaType.IMAGE, str(media / "sub" / "p.jpg")),
                                                     (MediaType.VIDEO, str(media / "v.mp4"))]
    assert all(i.local and i.license == "local" for i in items)
    assert expanded == [("https://youtu.be/abc", ""), ("ytsearch2:chef", ""), ("https://x.org/a.jpg", "cc-by-4.0")]


def test_windows_drive_paths_are_not_mistaken_for_links():
    assert not sources_mod._URL_PATTERN.match(r"C:\Videos\a.mp4")
    assert sources_mod._URL_PATTERN.match("ytsearch10:x") and sources_mod._URL_PATTERN.match("https://x")


def test_local_source_ignores_the_tools_own_result_folders(ctx_factory):
    ctx = ctx_factory()
    folder = ctx.paths.dataset("x") / "images"            # <tmp>/dataset/x/images: next to the project folder
    folder.mkdir(parents=True)
    (folder / "a.jpg").touch()
    ctx.paths.videos.mkdir(parents=True)                  # <tmp>/downloads/videos: the cache is NOT excluded
    (ctx.paths.videos / "cached.mp4").touch()
    ctx.paths.home.mkdir(parents=True)
    (ctx.paths.home / "b.jpg").touch()
    items = list(LocalSource(ctx).search(str(ctx.paths.home.parent)))
    assert sorted(i.title for i in items) == ["b.jpg", "cached.mp4"]


def test_unsupported_file_is_skipped(ctx_factory, tmp_path):
    (tmp_path / "doc.pdf").touch()
    assert list(resolve_inputs(ctx_factory(), [str(tmp_path / "doc.pdf")])) == []


# ============================================================================
# HTTP helper
# ============================================================================


class FakeSession:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def http_ctx(ctx_factory, session):
    ctx = ctx_factory()
    ctx.http = lambda: session
    return ctx


def test_http_get_backs_off_on_rate_limits(monkeypatch, ctx_factory):
    monkeypatch.setattr(sources_mod.time, "sleep", lambda s: None)
    session = FakeSession(FakeResponse(status=429, headers={"Retry-After": "1"}), FakeResponse({"ok": 1}))
    assert http_get(http_ctx(ctx_factory, session), "https://x").json() == {"ok": 1}
    assert session.calls == 2


def test_http_get_retries_timeouts_then_gives_up(ctx_factory):
    session = FakeSession(requests.Timeout(), requests.Timeout())
    assert http_get(http_ctx(ctx_factory, session), "https://x", retries=2) is None
    assert session.calls == 2


def test_http_get_returns_none_on_errors(monkeypatch, ctx_factory):
    monkeypatch.setattr(sources_mod.time, "sleep", lambda s: None)
    assert http_get(http_ctx(ctx_factory, FakeSession(FakeResponse(status=404))), "https://x") is None
    assert http_get(http_ctx(ctx_factory, FakeSession(requests.ConnectionError("down"))), "https://x") is None
    busy = FakeSession(*[FakeResponse(status=503)] * 3)
    assert http_get(http_ctx(ctx_factory, busy), "https://x") is None and busy.calls == 3


def test_http_session_is_per_thread(ctx_factory):
    import threading

    ctx = ctx_factory()
    sessions = [ctx.http()]
    thread = threading.Thread(target=lambda: sessions.append(ctx.http()))
    thread.start()
    thread.join()
    assert ctx.http() is sessions[0] and sessions[1] is not sessions[0]
