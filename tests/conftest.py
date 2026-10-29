"""pytest fixtures shared by all test files (the fakes and media builders live in helpers.py)."""

import functools
import http.server
import logging
import threading

import pytest
from helpers import FakeDetector, FakeEncoder, make_type

from mbfs_crawler.config import Paths, RunContext
from mbfs_crawler.datasets import DatasetType
from mbfs_crawler.utils import log


@pytest.fixture(autouse=True)
def restore_logging():
    """cli.main() reconfigures the logger; put it back so tests do not affect each other."""
    state = (log.handlers[:], log.filters[:], log.level, log.propagate)
    yield
    log.handlers[:], log.filters[:], log.level, log.propagate = state


@pytest.fixture
def log_lines():
    """[(LEVEL, message)] of everything logged during the test, whatever the terminal setup."""
    records: list[tuple[str, str]] = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append((record.levelname, record.getMessage()))

    handler = Collect(logging.DEBUG)
    level = log.level
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    yield records
    log.removeHandler(handler)
    log.setLevel(level)


@pytest.fixture
def ctx_factory(tmp_path):
    def build(dataset_type: DatasetType | None = None, detector=None, encoder=None, **kwargs) -> RunContext:
        return RunContext(Paths(tmp_path / "home"), dataset_type or make_type(), device="cpu",
                          detector_impl=detector if detector is not None else FakeDetector(),
                          encoder_impl=encoder if encoder is not None else FakeEncoder(), **kwargs)

    return build


@pytest.fixture
def web(tmp_path):
    """A local web server for "internet" tests. Put files in web.root; web.url(name) gives their link."""
    root = tmp_path / "web"
    root.mkdir()
    requests_seen: list[str] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests_seen.append(self.path)
            super().do_GET()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Web:
        def __init__(self):
            self.root = root
            self.requests = requests_seen

        def url(self, name: str) -> str:
            return f"http://127.0.0.1:{server.server_address[1]}/{name}"

    yield Web()
    server.shutdown()
    server.server_close()
