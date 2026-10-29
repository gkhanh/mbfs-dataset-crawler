"""Benchmark to run before a release: what the speed-ups save on one video.

    python tests/benchmark.py VIDEO [--type food_factory_workers]

Prints time, detector calls and images saved for mbfs-crawler with its speed-ups turned off and with default settings.
Both runs use the type from the project's datasets.yaml, copied into a temporary project folder.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def timed(command: list[str]) -> tuple[float, str]:
    start = time.monotonic()
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        sys.exit(f"failed: {' '.join(command)}\n{result.stdout}{result.stderr}")
    return time.monotonic() - start, result.stdout


def project(folder: Path, name: str, speed_ups: bool) -> Path:
    """A temporary project folder whose datasets.yaml holds the type, optionally with the speed-ups turned off."""
    settings = yaml.safe_load((REPO / "datasets.yaml").read_text(encoding="utf-8"))[name]
    model = Path(settings["detector"]["model"])
    settings["detector"]["model"] = str(model if model.is_absolute() else REPO / model)
    if not speed_ups:
        processor = settings["processor"]
        settings["processor"] = {**processor, "every_max": processor.get("every", 0.2), "stop_after": 0}
    folder.mkdir(parents=True)
    (folder / "datasets.yaml").write_text(yaml.safe_dump({name: settings}), encoding="utf-8")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--type", default="food_factory_workers", help="a dataset type from datasets.yaml")
    args = parser.parse_args()
    video = str(args.video.resolve())
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, speed_ups in (("speed-ups off", False), ("defaults", True)):
            home = project(Path(tmp, f"home{len(rows)}"), args.type, speed_ups)
            seconds, out = timed([sys.executable, "-m", "mbfs_crawler", "run", args.type, "-i", video,
                                  "--home", str(home), "--data-dir", f"{tmp}/data{len(rows)}",
                                  "--downloads-dir", f"{tmp}/downloads", "--no-progress"])
            calls = re.search(r"detector calls: (\d+)", out).group(1)
            saved = re.search(r"saved: (\d+)", out).group(1)
            rows.append((label, seconds, calls, saved))
    print(f"\n{'run':<20}{'time':>9}{'detector calls':>16}{'saved':>8}")
    for label, seconds, calls, saved in rows:
        print(f"{label:<20}{seconds:>8.1f}s{calls:>16}{saved:>8}")


if __name__ == "__main__":
    main()
