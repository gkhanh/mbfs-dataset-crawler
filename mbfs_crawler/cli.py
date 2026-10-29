"""mbfs-crawler: collect images for a detection dataset, entirely from the terminal.

WHAT is collected is set in datasets.yaml (project folder): one block per dataset type (birds, vehicle, ...).
The command only says which type to collect. It never asks questions and never opens a window, so it runs the same
on a laptop, over SSH or in a script.
Exit codes: 0 done, 1 error, 2 bad arguments or datasets.yaml, 3 nothing to do,
130 stopped with Ctrl+C (run the same command again to resume).
"""

import argparse
import glob
import os
import platform
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from mbfs_crawler import __version__
from mbfs_crawler.config import IMAGE_EXTS, Paths
from mbfs_crawler.utils import log, setup, short_error, warning_count

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NOTHING, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
LINKS_FILE = "found.txt"        # in links/<type>/: the links `search` finds

EXAMPLES = """examples:
  mbfs-crawler check-setup
  mbfs-crawler types                                  the dataset types in datasets.yaml
  mbfs-crawler search birds                           find links, save them to links/birds/found.txt
  mbfs-crawler run birds                              collect into ../dataset/birds/images/
  mbfs-crawler run food_factory_workers -i my_links.txt --cookies-from-browser firefox
  mbfs-crawler stats vehicle
"""


class UsageError(ValueError):
    """Bad command-line values (exit code 2)."""


# ============================================================================
# Arguments
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("general options")
    g.add_argument("--home", help="project folder holding datasets.yaml and models/ (default: $MBFS_HOME or .)")
    g.add_argument("--data-dir", metavar="DIR",
                   help="where results, references and logs go (default: $MBFS_DATA, else a folder called "
                        "'dataset' next to the project folder)")
    g.add_argument("--downloads-dir", metavar="DIR",
                   help="where downloaded videos and images are cached (default: $MBFS_DOWNLOADS, else a folder "
                        "called 'downloads' next to the project folder)")
    g.add_argument("--device", default="auto", help="auto, cpu, mps, or an NVIDIA GPU number (default: auto)")
    g.add_argument("--batch", type=int, default=16, help="still images per batch (default: 16)")
    g.add_argument("--threads", type=int, help="maximum CPU threads (for shared machines)")
    g.add_argument("--no-half", action="store_true", help="do not use FP16 on NVIDIA GPUs")
    g.add_argument("--log-file", help="also write messages to this file")
    g.add_argument("--no-progress", action="store_true", help="no progress bars (automatic when not a terminal)")
    g.add_argument("--quiet", action="store_true", help="only warnings and errors")
    g.add_argument("--debug", action="store_true",
                   help="show every detail, and write everything (with error tracebacks) to logs/<command>-<time>.log")

    parser = argparse.ArgumentParser(prog="mbfs-crawler", description=__doc__, epilog=EXAMPLES,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, help: str, with_type: bool = False) -> argparse.ArgumentParser:
        parser = sub.add_parser(name, parents=[common], help=help, description=help)
        if with_type:
            parser.add_argument("dataset_type", metavar="TYPE", help="a dataset type from datasets.yaml, e.g. birds")
        return parser

    command("check-setup", "check this machine's setup (packages, ffmpeg, GPU, models, datasets.yaml, internet)"
            ).add_argument("--offline", action="store_true", help="skip the internet checks")
    command("types", "list the dataset types in datasets.yaml")
    command("sources", "list the search sources")
    command("models", "list the models in models/ with their classes")

    s = command("search", "find links with a type's searches (no media is downloaded) and save them to "
                "links/TYPE/found.txt", with_type=True)
    s.add_argument("--limit", type=int, help="results per source and search (default: as in datasets.yaml)")
    s.add_argument("-o", "--output", type=Path, metavar="FILE.txt",
                   help="save the links to this .txt list instead of links/TYPE/found.txt")

    r = command("run", "collect images for a dataset type into ../dataset/<type>/", with_type=True)
    r.add_argument("-i", "--input", action="append", default=[], metavar="PATH|URL|LIST.txt",
                   help="collect only from these links, video files, folders or .txt lists (repeatable) "
                        "instead of the type's searches and links")
    r.add_argument("--limit", type=int, help="results per source and search (default: as in datasets.yaml)")
    r.add_argument("--force", action="store_true", help="process again sources already in processed.txt")
    r.add_argument("--delete-videos", action="store_true", help="delete each downloaded video after processing")
    r.add_argument("--cookies-from-browser", metavar="BROWSER",
                   help="use this browser's YouTube login if downloads are blocked (firefox, chrome, safari, ...)")

    v = command("review", "add good / noise examples that teach the noise filter", with_type=True)
    v.add_argument("--good", nargs="+", action="extend", default=[], metavar="FILES",
                   help="images, folders, globs (\"images/*_id3.jpg\") or .txt lists of wanted examples")
    v.add_argument("--noise", nargs="+", action="extend", default=[], metavar="FILES",
                   help="the same, for unwanted examples")

    c = command("clean", "move images that look like your noise examples to rejected/", with_type=True)
    c.add_argument("--dry-run", action="store_true", help="only list what would be moved")

    command("stats", "summary of a type's collected images", with_type=True)
    return parser


# ============================================================================
# Helpers
# ============================================================================

def _paths(args) -> Paths:
    return Paths.from_env(args.home, args.data_dir, args.downloads_dir)


def _dataset_type(args, paths: Paths):
    from mbfs_crawler.datasets import load_type

    return load_type(args.dataset_type, paths.config)


def _context(args, paths: Paths, dataset_type, **extra):
    from mbfs_crawler.config import RunContext

    return RunContext(paths, dataset_type, device=args.device, half=not args.no_half, batch=args.batch, **extra)


def plan_work(args, dataset_type) -> tuple[list[str], list[str]]:
    """(searches, inputs). With -i: only those inputs. Without: the type's searches and links from datasets.yaml."""
    given = getattr(args, "input", [])
    if given:
        return [], list(given)
    return list(dataset_type.search), list(dataset_type.links)


def _sources(args, ctx, searches: list[str]) -> list:
    from mbfs_crawler.pipeline import make_sources

    if not searches:
        return []
    if not ctx.dataset_type.sources:
        raise UsageError(f"type '{ctx.dataset_type.name}' has searches but no sources in datasets.yaml")
    try:
        return make_sources(ctx, list(ctx.dataset_type.sources), args.limit)
    except (ValueError, TypeError) as err:
        raise UsageError(str(err)) from None


def expand_paths(patterns: list[str], bases: list[Path]) -> list[Path]:
    """Files from paths, folders, globs (expanded here: PowerShell does not) and .txt lists."""
    found: list[Path] = []
    for pattern in patterns:
        for base in bases:
            candidate = Path(pattern).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            if any(ch in pattern for ch in "*?["):
                matches = sorted(glob.glob(str(candidate), recursive=True))
                if matches:
                    found += [Path(m) for m in matches]
                    break
            elif candidate.is_dir():
                found += sorted(p for p in candidate.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                break
            elif candidate.is_file():
                if candidate.suffix.lower() == ".txt":
                    lines = [x.strip() for x in candidate.read_text(encoding="utf-8").splitlines()]
                    found += expand_paths([x for x in lines if x and not x.startswith("#")], bases)
                else:
                    found.append(candidate)
                break
        else:
            log.warning(f"! not found: {pattern}")
    return found


# ============================================================================
# Commands
# ============================================================================

def cmd_check_setup(args) -> int:
    from mbfs_crawler.setup_check import format_checks, run_checks

    checks = run_checks(_paths(args), args.device, network=not args.offline)
    print(format_checks(checks))
    return EXIT_OK if all(c.ok or not c.required for c in checks) else EXIT_ERROR


def cmd_types(args) -> int:
    from mbfs_crawler.datasets import ConfigError, load_types

    paths = _paths(args)
    for name, dataset_type in load_types(paths.config).items():
        try:
            dataset_type.validate()
        except ConfigError as err:
            print(f"✗ {name:<22} {err}")
            continue
        model = (dataset_type.detector or {}).get("model", "-")
        print(f"{name:<22} {dataset_type.description}\n{'':<22} {len(dataset_type.search)} searches, "
              f"{len(dataset_type.links)} link lists, keeps: {dataset_type.processor_type}, model: {model}")
    print(f"\nTo add or change a type, edit {paths.config}")
    return EXIT_OK


def cmd_sources(args) -> int:
    from mbfs_crawler.config import SOURCES

    for name in SOURCES.names():
        doc = (SOURCES.get(name).__doc__ or "").strip().splitlines()
        print(f"{name:<18} {doc[0] if doc else ''}")
    return EXIT_OK


def cmd_models(args) -> int:
    paths = _paths(args)
    files = sorted(p for p in paths.models.glob("*") if p.suffix in (".pt", ".onnx")) if paths.models.is_dir() else []
    if not files:
        print(f"No models in {paths.models}")
        return EXIT_NOTHING
    from ultralytics import YOLO

    for path in files:
        names = YOLO(str(path), task="detect").names
        print(f"{path.name:<32} {', '.join(names[i] for i in sorted(names))}")
    return EXIT_OK


def cmd_search(args) -> int:
    from mbfs_crawler.pipeline import collect

    paths = _paths(args)
    dataset_type = _dataset_type(args, paths)
    searches = list(dataset_type.search)
    if not searches:
        log.error(f"Type '{dataset_type.name}' has no searches in datasets.yaml.")
        return EXIT_NOTHING
    ctx = _context(args, paths, dataset_type)
    items = []
    for item in collect(ctx, searches, [], _sources(args, ctx, searches)):
        items.append(item)
        print(f"{item.media_type.value:<5} {item.source_name:<16} {item.license:<14} {item.title[:50]:<50}  {item.uri}")
    videos = sum(item.media_type.value == "video" for item in items)
    log.info(f"\n{len(items)} found: {videos} videos, {len(items) - videos} images.")
    target = args.output or paths.links(dataset_type.name) / LINKS_FILE
    added = save_links(items, target, dataset_type.name)
    log.info(f"{added} new link(s) saved to {target} ({len(items) - added} already there). Edit it if you like "
             f"(delete the lines you do not want), then: mbfs-crawler run {dataset_type.name} -i {target}")
    return EXIT_OK


def save_links(items, path: Path, type_name: str) -> int:
    """Append the links that are not in `path` yet (the file is created with a header). Returns how many were added.

    Appending keeps what you edited by hand: lines you added stay, and a search never rewrites the file.
    """
    known = set()
    if path.exists():
        known = {line.split()[0] for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")}
    fresh = {}
    for item in items:
        if item.uri not in known:
            fresh.setdefault(item.uri, f"{item.uri}  # license={item.license} source={item.source_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        if not path.stat().st_size:
            out.write(f"# mbfs-crawler search {type_name}\n")
        out.writelines(line + "\n" for line in fresh.values())
    return len(fresh)


def cmd_run(args) -> int:
    from mbfs_crawler.pipeline import Pipeline, collect

    paths = _paths(args)
    dataset_type = _dataset_type(args, paths)
    searches, inputs = plan_work(args, dataset_type)
    if not searches and not inputs:
        log.error(f"Nothing to do: type '{dataset_type.name}' has no searches or links in datasets.yaml, "
                  "and no -i was given.")
        return EXIT_NOTHING
    ctx = _context(args, paths, dataset_type, cookies_from_browser=args.cookies_from_browser)
    sources = _sources(args, ctx, searches)
    try:
        pipeline = Pipeline(ctx, force=args.force, delete_videos=args.delete_videos)
    except (ValueError, TypeError) as err:
        raise UsageError(str(err)) from None
    log.info(f"Dataset type '{dataset_type.name}' → {ctx.dataset_dir}")
    stats = pipeline.run(collect(ctx, searches, inputs, sources))
    warnings = warning_count()
    log.info("\n" + stats.summary() + (f"\nwarnings: {warnings} (see the messages starting with ! above)"
                                       if warnings else ""))
    log.info(f"\nImages: {ctx.dataset_dir / 'images'}")
    return EXIT_OK


def cmd_review(args) -> int:
    from mbfs_crawler.filters import ReferenceSet

    paths = _paths(args)
    dataset_type = _dataset_type(args, paths)
    ctx = _context(args, paths, dataset_type)
    bases = [Path.cwd(), ctx.dataset_dir]
    good, noise = expand_paths(args.good, bases), expand_paths(args.noise, bases)
    if not good and not noise:
        raise UsageError("give --good and/or --noise with images, folders, globs or .txt lists")
    refs = ReferenceSet(paths.references(dataset_type.name), ctx.encoder)
    added_good, added_noise = refs.add("good", good), refs.add("noise", noise)
    print(f"Added {added_good} good and {added_noise} noise example(s) to {refs.folder}.")
    print(f"New runs use them automatically; apply them to existing images with: "
          f"mbfs-crawler clean {dataset_type.name}")
    return EXIT_OK


def cmd_clean(args) -> int:
    from mbfs_crawler.pipeline import clean_dataset

    paths = _paths(args)
    dataset_type = _dataset_type(args, paths)
    ctx = _context(args, paths, dataset_type)
    if not (ctx.dataset_dir / "crops.csv").exists():
        log.error(f"No dataset in {ctx.dataset_dir}")
        return EXIT_NOTHING
    reasons = clean_dataset(ctx, dry_run=args.dry_run)
    counts = ", ".join(f"{n} {reason}" for reason, n in Counter(reasons.values()).most_common()) or "nothing"
    if args.dry_run:
        for file, reason in sorted(reasons.items()):
            print(f"{reason:<10} {file}")
        print(f"Would move: {counts}")
    else:
        print(f"Moved to {ctx.dataset_dir / 'rejected'}: {counts}")
    return EXIT_OK


def cmd_stats(args) -> int:
    from mbfs_crawler.store import dataset_report

    paths = _paths(args)
    dataset_type = _dataset_type(args, paths)
    print(dataset_report(paths.dataset(dataset_type.name)))
    return EXIT_OK


COMMANDS = {"check-setup": cmd_check_setup, "types": cmd_types, "sources": cmd_sources, "models": cmd_models,
            "search": cmd_search, "run": cmd_run, "review": cmd_review, "clean": cmd_clean, "stats": cmd_stats}


def main(argv: list[str] | None = None) -> int:
    # Ultralytics would otherwise run `pip install` by itself when a package is missing. Report it instead:
    # a surprise install is unwelcome in a headless tool (and fails in environments without pip).
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from mbfs_crawler.config import RegistryError
    from mbfs_crawler.datasets import ConfigError
    from mbfs_crawler.utils import limit_threads

    args = build_parser().parse_args(argv)
    logs = _paths(args).logs
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = args.log_file or (logs / f"{args.command}-{stamp}.log" if args.debug else None)
    setup(log_file, args.quiet, args.debug, progress=not args.no_progress)
    if args.debug:
        log.info(f"Debug mode: everything is written to {log_file}")
        log.debug(f"mbfs-crawler {__version__}, Python {platform.python_version()}, {platform.platform()}")
        log.debug(f"command: {' '.join(sys.argv[1:] if argv is None else argv)}")
    limit_threads(args.threads)
    try:
        return COMMANDS[args.command](args)
    except (ConfigError, RegistryError, UsageError) as err:
        log.error(f"Error: {err}")
        return EXIT_USAGE
    except KeyboardInterrupt:
        log.warning("\nStopped. Run the same command again to continue where it left off.")
        return EXIT_INTERRUPTED
    except Exception as err:  # noqa: BLE001 - the terminal gets a clear message; the details go to a crash report
        report = _write_crash_report(logs / f"crash-{stamp}.log", argv)
        log.debug("Unexpected error", exc_info=True)
        log.error(f"Error: {short_error(err)}")
        if report:
            log.error(f"Details (send this file when reporting the problem): {report}")
        return EXIT_ERROR


def _write_crash_report(path: Path, argv: list[str] | None) -> Path | None:
    """Full traceback and context of an unexpected error, written even without --debug."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"mbfs-crawler {__version__}, Python {platform.python_version()}, {platform.platform()}\n"
            f"command: mbfs-crawler {' '.join(sys.argv[1:] if argv is None else argv)}\n"
            f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{traceback.format_exc()}", encoding="utf-8")
        return path
    except OSError:
        return None
