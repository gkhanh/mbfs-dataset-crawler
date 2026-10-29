"""mbfs-crawler check-setup: check that this machine can run the tool, with a fix for every problem found."""

import importlib
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mbfs_crawler.config import Paths

HOSTS = {"youtube": "https://www.youtube.com", "wikimedia": "https://commons.wikimedia.org",
         "openverse": "https://api.openverse.org", "internet_archive": "https://archive.org",
         "bing_images": "https://www.bing.com", "google_images": "https://www.google.com",
         "model downloads (GitHub)": "https://github.com"}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    required: bool = True


def run_checks(paths: Paths, device: str = "auto", network: bool = True) -> list[Check]:
    checks = [_python()]
    checks += [_package(name, required) for name, required in (
        ("numpy", True), ("cv2", True), ("yaml", True), ("requests", True), ("tqdm", True), ("yt_dlp", True),
        ("torch", True), ("ultralytics", True), ("onnx", True), ("onnxruntime", True),
        ("selenium", False))]
    checks.append(_ffmpeg())
    checks.append(_device(device))
    checks += _models(paths)
    checks += _dataset_types(paths)
    checks += [_writable(paths.data_root), _writable(paths.downloads)]
    checks.append(_dinov2_cache())
    if network:
        checks += _network()
    return checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "✓" if c.ok else ("✗" if c.required else "!")
        line = f"{mark} {c.name}" + (f": {c.detail}" if c.detail else "")
        if not c.ok and c.fix:
            line += f"\n    → {c.fix}"
        lines.append(line)
    failed = [c for c in checks if not c.ok and c.required]
    lines.append("\nAll required checks passed." if not failed else f"\n{len(failed)} required check(s) failed.")
    return "\n".join(lines)


def _python() -> Check:
    ok = sys.version_info >= (3, 10)
    return Check("Python", ok, sys.version.split()[0], "install Python 3.10 or newer")


def _package(name: str, required: bool) -> Check:
    try:
        module = importlib.import_module(name)
        return Check(f"package {name}", True, getattr(module, "__version__", ""), required=required)
    except Exception as err:  # noqa: BLE001 - any import failure is reported
        fix = 'pip install -e ".[google]"' if name == "selenium" else "pip install -e ."
        if name == "cv2" and "libGL" in str(err):
            fix = "sudo apt install libgl1   (Linux servers without a desktop)"
        return Check(f"package {name}", False, str(err), fix, required)


def _ffmpeg() -> Check:
    from mbfs_crawler.downloads import ffmpeg_path

    path = ffmpeg_path() or shutil.which("ffmpeg")
    return Check("ffmpeg", bool(path), path or "not found", "pip install imageio-ffmpeg")


def _device(device: str) -> Check:
    try:
        import torch

        from mbfs_crawler.utils import pick_device, torch_device

        picked = pick_device(device)
        torch.ones(2, device=torch_device(picked)).sum().item()
        hint = "" if picked != "cpu" else " (no GPU found: works, but slower)"
        return Check("compute device", True, picked + hint)
    except Exception as err:  # noqa: BLE001
        return Check("compute device", False, str(err), "use --device cpu")


def _dataset_types(paths: Paths) -> list[Check]:
    """datasets.yaml loads, and every type in it is valid (sources, options, model settings)."""
    from mbfs_crawler.datasets import ConfigError, load_types

    try:
        types = load_types(paths.config)
    except ConfigError as err:
        return [Check("datasets.yaml", False, str(err), "fix the file (see the comments at its top)")]
    checks = []
    for name, dataset_type in types.items():
        try:
            dataset_type.validate()
            checks.append(Check(f"dataset type {name}", True, dataset_type.description))
        except ConfigError as err:
            checks.append(Check(f"dataset type {name}", False, str(err), "fix this type in datasets.yaml"))
    return checks


def _models(paths: Paths) -> list[Check]:
    files = sorted(p for p in paths.models.glob("*") if p.suffix in (".pt", ".onnx")) if paths.models.is_dir() else []
    if not files:
        return [Check("models", False, f"no .pt/.onnx files in {paths.models}",
                      "run from the project folder, or set MBFS_HOME to it")]
    checks = []
    for path in files:
        try:
            from ultralytics import YOLO

            names = YOLO(str(path), task="detect").names
            checks.append(Check(f"model {path.name}", True, ", ".join(names[i] for i in sorted(names))))
        except Exception as err:  # noqa: BLE001
            checks.append(Check(f"model {path.name}", False, str(err), "re-download or replace this model file"))
    return checks


def _writable(folder) -> Check:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder):
            pass
        return Check(f"write access {folder.name}/", True, str(folder))
    except OSError as err:
        return Check(f"write access {folder.name}/", False, str(err),
                     "choose another folder with --data-dir / --downloads-dir")


def _dinov2_cache() -> Check:
    try:
        import torch

        cached = (Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vits14_pretrain.pth").exists()
    except Exception:  # noqa: BLE001
        cached = False
    return Check("DINOv2 weights (duplicate check)", cached, "cached" if cached else "not downloaded yet",
                 "downloaded automatically on the first run (needs internet once, ~90 MB)", required=False)


def _network() -> list[Check]:
    import requests

    def reach(item):
        name, url = item
        try:
            requests.head(url, timeout=6, allow_redirects=True)
            return Check(f"internet: {name}", True)
        except requests.RequestException as err:
            return Check(f"internet: {name}", False, type(err).__name__, "check the network / proxy", required=False)

    with ThreadPoolExecutor(len(HOSTS)) as pool:
        return list(pool.map(reach, HOSTS.items()))
