"""Dataset types: reads datasets.yaml (project root), the only place that says WHAT is collected.

Each top-level key of datasets.yaml is one dataset type (birds, food_factory_workers, vehicle, ...); the comments at
the top of that file describe every setting. This module only loads and checks the file.
"""

import difflib
import inspect
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from mbfs_crawler.config import RESERVED_NAMES

CONFIG_FILE = "datasets.yaml"
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_DOWNLOAD_KEYS = {"max_res", "max_duration", "max_minutes"}
_TEXT_LISTS = ("search", "links", "exclude_words")
_SECTIONS = ("sources", "download", "processor", "quality", "filters")


class ConfigError(ValueError):
    """datasets.yaml is missing or invalid, or a type refers to an unknown source, model setting or option."""


@dataclass(frozen=True)
class DatasetType:
    name: str
    description: str = ""
    search: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    exclude_words: tuple[str, ...] = ()
    sources: dict = field(default_factory=dict)
    download: dict = field(default_factory=dict)
    detector: dict | None = None
    processor: dict = field(default_factory=lambda: {"type": "crop"})
    quality: dict = field(default_factory=dict)
    filters: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data, base_dir: Path | None = None) -> "DatasetType":
        """One block of datasets.yaml. Link lists are resolved relative to `base_dir` (the file's folder)."""
        where = f"{CONFIG_FILE}, type '{name}'"
        if not _NAME.match(name):
            raise ConfigError(f"{where}: a type name may only use letters, digits, _ and -")
        if name in RESERVED_NAMES:
            raise ConfigError(f"{where}: '{name}' is a folder the tool uses itself; choose another name")
        if not isinstance(data, dict):
            raise ConfigError(f"{where}: expected settings (search:, sources:, ...) under the name")
        allowed = {f.name for f in fields(cls)} - {"name"}
        for key in data:
            if key not in allowed:
                close = difflib.get_close_matches(str(key), allowed, n=1)
                hint = f" Did you mean '{close[0]}'?" if close else ""
                raise ConfigError(f"{where}: unknown setting '{key}'.{hint}")
        values = dict(data)
        for key in _TEXT_LISTS:
            items = values.get(key) or ()
            if isinstance(items, str) or not all(isinstance(x, str) for x in items):
                raise ConfigError(f"{where}: '{key}' must be a list of text values")
            values[key] = tuple(items)
        if base_dir is not None:
            values["links"] = tuple(str(base_dir / x) if (base_dir / x).exists() else x for x in values["links"])
        for key in _SECTIONS:
            if values.get(key) is None:
                values.pop(key, None)
            elif not isinstance(values[key], dict):
                raise ConfigError(f"{where}: '{key}' must be a set of settings")
        for key in ("filters", "sources"):  # "duplicate:" with nothing after it simply means default options
            if key in values:
                values[key] = {n: dict(options or {}) for n, options in values[key].items()}
        if values.get("detector") is not None and not isinstance(values["detector"], dict):
            raise ConfigError(f"{where}: 'detector' must be a set of settings (model:, classes:, ...)")
        return cls(name=name, **values)

    @property
    def processor_type(self) -> str:
        return self.processor.get("type", "crop")

    @property
    def processor_options(self) -> dict:
        return {k: v for k, v in self.processor.items() if k != "type"}

    @property
    def detector_type(self) -> str:
        return (self.detector or {}).get("type", "ultralytics")

    @property
    def detector_options(self) -> dict:
        return {k: v for k, v in (self.detector or {}).items() if k != "type"}

    def validate(self) -> "DatasetType":
        """Check every source, processor, filter and detector option (imports light modules only, never a model)."""
        from mbfs_crawler.config import DETECTORS, FILTERS, PROCESSORS, SOURCES, RegistryError
        from mbfs_crawler.processors import QualityRules

        where = f"{CONFIG_FILE}, type '{self.name}'"
        try:
            _check_options(where, "processor", self.processor_type, PROCESSORS.get(self.processor_type),
                           self.processor_options)
            if self.detector is None and self.processor_type == "crop":
                raise ConfigError(f"{where}: processor 'crop' needs a detector (detector: {{model: ...}})")
            if self.detector is not None:
                if "model" not in self.detector:
                    raise ConfigError(f"{where}: detector needs a 'model'")
                _check_options(where, "detector", self.detector_type, DETECTORS.get(self.detector_type),
                               self.detector_options)
            for name, options in self.filters.items():
                _check_options(where, "filter", name, FILTERS.get(name), options)
            for name, options in self.sources.items():
                _check_options(where, "source", name, SOURCES.get(name), options)
        except RegistryError as err:
            raise ConfigError(f"{where}: {err}") from None
        _check_keys(where, "quality", self.quality, {f.name for f in fields(QualityRules)})
        _check_keys(where, "download", self.download, _DOWNLOAD_KEYS)
        return self


def _check_keys(where: str, section: str, values: dict, allowed: set[str]) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"{where}: {section} has unknown option(s) {sorted(unknown)}. Allowed: {sorted(allowed)}")


def _check_options(where: str, kind: str, name: str, cls: type, options: dict) -> None:
    """Options must match the plugin's constructor (everything except `self` and the injected `ctx`)."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    _check_keys(where, f"{kind} '{name}'", options, set(params) - {"self", "ctx"})


# ============================================================================================================
# Loading
# ============================================================================================================

def load_types(config: Path) -> dict[str, DatasetType]:
    """All dataset types in the file, in file order (not yet validated against the plugins)."""
    if not config.is_file():
        raise ConfigError(f"{config} not found. Run from the project folder (where {CONFIG_FILE} is), or use --home.")
    try:
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ConfigError(f"{config.name}: invalid YAML ({err})") from None
    if not isinstance(data, dict) or not data:
        raise ConfigError(f"{config.name}: no dataset types defined")
    return {str(name): DatasetType.from_dict(str(name), settings, config.parent) for name, settings in data.items()}


def load_type(name: str, config: Path) -> DatasetType:
    """One dataset type, validated. Unknown names list the types that exist."""
    types = load_types(config)
    if name not in types:
        close = difflib.get_close_matches(name, types, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ConfigError(f"unknown dataset type '{name}'. Types in {config.name}: {', '.join(types)}.{hint}")
    return types[name].validate()
