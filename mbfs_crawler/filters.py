"""Noise filters run on every sample, in the order of datasets.yaml: duplicates and reference retrieval (RAG)."""

import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mbfs_crawler.config import FILTERS, IMAGE_EXTS
from mbfs_crawler.processors import Sample
from mbfs_crawler.utils import read_image

# ==============================================================================================================
# Filter interface and the chain that runs a dataset type's filters in order, stopping at the first rejection.
# ==============================================================================================================

@dataclass(frozen=True)
class Verdict:
    keep: bool
    reason: str = ""    # why it was rejected: "duplicate", "noise", ...
    score: float = 0.0


KEEP = Verdict(True)


class Filter(ABC):
    """Register subclasses with @FILTERS.register("name"). Options come from the `filters` section of datasets.yaml."""

    registry_name = ""
    needs_vector = False  # True if check() reads sample.vector (computed once by the pipeline, then shared)

    def __init__(self, ctx):
        self.ctx = ctx

    @abstractmethod
    def check(self, sample: Sample) -> Verdict:
        """Keep or reject one sample."""


class FilterChain:
    def __init__(self, filters: list[Filter]):
        self.filters = filters

    @classmethod
    def from_type(cls, ctx) -> "FilterChain":
        from mbfs_crawler.config import FILTERS

        return cls([FILTERS.create(name, ctx=ctx, **options) for name, options in ctx.dataset_type.filters.items()])

    @property
    def needs_vector(self) -> bool:
        return any(f.needs_vector for f in self.filters)

    def check(self, sample: Sample) -> Verdict:
        for f in self.filters:
            verdict = f.check(sample)
            if not verdict.keep:
                return verdict
        return KEEP


# ==============================================================================================================
# Duplicate filter: reject a sample too similar to any image already in the dataset (including earlier runs).
# ==============================================================================================================

@FILTERS.register("duplicate")
class DuplicateFilter(Filter):
    """max_similarity: 0-1; lower is stricter, 1 turns it off."""

    needs_vector = True

    def __init__(self, ctx, max_similarity: float = 0.85):
        super().__init__(ctx)
        self.max_similarity = max_similarity

    def check(self, sample: Sample) -> Verdict:
        sample.max_sim = self.ctx.store.index.max_similarity(sample.vector)
        if sample.max_sim >= self.max_similarity:
            return Verdict(False, "duplicate", sample.max_sim)
        return KEEP


# ==============================================================================================================
# Reference retrieval (the "RAG" noise filter): compare each sample with examples you marked good or noise.
#
#     references/<type>/good/       images you want more of
#     references/<type>/noise/     images you do NOT want (posters, TV screens, mannequins, missing gear, ...)
#
# The sample's vector (already computed for the duplicate check, so this costs no extra model run) is compared with
# its k most similar examples. Only examples at least `min_similarity` alike vote, weighted by similarity; if noise
# wins, the sample is rejected. With no example similar enough, the sample is kept. Until you add examples
# (mbfs-crawler review), this filter does nothing.
# ==============================================================================================================

LABELS = {"good": 1.0, "noise": -1.0}
_CACHE = ".vectors.npz"


@FILTERS.register("reference")
class ReferenceFilter(Filter):
    """k: neighbours that vote; min_similarity: how alike an example must be to vote."""

    needs_vector = True

    def __init__(self, ctx, k: int = 7, min_similarity: float = 0.55):
        super().__init__(ctx)
        self.k = k
        self.min_similarity = min_similarity
        self._refs: tuple[np.ndarray, np.ndarray] | None = None

    def check(self, sample: Sample) -> Verdict:
        if self._refs is None:
            self._refs = ReferenceSet(self.ctx.paths.references(self.ctx.dataset_type.name), self.ctx.encoder).load()
        vectors, labels = self._refs
        if not len(vectors):
            return KEEP
        return vote(vectors, labels, sample.vector, self.k, self.min_similarity)


def vote(vectors: np.ndarray, labels: np.ndarray, vector: np.ndarray, k: int, min_similarity: float) -> Verdict:
    sims = vectors @ vector
    top = np.argsort(-sims)[:k]
    top = top[sims[top] >= min_similarity]
    if not top.size:
        return KEEP
    score = float((sims[top] * labels[top]).sum())
    return Verdict(True, "", score) if score >= 0 else Verdict(False, "noise", score)


class ReferenceSet:
    """Vectors of the example images, cached next to them so only new or changed files are encoded."""

    def __init__(self, folder: Path, encoder_factory: Callable):
        self.folder = folder
        self.encoder_factory = encoder_factory

    def files(self) -> list[tuple[str, Path]]:
        found = []
        for label in LABELS:
            folder = self.folder / label
            if folder.is_dir():
                found += [(label, p) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        return found

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        files = self.files()
        if not files:
            return np.zeros((0, 0), np.float32), np.zeros(0, np.float32)
        keys = [f"{label}/{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for label, p in files]
        cache = self._read_cache()
        missing = [(key, path) for key, (_, path) in zip(keys, files, strict=True) if key not in cache]
        if missing:
            images = [(key, read_image(path)) for key, path in missing]
            images = [(key, img) for key, img in images if img is not None]
            if images:
                vectors = self.encoder_factory().encode([img for _, img in images])
                cache.update(zip((key for key, _ in images), vectors, strict=True))
            self._write_cache({k: cache[k] for k in keys if k in cache})
        rows = [(cache[key], LABELS[label]) for key, (label, _) in zip(keys, files, strict=True) if key in cache]
        if not rows:
            return np.zeros((0, 0), np.float32), np.zeros(0, np.float32)
        return np.stack([v for v, _ in rows]).astype(np.float32), np.array([lab for _, lab in rows], np.float32)

    def add(self, label: str, files: Iterable[Path]) -> int:
        """Copy example images into good/ or noise/. Returns how many were added."""
        if label not in LABELS:
            raise ValueError(f"label must be one of {list(LABELS)}")
        folder = self.folder / label
        folder.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in files:
            if path.suffix.lower() in IMAGE_EXTS and path.is_file():
                shutil.copy2(path, folder / path.name)
                count += 1
        return count

    def _read_cache(self) -> dict[str, np.ndarray]:
        path = self.folder / _CACHE
        if not path.exists():
            return {}
        with np.load(path) as data:
            return dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    def _write_cache(self, cache: dict[str, np.ndarray]) -> None:
        if not cache:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        np.savez(self.folder / _CACHE, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
