"""In-process LRU cache layer for zarr stores + a factory for opening
weekly-cov artefacts from npz, local zarr, or s3:// URIs.

zarr 3 dropped LRUStoreCache from the public API, but the access
pattern that benefits from one — a backend that re-runs the same
quadratic-form queries against a remote object store — is exactly
what we have here. With the cache in place, only the first query for
a given (artefact, week range) eats the network cost; subsequent
queries collapse to compute-only.

Sized by bytes, not entry count, because zarr chunk sizes vary by
several orders of magnitude across artefacts (4 KB metadata vs 64 MB
matrix slices).

Usage:
    base = FsspecStore(s3fs_async, path="bucket/cov.zarr")
    cached = LRUWrapperStore(base, max_bytes=4 * 1024**3)  # 4 GB
    root = zarr.open_group(store=cached, mode="r")

The cache is shared across `with_read_only()` clones (zarr internally
re-wraps the store when opened read-only), so stats and cached chunks
survive the open call.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zarr.storage import WrapperStore


@dataclass
class LRUStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_held: int = 0
    bytes_max: int = 0  # peak watermark
    fetch_bytes: int = 0  # cumulative bytes fetched from inner store

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.fetch_bytes = 0
        # bytes_held / bytes_max stay — they reflect cache state, not query stats


class _SharedCache:
    """Mutable holder so cache + stats survive zarr's with_read_only()
    clone (which calls type(self)(store=...) — a fresh wrapper that
    needs to share state with its parent)."""

    def __init__(self, max_bytes: int) -> None:
        self.entries: OrderedDict[str, "object"] = OrderedDict()
        self.bytes: int = 0
        self.max_bytes: int = max_bytes
        self.stats = LRUStats(bytes_max=0)


def _buffer_size(buf) -> int:
    """Best-effort size in bytes for a zarr Buffer. The Buffer protocol
    in zarr 3 doesn't expose nbytes uniformly, so we materialise to
    bytes once and remember the length on the buffer for cheap reuse.
    """
    cached = getattr(buf, "_lru_size_cache", None)
    if cached is not None:
        return cached
    try:
        n = len(buf.to_bytes())
    except Exception:  # noqa: BLE001
        # Some buffer subclasses are sequence-like.
        try:
            n = len(buf)
        except Exception:  # noqa: BLE001
            n = 0
    try:
        object.__setattr__(buf, "_lru_size_cache", n)
    except Exception:  # noqa: BLE001 — Buffer may be immutable; we just lose the memo.
        pass
    return n


class LRUWrapperStore(WrapperStore):
    """Bounded chunk cache. `get` is the only path that hits zarr's
    chunk reads; `get_partial_values` is rare in our usage and goes
    straight to the inner store (caching partial reads correctly is
    fiddly and not worth it for this workload).

    The cache is keyed on the storage key string (e.g.
    "cov_full/c/12/0/0"). Byte-range gets are not cached.
    """

    def __init__(self, store, max_bytes: int = 0, _shared: _SharedCache | None = None) -> None:
        super().__init__(store)
        if _shared is None:
            _shared = _SharedCache(max_bytes)
        self._shared = _shared

    # zarr's `with_read_only` rebuilds the wrapper via type(self)(store=...).
    # Pass the shared cache through so stats and contents survive.
    def _with_store(self, store):
        return type(self)(store, _shared=self._shared)

    @property
    def stats(self) -> LRUStats:
        return self._shared.stats

    @property
    def max_bytes(self) -> int:
        return self._shared.max_bytes

    def reset_stats(self) -> None:
        self._shared.stats.reset()

    def clear_cache(self) -> None:
        self._shared.entries.clear()
        self._shared.bytes = 0
        self._shared.stats = LRUStats()

    async def get(self, key, prototype, byte_range=None):
        # Byte-range reads bypass the cache. The matrix-slice access
        # pattern in our queries always pulls full chunks, so this is
        # only hit for metadata-y partials — not worth the bookkeeping.
        if byte_range is not None or self._shared.max_bytes <= 0:
            return await self._store.get(key, prototype, byte_range)

        cache = self._shared.entries
        if key in cache:
            cache.move_to_end(key)
            self._shared.stats.hits += 1
            return cache[key]

        self._shared.stats.misses += 1
        result = await self._store.get(key, prototype)
        if result is None:
            return None

        size = _buffer_size(result)
        self._shared.stats.fetch_bytes += size

        # Evict until there's room. Items at the front are oldest.
        while self._shared.bytes + size > self._shared.max_bytes and cache:
            ev_key, ev_buf = cache.popitem(last=False)
            self._shared.bytes -= _buffer_size(ev_buf)
            self._shared.stats.evictions += 1

        # Skip caching if a single value exceeds the budget — store it
        # passthrough rather than thrashing the cache empty.
        if size <= self._shared.max_bytes:
            cache[key] = result
            self._shared.bytes += size
            if self._shared.bytes > self._shared.stats.bytes_max:
                self._shared.stats.bytes_max = self._shared.bytes
            self._shared.stats.bytes_held = self._shared.bytes

        return result


# ----- Backend integration: a Zarr-backed WeeklyCovStore --------------

def _import_zarr() -> Any:
    """Lazy import so the existing npz-only deployment doesn't pay the
    zarr import cost at startup."""
    import zarr  # noqa: PLC0415
    return zarr


def _open_zarr_store(path_or_uri: str, lru_mb: int):
    """Open the right kind of zarr store based on the artefact location.
    Returns (zarr_group, lru_wrapper_or_None).

      file://… or plain path  → LocalStore
      s3://bucket/key          → FsspecStore over async s3fs
    """
    zarr = _import_zarr()
    from zarr.storage import LocalStore, FsspecStore  # noqa: PLC0415

    if path_or_uri.startswith("s3://"):
        import s3fs  # noqa: PLC0415
        without_scheme = path_or_uri[len("s3://"):]
        # Credentials / endpoint come from env (boto3-standard) or
        # explicit env vars we honour:
        endpoint = os.environ.get("FRV_S3_ENDPOINT")
        client_kwargs: dict[str, Any] = {}
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        fs = s3fs.S3FileSystem(
            key=os.environ.get("FRV_S3_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("FRV_S3_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            client_kwargs=client_kwargs or None,
            asynchronous=True,
        )
        store = FsspecStore(fs, path=without_scheme)
    else:
        local_path = Path(path_or_uri)
        if not local_path.exists():
            raise FileNotFoundError(f"zarr artefact not found: {local_path}")
        store = LocalStore(str(local_path))

    cached: LRUWrapperStore | None = None
    if lru_mb > 0:
        cached = LRUWrapperStore(store, max_bytes=lru_mb * 1024 * 1024)
        store = cached

    group = zarr.open_group(store=store, mode="r")
    return group, cached


class ZarrWeeklyCovStore:
    """Mirror of the WeeklyCovStore interface, but backed by a zarr
    group instead of an mmap'd npz. Hot-path methods (`quadratic` /
    `quadratic_approx`) use the same numpy slicing semantics — zarr
    arrays return numpy arrays on `arr[s:e]`.

    The slicing patterns we use are the same as the local-mmap path:

      cov_full[s:e]                 → (e-s, N, N) numpy
      eig_V[s:e, :, -k_active:]     → (e-s, N, k_active) numpy
      eig_D[s:e, -k_active:]        → (e-s, k_active) numpy

    For S3-backed deployments wrap the underlying store in
    LRUWrapperStore (see `_open_zarr_store`); cold queries pay the
    network cost once, warm queries match local mmap perf.
    """

    def __init__(self, path_or_uri: str, lru_mb: int = 0) -> None:
        self.path_or_uri = path_or_uri
        self.lru_mb = lru_mb
        group, cached = _open_zarr_store(path_or_uri, lru_mb)
        self._group = group
        self._cache = cached
        self.cov_full = group["cov_full"]
        # factor_ids / week_dates / meta live as group attrs (lists)
        # rather than zarr arrays — much cheaper to load + simpler dtype
        # story. See bench/zarr_local_sim.py:ensure_zarr for the writer.
        self.factor_ids = list(group.attrs["factor_ids"])
        self.week_dates = list(group.attrs["week_dates"])
        meta = list(group.attrs["meta"])
        self.n = int(meta[0])
        self.w = int(meta[1])
        self.n_latent = int(meta[2])

        if "eig_V" in [str(p).split("/")[-1] for p in group]:
            self.eig_V = group["eig_V"]
            self.eig_D = group["eig_D"]
            self.eig_k = int(self.eig_D.shape[1])
        else:
            self.eig_V = None
            self.eig_D = None
            self.eig_k = 0

        self.factor_id_index: dict[str, int] = {
            str(fid): i for i, fid in enumerate(self.factor_ids)
        }
        self.week_index: dict[str, int] = {
            str(d): i for i, d in enumerate(self.week_dates)
        }

    @property
    def path(self) -> Path:
        # For local artefacts, return the actual Path so callers can
        # stat()/rglob() it. For s3:// URIs, return a Path made from
        # just the basename — `path.name` still surfaces meaningfully
        # in error messages, and is_file()/is_dir() correctly return
        # False for the remote case.
        if self.path_or_uri.startswith("s3://"):
            return Path(self.path_or_uri.rstrip("/").split("/")[-1])
        return Path(self.path_or_uri)

    def has_approx(self) -> bool:
        return self.eig_V is not None and self.eig_D is not None

    def quadratic(
        self,
        exposures,
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        import numpy as np  # noqa: PLC0415
        end = end_idx if end_idx is not None else self.w
        # Bulk slice → parallel chunk fetch under the hood. Per-week
        # loops would serialise the network. See report Phase 4.
        sub = np.asarray(self.cov_full[start_idx:end])
        x = exposures.astype(sub.dtype, copy=False)
        y_all = sub @ x
        return (y_all * x).sum(axis=1)

    def quadratic_approx(
        self,
        exposures,
        start_idx: int = 0,
        end_idx: int | None = None,
        k: int | None = None,
    ):
        import numpy as np  # noqa: PLC0415
        if not self.has_approx():
            raise RuntimeError("artefact has no eig arrays")
        end = end_idx if end_idx is not None else self.w
        k_active = self.eig_k if k is None else min(int(k), self.eig_k)
        V = np.asarray(self.eig_V[start_idx:end, :, -k_active:])
        D = np.asarray(self.eig_D[start_idx:end, -k_active:])
        y = np.einsum("wnk,n->wk", V, exposures, optimize=True)
        return np.einsum("wk,wk->w", D, y * y, optimize=True)

    def lru_stats(self) -> dict[str, int]:
        if self._cache is None:
            return {}
        s = self._cache.stats
        return {
            "hits": s.hits,
            "misses": s.misses,
            "evictions": s.evictions,
            "bytes_held": s.bytes_held,
            "bytes_max": s.bytes_max,
            "fetch_bytes": s.fetch_bytes,
        }

