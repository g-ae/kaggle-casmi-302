"""Tiny helpers to move tables in and out of flyte.io.File. Every file is parquet.

Small tables (shards, spectra) go through pyarrow. Big feature files go through polars: pyarrow's
parquet reader materialises two int16 "levels" per value of a fixed-size-list column, so 276k
molecules x 4860 fingerprint bits cost several GB on top of the data. polars' reader doesn't, and its
lazy API lets `scan_table` select and filter before anything is loaded. polars is imported inside
the functions that use it: only images that merge or read feature libraries need it.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from flyte.io import File

# Rows per row group in merged feature files. Small groups keep the streaming merge at ~1.5 GB for
# the full Klekota-Roth library; 16k-row groups needed 4.5 GB, 64k-row groups 8 GB.
MERGE_ROW_GROUP_SIZE = 2048


async def read_table(f: File, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(await f.download(), columns=columns)


async def scan_table(f: File):
    """Lazy polars frame over a parquet File: `select`/`filter`, then `collect(engine="streaming")`."""
    import polars as pl

    return pl.scan_parquet(await f.download())


def content_hash(path: str | Path, chunk: int = 64 << 20) -> str:
    import hashlib

    h = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


async def _upload(path: Path) -> File:
    """The File's cache key is its *content* hash, not its (random) storage path: identical outputs
    from a re-executed upstream task keep downstream caches valid."""
    return await File.from_local(str(path), hash_method=content_hash(path))


async def write_table(table: pa.Table, name: str) -> File:
    """Write `table` as parquet and upload it."""
    path = Path(tempfile.mkdtemp()) / name
    pq.write_table(table, path, compression="zstd")
    return await _upload(path)


def merge_local(paths: list[str], out: str | Path) -> None:
    """Concatenate parquet files into `out` with polars' streaming engine: memory stays at a few
    row groups, not the whole table."""
    import polars as pl

    pl.scan_parquet(paths).sink_parquet(out, compression="zstd", row_group_size=MERGE_ROW_GROUP_SIZE)


async def merge_tables(parts: list[File], name: str) -> File:
    """Download parquet Files, concatenate them (streaming, see `merge_local`) and upload the result."""
    paths = await asyncio.gather(*(p.download() for p in parts))
    out = Path(tempfile.mkdtemp()) / name
    merge_local(list(paths), out)
    print(f"{name}: {len(parts)} parts, {os.path.getsize(out) / 1e6:,.0f} MB")
    return await _upload(out)


def matrix_column(x: np.ndarray) -> pa.FixedSizeListArray:
    """(n, d) numpy array -> fixed-size-list column, the natural parquet layout for fingerprints.

    polars reads it back as an `Array(dtype, d)` column, and `.to_numpy()` gives the (n, d) array again.
    """
    x = np.ascontiguousarray(x)
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1])


def stable_fraction(keys: pa.Array) -> np.ndarray:
    """Deterministic pseudo-random number in [0, 1) per key (same on every machine and every run)."""
    import hashlib

    return np.array(
        [int.from_bytes(hashlib.blake2b(k.encode(), digest_size=8).digest()) / 2**64 for k in keys.to_pylist()]
    )
