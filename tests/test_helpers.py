"""Checks for the provided helpers. They need no data and no cluster."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from casmi_flyte.baseline import LOSS_BINS, MZ_BINS, bin_spectra
from casmi_flyte.tables import matrix_column, merge_local


def test_matrix_roundtrip(tmp_path):
    """matrix_column -> parquet -> polars Array column -> the same (n, d) numpy array."""
    import polars as pl

    x = np.random.default_rng(0).integers(0, 2, (10, 4860)).astype(np.uint8)
    pq.write_table(pa.table({"k": [str(i) for i in range(10)], "bits": matrix_column(x)}), tmp_path / "t.parquet")
    back = pl.scan_parquet(tmp_path / "t.parquet").select("bits").collect(engine="streaming")["bits"]
    assert back.dtype == pl.Array(pl.UInt8, 4860)
    assert (back.to_numpy() == x).all()


def test_merge_local_keeps_rows_in_order(tmp_path):
    import polars as pl

    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    for i, (lo, hi) in enumerate([(0, 4), (4, 7), (7, 10)]):
        table = pa.table({"k": [str(j) for j in range(lo, hi)], "v": matrix_column(x[lo:hi])})
        pq.write_table(table, tmp_path / f"part_{i}.parquet")
    merge_local([str(tmp_path / f"part_{i}.parquet") for i in range(3)], tmp_path / "merged.parquet")
    merged = pl.read_parquet(tmp_path / "merged.parquet")
    assert merged["k"].to_list() == [str(j) for j in range(10)]
    assert (merged["v"].to_numpy() == x).all()


def test_bin_spectra_synthetic():
    """Runs without the data: two spectra shaped like train.parquet rows."""
    table = pa.table(
        {
            "precursor_mz": [195.0877, 193.0731],
            "adduct": ["[M+H]+", "[M-H]-"],
            "collision_energy_ev": [[20.0, 40.0], []],  # the second has no collision energy
            "ms2_mzs": [[42.03, 110.07, 138.07, 195.09, 300.0], [122.02, 150.03]],  # 300 > precursor: dropped
            "ms2_normalized_intensities": [[0.005, 0.4, 1.0, 0.6, 0.9], [1.0, 0.2]],  # 0.005 < 1%: dropped
        }
    )
    x = bin_spectra(table)
    assert x.shape == (2, MZ_BINS + LOSS_BINS + 10 + 3)
    assert np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x[:, :MZ_BINS], axis=1), 1, atol=1e-3)
    assert x[0, 42] == 0 and x[0, 300] == 0 and x[0, 138] > 0
    assert x[0, MZ_BINS + 57] > 0  # neutral loss 195.09 - 138.07 = 57 Da
    assert x[1, MZ_BINS + LOSS_BINS + 10 + 1] == 1  # "no collision energy" flag
