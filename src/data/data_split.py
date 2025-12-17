from __future__ import annotations

import os
import json
import math
import tempfile
import hashlib
import re
from typing import Tuple

import polars as pl

from utils.basics import _log, verificate_portions

def _ensure_dir(path: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return path


def _final_parquet_path(p: str) -> str:
    return p if p.lower().endswith(".parquet") else p + ".parquet"


def _atomic_write_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _parse_dt(s: str):
    return pl.Series([s]).str.to_datetime(strict=False, time_zone="UTC")[0]


def _parse_duration(s: str) -> pl.Duration:
    s = str(s).strip().lower()
    if s in ("0", "0s", "0sec", "0secs", "0second", "0seconds", "0m", "0min", "0mins", "0h", "0d"):
        return pl.duration(seconds=0)
    m = re.fullmatch(r"(\d+)\s*(d|h|m|s)", s)
    if not m:
        raise ValueError('embargo_duration must look like "30m", "6h", "1d", "0s"')
    n = int(m.group(1))
    u = m.group(2)
    if u == "d":
        return pl.duration(days=n)
    if u == "h":
        return pl.duration(hours=n)
    if u == "m":
        return pl.duration(minutes=n)
    return pl.duration(seconds=n)


def _sha256_file(path: str, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _time_col(schema: dict) -> str:
    if "end_time" in schema:
        return "end_time"
    if "start_time" in schema:
        return "start_time"
    raise ValueError('Expected a time column "end_time" or "start_time" in parquet')


def _min_max_time(lf: pl.LazyFrame, time_col: str):
    out = lf.select(pl.col(time_col).min().alias("mn"), pl.col(time_col).max().alias("mx")).collect(streaming=True)
    mn = out["mn"][0]
    mx = out["mx"][0]
    if mn is None or mx is None:
        raise ValueError("Cannot determine time range (min/max is null)")
    return mn, mx


def _span_us(tmin, tmax) -> int:
    delta = tmax - tmin
    if hasattr(delta, "total_seconds"):
        return int(delta.total_seconds() * 1_000_000)
    raise ValueError("Unable to compute time span")


def _sink_parquet_atomic(lf: pl.LazyFrame, path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=parent)
    os.close(fd)
    try:
        lf.sink_parquet(tmp, compression="zstd", statistics=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _file_info(path: str) -> dict:
    return {
        "path": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "mtime": os.path.getmtime(path),
    }


def _check_monotonic(lf_sorted: pl.LazyFrame, time_col: str, sample_rows: int = 300_000) -> dict:
    df = lf_sorted.select(pl.col(time_col)).head(sample_rows).collect(streaming=True)
    if df.height <= 1:
        return {"checked_rows": int(df.height), "violations": 0}

    v = (
        df.select((pl.col(time_col) < pl.col(time_col).shift(1)).sum().alias("violations"))
        .item()
    )
    vio = int(v) if v is not None else 0
    if vio != 0:
        raise ValueError(f"Temporal monotonicity failed in sample: {vio} violations")
    return {"checked_rows": int(df.height), "violations": 0}


def data_split(data_split_config: dict):
    features_source_path = data_split_config.get("features_source_path", "sample_data/dev/featured_data.parquet")
    output_dir = data_split_config.get("output_path", "sample_data/dev/splits")

    starting_date = data_split_config.get("starting_date", "2025-01-01")
    end_date = data_split_config.get("end_date", "2025-12-01")

    train_portion = float(data_split_config.get("train_portion", 0.7))
    val_portion = float(data_split_config.get("val_portion", 0.15))
    test_portion = 1.0 - (train_portion + val_portion)

    verificate_portions(train_portion, val_portion)
    if test_portion <= 0:
        raise ValueError("Invalid portions: train_portion + val_portion must be < 1.0")

    split_mode = str(data_split_config.get("split_mode", "time")).lower()
    if split_mode not in ("time", "row"):
        raise ValueError('split_mode must be "time" or "row"')

    boundary_drop = int(data_split_config.get("boundary_drop", 0))
    if boundary_drop < 0:
        boundary_drop = 0

    embargo_str = data_split_config.get("embargo_duration", "0s")
    embargo = _parse_duration(embargo_str)

    monotonic_check = bool(data_split_config.get("monotonic_check", True))
    monotonic_sample_rows = int(data_split_config.get("monotonic_sample_rows", 300_000))
    if monotonic_sample_rows <= 0:
        monotonic_sample_rows = 300_000

    if not os.path.exists(features_source_path):
        raise FileNotFoundError(f"features_source_path not found: {features_source_path}")

    output_dir = _ensure_dir(output_dir)

    train_path = _final_parquet_path(os.path.join(output_dir, "train.parquet"))
    val_path = _final_parquet_path(os.path.join(output_dir, "val.parquet"))
    test_path = _final_parquet_path(os.path.join(output_dir, "test.parquet"))
    manifest_path = os.path.join(output_dir, "split_manifest.json")

    dt_start = _parse_dt(starting_date)
    dt_end = _parse_dt(end_date)

    _log(f"Source parquet: {features_source_path}")
    _log(f"Output dir: {output_dir}")
    _log(f"Mode: {split_mode} | embargo: {embargo_str} | boundary_drop: {boundary_drop}")
    _log(f"Date filter: {starting_date} -> {end_date}")
    _log(f"Portions: train={train_portion:.4f} val={val_portion:.4f} test={test_portion:.4f}")

    lf0 = pl.scan_parquet(features_source_path)
    schema = lf0.collect_schema()
    time_col = _time_col(schema)

    lf0 = (
        lf0.with_columns(pl.col(time_col).cast(pl.Datetime("us", time_zone="UTC"), strict=False))
        .filter(pl.col(time_col).is_not_null())
        .filter((pl.col(time_col) >= dt_start) & (pl.col(time_col) < dt_end))
        .sort(time_col)
    )

    monotonic_report = None
    if monotonic_check:
        _log("Validating temporal monotonicity (sample)...")
        monotonic_report = _check_monotonic(lf0, time_col, sample_rows=monotonic_sample_rows)

    if split_mode == "time":
        tmin, tmax = _min_max_time(lf0, time_col)
        span_us = _span_us(tmin, tmax)
        if span_us <= 0:
            raise ValueError("Time span is zero after date filter")

        train_end_raw = tmin + pl.duration(microseconds=int(span_us * train_portion))
        val_end_raw = tmin + pl.duration(microseconds=int(span_us * (train_portion + val_portion)))

        train_end = train_end_raw - embargo
        val_start = train_end_raw + embargo
        val_end = val_end_raw - embargo
        test_start = val_end_raw + embargo

        lf_train = lf0.filter(pl.col(time_col) < train_end)
        lf_val = lf0.filter((pl.col(time_col) >= val_start) & (pl.col(time_col) < val_end))
        lf_test = lf0.filter(pl.col(time_col) >= test_start)

    else:
        total = lf0.select(pl.len().alias("n")).collect(streaming=True)["n"][0]
        n = int(total) if total is not None else 0
        if n <= 0:
            raise ValueError("No rows after date filter")

        n_train = int(math.floor(n * train_portion))
        n_val = int(math.floor(n * val_portion))
        n_test = n - (n_train + n_val)
        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError(f"Split produced empty set: train={n_train} val={n_val} test={n_test}")

        lf_idx = lf0.with_row_count("__rid")
        lf_train = lf_idx.filter(pl.col("__rid") < n_train).drop("__rid")
        lf_val = lf_idx.filter((pl.col("__rid") >= n_train) & (pl.col("__rid") < n_train + n_val)).drop("__rid")
        lf_test = lf_idx.filter(pl.col("__rid") >= n_train + n_val).drop("__rid")

    if boundary_drop:
        ntr = int(lf_train.select(pl.len().alias("n")).collect(streaming=True)["n"][0])
        nva = int(lf_val.select(pl.len().alias("n")).collect(streaming=True)["n"][0])
        lf_train = lf_train.head(max(0, ntr - boundary_drop))
        lf_val = lf_val.head(max(0, nva - boundary_drop))

    _log(f"Writing: {train_path}")
    _sink_parquet_atomic(lf_train, train_path)
    _log(f"Writing: {val_path}")
    _sink_parquet_atomic(lf_val, val_path)
    _log(f"Writing: {test_path}")
    _sink_parquet_atomic(lf_test, test_path)

    def _split_info(path: str) -> dict:
        lf = pl.scan_parquet(path).with_columns(pl.col(time_col).cast(pl.Datetime("us", time_zone="UTC"), strict=False))
        out = lf.select(
            pl.len().alias("rows"),
            pl.col(time_col).min().alias("time_min"),
            pl.col(time_col).max().alias("time_max"),
        ).collect(streaming=True)
        return {
            "rows": int(out["rows"][0]),
            "time_min": str(out["time_min"][0]) if out["time_min"][0] is not None else None,
            "time_max": str(out["time_max"][0]) if out["time_max"][0] is not None else None,
            "size_bytes": os.path.getsize(path),
        }

    train_info = _split_info(train_path)
    val_info = _split_info(val_path)
    test_info = _split_info(test_path)

    def _ensure_nonoverlap(a: dict, b: dict, a_name: str, b_name: str):
        if a["time_max"] is None or b["time_min"] is None:
            return
        if a["time_max"] >= b["time_min"]:
            raise ValueError(f"Split overlap detected: {a_name}.time_max >= {b_name}.time_min")

    _ensure_nonoverlap(train_info, val_info, "train", "val")
    _ensure_nonoverlap(val_info, test_info, "val", "test")

    _atomic_write_json(
        manifest_path,
        {
            "version": "3.0",
            "features_source_path": os.path.abspath(features_source_path),
            "input_fingerprint": {
                **_file_info(features_source_path),
                "sha256_head64mb": _sha256_file(features_source_path, 64 * 1024 * 1024),
            },
            "time_col": time_col,
            "date_filter": {"start": starting_date, "end": end_date},
            "split_mode": split_mode,
            "embargo_duration": embargo_str,
            "boundary_drop": boundary_drop,
            "monotonic_check": {
                "enabled": monotonic_check,
                "sample_rows": monotonic_sample_rows,
                "result": monotonic_report,
            },
            "portions": {"train": train_portion, "val": val_portion, "test": test_portion},
            "paths": {
                "train": os.path.abspath(train_path),
                "val": os.path.abspath(val_path),
                "test": os.path.abspath(test_path),
            },
            "splits": {"train": train_info, "val": val_info, "test": test_info},
        },
    )

    _log("Data splitting finished.")
    print("")
