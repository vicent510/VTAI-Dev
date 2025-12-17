from __future__ import annotations

import os
import csv
import tempfile
import shutil
from typing import Tuple, Optional

import polars as pl

from utils.basics import _log

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)


PARQUET_COMPRESSION = "zstd"
PARQUET_STATS = True

PROBE_CHUNK = 200_000
TARGET_CHUNK_MB = 512
MIN_CHUNK = 100_000
MAX_CHUNK = 2_000_000


def verificate_data(dataframe: pl.DataFrame) -> pl.DataFrame:
    df = dataframe.clone()

    colmap_candidates = {
        "time": {"time", "timestamp", "ts", "datetime", "date", "t"},
        "bid": {"bid", "best_bid", "b", "bid_price", "px_bid"},
        "ask": {"ask", "best_ask", "a", "ask_price", "px_ask", "offer"},
    }

    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}
    rename_map = {}
    for std_col, aliases in colmap_candidates.items():
        for a in aliases:
            if a in lower_to_actual:
                src = lower_to_actual[a]
                if src != std_col:
                    rename_map[src] = std_col
                break

    if rename_map:
        df = df.rename(rename_map)

    required = ["time", "bid", "ask"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {df.columns}")

    df = df.select(required).drop_nulls(required)

    def parse_price(col_name: str) -> pl.Expr:
        s = pl.col(col_name).cast(pl.Utf8, strict=False).str.strip_chars()
        both = s.str.contains(r"\.") & s.str.contains(r",")
        euro_like = s.str.contains(r"\.\d{3},") | s.str.contains(r",\d{2}$")

        s_clean = (
            pl.when(both & euro_like)
            .then(s.str.replace_all(r"\.", "").str.replace_all(",", "."))
            .when(both & ~euro_like)
            .then(s.str.replace_all(",", ""))
            .when(s.str.contains(",") & ~s.str.contains(r"\."))
            .then(s.str.replace_all(",", "."))
            .otherwise(s)
        )
        return s_clean.cast(pl.Float64, strict=False)

    df = df.with_columns(
        parse_price("bid").alias("bid"),
        parse_price("ask").alias("ask"),
    ).drop_nulls(["bid", "ask"])

    time_dtype = df.schema["time"]

    if time_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32, pl.Float64, pl.Float32):
        median = df.select(pl.col("time").median()).item()
        if median is None:
            df = df.with_columns(pl.lit(None).cast(pl.Datetime("us", time_zone="UTC")).alias("time"))
        else:
            if median > 1e17:
                unit = "ns"
            elif median > 1e14:
                unit = "us"
            elif median > 1e11:
                unit = "ms"
            else:
                unit = "s"
            df = df.with_columns(
                pl.from_epoch(pl.col("time").cast(pl.Int64, strict=False), time_unit=unit)
                .dt.replace_time_zone("UTC")
                .alias("time")
            )
    elif time_dtype == pl.Datetime:
        df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC").alias("time"))
    else:
        df = df.with_columns(
            pl.col("time")
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.to_datetime(strict=False, time_zone="UTC")
            .alias("time")
        )

    df = df.drop_nulls(["time"])
    df = df.filter((pl.col("bid") > 0) & (pl.col("ask") > 0) & (pl.col("bid") <= pl.col("ask")))
    df = df.sort("time")
    df = df.unique(subset=["time", "bid", "ask"], keep="first")

    return df


def clean_data(dataframe: pl.DataFrame) -> pl.DataFrame:
    df = dataframe

    required = ["time", "bid", "ask"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {df.columns}")

    df = df.select(required).drop_nulls(required)

    df = df.filter(pl.col("bid").is_finite() & pl.col("ask").is_finite())

    df = df.with_columns(
        pl.col("bid").cast(pl.Float64, strict=False),
        pl.col("ask").cast(pl.Float64, strict=False),
    ).drop_nulls(["bid", "ask"])

    df = df.filter((pl.col("bid") > 0) & (pl.col("ask") > 0) & (pl.col("bid") <= pl.col("ask")))

    if df.schema["time"] != pl.Datetime:
        df = df.with_columns(pl.col("time").cast(pl.Datetime("us", time_zone="UTC"), strict=False))

    df = df.sort("time")
    df = df.unique(subset=["time", "bid", "ask"], keep="first")

    df = df.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))

    spread_q = df.select(
        pl.col("spread").quantile(0.99, "nearest").alias("q99"),
        pl.col("spread").quantile(0.01, "nearest").alias("q01"),
    )

    q01 = spread_q["q01"][0]
    q99 = spread_q["q99"][0]

    if q01 is not None and q99 is not None and q99 > q01:
        df = df.filter((pl.col("spread") >= q01) & (pl.col("spread") <= q99))

    return df.drop("spread")


def save_parquet_data(dataframe: pl.DataFrame, output_path: str) -> None:
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("output_path must be a non-empty string")

    output_path = os.path.expanduser(output_path)
    parent = os.path.dirname(output_path) or "."
    os.makedirs(parent, exist_ok=True)

    if not output_path.lower().endswith(".parquet"):
        output_path = output_path + ".parquet"

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=parent)
    os.close(fd)

    try:
        dataframe.write_parquet(tmp_path, compression=PARQUET_COMPRESSION, statistics=PARQUET_STATS)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def data_wrangling(data_wrangling_config: dict):
    data_path = data_wrangling_config["data_path"]
    output_path = data_wrangling_config["output_path"]
    tick_size = int(data_wrangling_config["tick_size"])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")
    if tick_size <= 0:
        raise ValueError("tick_size must be a positive integer")

    tmp_dir = tempfile.mkdtemp(prefix="data_wrangling_")
    parts_dir = os.path.join(tmp_dir, "parquet_parts")

    def _detect_encoding(head: bytes) -> str:
        if head.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if head.startswith(b"\xff\xfe\x00\x00"):
            return "utf-32-le"
        if head.startswith(b"\x00\x00\xfe\xff"):
            return "utf-32-be"
        if head.startswith(b"\xff\xfe"):
            return "utf-16-le"
        if head.startswith(b"\xfe\xff"):
            return "utf-16-be"

        n = max(1, len(head))
        z = head.count(b"\x00") / n
        if z > 0.15:
            even_zeros = sum(1 for i in range(0, len(head), 2) if head[i] == 0) / max(1, (len(head) + 1) // 2)
            odd_zeros = sum(1 for i in range(1, len(head), 2) if head[i] == 0) / max(1, len(head) // 2)
            return "utf-16-le" if odd_zeros > even_zeros else "utf-16-be"

        try:
            head.decode("utf-8", errors="strict")
            return "utf-8"
        except UnicodeDecodeError:
            return "cp1252"

    def _detect_separator(sample_text: str) -> str:
        lines = [ln for ln in sample_text.splitlines() if ln.strip()]
        if not lines:
            return ","
        s = lines[0]
        candidates = [",", ";", "\t", "|"]
        counts = {c: s.count(c) for c in candidates}
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            return best
        try:
            dialect = csv.Sniffer().sniff(s, delimiters="".join(candidates))
            return dialect.delimiter
        except Exception:
            return ","

    def _prepare_input(path_in: str) -> Tuple[str, str, Optional[str]]:
        with open(path_in, "rb") as f:
            head = f.read(64 * 1024)

        enc = _detect_encoding(head)
        sample = head.decode(enc, errors="replace")
        sep = _detect_separator(sample)

        _log(f"Detected encoding: {enc}")
        _log(f"Detected separator: {repr(sep)}")

        if enc in ("utf-8", "utf-8-sig"):
            return path_in, sep, None

        tmp_utf8 = os.path.join(tmp_dir, "input_utf8.csv")
        _log("Transcoding input to UTF-8 (temporary file)...")
        with open(path_in, "r", encoding=enc, errors="replace", newline="") as fin, open(
            tmp_utf8, "w", encoding="utf-8", errors="strict", newline=""
        ) as fout:
            for line in fin:
                fout.write(line)

        return tmp_utf8, sep, tmp_utf8

    def _auto_chunk_size(input_path: str, sep: str) -> int:
        _log("Profiling chunk size...")
        probe_reader = pl.read_csv_batched(
            input_path,
            batch_size=PROBE_CHUNK,
            separator=sep,
            infer_schema_length=10_000,
            ignore_errors=True,
            try_parse_dates=False,
        )
        probe_batches = probe_reader.next_batches(1)
        if not probe_batches:
            raise ValueError("Input CSV appears empty")
        probe_df = probe_batches[0]
        mem_bytes = max(1, int(probe_df.estimated_size()))
        rows = max(1, int(probe_df.height))
        bytes_per_row = mem_bytes / rows

        target_bytes = TARGET_CHUNK_MB * 1024 * 1024
        chunk = int(target_bytes / max(1.0, bytes_per_row))
        chunk = max(MIN_CHUNK, min(MAX_CHUNK, chunk))
        _log(f"Auto-selected chunk_size: {chunk} rows (target ~{TARGET_CHUNK_MB} MB)")
        return chunk

    def _write_part(bars: pl.DataFrame, idx: int) -> str:
        os.makedirs(parts_dir, exist_ok=True)
        path = os.path.join(parts_dir, f"part-{idx:06d}.parquet")
        bars.write_parquet(path, compression=PARQUET_COMPRESSION, statistics=PARQUET_STATS)
        return path

    def _final_output_path(p: str) -> str:
        return p if p.lower().endswith(".parquet") else p + ".parquet"

    try:
        input_path, separator, _tmp_utf8 = _prepare_input(data_path)
        chunk_size = _auto_chunk_size(input_path, separator)

        file_size = os.path.getsize(input_path)
        _log(f"Input size: {file_size} bytes")
        _log("Starting streaming pipeline...")

        reader = pl.read_csv_batched(
            input_path,
            batch_size=chunk_size,
            separator=separator,
            infer_schema_length=10_000,
            ignore_errors=True,
            try_parse_dates=False,
        )

        total_batches = getattr(reader, "n_batches", None)
        pbar = tqdm(
            total=total_batches if isinstance(total_batches, int) else None,
            desc="Processing chunks",
            unit="chunk",
        )

        carry_df: Optional[pl.DataFrame] = None
        global_row_offset = 0
        part_idx = 0
        total_bars = 0
        total_ticks_in_bars = 0

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break

            chunk = batches[0]
            v = verificate_data(chunk)
            c = clean_data(v)

            if carry_df is not None and carry_df.height > 0:
                c = pl.concat([carry_df, c], how="vertical", rechunk=False)

            if c.height == 0:
                carry_df = None
                pbar.update(1)
                continue

            c = (
                c.sort("time")
                .with_row_count(name="_row_id", offset=global_row_offset)
                .with_columns(
                    ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
                    (pl.col("_row_id") // pl.lit(tick_size)).cast(pl.Int64).alias("bar_id"),
                )
            )

            max_bar = c.select(pl.col("bar_id").max()).item()
            if max_bar is None:
                carry_df = None
                global_row_offset += c.height
                pbar.update(1)
                continue

            last_bar = int(max_bar)
            carry_df = c.filter(pl.col("bar_id") == last_bar).select(["time", "bid", "ask"])
            complete = c.filter(pl.col("bar_id") < last_bar)

            if complete.height > 0:
                bars = (
                    complete.group_by("bar_id", maintain_order=True)
                    .agg(
                        pl.col("time").first().alias("start_time"),
                        pl.col("time").last().alias("end_time"),
                        pl.col("mid").first().alias("open"),
                        pl.col("mid").max().alias("high"),
                        pl.col("mid").min().alias("low"),
                        pl.col("mid").last().alias("close"),
                        pl.col("bid").first().alias("bid_open"),
                        pl.col("bid").max().alias("bid_high"),
                        pl.col("bid").min().alias("bid_low"),
                        pl.col("bid").last().alias("bid_close"),
                        pl.col("ask").first().alias("ask_open"),
                        pl.col("ask").max().alias("ask_high"),
                        pl.col("ask").min().alias("ask_low"),
                        pl.col("ask").last().alias("ask_close"),
                        (pl.col("ask") - pl.col("bid")).mean().alias("spread_mean"),
                        pl.len().alias("ticks"),
                    )
                    .drop("bar_id")
                )

                _write_part(bars, part_idx)
                part_idx += 1
                total_bars += int(bars.height)
                total_ticks_in_bars += int(bars.select(pl.col("ticks").sum()).item())

            global_row_offset += c.height
            pbar.update(1)
            pbar.set_postfix_str(f"bars={total_bars}")

        pbar.close()

        if carry_df is not None and carry_df.height > 0:
            tail = (
                carry_df.sort("time")
                .with_row_count(name="_row_id", offset=global_row_offset)
                .with_columns(
                    ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
                    (pl.col("_row_id") // pl.lit(tick_size)).cast(pl.Int64).alias("bar_id"),
                )
            )

            tail_bars = (
                tail.group_by("bar_id", maintain_order=True)
                .agg(
                    pl.col("time").first().alias("start_time"),
                    pl.col("time").last().alias("end_time"),
                    pl.col("mid").first().alias("open"),
                    pl.col("mid").max().alias("high"),
                    pl.col("mid").min().alias("low"),
                    pl.col("mid").last().alias("close"),
                    pl.col("bid").first().alias("bid_open"),
                    pl.col("bid").max().alias("bid_high"),
                    pl.col("bid").min().alias("bid_low"),
                    pl.col("bid").last().alias("bid_close"),
                    pl.col("ask").first().alias("ask_open"),
                    pl.col("ask").max().alias("ask_high"),
                    pl.col("ask").min().alias("ask_low"),
                    pl.col("ask").last().alias("ask_close"),
                    (pl.col("ask") - pl.col("bid")).mean().alias("spread_mean"),
                    pl.len().alias("ticks"),
                )
                .drop("bar_id")
            )

            _write_part(tail_bars, part_idx)
            part_idx += 1
            total_bars += int(tail_bars.height)
            total_ticks_in_bars += int(tail_bars.select(pl.col("ticks").sum()).item())

        _log(f"Total bars (parts): {total_bars}")
        _log(f"Total ticks in bars: {total_ticks_in_bars}")

        out_parquet = _final_output_path(output_path)
        _log("Writing final parquet from parts (streaming scan)...")

        pattern = os.path.join(parts_dir, "*.parquet")
        lf = pl.scan_parquet(pattern)
        lf.sink_parquet(out_parquet, compression=PARQUET_COMPRESSION, statistics=PARQUET_STATS)

        _log(f"Final parquet size: {os.path.getsize(out_parquet)} bytes")
        _log("Data Wrangling Finished.")
        print("")
        return out_parquet

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
