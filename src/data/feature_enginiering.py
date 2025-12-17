from __future__ import annotations

import os
import math
import json
import shutil
import hashlib
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from utils.basics import _log

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)


PARQUET_COMPRESSION = "zstd"
PARQUET_STATS = True

def _default_windows() -> Dict[str, int]:
    return {"S": 5, "M": 20, "L": 50, "XL": 200}


def _zs(expr: pl.Expr, win: int) -> pl.Expr:
    mu = expr.rolling_mean(win)
    sd = expr.rolling_std(win).replace(0.0, None)
    return (expr - mu) / sd


def _pct_from_z(z: pl.Expr) -> pl.Expr:
    return pl.lit(1.0) / (pl.lit(1.0) + (-pl.lit(1.702) * z).exp())


def _range_position(price: pl.Expr, win: int) -> pl.Expr:
    lo = price.rolling_min(win)
    hi = price.rolling_max(win)
    denom = (hi - lo).replace(0.0, None)
    return ((price - lo) / denom).clip(0.0, 1.0)


def _realized_vol(logret: pl.Expr, win: int) -> pl.Expr:
    return (logret.pow(2).rolling_sum(win)).sqrt()


def _rolling_corr(x: pl.Expr, y: pl.Expr, win: int) -> pl.Expr:
    mx = x.rolling_mean(win)
    my = y.rolling_mean(win)
    mxy = (x * y).rolling_mean(win)
    vx = (x * x).rolling_mean(win) - mx * mx
    vy = (y * y).rolling_mean(win) - my * my
    denom = (vx * vy).sqrt().replace(0.0, None)
    return (mxy - mx * my) / denom


def _rolling_cov(x: pl.Expr, y: pl.Expr, win: int) -> pl.Expr:
    mx = x.rolling_mean(win)
    my = y.rolling_mean(win)
    mxy = (x * y).rolling_mean(win)
    return mxy - mx * my


def _rolling_var(x: pl.Expr, win: int) -> pl.Expr:
    mx = x.rolling_mean(win)
    return (x * x).rolling_mean(win) - mx * mx


def _trend_slope(y: pl.Expr, x: pl.Expr, win: int) -> pl.Expr:
    w = pl.lit(float(win))
    sx = x.rolling_sum(win)
    sy = y.rolling_sum(win)
    sxx = (x * x).rolling_sum(win)
    sxy = (x * y).rolling_sum(win)
    denom = (w * sxx - sx * sx).replace(0.0, None)
    return (w * sxy - sx * sy) / denom


def _trend_r2(y: pl.Expr, x: pl.Expr, win: int) -> pl.Expr:
    c = _rolling_corr(x, y, win)
    return (c * c).clip(0.0, 1.0)


def _ar1_phi(x: pl.Expr, win: int) -> pl.Expr:
    x1 = x.shift(1)
    cov = _rolling_cov(x, x1, win)
    var = _rolling_var(x1, win).replace(0.0, None)
    return cov / var


def _ar1_halflife(phi: pl.Expr) -> pl.Expr:
    aphi = phi.abs()
    return (
        pl.when((aphi > 0.0) & (aphi < 0.999999))
        .then(pl.lit(-math.log(2.0)) / aphi.log())
        .otherwise(None)
    )


def _efficiency_ratio(price: pl.Expr, win: int) -> pl.Expr:
    net = (price - price.shift(win)).abs()
    path = (price - price.shift(1)).abs().rolling_sum(win).replace(0.0, None)
    return net / path


def _variance_ratio(log_price: pl.Expr, win: int, k: int) -> pl.Expr:
    r1 = (log_price - log_price.shift(1)).fill_null(0.0)
    rk = (log_price - log_price.shift(k)).fill_null(0.0)
    v1 = _rolling_var(r1, win).replace(0.0, None)
    vk = _rolling_var(rk, win).replace(0.0, None)
    return vk / (pl.lit(float(k)) * v1)


def _sign_entropy(x: pl.Expr, win: int) -> pl.Expr:
    s = x.sign().fill_null(0.0)
    p_up = (s > 0).cast(pl.Float64).rolling_mean(win)
    p_dn = (s < 0).cast(pl.Float64).rolling_mean(win)
    p_0 = (s == 0).cast(pl.Float64).rolling_mean(win)

    def h(p: pl.Expr) -> pl.Expr:
        return pl.when(p > 0).then(-p * p.log()).otherwise(0.0)

    ent = h(p_up) + h(p_dn) + h(p_0)
    return ent / pl.lit(math.log(3.0))


def _sha256_file(path: str, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = max_bytes
        while True:
            if remaining <= 0:
                break
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _normalize_anchor(expr: pl.Expr, anchor_win: int, want_z: bool, want_pct: bool) -> List[Tuple[str, pl.Expr]]:
    out: List[Tuple[str, pl.Expr]] = []
    if want_z or want_pct:
        z = _zs(expr, anchor_win)
        if want_z:
            out.append(("z", z.fill_null(0.0)))
        if want_pct:
            out.append(("pct", _pct_from_z(z)))
    return out


def _is_heavy_tail_feature(name: str) -> bool:
    heavy_prefix = (
        "rv_",
        "spread_",
        "spread_rel_",
        "bid_ask_pressure_",
        "microprice_proxy",
        "variance_ratio_",
        "return_sign_entropy_",
        "efficiency_ratio_",
    )
    return name.startswith(heavy_prefix) or name in ("spread_rel_return_1", "spread_rel_range_1")


def _feature_exprs(wins: Dict[str, int], anchor_key: str = "XL") -> List[pl.Expr]:
    anchor_win = int(wins.get(anchor_key, max(wins.values())))

    open_ = pl.col("open").cast(pl.Float64)
    high = pl.col("high").cast(pl.Float64)
    low = pl.col("low").cast(pl.Float64)
    close = pl.col("close").cast(pl.Float64)

    bid_close = pl.col("bid_close").cast(pl.Float64)
    ask_close = pl.col("ask_close").cast(pl.Float64)

    bid_open = pl.col("bid_open").cast(pl.Float64)
    bid_high = pl.col("bid_high").cast(pl.Float64)
    bid_low = pl.col("bid_low").cast(pl.Float64)

    ask_open = pl.col("ask_open").cast(pl.Float64)
    ask_high = pl.col("ask_high").cast(pl.Float64)
    ask_low = pl.col("ask_low").cast(pl.Float64)

    spread = pl.col("spread_mean").cast(pl.Float64)
    ticks = pl.col("ticks").cast(pl.Float64)

    x = pl.col("_row_id").cast(pl.Float64)

    mid_expr = (bid_close + ask_close) / 2.0
    microprice_expr = mid_expr - close

    log_close = close.log()
    logret_1 = (log_close - log_close.shift(1)).fill_null(0.0)

    spread_rel_return_expr = logret_1 / spread.replace(0.0, None)
    spread_rel_range_expr = (high - low) / spread.replace(0.0, None)

    duration_ms = (
        (pl.col("end_time").cast(pl.Datetime("us", time_zone="UTC")) - pl.col("start_time").cast(pl.Datetime("us", time_zone="UTC")))
        .dt.total_microseconds()
        .cast(pl.Float64) / 1000.0
    ).alias("duration_ms")

    ticks_per_ms = (ticks / duration_ms.replace(0.0, None)).alias("ticks_per_ms")
    ret_per_ms = (logret_1 / duration_ms.replace(0.0, None)).alias("log_return_1_per_ms")

    feats: List[pl.Expr] = [
        pl.col("start_time"),
        pl.col("end_time"),
        open_.alias("open"),
        high.alias("high"),
        low.alias("low"),
        close.alias("close"),
        bid_open.alias("bid_open"),
        bid_high.alias("bid_high"),
        bid_low.alias("bid_low"),
        bid_close.alias("bid_close"),
        ask_open.alias("ask_open"),
        ask_high.alias("ask_high"),
        ask_low.alias("ask_low"),
        ask_close.alias("ask_close"),
        spread.alias("spread_mean"),
        ticks.alias("ticks"),
        duration_ms,
        ticks_per_ms,
        ret_per_ms,
        mid_expr.alias("mid"),
        microprice_expr.alias("microprice_proxy"),
        logret_1.alias("log_return_1"),
        spread_rel_return_expr.alias("spread_rel_return_1"),
        spread_rel_range_expr.alias("spread_rel_range_1"),
    ]

    for base_expr, base_name in (
        (microprice_expr, "microprice_proxy"),
        (spread_rel_return_expr, "spread_rel_return_1"),
        (spread_rel_range_expr, "spread_rel_range_1"),
        (ticks_per_ms, "ticks_per_ms"),
        (ret_per_ms, "log_return_1_per_ms"),
    ):
        want_z = True
        want_pct = _is_heavy_tail_feature(base_name)
        for suf, ex in _normalize_anchor(base_expr, anchor_win, want_z, want_pct):
            feats.append(ex.alias(f"{base_name}_{suf}"))

    for name, w in wins.items():
        r_expr = (log_close - log_close.shift(w)).fill_null(0.0)
        r_name = f"log_return_{name}"
        feats.append(r_expr.alias(r_name))
        for suf, ex in _normalize_anchor(r_expr, anchor_win, True, False):
            feats.append(ex.alias(f"{r_name}_{suf}"))

        dr_expr = (r_expr - r_expr.shift(1)).fill_null(0.0)
        dr_name = f"delta_log_return_{name}"
        feats.append(dr_expr.alias(dr_name))
        for suf, ex in _normalize_anchor(dr_expr, anchor_win, True, False):
            feats.append(ex.alias(f"{dr_name}_{suf}"))

        rv_expr = _realized_vol(logret_1, w)
        rv_name = f"rv_{name}"
        feats.append(rv_expr.alias(rv_name))
        for suf, ex in _normalize_anchor(rv_expr, anchor_win, True, True):
            feats.append(ex.alias(f"{rv_name}_{suf}"))

        rp_expr = _range_position(close, w)
        rp_name = f"range_position_{name}"
        feats.append(rp_expr.alias(rp_name))
        for suf, ex in _normalize_anchor(rp_expr, anchor_win, True, False):
            feats.append(ex.alias(f"{rp_name}_{suf}"))

        rw_expr = close.rolling_max(w) - close.rolling_min(w)
        rw_name = f"range_width_{name}"
        feats.append(rw_expr.alias(rw_name))
        for suf, ex in _normalize_anchor(rw_expr, anchor_win, True, True):
            feats.append(ex.alias(f"{rw_name}_{suf}"))

        sm_expr = spread.rolling_mean(w)
        sm_name = f"spread_mean_{name}"
        feats.append(sm_expr.alias(sm_name))
        for suf, ex in _normalize_anchor(sm_expr, anchor_win, True, True):
            feats.append(ex.alias(f"{sm_name}_{suf}"))

        schg_expr = (sm_expr - sm_expr.shift(1)).fill_null(0.0)
        schg_name = f"spread_change_{name}"
        feats.append(schg_expr.alias(schg_name))
        for suf, ex in _normalize_anchor(schg_expr, anchor_win, True, True):
            feats.append(ex.alias(f"{schg_name}_{suf}"))

        svol_expr = spread.rolling_std(w)
        svol_name = f"spread_vol_{name}"
        feats.append(svol_expr.alias(svol_name))
        for suf, ex in _normalize_anchor(svol_expr, anchor_win, True, True):
            feats.append(ex.alias(f"{svol_name}_{suf}"))

        tmean_expr = ticks.rolling_mean(w)
        tmean_name = f"ticks_mean_{name}"
        feats.append(tmean_expr.alias(tmean_name))
        for suf, ex in _normalize_anchor(tmean_expr, anchor_win, True, False):
            feats.append(ex.alias(f"{tmean_name}_{suf}"))

        tchg_expr = (tmean_expr - tmean_expr.shift(1)).fill_null(0.0)
        tchg_name = f"ticks_change_{name}"
        feats.append(tchg_expr.alias(tchg_name))
        for suf, ex in _normalize_anchor(tchg_expr, anchor_win, True, False):
            feats.append(ex.alias(f"{tchg_name}_{suf}"))

        sign = (close - close.shift(1)).sign().fill_null(0.0)
        imb_expr = sign.rolling_sum(w) / pl.lit(float(w))
        imb_name = f"tick_imbalance_{name}"
        feats.append(imb_expr.alias(imb_name))
        for suf, ex in _normalize_anchor(imb_expr, anchor_win, True, False):
            feats.append(ex.alias(f"{imb_name}_{suf}"))

        bchg = (bid_close - bid_close.shift(w)).fill_null(0.0)
        achg = (ask_close - ask_close.shift(w)).fill_null(0.0)
        press = achg - bchg
        bchg_name = f"bid_change_{name}"
        achg_name = f"ask_change_{name}"
        press_name = f"bid_ask_pressure_{name}"
        feats += [bchg.alias(bchg_name), achg.alias(achg_name), press.alias(press_name)]
        for base_expr, base_name in ((bchg, bchg_name), (achg, achg_name), (press, press_name)):
            for suf, ex in _normalize_anchor(base_expr, anchor_win, True, _is_heavy_tail_feature(base_name)):
                feats.append(ex.alias(f"{base_name}_{suf}"))

        er_expr = _efficiency_ratio(close, w)
        er_name = f"efficiency_ratio_{name}"
        feats.append(er_expr.alias(er_name))
        for suf, ex in _normalize_anchor(er_expr, anchor_win, True, True):
            feats.append(ex.alias(f"{er_name}_{suf}"))

        ent_expr = _sign_entropy(logret_1, w)
        ent_name = f"return_sign_entropy_{name}"
        feats.append(ent_expr.alias(ent_name))
        for suf, ex in _normalize_anchor(ent_expr, anchor_win, True, True):
            feats.append(ex.alias(f"{ent_name}_{suf}"))

    for name in ("M", "L", "XL"):
        if name in wins:
            w = wins[name]
            slope_expr = _trend_slope(close, x, w)
            r2_expr = _trend_r2(close, x, w)
            rv_expr = _realized_vol(logret_1, w).replace(0.0, None)
            strength_expr = slope_expr / rv_expr
            for base_expr, base_name in (
                (slope_expr, f"trend_slope_{name}"),
                (r2_expr, f"trend_r2_{name}"),
                (strength_expr, f"trend_strength_{name}"),
            ):
                feats.append(base_expr.alias(base_name))
                for suf, ex in _normalize_anchor(base_expr, anchor_win, True, False):
                    feats.append(ex.alias(f"{base_name}_{suf}"))

    for name in ("S", "M"):
        if name in wins:
            w = wins[name]
            ac_expr = _rolling_corr(logret_1, logret_1.shift(1), w)
            ac_name = f"return_autocorr_{name}"
            feats.append(ac_expr.alias(ac_name))
            for suf, ex in _normalize_anchor(ac_expr, anchor_win, True, False):
                feats.append(ex.alias(f"{ac_name}_{suf}"))

    for name in ("M", "L"):
        if name in wins:
            w = wins[name]
            phi_expr = _ar1_phi(logret_1, w)
            hl_expr = _ar1_halflife(phi_expr)
            for base_expr, base_name in (
                (phi_expr, f"ar1_phi_{name}"),
                (hl_expr, f"mean_reversion_halflife_{name}"),
            ):
                feats.append(base_expr.alias(base_name))
                for suf, ex in _normalize_anchor(base_expr, anchor_win, True, True):
                    feats.append(ex.alias(f"{base_name}_{suf}"))

    if "M" in wins:
        w = wins["M"]
        rvS = _realized_vol(logret_1, wins["S"])
        c1 = _rolling_corr(logret_1, rvS, w)
        c2 = _rolling_corr(logret_1, spread, w)
        c3 = _rolling_corr(rvS, spread, w)
        for base_expr, base_name in ((c1, "corr_ret1_rvS_M"), (c2, "corr_ret1_spread_M"), (c3, "corr_rvS_spread_M")):
            feats.append(base_expr.alias(base_name))
            for suf, ex in _normalize_anchor(base_expr, w, True, False):
                feats.append(ex.alias(f"{base_name}_{suf}"))

    if "L" in wins:
        w = wins["L"]
        rvS = _realized_vol(logret_1, wins["S"])
        c1 = _rolling_corr(logret_1, rvS, w)
        c2 = _rolling_corr(logret_1, spread, w)
        c3 = _rolling_corr(rvS, spread, w)
        for base_expr, base_name in ((c1, "corr_ret1_rvS_L"), (c2, "corr_ret1_spread_L"), (c3, "corr_rvS_spread_L")):
            feats.append(base_expr.alias(base_name))
            for suf, ex in _normalize_anchor(base_expr, w, True, False):
                feats.append(ex.alias(f"{base_name}_{suf}"))

    if "M" in wins:
        w = wins["M"]
        rvM = _realized_vol(logret_1, w)
        ch = (rvM - rvM.shift(w)).fill_null(0.0)
        for base_expr, base_name in ((ch, "rv_change_M"),):
            feats.append(base_expr.alias(base_name))
            for suf, ex in _normalize_anchor(base_expr, w, True, True):
                feats.append(ex.alias(f"{base_name}_{suf}"))

    if "L" in wins:
        w = wins["L"]
        rvL = _realized_vol(logret_1, w)
        ch = (rvL - rvL.shift(w)).fill_null(0.0)
        for base_expr, base_name in ((ch, "rv_change_L"),):
            feats.append(base_expr.alias(base_name))
            for suf, ex in _normalize_anchor(base_expr, w, True, True):
                feats.append(ex.alias(f"{base_name}_{suf}"))

    if "M" in wins:
        w = wins["M"]
        vr = _variance_ratio(log_close, w, k=5)
        feats.append(vr.alias("variance_ratio_M_k5"))
        for suf, ex in _normalize_anchor(vr, w, True, True):
            feats.append(ex.alias(f"variance_ratio_M_k5_{suf}"))

    if "L" in wins:
        w = wins["L"]
        vr = _variance_ratio(log_close, w, k=10)
        feats.append(vr.alias("variance_ratio_L_k10"))
        for suf, ex in _normalize_anchor(vr, w, True, True):
            feats.append(ex.alias(f"variance_ratio_L_k10_{suf}"))

    return feats


def _compute_features_df(df: pl.DataFrame, wins: Dict[str, int], row_id_start: int) -> pl.DataFrame:
    required = {
        "start_time", "end_time", "open", "high", "low", "close",
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
        "spread_mean", "ticks",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in source parquet: {sorted(missing)}")

    df2 = df.with_columns(
        pl.int_range(pl.lit(row_id_start), pl.lit(row_id_start) + pl.len(), dtype=pl.Int64).alias("_row_id")
    )

    out = df2.select(_feature_exprs(wins))
    numeric_cols = [c for c, t in out.schema.items() if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    if numeric_cols:
        out = out.with_columns([pl.col(c).replace([float("inf"), float("-inf")], None) for c in numeric_cols])
    return out


def _sample_head_rows(parquet_path: str, n_rows: int) -> pl.DataFrame:
    return pl.scan_parquet(parquet_path).head(n_rows).collect(streaming=True)


def _concat_blocks(blocks: List[pl.DataFrame]) -> pl.DataFrame:
    blocks = [b for b in blocks if b is not None and b.height > 0]
    if not blocks:
        return pl.DataFrame()
    return pl.concat(blocks, how="vertical", rechunk=True)


def _fit_blocks_sample(parquet_path: str, fit_rows: int, total_rows: int, n_blocks: int, rows_per_block: int) -> pl.DataFrame:
    lf = pl.scan_parquet(parquet_path)
    if total_rows <= 0:
        return pl.DataFrame()

    fit_rows = max(1, min(fit_rows, total_rows))
    n_blocks = max(1, int(n_blocks))

    blocks: List[pl.DataFrame] = []
    if n_blocks == 1:
        return _sample_head_rows(parquet_path, min(fit_rows, rows_per_block))

    for i in range(n_blocks):
        start = int((fit_rows - 1) * (i / (n_blocks - 1))) if n_blocks > 1 else 0
        start = max(0, min(start, fit_rows - 1))
        blocks.append(lf.slice(start, min(rows_per_block, fit_rows - start)).collect(streaming=True))

    return _concat_blocks(blocks)


def _compute_clip_bounds(sample_df: pl.DataFrame, protect: List[str], q_low: float, q_high: float) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    numeric_cols = [
        c for c, t in sample_df.schema.items()
        if c not in protect and t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    if not numeric_cols:
        return bounds

    qs = sample_df.select(
        [pl.col(c).quantile(q_low, "nearest").alias(f"{c}__lo") for c in numeric_cols] +
        [pl.col(c).quantile(q_high, "nearest").alias(f"{c}__hi") for c in numeric_cols]
    )

    row = qs.row(0)
    n = len(numeric_cols)
    for i, c in enumerate(numeric_cols):
        lo = row[i]
        hi = row[i + n]
        if lo is None or hi is None:
            continue
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        bounds[c] = (lo_f, hi_f)
    return bounds


def _nan_rate(sample_df: pl.DataFrame, cols: List[str]) -> Dict[str, float]:
    if not cols:
        return {}
    row = sample_df.select([pl.col(c).is_null().mean().alias(c) for c in cols]).row(0)
    return {c: float(v) if v is not None else 1.0 for c, v in zip(cols, row)}


def _variance(sample_df: pl.DataFrame, cols: List[str]) -> Dict[str, float]:
    if not cols:
        return {}
    row = sample_df.select([pl.col(c).var().alias(c) for c in cols]).row(0)
    return {c: float(v) if v is not None else 0.0 for c, v in zip(cols, row)}


def _score_feature(col: str, var_map: Dict[str, float], nan_map: Dict[str, float]) -> float:
    v = var_map.get(col, 0.0)
    n = nan_map.get(col, 1.0)
    s = math.log1p(max(v, 0.0)) - 2.0 * n
    if col.endswith("_z"):
        s += 0.05
    if col.endswith("_pct"):
        s += 0.02
    if "_XL" in col or col.endswith("_1") or col.endswith("_per_ms"):
        s += 0.02
    if col.startswith(("duration_", "ticks_per_", "spread_rel_")):
        s += 0.03
    return s


def _prune_corr_group_aware(df: pl.DataFrame, threshold: float, protect: List[str]) -> List[str]:
    candidates = [c for c in df.columns if c not in protect]
    numeric = [c for c in candidates if df.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    if len(numeric) <= 1:
        return protect + numeric

    X = df.select(numeric).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    std = X.std(axis=0)
    keep_mask = std > 0
    numeric = [c for c, k in zip(numeric, keep_mask) if k]
    X = X[:, keep_mask]
    if X.shape[1] <= 1:
        return protect + numeric

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    var_map = _variance(df, numeric)
    nan_map = _nan_rate(df, numeric)

    kept: List[int] = []
    dropped = set()

    for i in range(corr.shape[0]):
        if i in dropped:
            continue
        kept.append(i)
        for j in range(i + 1, corr.shape[0]):
            if j in dropped:
                continue
            if abs(corr[i, j]) >= threshold:
                ci = numeric[i]
                cj = numeric[j]
                if _score_feature(ci, var_map, nan_map) >= _score_feature(cj, var_map, nan_map):
                    dropped.add(j)
                else:
                    dropped.add(i)
                    kept.pop()
                    break

    selected_numeric = [numeric[i] for i in kept if i not in dropped]
    return protect + selected_numeric


def _iter_parquet_rowgroups(path: str):
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        yield i, pf.read_row_group(i), pf.num_row_groups


def _write_part_parquet(df: pl.DataFrame, parts_dir: str, idx: int) -> str:
    os.makedirs(parts_dir, exist_ok=True)
    p = os.path.join(parts_dir, f"part-{idx:06d}.parquet")
    df.write_parquet(p, compression=PARQUET_COMPRESSION, statistics=PARQUET_STATS)
    return p


def _final_parquet_path(output_path: str) -> str:
    return output_path if output_path.lower().endswith(".parquet") else output_path + ".parquet"


def _select_with_clipping(cols: List[str], schema: Dict[str, pl.DataType], clip_bounds: Dict[str, Tuple[float, float]]) -> List[pl.Expr]:
    exprs: List[pl.Expr] = []
    for c in cols:
        if c in clip_bounds and schema.get(c) in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            lo, hi = clip_bounds[c]
            exprs.append(pl.col(c).clip(lo, hi).alias(c))
        else:
            exprs.append(pl.col(c))
    return exprs


def _save_transform_params(path: str, payload: dict) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def feature_enginiering(feature_enginiering_config: dict):
    data_source_path = feature_enginiering_config.get("data_source_path", "sample_data/dev/cleaned_data.parquet")
    output_path = feature_enginiering_config.get("output_path", "sample_data/dev/features.parquet")

    corr_threshold = float(feature_enginiering_config.get("corr_threshold", 0.85))
    fit_fraction = float(feature_enginiering_config.get("fit_fraction", 0.7))

    clip_q_low = float(feature_enginiering_config.get("clip_q_low", 0.001))
    clip_q_high = float(feature_enginiering_config.get("clip_q_high", 0.999))

    fit_sample_rows = int(feature_enginiering_config.get("fit_sample_rows", 300_000))
    corr_sample_rows = int(feature_enginiering_config.get("corr_sample_rows", 250_000))

    fit_blocks = int(feature_enginiering_config.get("fit_blocks", 5))

    wins_cfg = feature_enginiering_config.get("windows", None)
    wins = _default_windows() if not isinstance(wins_cfg, dict) else {k: int(v) for k, v in wins_cfg.items()}

    if not os.path.exists(data_source_path):
        raise FileNotFoundError(f"data_source_path not found: {data_source_path}")

    protect_cols = [
        "start_time", "end_time",
        "open", "high", "low", "close",
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
        "spread_mean", "ticks",
        "duration_ms", "ticks_per_ms", "log_return_1_per_ms",
        "mid", "microprice_proxy",
        "log_return_1",
        "spread_rel_return_1",
        "spread_rel_range_1",
    ]

    tmp_dir = tempfile.mkdtemp(prefix="feature_engineering_")
    parts_dir = os.path.join(tmp_dir, "parts_features")
    tmp_features = os.path.join(tmp_dir, "features_all.parquet")
    out_parquet = _final_parquet_path(output_path)

    transform_path = feature_enginiering_config.get(
        "transform_params_path",
        os.path.join(os.path.dirname(out_parquet) or ".", "feature_transform_params.json"),
    )

    max_win = max(wins.values())
    overlap = max_win + 2

    try:
        _log(f"Source parquet: {data_source_path}")
        _log(f"Windows: {wins}")
        _log("Stage 1/4: computing features -> parquet parts")

        pf = pq.ParquetFile(data_source_path)
        pbar = tqdm(total=pf.num_row_groups, desc="Feature chunks", unit="rowgroup")

        tail: Optional[pl.DataFrame] = None
        part_idx = 0
        total_rows_out = 0
        global_offset = 0

        for rg_idx, rg_table, rg_total in _iter_parquet_rowgroups(data_source_path):
            rg_df = pl.from_arrow(rg_table)

            tail_h = 0 if tail is None else int(tail.height)
            row_id_start = global_offset - tail_h

            df_in = pl.concat([tail, rg_df], how="vertical", rechunk=False) if tail_h > 0 else rg_df
            feats = _compute_features_df(df_in, wins, row_id_start=row_id_start)

            is_first = rg_idx == 0
            is_last = rg_idx == (rg_total - 1)

            start_trim = max_win if is_first else overlap
            end_trim = 0 if is_last else overlap

            out_len = feats.height - start_trim - end_trim
            feats_out = feats.slice(start_trim, out_len) if out_len > 0 else feats.head(0)

            if feats_out.height > 0:
                _write_part_parquet(feats_out, parts_dir, part_idx)
                total_rows_out += feats_out.height
                part_idx += 1

            tail = df_in.tail(overlap)
            global_offset += int(rg_df.height)
            pbar.update(1)

        pbar.close()

        if part_idx == 0:
            raise ValueError("No feature parts were produced.")

        _log(f"Parts: {part_idx} | rows: {total_rows_out}")
        _log("Stage 2/4: building unified parquet (temporary)")

        pl.scan_parquet(os.path.join(parts_dir, "*.parquet")).sink_parquet(
            tmp_features,
            compression=PARQUET_COMPRESSION,
            statistics=PARQUET_STATS,
        )

        _log("Stage 3/4: fit transform params on multi-block early slice")

        total_rows = pq.ParquetFile(tmp_features).metadata.num_rows
        fit_rows = int(max(1, min(total_rows, math.floor(total_rows * max(0.05, min(0.95, fit_fraction))))))

        rows_per_block = max(10_000, int(min(fit_rows, fit_sample_rows) / max(1, fit_blocks)))
        fit_df = _fit_blocks_sample(tmp_features, fit_rows, total_rows, fit_blocks, rows_per_block)

        corr_rows = int(min(fit_rows, corr_sample_rows))
        corr_df = _fit_blocks_sample(tmp_features, fit_rows, total_rows, max(2, min(fit_blocks, 5)), max(10_000, corr_rows // max(2, min(fit_blocks, 5))))

        clip_bounds = _compute_clip_bounds(fit_df, protect_cols, clip_q_low, clip_q_high)
        selected_cols = _prune_corr_group_aware(corr_df, threshold=corr_threshold, protect=protect_cols)

        lf = pl.scan_parquet(tmp_features)
        schema = lf.collect_schema()

        select_exprs = _select_with_clipping(selected_cols, schema, clip_bounds)

        qa_cols = [c for c in selected_cols if c not in protect_cols and c in fit_df.columns]
        nan_map = _nan_rate(fit_df, qa_cols)
        var_map = _variance(fit_df, qa_cols)

        in_abs = os.path.abspath(data_source_path)
        out_abs = os.path.abspath(out_parquet)

        input_stat = {
            "path": in_abs,
            "size_bytes": os.path.getsize(data_source_path),
            "mtime": os.path.getmtime(data_source_path),
            "sha256_head64mb": _sha256_file(data_source_path, 64 * 1024 * 1024),
        }

        _save_transform_params(
            transform_path,
            {
                "version": "1.2",
                "input": input_stat,
                "output_parquet": out_abs,
                "windows": wins,
                "fit_fraction": fit_fraction,
                "fit_rows": fit_rows,
                "fit_blocks": fit_blocks,
                "fit_rows_per_block": rows_per_block,
                "corr_threshold": corr_threshold,
                "clip_q_low": clip_q_low,
                "clip_q_high": clip_q_high,
                "clip_bounds": {k: [v[0], v[1]] for k, v in clip_bounds.items()},
                "selected_columns": selected_cols,
                "qa": {
                    "nan_rate": nan_map,
                    "variance": var_map,
                },
            },
        )

        _log(f"Transform params saved: {transform_path}")
        _log("Stage 4/4: apply clipping + select columns -> final parquet")

        lf.select(select_exprs).sink_parquet(
            out_parquet,
            compression=PARQUET_COMPRESSION,
            statistics=PARQUET_STATS,
        )

        _log(f"Selected columns: {len(selected_cols)} (threshold={corr_threshold})")
        _log(f"Output: {out_parquet} | size: {os.path.getsize(out_parquet)} bytes")
        _log("Feature Enginiering Finished.")
        print("")
        return out_parquet

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
