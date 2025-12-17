from __future__ import annotations

import os
import json
import math
import tempfile
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import polars as pl

from utils.basics import _log


@dataclass(frozen=True)
class TargetingParams:
    horizon_bars: int

    vol_window: int
    vol_mode: str
    vol_ret_clip: float

    atr_window: int
    atr_mode: str

    pt_mult: float
    sl_mult: float

    min_move_cost_mult: float
    min_move_buffer_bps: float

    cost_k_spread: float
    slippage_k: float
    slippage_pow_speed: float
    min_cost_bps: float
    max_cost_bps: float
    cost_roundtrip: bool

    min_vol: float
    ret_clip: float

    tie_break: str

    purge_mode: str
    purge_embargo_bars: int

    weight_cap: float


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


def _atomic_write_parquet(df: pl.DataFrame, path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=parent)
    os.close(fd)
    try:
        df.write_parquet(tmp, compression="zstd", statistics=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _require_cols(df: pl.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {df.columns}")


def _to_float(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float64, copy=False)


def _ewm_vol_past(r: np.ndarray, win: int) -> np.ndarray:
    n = r.size
    out = np.full(n, np.nan, dtype=np.float64)
    if win <= 1 or n == 0:
        return out
    alpha = 2.0 / (win + 1.0)
    v = 0.0
    warmed = False
    for i in range(n):
        ri = r[i]
        if not np.isfinite(ri):
            out[i] = np.nan
            continue
        if not warmed:
            v = ri * ri
            warmed = True
        else:
            v = alpha * (ri * ri) + (1.0 - alpha) * v
        out[i] = math.sqrt(v) if v > 0 else 0.0
    return out


def _rolling_std_past(r: np.ndarray, win: int) -> np.ndarray:
    n = r.size
    out = np.full(n, np.nan, dtype=np.float64)
    if win <= 1 or n == 0:
        return out
    xs = np.cumsum(np.insert(r, 0, 0.0))
    x2 = np.cumsum(np.insert(r * r, 0, 0.0))
    for i in range(win, n + 1):
        s = xs[i] - xs[i - win]
        s2 = x2[i] - x2[i - win]
        m = s / win
        v = (s2 / win) - m * m
        out[i - 1] = math.sqrt(v) if v > 0 else 0.0
    return out


def _vol_target(logret1: np.ndarray, params: TargetingParams) -> np.ndarray:
    r = logret1.copy()
    if params.vol_ret_clip and params.vol_ret_clip > 0:
        r = np.clip(r, -params.vol_ret_clip, params.vol_ret_clip)

    mode = params.vol_mode.lower()
    if mode == "ewm":
        vol = _ewm_vol_past(r, params.vol_window)
    elif mode == "rolling_std":
        vol = _rolling_std_past(r, params.vol_window)
    else:
        vol = _ewm_vol_past(r, params.vol_window)

    return np.where(np.isfinite(vol), vol, np.nan)


def _atr_past(high: np.ndarray, low: np.ndarray, close: np.ndarray, win: int, mode: str) -> np.ndarray:
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    prev = np.roll(close, 1)
    prev[0] = np.nan

    tr1 = high - low
    tr2 = np.abs(high - prev)
    tr3 = np.abs(low - prev)

    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.where(np.isfinite(tr) & (tr >= 0), tr, np.nan)

    win = int(win)
    if win <= 1:
        return tr

    m = mode.lower()
    if m == "ewm":
        alpha = 2.0 / (win + 1.0)
        v = np.nan
        for i in range(n):
            x = tr[i]
            if not np.isfinite(x):
                out[i] = np.nan
                continue
            if not np.isfinite(v):
                v = x
            else:
                v = alpha * x + (1.0 - alpha) * v
            out[i] = v
        return out

    xs = np.cumsum(np.insert(np.where(np.isfinite(tr), tr, 0.0), 0, 0.0))
    cnt = np.cumsum(np.insert(np.where(np.isfinite(tr), 1.0, 0.0), 0, 0.0))
    for i in range(win, n + 1):
        s = xs[i] - xs[i - win]
        c = cnt[i] - cnt[i - win]
        if c > 0:
            out[i - 1] = s / c
        else:
            out[i - 1] = np.nan
    return out


def _slippage_bps(
    spread: np.ndarray,
    ticks: np.ndarray,
    duration_ms: Optional[np.ndarray],
    entry_px: np.ndarray,
    k: float,
    pow_speed: float,
) -> np.ndarray:
    if duration_ms is None:
        speed = ticks
    else:
        denom = np.maximum(duration_ms, 1.0)
        speed = ticks / denom
    speed = np.maximum(speed, 0.0)

    entry = np.maximum(entry_px, 1e-12)
    spread_bps = 1e4 * (spread / entry)
    slip_bps = k * spread_bps * np.power(1.0 + speed, pow_speed)
    return np.maximum(slip_bps, 0.0)


def _apply_purge_nonoverlap(t1: np.ndarray, valid: np.ndarray, embargo_bars: int) -> np.ndarray:
    n = t1.size
    keep = valid.copy()
    last_end = -1
    emb = max(0, int(embargo_bars))
    for i in range(n):
        if not keep[i]:
            continue
        if i <= last_end + emb:
            keep[i] = False
            continue
        j = int(t1[i])
        if j <= i:
            keep[i] = False
            continue
        last_end = max(last_end, j)
    return keep


def _cost_bps_roundtrip(
    entry_px: np.ndarray,
    spread: np.ndarray,
    ticks: np.ndarray,
    duration_ms: Optional[np.ndarray],
    params: TargetingParams,
) -> np.ndarray:
    spread_bps = 1e4 * (spread / np.maximum(entry_px, 1e-12))
    slip_bps = _slippage_bps(spread, ticks, duration_ms, entry_px, params.slippage_k, params.slippage_pow_speed)
    one_way = params.cost_k_spread * spread_bps + slip_bps
    cbps = 2.0 * one_way if params.cost_roundtrip else one_way
    cbps = np.clip(cbps, params.min_cost_bps, params.max_cost_bps)
    return cbps


def _forward_return_net_timeout(
    entry_px: np.ndarray,
    exit_bid: np.ndarray,
    exit_ask: np.ndarray,
    cost_bps: np.ndarray,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = entry_px.size
    idx = np.arange(n, dtype=np.int64)
    t1 = np.minimum(idx + h, n - 1)

    ex_mid = (exit_bid[t1] + exit_ask[t1]) / 2.0

    valid = np.isfinite(entry_px) & (entry_px > 0) & np.isfinite(ex_mid) & (ex_mid > 0) & np.isfinite(cost_bps)
    gross = np.full(n, np.nan, dtype=np.float64)
    gross[valid] = (ex_mid[valid] - entry_px[valid]) / entry_px[valid]
    ret_net = np.full(n, np.nan, dtype=np.float64)
    ret_net[valid] = gross[valid] - (cost_bps[valid] / 1e4)
    return ret_net, t1


def _forward_up_down(ret_net: np.ndarray) -> np.ndarray:
    y = np.full(ret_net.size, -1, dtype=np.int8)
    ok = np.isfinite(ret_net)
    y[(ok) & (ret_net > 0)] = 1
    y[(ok) & (ret_net <= 0)] = 0
    return y


def _triple_barrier_two_sides_atr_costaware(
    entry_mid: np.ndarray,
    close: np.ndarray,
    fut_high: np.ndarray,
    fut_low: np.ndarray,
    exit_bid: np.ndarray,
    exit_ask: np.ndarray,
    atr_ret: np.ndarray,
    spread: np.ndarray,
    ticks: np.ndarray,
    duration_ms: Optional[np.ndarray],
    params: TargetingParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = entry_mid.size
    H = int(params.horizon_bars)

    t1 = np.full(n, -1, dtype=np.int64)
    touch = np.full(n, 0, dtype=np.int8)
    side = np.full(n, 0, dtype=np.int8)
    ret_net = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    cost_bps = np.full(n, np.nan, dtype=np.float64)

    tie = params.tie_break.lower()

    for i in range(n):
        entry = entry_mid[i]
        if not (np.isfinite(entry) and entry > 0):
            continue

        v = atr_ret[i]
        if not np.isfinite(v):
            continue
        v = max(v, params.min_vol)

        end = min(n - 1, i + H)
        if end <= i:
            continue

        cbps = _cost_bps_roundtrip(
            entry_px=np.array([entry], dtype=np.float64),
            spread=np.array([spread[i]], dtype=np.float64),
            ticks=np.array([ticks[i]], dtype=np.float64),
            duration_ms=None if duration_ms is None else np.array([duration_ms[i]], dtype=np.float64),
            params=params,
        )[0]
        cost_bps[i] = cbps

        pt = params.pt_mult * v
        sl = params.sl_mult * v

        pt_bps = 1e4 * pt
        min_required_bps = (params.min_move_cost_mult * cbps) + params.min_move_buffer_bps
        if not (np.isfinite(pt_bps) and pt_bps >= min_required_bps):
            continue

        cost_r = cbps / 1e4

        hit_j = -1
        hit_type = 0
        hit_side = 0

        for j in range(i + 1, end + 1):
            hi = fut_high[j]
            lo = fut_low[j]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                continue

            long_pt = ((hi / entry) - 1.0) >= pt
            long_sl = ((lo / entry) - 1.0) <= -sl

            short_pt = ((entry / lo) - 1.0) >= pt
            short_sl = ((hi / entry) - 1.0) >= sl

            long_hit = long_pt or long_sl
            short_hit = short_pt or short_sl

            if not (long_hit or short_hit):
                continue

            hit_j = j

            if long_hit and not short_hit:
                hit_side = 1
                hit_type = 1 if long_pt else -1
                break

            if short_hit and not long_hit:
                hit_side = -1
                hit_type = 1 if short_pt else -1
                break

            if tie == "best_case":
                hit_side = 1
                hit_type = 1
            elif tie == "close_based":
                cj = close[j]
                if not (np.isfinite(cj) and cj > 0):
                    hit_side = 0
                    hit_type = -1
                else:
                    hit_side = 1 if cj >= entry else -1
                    hit_type = 1
            else:
                hit_side = 0
                hit_type = -1
            break

        if hit_j == -1:
            hit_j = end
            hit_type = 0
            hit_side = 0

        ex_mid = (exit_bid[hit_j] + exit_ask[hit_j]) / 2.0
        if not (np.isfinite(ex_mid) and ex_mid > 0):
            continue

        t1[i] = hit_j
        touch[i] = hit_type
        side[i] = hit_side
        valid[i] = True

        if hit_side == 1:
            gross = (ex_mid - entry) / entry
        elif hit_side == -1:
            gross = (entry - ex_mid) / entry
        else:
            gross = abs(ex_mid - entry) / entry

        ret_net[i] = float(np.clip(gross - cost_r, -params.ret_clip, params.ret_clip))

    return t1, touch, side, ret_net, valid, cost_bps


def _uniqueness_weights(t1: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = t1.size
    diff = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        if not valid[i]:
            continue
        j = int(t1[i])
        if j <= i or j < 0:
            continue
        diff[i] += 1
        if j + 1 <= n:
            diff[j + 1] -= 1

    conc = np.cumsum(diff[:-1]).astype(np.float64)
    conc[conc <= 0] = 1.0
    inv = 1.0 / conc

    w = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not valid[i]:
            continue
        j = int(t1[i])
        if j <= i:
            continue
        w[i] = float(np.nanmean(inv[i : j + 1]))
    return w


def _build_labeled_df(df: pl.DataFrame, params: TargetingParams) -> pl.DataFrame:
    req = [
        "start_time", "end_time",
        "open", "high", "low", "close",
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
        "spread_mean", "ticks",
    ]
    _require_cols(df, req)

    d = df.with_columns(
        pl.col("start_time").cast(pl.Datetime("us", time_zone="UTC"), strict=False),
        pl.col("end_time").cast(pl.Datetime("us", time_zone="UTC"), strict=False),
    )

    if "duration_ms" in d.columns:
        duration_ms = _to_float(d["duration_ms"].to_numpy())
    else:
        dur = d.select(((pl.col("end_time") - pl.col("start_time")).dt.total_microseconds() / 1000.0).alias("_dur")).to_numpy()
        duration_ms = _to_float(dur[:, 0])

    close = _to_float(d["close"].to_numpy())
    high = _to_float(d["high"].to_numpy())
    low = _to_float(d["low"].to_numpy())

    ask_open = _to_float(d["ask_open"].to_numpy())
    bid_open = _to_float(d["bid_open"].to_numpy())
    bid_close = _to_float(d["bid_close"].to_numpy())
    ask_close = _to_float(d["ask_close"].to_numpy())

    spread = _to_float(d["spread_mean"].to_numpy())
    ticks = _to_float(d["ticks"].to_numpy())

    n = close.size
    if n < (params.horizon_bars + max(params.vol_window, params.atr_window) + 5):
        raise ValueError("Not enough rows in split for targeting params")

    logret1 = np.zeros(n, dtype=np.float64)
    logret1[1:] = np.log(close[1:]) - np.log(close[:-1])
    vol = _vol_target(logret1, params)

    atr = _atr_past(high, low, close, int(params.atr_window), str(params.atr_mode))
    entry_mid = np.full(n, np.nan, dtype=np.float64)
    entry_mid[:-1] = (bid_open[1:] + ask_open[1:]) / 2.0

    atr_ret = np.full(n, np.nan, dtype=np.float64)
    ok = np.isfinite(atr) & np.isfinite(entry_mid) & (entry_mid > 0)
    atr_ret[ok] = atr[ok] / entry_mid[ok]

    cost_bps_rt = _cost_bps_roundtrip(entry_mid, spread, ticks, duration_ms, params)

    y_ret_net_h, t1_timeout = _forward_return_net_timeout(
        entry_px=entry_mid,
        exit_bid=bid_close,
        exit_ask=ask_close,
        cost_bps=cost_bps_rt,
        h=int(params.horizon_bars),
    )

    y_up_h = _forward_up_down(y_ret_net_h)

    t1_tb, touch_tb, side_tb, y_ret_net_tb, valid_tb, cost_bps_tb = _triple_barrier_two_sides_atr_costaware(
        entry_mid=entry_mid,
        close=close,
        fut_high=high,
        fut_low=low,
        exit_bid=bid_close,
        exit_ask=ask_close,
        atr_ret=atr_ret,
        spread=spread,
        ticks=ticks,
        duration_ms=duration_ms,
        params=params,
    )

    last_valid = n - (params.horizon_bars + 1)
    if last_valid < 0:
        last_valid = 0

    valid_h = np.isfinite(y_ret_net_h) & (np.arange(n) <= last_valid) & np.isfinite(cost_bps_rt)
    valid_tb = valid_tb & (np.arange(n) <= last_valid) & np.isfinite(y_ret_net_tb) & np.isfinite(cost_bps_tb)

    if params.purge_mode.lower() == "nonoverlap":
        valid_h = _apply_purge_nonoverlap(t1_timeout, valid_h, params.purge_embargo_bars)
        valid_tb = _apply_purge_nonoverlap(t1_tb, valid_tb, params.purge_embargo_bars)

    uniq_h = _uniqueness_weights(t1_timeout, valid_h)
    uniq_tb = _uniqueness_weights(t1_tb, valid_tb)

    vol_eff = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
    atr_eff = np.where(np.isfinite(atr_ret) & (atr_ret > 0), atr_ret, np.nan)

    info_h = np.abs(y_ret_net_h) / (vol_eff + 1e-12)
    info_h = np.where(np.isfinite(info_h), info_h, 0.0)
    w_h = uniq_h * np.clip(info_h, 0.0, params.weight_cap)

    info_tb = np.abs(y_ret_net_tb) / (atr_eff + 1e-12)
    info_tb = np.where(np.isfinite(info_tb), info_tb, 0.0)
    w_tb = uniq_tb * np.clip(info_tb, 0.0, params.weight_cap)

    y_meta_tb = np.where(valid_tb & (y_ret_net_tb > 0.0), 1, 0).astype(np.int8)

    end_us = d.select(pl.col("end_time").dt.epoch("us").alias("_e")).to_numpy()[:, 0].astype(np.int64, copy=False)

    t1_time_us_h = np.full(n, -1, dtype=np.int64)
    idx_ok = np.where(valid_h & (t1_timeout >= 0))[0]
    for i in idx_ok:
        j = int(t1_timeout[i])
        if 0 <= j < n:
            t1_time_us_h[i] = end_us[j]

    t1_time_us_tb = np.full(n, -1, dtype=np.int64)
    idx_ok2 = np.where(valid_tb & (t1_tb >= 0))[0]
    for i in idx_ok2:
        j = int(t1_tb[i])
        if 0 <= j < n:
            t1_time_us_tb[i] = end_us[j]

    out = d.with_columns(
        pl.Series("vol_target", vol).cast(pl.Float64),
        pl.Series("atr", atr).cast(pl.Float64),
        pl.Series("atr_ret", atr_ret).cast(pl.Float64),

        pl.Series("t1_timeout_index", t1_timeout).cast(pl.Int64),
        pl.Series("t1_timeout_time_us", t1_time_us_h).cast(pl.Int64),
        pl.Series("y_ret_net_h", y_ret_net_h).cast(pl.Float64),
        pl.Series("y_up_h", y_up_h).cast(pl.Int8),
        pl.Series("cost_bps_h", cost_bps_rt).cast(pl.Float64),
        pl.Series("sample_weight_h", w_h).cast(pl.Float64),
        pl.Series("is_valid_label_h", valid_h).cast(pl.Boolean),

        pl.Series("t1_tb_index", t1_tb).cast(pl.Int64),
        pl.Series("t1_tb_time_us", t1_time_us_tb).cast(pl.Int64),
        pl.Series("barrier_touch_tb", touch_tb).cast(pl.Int8),
        pl.Series("side_tb", side_tb).cast(pl.Int8),
        pl.Series("y_ret_net_tb", y_ret_net_tb).cast(pl.Float64),
        pl.Series("y_meta_execute_tb", y_meta_tb).cast(pl.Int8),
        pl.Series("cost_bps_tb", cost_bps_tb).cast(pl.Float64),
        pl.Series("sample_weight_tb", w_tb).cast(pl.Float64),
        pl.Series("is_valid_label_tb", valid_tb).cast(pl.Boolean),
    ).with_columns(
        pl.when(pl.col("t1_timeout_time_us") >= 0)
        .then(pl.from_epoch(pl.col("t1_timeout_time_us"), time_unit="us").dt.replace_time_zone("UTC"))
        .otherwise(None)
        .alias("t1_timeout_time"),
        pl.when(pl.col("t1_tb_time_us") >= 0)
        .then(pl.from_epoch(pl.col("t1_tb_time_us"), time_unit="us").dt.replace_time_zone("UTC"))
        .otherwise(None)
        .alias("t1_tb_time"),
    ).drop(["t1_timeout_time_us", "t1_tb_time_us"])

    return out

def _stats(df: pl.DataFrame) -> dict:
    rows = int(df.height)
    if rows == 0:
        return {"rows": 0}

    out = {"rows": rows}

    if "is_valid_label_h" in df.columns:
        vh = df.filter(pl.col("is_valid_label_h") == True)
        out["H_valid_rows"] = int(vh.height)
        if vh.height:
            ret = vh["y_ret_net_h"]
            out["H_ret_mean"] = float(ret.mean())
            out["H_ret_std"] = float(ret.std())
            out["H_ret_p50"] = float(ret.quantile(0.50, "nearest"))
            out["H_hit_rate"] = float(vh.select((pl.col("y_ret_net_h") > 0).mean()).item() or 0.0)
            if "cost_bps_h" in vh.columns:
                cost = vh["cost_bps_h"]
                out["H_cost_bps_mean"] = float(cost.mean())
                out["H_cost_bps_p95"] = float(cost.quantile(0.95, "nearest"))
            if "sample_weight_h" in vh.columns:
                out["H_w_mean"] = float(vh["sample_weight_h"].mean())

    if "is_valid_label_tb" in df.columns:
        vtb = df.filter(pl.col("is_valid_label_tb") == True)
        out["TB_valid_rows"] = int(vtb.height)
        if vtb.height:
            ret = vtb["y_ret_net_tb"]
            out["TB_ret_mean"] = float(ret.mean())
            out["TB_ret_std"] = float(ret.std())
            out["TB_ret_p50"] = float(ret.quantile(0.50, "nearest"))
            out["TB_hit_rate"] = float(vtb.select((pl.col("y_ret_net_tb") > 0).mean()).item() or 0.0)
            if "cost_bps_tb" in vtb.columns:
                cost = vtb["cost_bps_tb"]
                out["TB_cost_bps_mean"] = float(cost.mean())
                out["TB_cost_bps_p95"] = float(cost.quantile(0.95, "nearest"))
            if "sample_weight_tb" in vtb.columns:
                out["TB_w_mean"] = float(vtb["sample_weight_tb"].mean())

            if "side_tb" in vtb.columns:
                out["TB_trades"] = int(vtb.filter(pl.col("side_tb") != 0).height)
                out["TB_trade_rate"] = float(out["TB_trades"] / max(1, rows))

            if "y_meta_execute_tb" in vtb.columns:
                out["TB_meta_execute_rate"] = float(vtb.select(pl.col("y_meta_execute_tb").mean()).item() or 0.0)

            if "barrier_touch_tb" in vtb.columns:
                touch = vtb["barrier_touch_tb"].to_list()
                tp = sum(1 for x in touch if x == 1)
                sl = sum(1 for x in touch if x == -1)
                to = sum(1 for x in touch if x == 0)
                out["TB_tp_rate"] = tp / int(vtb.height)
                out["TB_sl_rate"] = sl / int(vtb.height)
                out["TB_timeout_rate"] = to / int(vtb.height)

    return out


def targerting(targeting_config: dict):
    train_source_path = targeting_config.get("train_source_path", "sample_data/Dev/splits/train.parquet")
    val_source_path = targeting_config.get("val_source_path", "sample_data/Dev/splits/val.parquet")
    test_source_path = targeting_config.get("test_source_path", "sample_data/Dev/splits/test.parquet")
    output_path = targeting_config.get("output_path", "sample_data/Dev/targeted_data/")

    output_path = _ensure_dir(output_path)
    train_out = _final_parquet_path(os.path.join(output_path, "train_targeted.parquet"))
    val_out = _final_parquet_path(os.path.join(output_path, "val_targeted.parquet"))
    test_out = _final_parquet_path(os.path.join(output_path, "test_targeted.parquet"))
    params_out = os.path.join(output_path, "targeting_params.json")

    params = TargetingParams(
        horizon_bars=int(targeting_config.get("horizon_bars", 50)),

        vol_window=int(targeting_config.get("vol_window", 50)),
        vol_mode=str(targeting_config.get("vol_mode", "ewm")),
        vol_ret_clip=float(targeting_config.get("vol_ret_clip", 0.02)),

        atr_window=int(targeting_config.get("atr_window", 50)),
        atr_mode=str(targeting_config.get("atr_mode", "ewm")),

        pt_mult=float(targeting_config.get("pt_mult", 1.5)),
        sl_mult=float(targeting_config.get("sl_mult", 2.0)),

        min_move_cost_mult=float(targeting_config.get("min_move_cost_mult", 3.0)),
        min_move_buffer_bps=float(targeting_config.get("min_move_buffer_bps", 0.5)),

        cost_k_spread=float(targeting_config.get("cost_k_spread", 0.8)),
        slippage_k=float(targeting_config.get("slippage_k", 0.20)),
        slippage_pow_speed=float(targeting_config.get("slippage_pow_speed", 0.50)),
        min_cost_bps=float(targeting_config.get("min_cost_bps", 0.6)),
        max_cost_bps=float(targeting_config.get("max_cost_bps", 6.0)),
        cost_roundtrip=bool(targeting_config.get("cost_roundtrip", True)),

        min_vol=float(targeting_config.get("min_vol", 1e-6)),
        ret_clip=float(targeting_config.get("ret_clip", 0.20)),

        tie_break=str(targeting_config.get("tie_break", "close_based")),

        purge_mode=str(targeting_config.get("purge_mode", "none")),
        purge_embargo_bars=int(targeting_config.get("purge_embargo_bars", 0)),

        weight_cap=float(targeting_config.get("weight_cap", 50.0)),
    )

    if params.tie_break.lower() not in ("worst_case", "best_case", "close_based"):
        raise ValueError('tie_break must be "worst_case", "best_case", or "close_based"')
    if params.vol_mode.lower() not in ("ewm", "rolling_std"):
        raise ValueError('vol_mode must be "ewm" or "rolling_std"')
    if params.atr_mode.lower() not in ("ewm", "sma"):
        raise ValueError('atr_mode must be "ewm" or "sma"')
    if params.purge_mode.lower() not in ("none", "nonoverlap"):
        raise ValueError('purge_mode must be "none" or "nonoverlap"')
    if params.min_cost_bps < 0 or params.max_cost_bps <= 0 or params.max_cost_bps < params.min_cost_bps:
        raise ValueError("Invalid min_cost_bps/max_cost_bps")
    if params.horizon_bars <= 1:
        raise ValueError("horizon_bars must be > 1")
    if params.vol_window <= 1:
        raise ValueError("vol_window must be > 1")
    if params.atr_window <= 1:
        raise ValueError("atr_window must be > 1")
    if params.pt_mult <= 0 or params.sl_mult <= 0:
        raise ValueError("pt_mult and sl_mult must be > 0")
    if params.min_move_cost_mult < 0 or params.min_move_buffer_bps < 0:
        raise ValueError("min_move_cost_mult and min_move_buffer_bps must be >= 0")

    _log("Targeting: H forward net return + TB (two-sided) cost-aware ATR barriers + min-move filter")
    _log(f"Output dir: {output_path}")
    _log(
        f"Params: H={params.horizon_bars} ATR={params.atr_mode}/{params.atr_window} "
        f"pt={params.pt_mult} sl={params.sl_mult} "
        f"min_move={params.min_move_cost_mult}x_cost+{params.min_move_buffer_bps}bps "
        f"tie={params.tie_break} purge={params.purge_mode} roundtrip={params.cost_roundtrip}"
    )

    _log(f"Loading train: {train_source_path}")
    train_df = pl.read_parquet(train_source_path)
    _log(f"Computing targets: train rows={train_df.height}")
    train_t = _build_labeled_df(train_df, params)
    _atomic_write_parquet(train_t, train_out)
    train_stats = _stats(train_t)
    _log(f"Saved: {train_out}")

    _log(f"Loading val: {val_source_path}")
    val_df = pl.read_parquet(val_source_path)
    _log(f"Computing targets: val rows={val_df.height}")
    val_t = _build_labeled_df(val_df, params)
    _atomic_write_parquet(val_t, val_out)
    val_stats = _stats(val_t)
    _log(f"Saved: {val_out}")

    _log(f"Loading test: {test_source_path}")
    test_df = pl.read_parquet(test_source_path)
    _log(f"Computing targets: test rows={test_df.height}")
    test_t = _build_labeled_df(test_df, params)
    _atomic_write_parquet(test_t, test_out)
    test_stats = _stats(test_t)
    _log(f"Saved: {test_out}")

    payload = {
        "version": "4.0",
        "targeting_type": "forward_return_net_h + TB_two_sided_ATR_costaware_minmove",
        "inputs": {
            "train_source_path": os.path.abspath(train_source_path),
            "val_source_path": os.path.abspath(val_source_path),
            "test_source_path": os.path.abspath(test_source_path),
        },
        "outputs": {
            "train_targeted": os.path.abspath(train_out),
            "val_targeted": os.path.abspath(val_out),
            "test_targeted": os.path.abspath(test_out),
        },
        "params": {
            "horizon_bars": params.horizon_bars,
            "vol_window": params.vol_window,
            "vol_mode": params.vol_mode,
            "vol_ret_clip": params.vol_ret_clip,
            "atr_window": params.atr_window,
            "atr_mode": params.atr_mode,
            "pt_mult": params.pt_mult,
            "sl_mult": params.sl_mult,
            "min_move_cost_mult": params.min_move_cost_mult,
            "min_move_buffer_bps": params.min_move_buffer_bps,
            "tie_break": params.tie_break,
            "cost_k_spread": params.cost_k_spread,
            "slippage_k": params.slippage_k,
            "slippage_pow_speed": params.slippage_pow_speed,
            "min_cost_bps": params.min_cost_bps,
            "max_cost_bps": params.max_cost_bps,
            "cost_roundtrip": params.cost_roundtrip,
            "min_vol": params.min_vol,
            "ret_clip": params.ret_clip,
            "purge_mode": params.purge_mode,
            "purge_embargo_bars": params.purge_embargo_bars,
            "weight_cap": params.weight_cap,
            "entry_rule": "enter_next_bar_open(mid of bid/ask)",
            "exit_rule_H": "exit_at_t+H_bar_close(mid of bid/ask)",
            "exit_rule_TB": "exit_at_touch_or_timeout(t1) close(mid), tie_break configurable",
            "tb_barriers": "pt/sl distances = pt_mult/sl_mult * atr_ret",
            "min_move_filter": "require pt_bps >= min_move_cost_mult*cost_bps + min_move_buffer_bps",
            "cost_rule": "cost_bps(one_way)=cost_k_spread*spread_bps(entry)+slippage_bps(speed); roundtrip optional; clamped",
        },
        "stats": {"train": train_stats, "val": val_stats, "test": test_stats},
    }

    train_stats = _stats(train_t)
    _log(f"Train targets: H_valid={train_stats.get('H_valid_rows', 0)} TB_valid={train_stats.get('TB_valid_rows', 0)} TB_trades={train_stats.get('TB_trades', 0)}")
    val_stats = _stats(val_t)
    _log(f"Val targets: H_valid={val_stats.get('H_valid_rows', 0)} TB_valid={val_stats.get('TB_valid_rows', 0)} TB_trades={val_stats.get('TB_trades', 0)}")
    test_stats = _stats(test_t)
    _log(f"Test targets: H_valid={test_stats.get('H_valid_rows', 0)} TB_valid={test_stats.get('TB_valid_rows', 0)} TB_trades={test_stats.get('TB_trades', 0)}")

    _atomic_write_json(params_out, payload)
    _log("Targeting finished.")
    print("")
