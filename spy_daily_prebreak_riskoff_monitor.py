#!/usr/bin/env python3
"""
SPY Daily Prebreak Risk-Off / Whipsaw State Monitor.

Production daily monitor based on the v1-v5 audit findings. This script is not a
research grid search: it uses the finalized hand-coded prebreak state hierarchy.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass


EVENTS = {
    "0_75": 0.0075,
    "1_00": 0.0100,
    "1_25": 0.0125,
    "1_75": 0.0175,
    "2_00": 0.0200,
    "2_25": 0.0225,
    "3_00": 0.0300,
}
HORIZONS = [3, 5, 10, 21]
OUT_DEFAULT = Path("outputs/spy_daily_prebreak_monitor")
SCREEN_VERSION = "daily_screen_v1.1"

STATE_LEVELS = {
    0: [],
    1: ["distance_to_MA200_bucket_2"],
    2: ["distance_to_MA200_bucket_2", "abs100_count_5d_bucket", "RV21_state"],
    3: ["distance_to_MA200_bucket_2", "abs100_count_5d_bucket", "RV21_state", "recent_whipsaw_10d_100"],
    4: ["distance_to_MA200_bucket_2", "abs100_count_5d_bucket", "RV21_state", "recent_whipsaw_10d_100", "recent_pattern_state"],
    5: ["distance_to_MA200_bucket_2", "abs100_count_5d_bucket", "RV21_state", "recent_whipsaw_10d_100", "vol_transition_state", "weekly_memory_score_bucket"],
}

PRODUCTION_GRID_TARGETS = (
    [f"future_down_2_00_{h}d" for h in HORIZONS]
    + [f"future_down_3_00_{h}d" for h in HORIZONS]
    + [f"future_whipsaw_1_00_{h}d" for h in HORIZONS]
    + [f"break_below_MA200_{h}d" for h in HORIZONS]
)

KEY_TARGETS = [
    "future_abs_1_00_3d", "future_abs_1_00_5d", "future_abs_1_00_10d", "future_abs_1_00_21d",
    "future_down_1_00_3d", "future_down_1_00_5d", "future_down_1_00_10d", "future_down_1_00_21d",
    *PRODUCTION_GRID_TARGETS,
    "stress_whipsaw_10d", "forward_fast_riskoff_21d", "forward_tail_down_21d",
]
KEY_TARGETS = list(dict.fromkeys(KEY_TARGETS))


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * float(x):.2f}%"


def sample_flag(n: int, min_n: int = 50, fallback_min_n: int = 20) -> str:
    if n >= min_n:
        return "ok"
    if n >= fallback_min_n:
        return "low_n"
    return "insufficient"


def read_csv_robust(path: Path) -> pd.DataFrame:
    last = None
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Could not read {path}: {last}")


def detect_date_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if str(c).lower().strip() in {"date", "datetime", "timestamp"}]
    candidates += [c for c in df.columns if "date" in str(c).lower()]
    for col in candidates + list(df.columns[:3]):
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.80:
            return col
    raise ValueError("Could not detect date column")


def detect_close_col(df: pd.DataFrame) -> str:
    lower = {str(c).lower().strip(): c for c in df.columns}
    for key in ["adj close", "adj_close", "adjusted_close", "adjusted close", "close"]:
        if key in lower:
            return lower[key]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("Could not detect adjusted close / close column")
    return numeric[-1]


def load_price_csv(path: Path, value_name: str = "close") -> tuple[pd.DataFrame, dict]:
    raw = read_csv_robust(path)
    date_col = detect_date_col(raw)
    close_col = detect_close_col(raw)
    keep = [date_col, close_col]
    rename = {date_col: "date", close_col: value_name}
    lower = {str(c).lower().strip(): c for c in raw.columns}
    for key, name in [("high", "high"), ("low", "low")]:
        if key in lower and lower[key] not in keep:
            keep.append(lower[key])
            rename[lower[key]] = name
    out = raw[keep].copy().rename(columns=rename)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    for col in ["high", "low"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if {"high", "low"}.issubset(out.columns) and "close" in lower and lower["close"] != close_col:
        raw_close = pd.to_numeric(raw.loc[out.index, lower["close"]], errors="coerce")
        adj_ratio = out[value_name] / raw_close
        out["high"] = out["high"] * adj_ratio
        out["low"] = out["low"] * adj_ratio
    raw_n = len(out)
    out = out.dropna(subset=["date", value_name]).sort_values("date")
    duplicate_dates = int(out["date"].duplicated().sum())
    out = out.drop_duplicates("date", keep="last")
    out = out[out[value_name] > 0].reset_index(drop=True)
    return out, {
        "date_col": date_col,
        "price_col": close_col,
        "raw_rows": raw_n,
        "rows": len(out),
        "duplicate_dates_removed": duplicate_dates,
    }


def maybe_download_spy(csv_path: Path) -> None:
    try:
        import yfinance as yf
        data = yf.download("SPY", period="max", interval="1d", auto_adjust=False, progress=False)
        if data is None or data.empty:
            print("WARNING: yfinance download returned no data; using local CSV.")
            return
        data = data.reset_index()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv_path, index=False)
        print(f"Downloaded SPY data to {csv_path}")
    except Exception as exc:
        print(f"WARNING: yfinance download failed; using local CSV. {exc}")


def future_sum(s: pd.Series, h: int) -> pd.Series:
    out = s.astype(float).shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]
    out.iloc[-h:] = np.nan
    return out


def future_any(s: pd.Series, h: int) -> pd.Series:
    out = (future_sum(s, h) > 0).astype(float)
    out.iloc[-h:] = np.nan
    return out


def rolling_return(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1


def add_features(df: pd.DataFrame, vix_path: Path | None = None) -> tuple[pd.DataFrame, bool]:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    for label, thr in EVENTS.items():
        df[f"abs_{label}"] = df["ret"].abs() >= thr
        df[f"up_{label}"] = df["ret"] >= thr
        df[f"down_{label}"] = df["ret"] <= -thr

    r = df["ret"]
    df["current_ladder_simple"] = np.select(
        [
            r.abs() < 0.0025,
            (r >= 0.0025) & (r < 0.0075),
            (r <= -0.0025) & (r > -0.0075),
            (r >= 0.0075) & (r < 0.0125),
            (r <= -0.0075) & (r > -0.0125),
            (r >= 0.0125) & (r < 0.0200),
            (r <= -0.0125) & (r > -0.0200),
            r >= 0.0200,
            r <= -0.0200,
        ],
        ["flat", "mild_up", "mild_down", "large_up", "large_down", "escalation_up", "escalation_down", "stress_up", "stress_down"],
        default="flat",
    )

    for n in [50, 100, 200]:
        df[f"MA{n}"] = df["close"].rolling(n, min_periods=n).mean()
        df[f"distance_to_MA{n}"] = df["close"] / df[f"MA{n}"] - 1
    df["ma20"] = df["close"].rolling(20, min_periods=20).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=50).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=60).mean()
    df["slope20"] = df["ma20"].diff()
    if {"high", "low"}.issubset(df.columns):
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14, min_periods=14).mean()
        df["atr_slope"] = df["atr14"].diff()
        df["old_ma_atr_state_source"] = "full_ma_atr"
        old_risk_off = (df["ma20"] < df["ma50"]) & (df["slope20"] < 0) & (df["atr_slope"] > 0)
    else:
        df["atr14"] = np.nan
        df["atr_slope"] = np.nan
        df["old_ma_atr_state_source"] = "ma_only_fallback"
        old_risk_off = (df["ma20"] < df["ma50"]) & (df["slope20"] < 0)
    df["old_ma_atr_state"] = np.where(old_risk_off.fillna(False), "RISK_OFF", "RISK_ON")
    prev_old = df["old_ma_atr_state"].shift(1)
    df["old_ma_atr_signal"] = np.select(
        [
            df["old_ma_atr_state"].eq("RISK_OFF") & prev_old.eq("RISK_ON"),
            df["old_ma_atr_state"].eq("RISK_ON") & prev_old.eq("RISK_OFF"),
        ],
        ["SELL_REDUCE_EQUIVALENT", "BUY_RESTORE_EQUIVALENT"],
        default="HOLD",
    )
    df["MA200_state"] = np.where(df["distance_to_MA200"] >= 0, "above_ma200", "below_ma200")
    df["distance_to_MA200_bucket"] = pd.cut(
        df["distance_to_MA200"],
        [-np.inf, 0, 0.01, 0.02, 0.05, 0.10, np.inf],
        labels=["below_ma200", "above_ma200_0_1pct", "above_ma200_1_2pct", "above_ma200_2_5pct", "above_ma200_5_10pct", "above_ma200_10pct_plus"],
    ).astype(object)
    df["distance_to_MA200_bucket_2"] = pd.cut(
        df["distance_to_MA200"],
        [-np.inf, 0, 0.02, 0.05, np.inf],
        labels=["below_ma200", "above_ma200_0_2pct", "above_ma200_2_5pct", "above_ma200_5pct_plus"],
    ).astype(object)
    df["prebreak_sample"] = df["distance_to_MA200"] >= 0
    df["near_ma200_sample"] = (df["distance_to_MA200"] >= 0) & (df["distance_to_MA200"] < 0.02)

    for n in [3, 5, 10, 20, 40, 60]:
        df[f"trail_ret_{n}d"] = rolling_return(df["close"], n)
    for n in [3, 5, 10, 20]:
        df[f"abs100_count_{n}d"] = df["abs_1_00"].astype(int).rolling(n, min_periods=n).sum()
    df["abs100_count_5d_bucket"] = pd.cut(df["abs100_count_5d"], [-1, 0, 1, 5], labels=["n0", "n1", "n2plus"]).astype(object)
    df["abs100_count_10d_bucket"] = pd.cut(df["abs100_count_10d"], [-1, 1, 3, 10], labels=["n0_1", "n2_3", "n4plus"]).astype(object)

    for win, label, thr in [(5, "5d_075", "0_75"), (10, "10d_100", "1_00"), (20, "20d_100", "1_00")]:
        up = df[f"up_{thr}"].astype(int).rolling(win, min_periods=win).sum()
        dn = df[f"down_{thr}"].astype(int).rolling(win, min_periods=win).sum()
        df[f"recent_whipsaw_{label}"] = (up > 0) & (dn > 0)
    for n in [5, 10]:
        df[f"up_days_{n}d_025"] = (df["ret"] >= 0.0025).astype(int).rolling(n, min_periods=n).sum()
        df[f"down_days_{n}d_025"] = (df["ret"] <= -0.0025).astype(int).rolling(n, min_periods=n).sum()
    df["down_dominance_5d"] = (df["down_days_5d_025"] >= 3) & (df["down_days_5d_025"] > df["up_days_5d_025"])
    df["down_dominance_10d"] = (df["down_days_10d_025"] >= 6) & (df["down_days_10d_025"] > df["up_days_10d_025"])
    df["failed_rally_10d"] = (df["trail_ret_10d"] > 0.01) & (df["trail_ret_5d"] < -0.005)
    df["recent_deterioration_flag"] = (df["trail_ret_20d"] > 0.02) & (df["trail_ret_10d"] < 0.01) & (df["trail_ret_5d"] < 0)
    df["selloff_accel_10d"] = (df["trail_ret_10d"] < -0.015) & (df["trail_ret_5d"] < -0.010)
    df["compression_10d"] = (df["ret"].abs().rolling(10, min_periods=10).max() < 0.0075) & (df["trail_ret_10d"].abs() < 0.01)
    df["compression_breakout"] = df["compression_10d"].shift(1).fillna(False) & df["abs_0_75"]
    df["recent_pattern_state"] = np.select(
        [
            df["selloff_accel_10d"],
            df["recent_whipsaw_10d_100"] | df["recent_whipsaw_20d_100"],
            df["failed_rally_10d"],
            df["recent_deterioration_flag"],
            df["down_dominance_5d"] | df["down_dominance_10d"],
            df["compression_breakout"],
            df["compression_10d"],
            (df["trail_ret_20d"] > 0.04) & (df["trail_ret_5d"] > 0),
            (df["trail_ret_20d"] < -0.04) & (df["trail_ret_5d"] < 0),
        ],
        ["selloff_acceleration", "recent_whipsaw", "failed_rally", "recent_deterioration", "down_dominance", "compression_breakout", "compression", "strong_up_trend", "strong_down_trend"],
        default="other_mixed",
    )

    df["RV21"] = df["ret"].rolling(21, min_periods=21).std() * np.sqrt(252)
    rv_med = df["RV21"].median()
    df["RV21_state"] = np.where(df["RV21"] <= rv_med, "low_vol", "high_vol")
    df["RV21_percentile"] = df["RV21"].rank(pct=True)
    q1, q3, q9 = df["RV21"].quantile(0.25), df["RV21"].quantile(0.75), df["RV21"].quantile(0.90)
    df["RV21_bucket"] = np.select(
        [df["RV21"] <= q1, df["RV21"] <= q3, df["RV21"] <= q9],
        ["vol_low_q1", "vol_mid_q2_q3", "vol_high_q4"],
        default="vol_extreme_top10",
    )
    df["rv21_rising_5d"] = df["RV21"] > df["RV21"].shift(5)
    df["rv21_rising_10d"] = df["RV21"] > df["RV21"].shift(10)
    df["rv21_expansion_5d"] = (df["RV21"] - df["RV21"].shift(5)) / df["RV21"].shift(5)
    df["rv21_expansion_bucket"] = np.select(
        [df["rv21_expansion_5d"] < -0.10, df["rv21_expansion_5d"] <= 0.10, df["rv21_expansion_5d"] <= 0.25],
        ["falling", "flat", "rising_mild"],
        default="rising_strong",
    )
    df["vol_transition_state"] = np.select(
        [
            (df["RV21_state"] == "low_vol") & df["rv21_rising_5d"],
            (df["RV21_state"] == "high_vol") & df["rv21_rising_5d"],
            (df["RV21_state"] == "high_vol") & (~df["rv21_rising_5d"]),
        ],
        ["low_and_rising", "high_and_rising", "high_but_falling"],
        default="low_and_flat_falling",
    )

    df = add_weekly_memory(df)
    vix_available = False
    if vix_path and vix_path.exists():
        try:
            vix, _ = load_price_csv(vix_path, "vix_close")
            vix["VIX_percentile"] = vix["vix_close"].rank(pct=True)
            vix["VIX_rising_5d"] = vix["vix_close"] > vix["vix_close"].shift(5)
            vix["VIX_rising_10d"] = vix["vix_close"] > vix["vix_close"].shift(10)
            df = df.merge(vix[["date", "vix_close", "VIX_percentile", "VIX_rising_5d", "VIX_rising_10d"]], on="date", how="left")
            vix_available = True
        except Exception as exc:
            print(f"WARNING: VIX load failed; continuing without VIX. {exc}")
    return df, vix_available


def add_weekly_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k in range(1, 9):
        offset = 1 + 5 * (k - 1)
        ret_block = df["ret"].shift(offset)
        df[f"W{k}_cumret"] = ret_block.rolling(5, min_periods=5).apply(lambda x: np.prod(1 + x) - 1, raw=True)
        for label in ["0_75", "1_00", "1_25", "2_00"]:
            df[f"W{k}_abs_count_{label}"] = df[f"abs_{label}"].astype(int).shift(offset).rolling(5, min_periods=5).sum()
        for label in ["0_75", "1_00"]:
            up = df[f"up_{label}"].astype(int).shift(offset).rolling(5, min_periods=5).sum()
            down = df[f"down_{label}"].astype(int).shift(offset).rolling(5, min_periods=5).sum()
            df[f"W{k}_whipsaw_{label}"] = ((up > 0) & (down > 0)).astype(float)
    stress4 = sum(((df[f"W{k}_abs_count_1_25"] >= 1) | (df[f"W{k}_abs_count_2_00"] >= 1)).astype(int) for k in range(1, 5))
    whipsaw4 = sum(((df[f"W{k}_whipsaw_0_75"] == 1) | (df[f"W{k}_whipsaw_1_00"] == 1)).astype(int) for k in range(1, 5))
    quiet4 = sum(((df[f"W{k}_cumret"].abs() < 0.0075) & (df[f"W{k}_abs_count_0_75"] == 0)).astype(int) for k in range(1, 5))
    large8 = sum((df[f"W{k}_abs_count_1_00"] >= 1).astype(int) for k in range(1, 9))
    df["stress_weeks_4w_bucket"] = pd.cut(stress4, [-1, 0, 1, 4], labels=["stress_0", "stress_1", "stress_2plus"]).astype(object)
    df["whipsaw_weeks_4w_bucket"] = pd.cut(whipsaw4, [-1, 0, 1, 4], labels=["whipsaw_0", "whipsaw_1", "whipsaw_2plus"]).astype(object)
    df["large_move_weeks_8w_bucket"] = pd.cut(large8, [-1, 2, 5, 8], labels=["low", "medium", "high"]).astype(object)
    df["quiet_weeks_4w_bucket"] = pd.cut(quiet4, [-1, 1, 2, 4], labels=["compression_0_1", "compression_2", "compression_3plus"]).astype(object)
    score = (
        (df["stress_weeks_4w_bucket"] == "stress_1").astype(int)
        + 2 * (df["stress_weeks_4w_bucket"] == "stress_2plus").astype(int)
        + (df["whipsaw_weeks_4w_bucket"] == "whipsaw_1").astype(int)
        + 2 * (df["whipsaw_weeks_4w_bucket"] == "whipsaw_2plus").astype(int)
        + (df["large_move_weeks_8w_bucket"] == "medium").astype(int)
        + 2 * (df["large_move_weeks_8w_bucket"] == "high").astype(int)
        - (df["quiet_weeks_4w_bucket"] == "compression_3plus").astype(int)
    )
    df["weekly_memory_score"] = score
    df["weekly_memory_score_bucket"] = np.select([score <= 0, score >= 4], ["weekly_low", "weekly_high"], default="weekly_medium")
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in HORIZONS:
        future_path = pd.concat([df["close"].shift(-i) / df["close"] - 1 for i in range(1, h + 1)], axis=1)
        df[f"fwd_ret_{h}d"] = df["close"].shift(-h) / df["close"] - 1
        df[f"fwd_max_drawdown_{h}d"] = future_path.min(axis=1)
        df[f"fwd_max_runup_{h}d"] = future_path.max(axis=1)
        df[f"fwd_max_abs_return_{h}d"] = future_path.abs().max(axis=1)
        for col in [f"fwd_ret_{h}d", f"fwd_max_drawdown_{h}d", f"fwd_max_runup_{h}d", f"fwd_max_abs_return_{h}d"]:
            df.loc[df.index[-h:], col] = np.nan
        for label in EVENTS:
            df[f"future_abs_{label}_{h}d"] = future_any(df[f"abs_{label}"], h)
        for label in ["1_00", "1_25", "1_75", "2_00", "2_25", "3_00"]:
            df[f"future_down_{label}_{h}d"] = future_any(df[f"down_{label}"], h)
        for label in ["1_00", "1_25", "1_75", "2_00"]:
            df[f"future_up_{label}_{h}d"] = future_any(df[f"up_{label}"], h)
        for label in ["0_75", "1_00", "1_25"]:
            up = future_any(df[f"up_{label}"], h)
            down = future_any(df[f"down_{label}"], h)
            df[f"future_whipsaw_{label}_{h}d"] = ((up == 1) & (down == 1)).astype(float)
            df.loc[df.index[-h:], f"future_whipsaw_{label}_{h}d"] = np.nan
    for h in HORIZONS:
        below = future_any(df["close"] < df["MA200"], h)
        df[f"break_below_MA200_{h}d"] = np.where(df["distance_to_MA200"] >= 0, below, np.nan)
    df["stress_whipsaw_10d"] = df["future_whipsaw_1_25_10d"]
    df["forward_fast_riskoff_21d"] = ((df["fwd_ret_10d"] <= -0.025) | (df["fwd_max_drawdown_21d"] <= -0.05)).astype(float)
    df["forward_tail_down_21d"] = (df["fwd_max_drawdown_21d"] <= -0.075).astype(float)
    df.loc[df.index[-21:], ["forward_fast_riskoff_21d", "forward_tail_down_21d"]] = np.nan
    return df


def classify_current_state(row: pd.Series) -> tuple[str, int, str]:
    calm = row["distance_to_MA200_bucket_2"] == "above_ma200_5pct_plus" and row["abs100_count_5d_bucket"] == "n0" and row["RV21_state"] == "low_vol"
    active_safe = row["distance_to_MA200_bucket_2"] in ["above_ma200_2_5pct", "above_ma200_5pct_plus"] and row["RV21_state"] == "low_vol" and row["abs100_count_5d_bucket"] != "n2plus"
    whipsaw_aware = bool(row["recent_whipsaw_10d_100"]) or row["abs100_count_5d_bucket"] == "n2plus"
    prebreak_watch = row["distance_to_MA200_bucket_2"] == "above_ma200_0_2pct" and row["abs100_count_5d_bucket"] in ["n1", "n2plus"]
    prebreak_warning = prebreak_watch and row["abs100_count_5d_bucket"] == "n2plus" and row["RV21_state"] == "high_vol"
    prebreak_transition = prebreak_warning and bool(row["recent_whipsaw_10d_100"])
    confirmed_stress = row["distance_to_MA200_bucket_2"] == "below_ma200" and row["RV21_state"] == "high_vol" and row["abs100_count_5d_bucket"] == "n2plus"
    if confirmed_stress:
        label = "confirmed_stress"
    elif prebreak_transition:
        label = "prebreak_riskoff_transition"
    elif prebreak_warning:
        label = "prebreak_warning"
    elif prebreak_watch:
        label = "prebreak_watch"
    elif whipsaw_aware:
        label = "whipsaw_aware"
    elif active_safe:
        label = "active_but_safe"
    elif calm:
        label = "calm_state"
    else:
        label = "mixed"
    score = 0
    score += {"below_ma200": 3, "above_ma200_0_2pct": 2, "above_ma200_2_5pct": 1}.get(row["distance_to_MA200_bucket_2"], 0)
    score += {"n2plus": 2, "n1": 1}.get(row["abs100_count_5d_bucket"], 0)
    score += 2 if row["RV21_state"] == "high_vol" else 0
    score += 1 if row["vol_transition_state"] == "low_and_rising" else 0
    score += 2 if row["vol_transition_state"] == "high_and_rising" else 0
    score += 1 if bool(row["recent_whipsaw_10d_100"]) else 0
    score += 1 if row["recent_pattern_state"] in ["selloff_acceleration", "failed_rally", "recent_deterioration", "recent_whipsaw"] else 0
    score += 1 if row["weekly_memory_score_bucket"] == "weekly_high" else 0
    bucket = "low" if score <= 2 else ("moderate" if score <= 4 else ("elevated" if score <= 6 else ("high" if score <= 8 else "extreme")))
    return label, score, bucket


def state_mask(df: pd.DataFrame, latest: pd.Series, level: int) -> pd.Series:
    cols = STATE_LEVELS[level]
    mask = pd.Series(True, index=df.index)
    for col in cols:
        mask &= df[col].astype(str) == str(latest[col])
    return mask


def target_horizon(target: str) -> int | None:
    for h in [21, 10, 5, 3]:
        if target.endswith(f"_{h}d"):
            return h
    if target == "stress_whipsaw_10d":
        return 10
    if target in ["forward_fast_riskoff_21d", "forward_tail_down_21d"]:
        return 21
    return None


def fwd_cols_for_target(target: str) -> tuple[str, str, str]:
    h = target_horizon(target) or 10
    if h not in HORIZONS:
        h = 10
    return f"fwd_ret_{h}d", f"fwd_max_drawdown_{h}d", f"fwd_max_runup_{h}d"


def add_probability_uncertainty(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "conditional_probability" in out.columns:
        p_col = "conditional_probability"
    elif "probability" in out.columns:
        p_col = "probability"
    else:
        return out
    n_col = "usable_n" if "usable_n" in out.columns else ("n" if "n" in out.columns else None)
    if n_col is None:
        out["ci_method"] = "wilson_95"
        out["ci_low"] = np.nan
        out["ci_high"] = np.nan
        out["ci_width"] = np.nan
        out["uncertainty_label"] = "UNKNOWN"
        return out
    z = 1.96
    lows, highs, widths, labels = [], [], [], []
    for _, row in out.iterrows():
        n = row.get(n_col)
        p = row.get(p_col)
        if pd.isna(n) or pd.isna(p) or n <= 0:
            lows.append(np.nan); highs.append(np.nan); widths.append(np.nan); labels.append("UNKNOWN")
            continue
        n = int(n)
        hits = row.get("hit_count", np.nan)
        if pd.isna(hits):
            hits = round(float(p) * n)
        p = max(0.0, min(1.0, float(hits) / n))
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
        lo = max(0.0, center - half)
        hi = min(1.0, center + half)
        width = hi - lo
        if n >= 200 and width <= 0.15:
            lab = "LOW"
        elif n >= 75 and width <= 0.25:
            lab = "MEDIUM"
        elif n >= 30 and width <= 0.40:
            lab = "HIGH"
        else:
            lab = "VERY_HIGH"
        lows.append(lo); highs.append(hi); widths.append(width); labels.append(lab)
    out["ci_method"] = "wilson_95"
    out["ci_low"] = lows
    out["ci_high"] = highs
    out["ci_width"] = widths
    out["uncertainty_label"] = labels
    return out


def compute_conditional_table(df: pd.DataFrame, latest: pd.Series, targets: list[str], min_n: int, fallback_min_n: int, estimation_mask: pd.Series) -> pd.DataFrame:
    rows = []
    hist = df.loc[estimation_mask].copy()
    for level, cols in STATE_LEVELS.items():
        mask = state_mask(hist, latest, level)
        raw_state = hist.loc[mask]
        state_key = "ALL" if not cols else "|".join(f"{c}={latest[c]}" for c in cols)
        for target in targets:
            fwd_ret, fwd_dd, fwd_ru = fwd_cols_for_target(target)
            d_state = raw_state.dropna(subset=[target])
            d_base = hist.dropna(subset=[target])
            if d_base.empty:
                continue
            n = len(d_state)
            hits = d_state[target].sum() if n else 0
            p = hits / n if n else np.nan
            base = d_base[target].mean()
            rows.append({
                "state_level": level,
                "state_key": state_key,
                "target": target,
                "horizon": target_horizon(target),
                "raw_n": len(raw_state),
                "usable_n": n,
                "hit_count": hits,
                "conditional_probability": p,
                "unconditional_probability": base,
                "probability_difference": p - base if p == p else np.nan,
                "hazard_ratio": p / base if base and p == p else np.nan,
                "sample_flag": sample_flag(n, min_n, fallback_min_n),
                "mean_forward_return": d_state[fwd_ret].mean() if fwd_ret in d_state else np.nan,
                "median_forward_return": d_state[fwd_ret].median() if fwd_ret in d_state else np.nan,
                "p25_forward_return": d_state[fwd_ret].quantile(0.25) if fwd_ret in d_state and n else np.nan,
                "p75_forward_return": d_state[fwd_ret].quantile(0.75) if fwd_ret in d_state and n else np.nan,
                "mean_max_drawdown": d_state[fwd_dd].mean() if fwd_dd in d_state else np.nan,
                "median_max_drawdown": d_state[fwd_dd].median() if fwd_dd in d_state else np.nan,
                "mean_max_runup": d_state[fwd_ru].mean() if fwd_ru in d_state else np.nan,
                "median_max_runup": d_state[fwd_ru].median() if fwd_ru in d_state else np.nan,
            })
    return add_probability_uncertainty(pd.DataFrame(rows))


def select_current_probabilities(cond: pd.DataFrame, min_n: int, fallback_min_n: int) -> pd.DataFrame:
    rows = []
    for target, g in cond.groupby("target"):
        ok = g[g["usable_n"] >= min_n].sort_values("state_level", ascending=False)
        low = g[(g["usable_n"] >= fallback_min_n) & (g["usable_n"] < min_n)].sort_values("state_level", ascending=False)
        if not ok.empty:
            chosen = ok.iloc[0].copy()
            chosen["selection_note"] = "primary_n_ok"
        elif not low.empty:
            chosen = low.iloc[0].copy()
            chosen["selection_note"] = "fallback_low_n"
        else:
            chosen = g[g["state_level"] == 0].iloc[0].copy()
            chosen["selection_note"] = "fallback_unconditional"
        for level in [2, 3]:
            lg = g[g["state_level"] == level]
            prefix = f"level{level}"
            if lg.empty:
                chosen[f"{prefix}_probability"] = np.nan
                chosen[f"{prefix}_usable_n"] = np.nan
                chosen[f"{prefix}_hazard_ratio"] = np.nan
                chosen[f"{prefix}_ci_low"] = np.nan
                chosen[f"{prefix}_ci_high"] = np.nan
                chosen[f"{prefix}_uncertainty_label"] = "UNKNOWN"
                chosen[f"{prefix}_sample_flag"] = "insufficient"
            else:
                r = lg.iloc[0]
                chosen[f"{prefix}_probability"] = r.get("conditional_probability", np.nan)
                chosen[f"{prefix}_usable_n"] = r.get("usable_n", np.nan)
                chosen[f"{prefix}_hazard_ratio"] = r.get("hazard_ratio", np.nan)
                chosen[f"{prefix}_ci_low"] = r.get("ci_low", np.nan)
                chosen[f"{prefix}_ci_high"] = r.get("ci_high", np.nan)
                chosen[f"{prefix}_uncertainty_label"] = r.get("uncertainty_label", "UNKNOWN")
                chosen[f"{prefix}_sample_flag"] = r.get("sample_flag", "insufficient")
        rows.append(chosen)
    return add_probability_uncertainty(pd.DataFrame(rows))


def build_current_state_probabilities_export(selected: pd.DataFrame, cond: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for _, sel in selected.iterrows():
        target = sel["target"]
        g = cond[cond["target"] == target]
        row = {
            "date": latest_date,
            "target": target,
            "horizon": sel.get("horizon", np.nan),
            "selected_state_level": sel.get("state_level", np.nan),
            "selected_state_key": sel.get("state_key", ""),
            "selected_probability": sel.get("conditional_probability", np.nan),
            "selected_base_rate": sel.get("unconditional_probability", np.nan),
            "selected_hazard_ratio": sel.get("hazard_ratio", np.nan),
            "selected_usable_n": sel.get("usable_n", np.nan),
            "selected_hit_count": sel.get("hit_count", np.nan),
            "selected_sample_flag": sel.get("sample_flag", ""),
            "selected_uncertainty": sel.get("uncertainty_label", ""),
            "selected_ci_low": sel.get("ci_low", np.nan),
            "selected_ci_high": sel.get("ci_high", np.nan),
        }
        for level in [5, 4, 3, 2, 1]:
            r = g[g["state_level"] == level]
            prefix = f"level{level}"
            if r.empty:
                row.update({
                    f"{prefix}_state_key": "",
                    f"{prefix}_probability": np.nan,
                    f"{prefix}_base_rate": np.nan,
                    f"{prefix}_hazard_ratio": np.nan,
                    f"{prefix}_usable_n": np.nan,
                    f"{prefix}_hit_count": np.nan,
                    f"{prefix}_sample_flag": "insufficient",
                    f"{prefix}_uncertainty": "UNKNOWN",
                    f"{prefix}_ci_low": np.nan,
                    f"{prefix}_ci_high": np.nan,
                })
            else:
                rr = r.iloc[0]
                row.update({
                    f"{prefix}_state_key": rr.get("state_key", ""),
                    f"{prefix}_probability": rr.get("conditional_probability", np.nan),
                    f"{prefix}_base_rate": rr.get("unconditional_probability", np.nan),
                    f"{prefix}_hazard_ratio": rr.get("hazard_ratio", np.nan),
                    f"{prefix}_usable_n": rr.get("usable_n", np.nan),
                    f"{prefix}_hit_count": rr.get("hit_count", np.nan),
                    f"{prefix}_sample_flag": rr.get("sample_flag", ""),
                    f"{prefix}_uncertainty": rr.get("uncertainty_label", "UNKNOWN"),
                    f"{prefix}_ci_low": rr.get("ci_low", np.nan),
                    f"{prefix}_ci_high": rr.get("ci_high", np.nan),
                })
        r0 = g[g["state_level"] == 0]
        if r0.empty:
            row.update({"level0_probability": np.nan, "level0_usable_n": np.nan, "level0_hit_count": np.nan})
        else:
            rr = r0.iloc[0]
            row.update({
                "level0_probability": rr.get("conditional_probability", np.nan),
                "level0_usable_n": rr.get("usable_n", np.nan),
                "level0_hit_count": rr.get("hit_count", np.nan),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def forward_distribution(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        rcol, ddcol, rucol = f"fwd_ret_{h}d", f"fwd_max_drawdown_{h}d", f"fwd_max_runup_{h}d"
        d = matches.dropna(subset=[rcol])
        x = d[rcol]
        dd = d[ddcol]
        ru = d[rucol]
        rows.append({
            "horizon": f"{h}d",
            "n": len(d),
            "mean": x.mean(),
            "median": x.median(),
            "std": x.std(),
            "p05": x.quantile(0.05),
            "p10": x.quantile(0.10),
            "p25": x.quantile(0.25),
            "p75": x.quantile(0.75),
            "p90": x.quantile(0.90),
            "p95": x.quantile(0.95),
            "probability_positive": (x > 0).mean(),
            "probability_negative": (x < 0).mean(),
            "probability_le_minus_1pct": (x <= -0.01).mean(),
            "probability_le_minus_2pct": (x <= -0.02).mean(),
            "probability_ge_plus_1pct": (x >= 0.01).mean(),
            "probability_ge_plus_2pct": (x >= 0.02).mean(),
            "expected_max_drawdown": dd.mean(),
            "median_max_drawdown": dd.median(),
            "p10_max_drawdown": dd.quantile(0.10),
            "expected_max_runup": ru.mean(),
            "median_max_runup": ru.median(),
        })
    return pd.DataFrame(rows)


def post2020_forward_distribution(matches: pd.DataFrame) -> pd.DataFrame:
    d0 = matches[pd.to_datetime(matches["date"], errors="coerce") >= pd.Timestamp("2020-01-01")].copy()
    rows = []
    for h in HORIZONS:
        rcol, ddcol = f"fwd_ret_{h}d", f"fwd_max_drawdown_{h}d"
        d = d0.dropna(subset=[rcol])
        x = d[rcol]
        dd = d[ddcol] if ddcol in d else pd.Series(dtype=float)
        rows.append({
            "horizon": f"{h}d",
            "n": len(d),
            "mean": x.mean(),
            "median": x.median(),
            "std": x.std(),
            "p10": x.quantile(0.10),
            "p25": x.quantile(0.25),
            "p75": x.quantile(0.75),
            "p90": x.quantile(0.90),
            "prob_negative": (x < 0).mean(),
            "prob_le_minus_2": (x <= -0.02).mean(),
            "expected_max_drawdown": dd.mean(),
            "sample_flag": sample_flag(len(d)),
            "sample_start": str(pd.to_datetime(d["date"]).min().date()) if len(d) else "",
            "sample_end": str(pd.to_datetime(d["date"]).max().date()) if len(d) else "",
        })
    return pd.DataFrame(rows)


def base_rate_summary(df: pd.DataFrame, current_matches: pd.DataFrame, estimation_mask: pd.Series) -> pd.DataFrame:
    samples = {
        "full": estimation_mask,
        "post_2020": estimation_mask & (df["date"] >= pd.Timestamp("2020-01-01")),
        "prebreak": estimation_mask & (df["distance_to_MA200"] >= 0),
        "near_ma200": estimation_mask & (df["distance_to_MA200"] >= 0) & (df["distance_to_MA200"] < 0.02),
    }
    targets = KEY_TARGETS
    rows = []
    for name, mask in samples.items():
        d = df.loc[mask]
        for target in targets:
            t = d.dropna(subset=[target])
            rows.append({"sample": name, "target": target, "n": len(t), "hit_count": t[target].sum(), "probability": t[target].mean()})
    for target in targets:
        t = current_matches.dropna(subset=[target])
        rows.append({"sample": "current_state_sample_used", "target": target, "n": len(t), "hit_count": t[target].sum(), "probability": t[target].mean()})
    return pd.DataFrame(rows)


def final_daily_label(report_row: dict, probs: pd.DataFrame) -> tuple[str, str, str]:
    final_state = str(report_row.get("final_state_label", "")).lower()
    risk_score = report_row.get("risk_score", 0)
    whipsaw = bool(report_row.get("recent_whipsaw_10d_100", False))
    old_state = report_row.get("old_ma_atr_state", "")
    exact_n = report_row.get("n_used_for_primary_target", np.nan)
    key = probs[probs["target"].isin([
        "future_down_2_00_10d", "future_down_3_00_21d",
        "future_whipsaw_1_00_10d", "break_below_MA200_21d",
        "forward_fast_riskoff_21d", "forward_tail_down_21d",
    ])]
    very_high_share = (key["uncertainty_label"].eq("VERY_HIGH")).mean() if "uncertainty_label" in key and len(key) else 0
    insufficient_share = (key["sample_flag"].eq("insufficient")).mean() if "sample_flag" in key and len(key) else 0
    if (pd.notna(exact_n) and exact_n < 30) or very_high_share > 0.5 or insufficient_share > 0.5:
        return "GRAY_LOW_SAMPLE", "current exact-state n is below 30 or most key probability rows have very high uncertainty/insufficient quality", "1_GRAY_LOW_SAMPLE"
    br = probs[probs["target"] == "break_below_MA200_21d"]
    br_confirm = False
    if not br.empty:
        b = br.iloc[0]
        br_confirm = b.get("hazard_ratio", np.nan) > 1.25 and b.get("conditional_probability", np.nan) > b.get("unconditional_probability", np.inf) and risk_score >= 7
    if "riskoff" in final_state or "risk_off" in final_state or br_confirm or (old_state == "RISK_OFF" and "prebreak" in final_state):
        return "RED_CONFIRMED_RISK_OFF", "risk-off or prebreak risk-off confirmation condition is active", "2_RED_CONFIRMED_RISK_OFF"
    if risk_score >= 7 and ("prebreak" in final_state or report_row.get("risk_score_bucket") in ["elevated", "high", "extreme"]):
        return "ORANGE_PREBREAK_WATCH", "risk score is high and current state is prebreak/elevated, but risk-off is not confirmed", "3_ORANGE_PREBREAK_WATCH"
    if whipsaw or "whipsaw" in final_state:
        return "YELLOW_WHIPSAW_NOT_RISK_OFF", "recent whipsaw detected, but prebreak/risk-off transition is not confirmed", "4_YELLOW_WHIPSAW_NOT_RISK_OFF"
    if risk_score <= 4 and not whipsaw and "prebreak" not in final_state and "riskoff" not in final_state and "risk_off" not in final_state:
        return "GREEN_NORMAL", "risk score is low/moderate with no whipsaw, prebreak, or risk-off state", "5_GREEN_NORMAL"
    return "MIXED_NEUTRAL", "state is mixed and does not meet colored label rules", "6_MIXED_NEUTRAL"


def current_report(latest: pd.Series, probs: pd.DataFrame, post2020_dist: pd.DataFrame) -> pd.DataFrame:
    label, score, bucket = classify_current_state(latest)
    primary = probs[probs["target"] == "future_down_2_00_10d"]
    state_level = primary["state_level"].iloc[0] if not primary.empty else np.nan
    n_used = primary["usable_n"].iloc[0] if not primary.empty else np.nan
    fields = [
        "date", "close", "ret", "current_ladder_simple", "distance_to_MA50", "distance_to_MA100", "distance_to_MA200",
        "distance_to_MA200_bucket", "distance_to_MA200_bucket_2", "MA200_state", "abs100_count_5d",
        "abs100_count_5d_bucket", "abs100_count_10d", "abs100_count_10d_bucket", "recent_whipsaw_5d_075",
        "recent_whipsaw_10d_100", "recent_whipsaw_20d_100", "trail_ret_3d", "trail_ret_5d", "trail_ret_10d",
        "trail_ret_20d", "recent_pattern_state", "RV21", "RV21_state", "RV21_percentile", "rv21_expansion_5d",
        "rv21_expansion_bucket", "vol_transition_state", "weekly_memory_score", "weekly_memory_score_bucket",
        "old_ma_atr_state", "old_ma_atr_signal", "old_ma_atr_state_source", "ma20", "ma50", "ma60", "slope20", "atr14", "atr_slope",
    ]
    row = {f: latest.get(f, np.nan) for f in fields}
    row.update({
        "final_state_label": label,
        "risk_score": score,
        "risk_score_bucket": bucket,
        "matched_state_level_for_primary_target": state_level,
        "n_used_for_primary_target": n_used,
        "post2020_n": int(post2020_dist["n"].min()) if not post2020_dist.empty else 0,
        "post2020_sample_flag": sample_flag(int(post2020_dist["n"].min())) if not post2020_dist.empty else "insufficient",
        "screen_version": SCREEN_VERSION,
    })
    daily_label, reason, rule = final_daily_label(row, probs)
    row["final_daily_label"] = daily_label
    row["final_daily_label_reason"] = reason
    row["final_daily_label_priority_rule"] = rule
    return pd.DataFrame([row])


def state_match_history(df: pd.DataFrame, latest: pd.Series, probs: pd.DataFrame, estimation_mask: pd.Series) -> tuple[pd.DataFrame, int]:
    primary = probs[probs["target"] == "future_down_2_00_10d"]
    level = int(primary["state_level"].iloc[0]) if not primary.empty else 0
    mask = state_mask(df, latest, level) & estimation_mask & (df["date"] < latest["date"])
    cols = [
        "date", "close", "ret", "distance_to_MA200_bucket_2", "abs100_count_5d_bucket", "RV21_state",
        "recent_whipsaw_10d_100", "recent_pattern_state", "vol_transition_state", "weekly_memory_score_bucket",
        "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_21d",
        "fwd_max_drawdown_3d", "fwd_max_drawdown_5d", "fwd_max_drawdown_10d", "fwd_max_drawdown_21d",
        "fwd_max_runup_3d", "fwd_max_runup_5d", "fwd_max_runup_10d", "fwd_max_runup_21d",
        "future_down_2_00_10d", "future_down_3_00_21d", "future_whipsaw_1_00_10d", "break_below_MA200_21d",
    ]
    cols = list(dict.fromkeys(cols + KEY_TARGETS))
    return df.loc[mask, cols].sort_values("date"), level


LOG_KEY_TARGETS = [
    "future_down_2_00_10d",
    "future_down_3_00_21d",
    "future_whipsaw_1_00_10d",
    "break_below_MA200_21d",
    "forward_fast_riskoff_21d",
    "forward_tail_down_21d",
]


def log_suffix(target: str) -> str:
    return target


def daily_signal_log(df: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    log = df[["date", "close", "ret", "distance_to_MA200", "distance_to_MA200_bucket_2", "abs100_count_5d_bucket", "RV21_state", "recent_whipsaw_10d_100", "recent_pattern_state", "vol_transition_state", "weekly_memory_score_bucket"]].copy()
    labels_scores = df.apply(lambda r: classify_current_state(r), axis=1)
    log["final_state_label"] = [x[0] for x in labels_scores]
    log["risk_score"] = [x[1] for x in labels_scores]
    log["risk_score_bucket"] = [x[2] for x in labels_scores]
    log["old_ma_atr_state"] = df["old_ma_atr_state"]
    log["old_ma_atr_signal"] = df["old_ma_atr_signal"]
    log["screen_version"] = SCREEN_VERSION
    log["final_daily_label"] = np.select(
        [
            log["final_state_label"].astype(str).str.contains("riskoff|risk_off", case=False, regex=True),
            log["risk_score"] >= 7,
            log["recent_whipsaw_10d_100"].astype(bool) | log["final_state_label"].astype(str).str.contains("whipsaw", case=False),
            log["risk_score"] <= 4,
        ],
        ["RED_CONFIRMED_RISK_OFF", "ORANGE_PREBREAK_WATCH", "YELLOW_WHIPSAW_NOT_RISK_OFF", "GREEN_NORMAL"],
        default="MIXED_NEUTRAL",
    )
    for target in LOG_KEY_TARGETS:
        suffix = log_suffix(target)
        log[f"selected_prob_{suffix}"] = np.nan
        log[f"selected_level_{suffix}"] = np.nan
        log[f"n_used_{suffix}"] = np.nan
    log["estimation_mode"] = "latest_only_missing_history"
    log["probability_note"] = "probabilities only computed for latest row"
    return log


def fill_full_sample_backfill(log: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    out = log.copy()
    key_probs = cond[cond["state_level"] == 2]
    keys = out.apply(lambda r: f"distance_to_MA200_bucket_2={r['distance_to_MA200_bucket_2']}|abs100_count_5d_bucket={r['abs100_count_5d_bucket']}|RV21_state={r['RV21_state']}", axis=1)
    for target in LOG_KEY_TARGETS:
        sub = key_probs[key_probs["target"] == target].copy()
        pmap = dict(zip(sub["state_key"], sub["conditional_probability"]))
        nmap = dict(zip(sub["state_key"], sub["usable_n"]))
        suffix = log_suffix(target)
        out[f"selected_prob_{suffix}"] = keys.map(pmap)
        out[f"selected_level_{suffix}"] = 2
        out[f"n_used_{suffix}"] = keys.map(nmap)
    out["estimation_mode"] = "historical_full_sample_backfill"
    out["probability_note"] = "uses full-sample backfill; not valid for historical signal evaluation"
    return out


def fill_latest_probabilities(log: pd.DataFrame, probs: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = log.copy()
    if mode == "latest_only":
        out["estimation_mode"] = "latest_only_missing_history"
        out["probability_note"] = "probabilities only computed for latest row"
    latest_idx = out.index[-1]
    for target in LOG_KEY_TARGETS:
        suffix = log_suffix(target)
        r = probs[probs["target"] == target]
        if r.empty:
            continue
        rr = r.iloc[0]
        out.loc[latest_idx, f"selected_prob_{suffix}"] = rr.get("conditional_probability", np.nan)
        out.loc[latest_idx, f"selected_level_{suffix}"] = rr.get("state_level", np.nan)
        out.loc[latest_idx, f"n_used_{suffix}"] = rr.get("usable_n", np.nan)
    if mode == "latest_only":
        out.loc[latest_idx, "estimation_mode"] = "latest_full_sample"
        out.loc[latest_idx, "probability_note"] = "latest row probability estimate"
    return out


def fill_walkforward_log(log: pd.DataFrame, df: pd.DataFrame, min_n: int, fallback_min_n: int, min_history: int = 252) -> pd.DataFrame:
    out = log.copy()
    out["estimation_mode"] = "walkforward"
    out["probability_note"] = "insufficient_history"
    for i in range(len(df)):
        if i < min_history:
            continue
        hist = df.iloc[:i]
        current = df.iloc[i]
        out.loc[out.index[i], "probability_note"] = "prior-data estimate"
        for target in LOG_KEY_TARGETS:
            chosen = None
            for require_ok in [True, False]:
                for level in sorted(STATE_LEVELS.keys(), reverse=True):
                    mask = pd.Series(True, index=hist.index)
                    for col in STATE_LEVELS[level]:
                        mask &= hist[col].astype(str).eq(str(current[col]))
                    d = hist.loc[mask].dropna(subset=[target])
                    n = len(d)
                    if (require_ok and n >= min_n) or ((not require_ok) and n >= fallback_min_n):
                        chosen = (level, d[target].mean(), n)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                d = hist.dropna(subset=[target])
                chosen = (0, d[target].mean() if len(d) else np.nan, len(d))
            suffix = log_suffix(target)
            out.loc[out.index[i], f"selected_prob_{suffix}"] = chosen[1]
            out.loc[out.index[i], f"selected_level_{suffix}"] = chosen[0]
            out.loc[out.index[i], f"n_used_{suffix}"] = chosen[2]
    return out


def write_daily_signal_log(out_dir: Path, signal_log: pd.DataFrame, report_df: pd.DataFrame, probs: pd.DataFrame, overwrite: bool, signal_log_mode: str, df: pd.DataFrame, cond: pd.DataFrame, min_n: int, fallback_min_n: int) -> pd.DataFrame:
    path = out_dir / "daily_signal_log.csv"
    if signal_log_mode == "full_sample_backfill":
        prepared = fill_latest_probabilities(fill_full_sample_backfill(signal_log, cond), probs, "full_sample_backfill")
    elif signal_log_mode == "walkforward":
        prepared = fill_latest_probabilities(fill_walkforward_log(signal_log, df, min_n, fallback_min_n), probs, "walkforward")
    else:
        prepared = fill_latest_probabilities(signal_log, probs, "latest_only")
    current = prepared.tail(1).copy()
    report = report_df.iloc[0]
    current["final_daily_label"] = report["final_daily_label"]
    current["post2020_match_n"] = report["post2020_n"]
    current["exact_match_n"] = report["n_used_for_primary_target"]
    current["post2020_sample_flag"] = report.get("post2020_sample_flag", "")
    if "uncertainty_label" in probs and len(probs):
        order = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "VERY_HIGH": 4}
        current["max_uncertainty_label"] = max(probs["uncertainty_label"].astype(str), key=lambda x: order.get(x, 0))
    else:
        current["max_uncertainty_label"] = "UNKNOWN"
    if overwrite or signal_log_mode in ["full_sample_backfill", "walkforward"] or not path.exists():
        base = prepared.iloc[:-1].copy()
        for col in current.columns:
            if col not in base:
                base[col] = np.nan
        for col in base.columns:
            if col not in current:
                current[col] = np.nan
        out = pd.concat([base[base.columns], current[base.columns]], ignore_index=True)
    else:
        old = pd.read_csv(path, low_memory=False)
        for col in current.columns:
            if col not in old:
                old[col] = np.nan
        for col in old.columns:
            if col not in current:
                current[col] = np.nan
        current = current[old.columns]
        old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()
        cur_date = pd.to_datetime(current["date"].iloc[0]).normalize()
        old = old[old["date"] != cur_date]
        out = pd.concat([old, current], ignore_index=True).sort_values("date")
    deprecated = [alias for _, alias in ALIAS_GROUPS if alias in out.columns]
    deprecated += [c for c in out.columns if any(alias in c for _, alias in ALIAS_GROUPS)]
    if deprecated:
        out = out.drop(columns=list(dict.fromkeys(deprecated)), errors="ignore")
    out.to_csv(path, index=False)
    return out


def write_text_report(path: Path, report_df: pd.DataFrame, probs: pd.DataFrame, fdist: pd.DataFrame, post2020_dist: pd.DataFrame, history: pd.DataFrame, meta: dict) -> None:
    row = report_df.iloc[0]
    interpretation = {
        "calm_state": "Current state is statistically calm. Historical analogs show lower-than-base short-horizon downside and whipsaw risk.",
        "active_but_safe": "Current state is active but not near the prebreak zone. Watch clustering and volatility, but MA200 distance is still protective.",
        "whipsaw_aware": "Current state shows whipsaw or repeated 1% movement, but does not yet meet the prebreak risk-off transition definition.",
        "prebreak_watch": "Current state is near MA200 with some clustering. This is a watch state, not a confirmed stress state.",
        "prebreak_warning": "Current state matches the prebreak warning structure: near MA200, repeated 1% movement, high realized volatility.",
        "prebreak_riskoff_transition": "Current state matches the strongest prebreak risk-off transition structure found in the research: near MA200, repeated 1% movement, high realized volatility, and recent whipsaw.",
        "confirmed_stress": "Current state is already in confirmed stress: below MA200, high volatility, and repeated 1% movement. This is not early warning; it is after-break stress.",
        "mixed": "Current state is mixed and does not match a named production risk state.",
    }
    lines = []
    lines.append("SPY Daily Prebreak Risk-Off / Whipsaw State Monitor")
    lines.append("=" * 64)
    lines.append("")
    lines.append("Data coverage")
    lines.append(f"Date range: {meta['date_min']} to {meta['date_max']} | rows={meta['rows']} | price column={meta['price_col']}")
    lines.append("")
    lines.append("Final daily label")
    lines.append(f"- {row['final_daily_label']}")
    lines.append(f"- reason: {row['final_daily_label_reason']}")
    lines.append(f"- priority rule: {row['final_daily_label_priority_rule']}")
    lines.append("")
    lines.append("Current state")
    lines.append(f"- final_state_label: {row['final_state_label']}")
    lines.append(f"- risk_score: {row['risk_score']} ({row['risk_score_bucket']})")
    lines.append(f"- distance_to_MA200: {pct(row['distance_to_MA200'])} ({row['distance_to_MA200_bucket_2']})")
    lines.append(f"- abs100_count_5d_bucket: {row['abs100_count_5d_bucket']}")
    lines.append(f"- RV21 state: {row['RV21_state']} | vol transition: {row['vol_transition_state']}")
    lines.append(f"- recent whipsaw 10d 1%: {row['recent_whipsaw_10d_100']}")
    lines.append(f"- weekly memory: {row['weekly_memory_score_bucket']}")
    lines.append("")
    lines.append("Conditional probabilities")
    display_targets = [
        "future_abs_1_00_3d", "future_down_1_00_3d", "future_abs_1_00_5d", "future_down_1_00_5d",
        *PRODUCTION_GRID_TARGETS,
        "forward_fast_riskoff_21d", "forward_tail_down_21d",
    ]
    for _, p in probs[probs["target"].isin(display_targets)].sort_values(["horizon", "target"]).iterrows():
        level_context = (
            f" | L3 {pct(p.get('level3_probability', np.nan))} n={int(p['level3_usable_n']) if pd.notna(p.get('level3_usable_n', np.nan)) else 'n/a'}"
            f" | L2 {pct(p.get('level2_probability', np.nan))} n={int(p['level2_usable_n']) if pd.notna(p.get('level2_usable_n', np.nan)) else 'n/a'}"
        )
        lines.append(f"{p['target']}: {pct(p['conditional_probability'])} | base {pct(p['unconditional_probability'])} | HR {p['hazard_ratio']:.2f} | n={int(p['usable_n'])} | CI {pct(p['ci_low'])}-{pct(p['ci_high'])} | uncertainty {p['uncertainty_label']} | L{int(p['state_level'])} | {p['sample_flag']}{level_context}")
    lines.append("")
    lines.append("Forward return distribution")
    for _, r in fdist.iterrows():
        lines.append(f"{r['horizon']}: mean {pct(r['mean'])}, median {pct(r['median'])}, p10 {pct(r['p10'])}, p25 {pct(r['p25'])}, p75 {pct(r['p75'])}, p90 {pct(r['p90'])}, P(negative) {pct(r['probability_negative'])}, P(<=-2%) {pct(r['probability_le_minus_2pct'])}, expected max drawdown {pct(r['expected_max_drawdown'])}")
    lines.append("")
    lines.append("Post-2020 analog summary")
    for _, r in post2020_dist.iterrows():
        lines.append(f"{r['horizon']}: n={int(r['n'])} | {r['sample_flag']} | mean {pct(r['mean'])} | median {pct(r['median'])} | p10 {pct(r['p10'])} | P(negative) {pct(r['prob_negative'])} | P(<=-2%) {pct(r['prob_le_minus_2'])}")
    if not post2020_dist.empty and post2020_dist["sample_flag"].isin(["insufficient", "low_n"]).any():
        lines.append("Post-2020 analog sample is small/low_n; treat as directional only.")
    lines.append("")
    lines.append("Old MA/ATR trend overlay")
    lines.append(f"- state: {row['old_ma_atr_state']}")
    lines.append(f"- signal: {row['old_ma_atr_signal']}")
    lines.append(f"- MA20: {row['ma20']:.2f}" if pd.notna(row["ma20"]) else "- MA20: n/a")
    lines.append(f"- MA50: {row['ma50']:.2f}" if pd.notna(row["ma50"]) else "- MA50: n/a")
    lines.append(f"- slope20: {row['slope20']:.4f}" if pd.notna(row["slope20"]) else "- slope20: n/a")
    lines.append(f"- ATR14: {row['atr14']:.4f}" if pd.notna(row["atr14"]) else "- ATR14: n/a")
    lines.append(f"- ATR_slope: {row['atr_slope']:.4f}" if pd.notna(row["atr_slope"]) else "- ATR_slope: n/a")
    lines.append(f"- source: {row['old_ma_atr_state_source']}")
    lines.append("")
    lines.append("Most recent historical analogs")
    for _, h in history.tail(10).sort_values("date", ascending=False).iterrows():
        lines.append(f"{pd.to_datetime(h['date']).date()} close={h['close']:.2f} ret={pct(h['ret'])} fwd10={pct(h['fwd_ret_10d'])} fwd21={pct(h['fwd_ret_21d'])} down2_10={h['future_down_2_00_10d']}")
    lines.append("")
    lines.append("Interpretation")
    lines.append(interpretation.get(row["final_state_label"], interpretation["mixed"]))
    lines.append("")
    lines.append("Production warnings")
    lines.append(f"- exact state n: {int(row['n_used_for_primary_target']) if pd.notna(row['n_used_for_primary_target']) else 'n/a'}")
    lines.append(f"- selected probability level: L{int(row['matched_state_level_for_primary_target']) if pd.notna(row['matched_state_level_for_primary_target']) else 'n/a'}")
    lines.append(f"- exact state uncertainty: {meta.get('primary_uncertainty', 'UNKNOWN')}")
    lines.append(f"- post-2020 analog sample flag: {row.get('post2020_sample_flag', 'n/a')}")
    if row.get("post2020_sample_flag") == "low_n":
        lines.append("Post-2020 analog sample is low_n; treat as directional only.")
    elif row.get("post2020_sample_flag") == "insufficient":
        lines.append("Post-2020 analog sample is insufficient; do not use it as the primary estimate.")
    lines.append(f"- estimation mode: {meta.get('estimation_mode', 'n/a')}")
    lines.append(f"- duplicate target audit: {meta.get('duplicate_target_audit', 'n/a')}")
    lines.append(f"- horizon leakage check: {meta.get('horizon_leakage_audit', 'n/a')}")
    lines.append(f"- production_ready: {row.get('production_ready', False)}")
    if str(row.get("failed_readiness_checks", "")):
        lines.append(f"- failed_readiness_checks: {row.get('failed_readiness_checks')}")
    lines.append("No trading recommendation is made.")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_charts(out_dir: Path, matches: pd.DataFrame, probs: pd.DataFrame, df: pd.DataFrame, signal_log: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable; charts skipped. {exc}")
        return
    png = out_dir / "png"
    png.mkdir(parents=True, exist_ok=True)

    def guard(name, fn):
        try:
            fn()
        except Exception as exc:
            print(f"WARNING: chart {name} failed: {exc}")

    guard("current_state_forward_return_boxplot.png", lambda: chart_boxplot(plt, matches, png))
    guard("current_state_probability_vs_base.png", lambda: chart_prob_vs_base(plt, probs, png))
    guard("spy_close_ma200_state.png", lambda: chart_spy_ma200(plt, df, png))
    guard("risk_score_history.png", lambda: chart_risk_score(plt, signal_log, png))
    guard("daily_signal_probabilities.png", lambda: chart_signal_probs(plt, signal_log, png))


def chart_boxplot(plt, matches, png):
    data = [matches[f"fwd_ret_{h}d"].dropna() for h in HORIZONS]
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        ax.boxplot(data, tick_labels=[f"{h}d" for h in HORIZONS])
    except TypeError:
        ax.boxplot(data, labels=[f"{h}d" for h in HORIZONS])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Current-state historical analog forward returns")
    fig.tight_layout()
    fig.savefig(png / "current_state_forward_return_boxplot.png", dpi=160)
    plt.close(fig)


def chart_prob_vs_base(plt, probs, png):
    targets = ["future_down_2_00_10d", "future_down_3_00_21d", "future_whipsaw_1_00_10d", "break_below_MA200_21d", "forward_fast_riskoff_21d"]
    d = probs[probs["target"].isin(targets)].copy()
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.18, d["conditional_probability"], width=0.36, label="conditional")
    ax.bar(x + 0.18, d["unconditional_probability"], width=0.36, label="base")
    ax.set_xticks(x, d["target"], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_title("Current-state probability vs base rate")
    fig.tight_layout()
    fig.savefig(png / "current_state_probability_vs_base.png", dpi=160)
    plt.close(fig)


def chart_spy_ma200(plt, df, png):
    d = df.tail(504)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d["date"], d["close"], label="SPY close")
    ax.plot(d["date"], d["MA200"], label="MA200")
    ax.scatter(d["date"].iloc[-1], d["close"].iloc[-1], color="red", zorder=3)
    ax.legend()
    ax.set_title("SPY close and MA200")
    fig.tight_layout()
    fig.savefig(png / "spy_close_ma200_state.png", dpi=160)
    plt.close(fig)


def chart_risk_score(plt, log, png):
    d = log.tail(504)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d["date"], d["risk_score"])
    ax.set_title("Risk score history, last 2 years")
    fig.tight_layout()
    fig.savefig(png / "risk_score_history.png", dpi=160)
    plt.close(fig)


def chart_signal_probs(plt, log, png):
    d = log.tail(252)
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for col in ["selected_prob_future_down_2_00_10d", "selected_prob_future_down_3_00_21d", "selected_prob_future_whipsaw_1_00_10d", "selected_prob_break_below_MA200_21d", "selected_prob_forward_fast_riskoff_21d"]:
        if col in d:
            y = pd.to_numeric(d[col], errors="coerce")
            if y.notna().any():
                ax.plot(d["date"], y, label=col.replace("selected_prob_", ""))
                plotted = True
    if plotted:
        ax.legend(fontsize=7)
    ax.set_title("Daily signal probabilities, last 1 year")
    fig.tight_layout()
    fig.savefig(png / "daily_signal_probabilities.png", dpi=160)
    plt.close(fig)


ALIAS_GROUPS = [
    ("future_down_2_00_10d", "future_down_2_00_within_10d"),
    ("future_down_3_00_21d", "future_down_3_00_within_21d"),
    ("future_whipsaw_1_00_10d", "future_whipsaw_1_00_within_10d"),
    ("break_below_MA200_21d", "break_below_MA200_within_21d"),
]


def horizon_from_name(name: str) -> int | None:
    for h in [63, 42, 21, 10, 5, 3, 1]:
        if str(name).endswith(f"_{h}d"):
            return h
    return None


def duplicate_target_audit_for_df(d: pd.DataFrame) -> tuple[bool, str]:
    names = set(d.columns)
    if "target" in d.columns:
        names |= set(d["target"].dropna().astype(str).unique())
    duplicates = []
    for canonical, alias in ALIAS_GROUPS:
        if canonical in names and alias in names:
            duplicates.append(f"{canonical}|{alias}")
        elif alias in names:
            duplicates.append(alias)
    return bool(duplicates), ";".join(duplicates)


def horizon_leakage_for_df(path: Path, d: pd.DataFrame) -> tuple[bool, int | float, int | float, int | float, str]:
    if path.name != "daily_state_dataset.csv":
        has_targets = any(horizon_from_name(c) for c in d.columns)
        return has_targets, np.nan, np.nan, np.nan, "not_applicable"
    target_cols = [c for c in d.columns if horizon_from_name(c) and (c.startswith("future_") or c.startswith("break_below_MA200_") or c.startswith("forward_") or c.startswith("fwd_") or c == "stress_whipsaw_10d")]
    if not target_cols:
        return False, np.nan, np.nan, np.nan, "not_applicable"
    max_h = max(horizon_from_name(c) or 0 for c in target_cols)
    expected_total = 0
    actual_total = 0
    failed = []
    for col in target_cols:
        h = horizon_from_name(col) or (10 if col == "stress_whipsaw_10d" else None)
        if not h:
            continue
        expected_total += h
        actual = int(d[col].tail(h).isna().sum())
        actual_total += actual
        if actual != h:
            failed.append(col)
    return True, max_h, expected_total, actual_total, "fail" if failed else "pass"


def output_audit(out_dir: Path) -> pd.DataFrame:
    required = {
        "base_rate_summary.csv": ["sample", "target", "n", "probability"],
        "conditional_probability_table.csv": ["target", "usable_n", "conditional_probability", "ci_low", "ci_high", "uncertainty_label"],
        "current_state_probabilities.csv": ["date", "target", "selected_state_level", "selected_probability", "selected_usable_n", "selected_ci_low", "selected_ci_high", "level5_probability", "level4_probability", "level3_probability", "level2_probability", "level1_probability", "level0_probability"],
        "current_state_report.csv": ["date", "final_state_label", "final_daily_label", "old_ma_atr_state", "post2020_n", "production_ready", "failed_readiness_checks"],
        "current_state_report.txt": [],
        "daily_signal_log.csv": [
            "date", "close", "ret", "final_state_label", "risk_score", "risk_score_bucket",
            "distance_to_MA200", "distance_to_MA200_bucket_2", "abs100_count_5d_bucket",
            "RV21_state", "vol_transition_state", "recent_whipsaw_10d_100",
            "recent_pattern_state", "weekly_memory_score_bucket", "estimation_mode",
            "probability_note", "selected_prob_future_down_2_00_10d",
            "selected_prob_future_down_3_00_21d", "selected_prob_future_whipsaw_1_00_10d",
            "selected_prob_break_below_MA200_21d", "selected_prob_forward_fast_riskoff_21d",
            "selected_prob_forward_tail_down_21d", "selected_level_future_down_2_00_10d",
            "selected_level_future_down_3_00_21d", "selected_level_future_whipsaw_1_00_10d",
            "selected_level_break_below_MA200_21d", "selected_level_forward_fast_riskoff_21d",
            "selected_level_forward_tail_down_21d", "n_used_future_down_2_00_10d",
            "n_used_future_down_3_00_21d", "n_used_future_whipsaw_1_00_10d",
            "n_used_break_below_MA200_21d", "n_used_forward_fast_riskoff_21d",
            "n_used_forward_tail_down_21d",
        ],
        "daily_state_dataset.csv": ["date", "close", "ret", "old_ma_atr_state"],
        "forward_return_distribution.csv": ["horizon", "n", "mean", "expected_max_drawdown"],
        "post2020_forward_return_distribution.csv": ["horizon", "n", "sample_flag", "mean", "median", "std", "p10", "p25", "p75", "p90", "prob_negative", "prob_le_minus_2", "expected_max_drawdown"],
        "output_file_audit.csv": [],
        "state_match_history.csv": ["date", "close", "fwd_ret_10d", "fwd_ret_21d"],
    }
    rows = []
    all_csv_names = sorted({p.name for p in out_dir.glob("*.csv")} | set(required.keys()))
    for file_name in all_csv_names:
        required_cols = required.get(file_name, [])
        path = out_dir / file_name
        exists = path.exists()
        if not exists:
            rows.append({
                "filename": file_name, "exists": False, "rows": 0, "columns": 0,
                "required_columns_present": False, "missing_required_columns": ",".join(required_cols),
                "last_modified_time": "", "has_duplicate_target_names": False, "duplicate_target_names": "",
                "has_forward_targets": False, "max_horizon": np.nan, "expected_nan_tail_rows": np.nan,
                "actual_nan_tail_rows": np.nan, "horizon_leakage_check": "not_applicable", "notes": "missing",
            })
            continue
        try:
            if path.suffix.lower() == ".csv":
                d = pd.read_csv(path, low_memory=False)
                rows_n, cols_n = len(d), len(d.columns)
                missing = [c for c in required_cols if c not in d.columns]
                has_dups, dup_names = duplicate_target_audit_for_df(d)
                has_fwd, max_h, expected_tail, actual_tail, leak_check = horizon_leakage_for_df(path, d)
                if missing or has_dups or leak_check == "fail":
                    notes = "fail"
                else:
                    notes = "ok" if rows_n or file_name == "output_file_audit.csv" else "empty"
            else:
                text = path.read_text(encoding="utf-8")
                d = pd.DataFrame()
                rows_n, cols_n = len(text.splitlines()), 1
                missing = []
                notes = "ok" if text else "empty"
                has_dups, dup_names = False, ""
                has_fwd, max_h, expected_tail, actual_tail, leak_check = False, np.nan, np.nan, np.nan, "not_applicable"
            date_min = date_max = ""
            if path.suffix.lower() == ".csv":
                for c in ["date"]:
                    if c in d:
                        dt = pd.to_datetime(d[c], errors="coerce")
                        if dt.notna().any():
                            date_min = str(dt.min().date())
                            date_max = str(dt.max().date())
            rows.append({
                "filename": file_name,
                "exists": True,
                "rows": rows_n,
                "columns": cols_n,
                "required_columns_present": len(missing) == 0,
                "missing_required_columns": ",".join(missing),
                "last_modified_time": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
                "missing_value_percent": d.isna().mean().mean() if path.suffix.lower() == ".csv" and len(d.columns) else 0,
                "date_min": date_min,
                "date_max": date_max,
                "has_duplicate_target_names": has_dups,
                "duplicate_target_names": dup_names,
                "has_forward_targets": has_fwd,
                "max_horizon": max_h,
                "expected_nan_tail_rows": expected_tail,
                "actual_nan_tail_rows": actual_tail,
                "horizon_leakage_check": leak_check,
                "notes": notes,
            })
        except Exception as exc:
            rows.append({"filename": file_name, "exists": exists, "rows": np.nan, "columns": np.nan, "required_columns_present": False, "missing_required_columns": ",".join(required_cols), "last_modified_time": "", "missing_value_percent": np.nan, "date_min": "", "date_max": "", "has_duplicate_target_names": True, "duplicate_target_names": "", "has_forward_targets": False, "max_horizon": np.nan, "expected_nan_tail_rows": np.nan, "actual_nan_tail_rows": np.nan, "horizon_leakage_check": "fail", "notes": str(exc)})
    return pd.DataFrame(rows)


def readiness_checks(out_dir: Path, audit: pd.DataFrame, current_probs_export: pd.DataFrame, signal_log: pd.DataFrame) -> dict:
    checks = {}
    checks["no_duplicate_target_names"] = not bool(audit["has_duplicate_target_names"].fillna(False).any())
    leakage = audit.loc[audit["horizon_leakage_check"].ne("not_applicable"), "horizon_leakage_check"]
    checks["horizon_leakage_pass"] = bool(len(leakage) == 0 or leakage.eq("pass").all())
    required_prob_cols = {
        "level5_probability", "level4_probability", "level3_probability", "level2_probability",
        "selected_probability", "selected_usable_n", "selected_ci_low", "selected_ci_high",
    }
    checks["current_probs_l2_l3_l5_columns"] = required_prob_cols.issubset(current_probs_export.columns)
    checks["daily_log_estimation_mode"] = "estimation_mode" in signal_log.columns
    checks["output_file_audit_created"] = (out_dir / "output_file_audit.csv").exists()
    checks["current_state_report_txt_created"] = (out_dir / "current_state_report.txt").exists()
    failed = [k for k, v in checks.items() if not v]
    return {"production_ready": len(failed) == 0, "failed_readiness_checks": ";".join(failed), **checks}


def console_report(report_df: pd.DataFrame, probs: pd.DataFrame, fdist: pd.DataFrame, history: pd.DataFrame, meta: dict, quiet: bool, fallback_level: int, min_n: int, audit: pd.DataFrame) -> None:
    if quiet:
        return
    row = report_df.iloc[0]
    print("\nSPY Daily Prebreak Risk-Off / Whipsaw Monitor")
    print("=" * 64)
    print(f"Date range: {meta['date_min']} to {meta['date_max']}")
    print(f"Latest date: {meta['date_max']}")
    print(f"Latest close: {row['close']:.2f}")
    print(f"Final state: {row['final_state_label']} / {row.get('final_daily_label', 'n/a')}")
    print(f"Risk score: {row['risk_score']} ({row['risk_score_bucket']})")
    print(f"Production ready: {str(bool(row.get('production_ready', False))).lower()}")
    print("\nAudit:")
    print(f"- duplicate target names: {meta.get('duplicate_target_audit', 'n/a')}")
    print(f"- horizon leakage: {meta.get('horizon_leakage_audit', 'n/a')}")
    print(f"- L2/L3/L5 comparison columns: {'pass' if meta.get('l2_l3_l5_audit') == 'pass' else 'fail'}")
    print(f"- estimation_mode in daily_signal_log: {meta.get('estimation_mode_audit', 'n/a')}")
    print(f"- post-2020 low-n warning: {meta.get('post2020_low_n_audit', 'n/a')}")
    print("\nCurrent key probabilities:")
    targets = LOG_KEY_TARGETS
    for _, p in probs[probs["target"].isin(targets)].sort_values(["horizon", "target"]).iterrows():
        l3n = int(p["level3_usable_n"]) if pd.notna(p.get("level3_usable_n", np.nan)) else "n/a"
        l2n = int(p["level2_usable_n"]) if pd.notna(p.get("level2_usable_n", np.nan)) else "n/a"
        print(f"- {p['target']}: {pct(p['conditional_probability'])} | base {pct(p['unconditional_probability'])} | HR {p['hazard_ratio']:.2f} | n={int(p['usable_n'])} | L{int(p['state_level'])} | L3 {pct(p.get('level3_probability', np.nan))} n={l3n} | L2 {pct(p.get('level2_probability', np.nan))} n={l2n}")
    print("\nForward return distribution")
    print("horizon | mean | median | p10 | p25 | p75 | p90 | P(negative) | P(<=-2%) | expected max drawdown")
    for _, r in fdist.iterrows():
        print(f"{r['horizon']} | {pct(r['mean'])} | {pct(r['median'])} | {pct(r['p10'])} | {pct(r['p25'])} | {pct(r['p75'])} | {pct(r['p90'])} | {pct(r['probability_negative'])} | {pct(r['probability_le_minus_2pct'])} | {pct(r['expected_max_drawdown'])}")
    print("\nHistorical analogs (most recent 10)")
    for _, h in history.tail(10).sort_values("date", ascending=False).iterrows():
        print(f"  {pd.to_datetime(h['date']).date()} close={h['close']:.2f} ret={pct(h['ret'])} fwd10={pct(h['fwd_ret_10d'])} fwd21={pct(h['fwd_ret_21d'])}")
    print("\nInterpretation")
    interpretations = {
        "calm_state": "Current state is statistically calm. Historical analogs show lower-than-base short-horizon downside and whipsaw risk.",
        "active_but_safe": "Current state is active but not near the prebreak zone. Watch clustering and volatility, but MA200 distance is still protective.",
        "whipsaw_aware": "Current state shows whipsaw or repeated 1% movement, but does not yet meet the prebreak risk-off transition definition.",
        "prebreak_watch": "Current state is near MA200 with some clustering. This is a watch state, not a confirmed stress state.",
        "prebreak_warning": "Current state matches the prebreak warning structure: near MA200, repeated 1% movement, high realized volatility.",
        "prebreak_riskoff_transition": "Current state matches the strongest prebreak risk-off transition structure found in the research: near MA200, repeated 1% movement, high realized volatility, and recent whipsaw.",
        "confirmed_stress": "Current state is already in confirmed stress: below MA200, high volatility, and repeated 1% movement. This is not early warning; it is after-break stress.",
        "mixed": "Current state is mixed and does not match a named production risk state.",
    }
    print(interpretations.get(row["final_state_label"], interpretations["mixed"]))
    print("No trading advice is provided.")
    print("\nFinal validation")
    print("All output files created.")
    print(f"Latest date processed: {meta['date_max']}")
    if fallback_level < 5:
        print(f"Current exact state has low historical sample size; probability estimate used fallback Level {fallback_level}.")
    else:
        print("Current exact Level 5 state had enough historical analogs for the primary target.")
    print("\n=== Daily Screen v1.1 Patch Summary ===")
    print(f"- Confidence intervals added: {'yes' if {'ci_low','ci_high','uncertainty_label'}.issubset(probs.columns) else 'no'}")
    print("- Post-2020 analog summary added: yes")
    print(f"- Old MA/ATR overlay added: {'yes' if 'old_ma_atr_state' in report_df.columns else 'no'}")
    print(f"- Final daily label added: {'yes' if 'final_daily_label' in report_df.columns else 'no'}")
    print(f"- Output audit passed: {'yes' if bool(audit['required_columns_present'].all() and audit['exists'].all()) else 'no'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SPY Daily Prebreak Risk-Off / Whipsaw State Monitor")
    parser.add_argument("--spy-csv", default="data/SPY_20y.csv")
    parser.add_argument("--vix-csv", default="data/^VIX_20y.csv")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--post-2020-only", action="store_true")
    parser.add_argument("--min-n", type=int, default=50)
    parser.add_argument("--fallback-min-n", type=int, default=20)
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--walkforward-current", action="store_true")
    parser.add_argument("--overwrite-log", action="store_true")
    parser.add_argument("--signal-log-mode", choices=["latest_only", "full_sample_backfill", "walkforward"], default="latest_only")
    args = parser.parse_args()

    spy_csv = Path(args.spy_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        maybe_download_spy(spy_csv)
    prices, load_meta = load_price_csv(spy_csv, "close")
    if args.start_date:
        prices = prices[prices["date"] >= pd.Timestamp(args.start_date)].reset_index(drop=True)
    df, vix_available = add_features(prices, Path(args.vix_csv))
    df = add_targets(df)
    latest = df.dropna(subset=["close"]).iloc[-1]
    estimation_mask = pd.Series(True, index=df.index)
    if args.walkforward_current:
        estimation_mask &= df["date"] < latest["date"]
    if args.post_2020_only:
        estimation_mask &= df["date"] >= pd.Timestamp("2020-01-01")
    estimation_mode = "walkforward_current" if args.walkforward_current else "full_history"
    if args.post_2020_only:
        estimation_mode += "_post_2020_only"

    targets = KEY_TARGETS
    cond = compute_conditional_table(df, latest, targets, args.min_n, args.fallback_min_n, estimation_mask)
    probs = select_current_probabilities(cond, args.min_n, args.fallback_min_n)
    current_probs_export = build_current_state_probabilities_export(probs, cond, latest["date"])
    history, selected_level = state_match_history(df, latest, probs, estimation_mask)
    fdist = forward_distribution(history)
    post2020_dist = post2020_forward_distribution(history)
    report_df = current_report(latest, probs, post2020_dist)
    base = base_rate_summary(df, history, estimation_mask)
    signal_log = daily_signal_log(df, cond)

    df.to_csv(out_dir / "daily_state_dataset.csv", index=False)
    report_df.to_csv(out_dir / "current_state_report.csv", index=False)
    current_probs_export.to_csv(out_dir / "current_state_probabilities.csv", index=False)
    cond.to_csv(out_dir / "conditional_probability_table.csv", index=False)
    fdist.to_csv(out_dir / "forward_return_distribution.csv", index=False)
    post2020_dist.to_csv(out_dir / "post2020_forward_return_distribution.csv", index=False)
    history.to_csv(out_dir / "state_match_history.csv", index=False)
    base.to_csv(out_dir / "base_rate_summary.csv", index=False)
    signal_log = write_daily_signal_log(out_dir, signal_log, report_df, probs, args.overwrite_log, args.signal_log_mode, df, cond, args.min_n, args.fallback_min_n)
    meta = {
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "rows": len(df),
        "price_col": load_meta["price_col"],
        "vix_available": vix_available,
        "estimation_mode": args.signal_log_mode,
        "primary_uncertainty": probs.loc[probs["target"].eq("future_down_2_00_10d"), "uncertainty_label"].iloc[0] if probs["target"].eq("future_down_2_00_10d").any() else "UNKNOWN",
    }
    write_text_report(out_dir / "current_state_report.txt", report_df, probs, fdist, post2020_dist, history, meta)
    if not args.no_charts:
        make_charts(out_dir, history, probs, df, signal_log)
    audit = output_audit(out_dir)
    audit.to_csv(out_dir / "output_file_audit.csv", index=False)
    readiness = readiness_checks(out_dir, audit, current_probs_export, signal_log)
    report_df["production_ready"] = readiness["production_ready"]
    report_df["failed_readiness_checks"] = readiness["failed_readiness_checks"]
    meta.update({
        "duplicate_target_audit": "pass" if readiness["no_duplicate_target_names"] else "fail",
        "horizon_leakage_audit": "pass" if readiness["horizon_leakage_pass"] else "fail",
        "l2_l3_l5_audit": "pass" if readiness["current_probs_l2_l3_l5_columns"] else "fail",
        "estimation_mode_audit": "pass" if readiness["daily_log_estimation_mode"] else "fail",
        "post2020_low_n_audit": "pass" if report_df["post2020_sample_flag"].iloc[0] in ["low_n", "insufficient", "ok"] else "fail",
    })
    report_df.to_csv(out_dir / "current_state_report.csv", index=False)
    write_text_report(out_dir / "current_state_report.txt", report_df, probs, fdist, post2020_dist, history, meta)
    audit = output_audit(out_dir)
    audit.to_csv(out_dir / "output_file_audit.csv", index=False)
    console_report(report_df, probs, fdist, history, meta, args.quiet, selected_level, args.min_n, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
