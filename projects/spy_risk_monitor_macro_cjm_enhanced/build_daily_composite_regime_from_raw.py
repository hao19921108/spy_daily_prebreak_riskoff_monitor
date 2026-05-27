#!/usr/bin/env python3
"""
Build the daily Composite / CJM-enhanced SPY regime monitor directly from raw
local market files and cached/FRED macro data.

Regime-classification research only. No trading advice, no portfolio action,
no position sizing, and no trading inference. Forward returns/drawdowns are
descriptive only and are computed after regime labels are fixed.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from scipy.special import softmax

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer, StandardScaler

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.*")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SIBLING_MACRO_DATA_DIR = BASE_DIR.parent / "spy_risk_monitor_macro_enhanced" / "data"
NO_ADVICE = "No trading advice generated. Regime classification only."
UNIVERSES = ["VIX_ONLY", "FULL_PUBLIC_MACRO", "NO_VIX", "RATES_CREDIT_ONLY", "GROWTH_ONLY"]
FORWARD_HORIZONS = [5, 10, 21, 42, 63]

FRED_SERIES = {
    "INDPRO": ("Industrial Production Index", "monthly", "1M"),
    "CFNAI": ("Chicago Fed National Activity Index", "monthly", "1M"),
    "UNRATE": ("Unemployment Rate", "monthly", "1M"),
    "ICSA": ("Initial Claims", "weekly", "1W"),
    "CPIAUCSL": ("CPI", "monthly", "1M"),
    "CORESTICKM159SFRBATL": ("Sticky CPI less Food and Energy", "monthly", "1M"),
    "T5YIE": ("5-Year Breakeven Inflation", "daily", "0D"),
    "T10YIE": ("10-Year Breakeven Inflation", "daily", "0D"),
    "DGS2": ("2-Year Treasury", "daily", "0D"),
    "DGS10": ("10-Year Treasury", "daily", "0D"),
    "FEDFUNDS": ("Effective Fed Funds", "monthly", "1M"),
    "DFII10": ("10-Year TIPS Real Yield", "daily", "0D"),
    "BAA10Y": ("BAA less 10-Year Treasury", "daily", "0D"),
    "BAMLH0A0HYM2": ("High Yield OAS", "daily", "0D"),
    "NFCI": ("National Financial Conditions Index", "weekly", "1W"),
    "ANFCI": ("Adjusted NFCI", "weekly", "1W"),
    "WALCL": ("Fed Balance Sheet", "weekly", "1W"),
    "M2SL": ("M2 Money Stock", "monthly", "1M"),
    "UMCSENT": ("Michigan Consumer Sentiment", "monthly", "1M"),
    "HOUST": ("Housing Starts", "monthly", "1M"),
    "PERMIT": ("Building Permits", "monthly", "1M"),
    "DCOILWTICO": ("WTI Crude Oil", "daily", "0D"),
    "VIXCLS": ("CBOE VIX", "daily", "0D"),
}


@dataclass
class CJMFit:
    labels: pd.Series
    probabilities: pd.DataFrame
    features: List[str]


def log(msg: str) -> None:
    print(f"[from-raw] {msg}")


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_dirs(out: Path, fred_cache: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for sub in ["raw", "processed", "metadata"]:
        (fred_cache / sub).mkdir(parents=True, exist_ok=True)


def find_date_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() in {"date", "datetime", "timestamp", "observation_date", "unnamed: 0"}:
            return col
    for col in df.columns:
        if "date" in str(col).lower():
            return col
    return df.columns[0]


def read_date_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    date_col = find_date_col(df)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    if date_col != "date":
        df = df.drop(columns=[date_col], errors="ignore")
    return df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def find_files(names: Sequence[str]) -> List[Path]:
    roots = [DATA_DIR, BASE_DIR, DATA_DIR / "core", SIBLING_MACRO_DATA_DIR]
    return [path for path in [r / n for r in roots for n in names] if path.exists()]


def find_file(names: Sequence[str]) -> Optional[Path]:
    paths = find_files(names)
    return paths[0] if paths else None


def close_col(df: pd.DataFrame) -> str:
    for col in ["Adj Close", "adj_close", "adjusted_close", "Close", "close", "value"]:
        if col in df.columns:
            return col
    numeric = [c for c in df.columns if c != "date" and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
    if not numeric:
        raise ValueError("No close/value column found.")
    return numeric[0]


def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0] or col[-1]) for col in out.columns]
    return out


def read_price_file(path: Path, symbol: str) -> pd.DataFrame:
    df = read_date_csv(path)
    if df.empty:
        raise ValueError(f"No dated rows found in {path}")
    c = close_col(df)
    out = pd.DataFrame({"date": df["date"], f"{symbol}_close": pd.to_numeric(df[c], errors="coerce")})
    for src, dst in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "raw_close"), ("Volume", "volume")]:
        if src in df.columns:
            out[f"{symbol}_{dst}"] = pd.to_numeric(df[src], errors="coerce")
    out = out.dropna(subset=[f"{symbol}_close"]).sort_values("date")
    if out.empty:
        raise ValueError(f"No usable close rows found in {path}")
    return out


def annotate_source(df: pd.DataFrame, path_or_provider: str, status: str, warning: str = "") -> pd.DataFrame:
    df.attrs["path_or_provider"] = path_or_provider
    df.attrs["source_status"] = status
    df.attrs["source_warning"] = warning
    return df


def price_audit_row(source_name: str, df: pd.DataFrame, fallback_path: str = "") -> Dict[str, object]:
    latest = pd.to_datetime(df["date"], errors="coerce").max() if len(df) and "date" in df else pd.NaT
    return {
        "source_name": source_name,
        "path_or_provider": df.attrs.get("path_or_provider", fallback_path),
        "latest_date": latest.date().isoformat() if pd.notna(latest) else "",
        "rows": int(len(df)),
        "status": df.attrs.get("source_status", "missing" if df.empty else "ok"),
        "warning": df.attrs.get("source_warning", ""),
    }


def load_price(symbol: str, names: Sequence[str], allow_yfinance: bool = False) -> pd.DataFrame:
    refresh_warning = ""
    if symbol == "SPY" and allow_yfinance:
        try:
            import yfinance as yf

            log("Refreshing SPY from yfinance before reading local CSV.")
            raw = yf.download("SPY", start="2000-01-01", auto_adjust=False, progress=False)
            raw = normalize_yfinance_columns(raw)
            if raw.empty:
                raise RuntimeError("yfinance returned no SPY rows")
            raw = raw.reset_index()
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            target = DATA_DIR / "SPY_20y.csv"
            raw.to_csv(target, index=False)
            out = read_price_file(target, symbol)
            return annotate_source(out, f"yfinance:{target}", "fresh", "")
        except Exception as exc:
            refresh_warning = f"WARNING: SPY yfinance refresh failed; falling back to local CSV. Error: {exc}"
            log(refresh_warning)
    for path in find_files(names):
        try:
            status = "warning" if refresh_warning else "ok"
            return annotate_source(read_price_file(path, symbol), str(path), status, refresh_warning)
        except Exception as exc:
            log(f"Skipping unusable {symbol} price file {path}: {exc}")
    if symbol == "SPY":
        raise FileNotFoundError(f"SPY price data is required. {refresh_warning}".strip())
    return annotate_source(pd.DataFrame(columns=["date"]), "local CSV", "missing", f"No local {symbol} file found.")


def fred_raw_to_series(path: Path, sid: str) -> pd.Series:
    df = pd.read_csv(path, low_memory=False)
    date_col = find_date_col(df)
    value_col = sid if sid in df.columns else close_col(df.rename(columns={df.columns[-1]: "value"}))
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", sid]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out[sid] = pd.to_numeric(out[sid].replace(".", np.nan), errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    return out.set_index("date")[sid]


def update_manifest(fred_cache: Path, row: Dict[str, object]) -> None:
    path = fred_cache / "metadata" / "fred_series_manifest.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old) and "series_id" in old:
        old = old[old["series_id"].astype(str) != str(row["series_id"])]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).sort_values("series_id").to_csv(path, index=False)


def fetch_fred_series(series_id: str, start_date: str, end_date: str, fred_cache: Path, refresh: bool = False) -> Tuple[pd.Series, str]:
    raw_path = fred_cache / "raw" / f"{series_id}.csv"
    desc, freq, lag = FRED_SERIES.get(series_id, (series_id, "unknown", "unknown"))
    status = "cache"
    try:
        if refresh or not raw_path.exists():
            if requests is None:
                raise RuntimeError("requests is unavailable")
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            raw_path.write_bytes(resp.content)
            status = "fresh"
        s = fred_raw_to_series(raw_path, series_id)
        s = s.loc[(s.index >= pd.Timestamp(start_date)) & (s.index <= pd.Timestamp(end_date))]
        load_status = status
    except Exception as exc:
        s = pd.Series(dtype=float, name=series_id)
        load_status = f"failed: {exc}"
    update_manifest(
        fred_cache,
        {
            "series_id": series_id,
            "description": desc,
            "local_path": str(raw_path),
            "first_date": s.index.min().date().isoformat() if len(s) else "",
            "last_date": s.index.max().date().isoformat() if len(s) else "",
            "rows": int(s.notna().sum()),
            "last_fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frequency_assumption": freq,
            "lag_assumption": f"{lag}; conservative publication-lag approximation",
            "load_status": load_status,
        },
    )
    return s, load_status


def lag_series(s: pd.Series, sid: str) -> pd.Series:
    if s.empty:
        return s
    lag = FRED_SERIES[sid][2]
    out = s.copy()
    if lag == "1W":
        out.index = out.index + pd.DateOffset(weeks=1)
    elif lag == "1M":
        out.index = out.index + pd.DateOffset(months=1)
    return out[~out.index.duplicated(keep="last")].sort_index()


def build_fred_panel(spy_dates: pd.DatetimeIndex, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    series = {}
    for sid in FRED_SERIES:
        s, status = fetch_fred_series(sid, args.start_date, spy_dates.max().date().isoformat(), Path(args.fred_cache_dir), args.refresh_fred)
        rows.append({"series_id": sid, "status": status, "rows": int(s.notna().sum())})
        if len(s):
            series[sid] = lag_series(s, sid)
    daily = pd.DataFrame(index=spy_dates)
    for sid, s in series.items():
        daily[sid] = s.reindex(spy_dates)
    daily = daily.ffill()
    daily.index.name = "date"
    weekly = daily.resample("W-FRI").last()
    if getattr(args, "keep_cache", True):
        (Path(args.fred_cache_dir) / "processed").mkdir(parents=True, exist_ok=True)
        daily.to_csv(Path(args.fred_cache_dir) / "processed" / "fred_macro_daily_panel.csv")
        weekly.to_csv(Path(args.fred_cache_dir) / "processed" / "fred_macro_weekly_panel.csv")
    return daily, weekly, pd.DataFrame(rows)


def build_market_panel(args: argparse.Namespace) -> pd.DataFrame:
    spy = load_price("SPY", ["SPY_20y.csv", "SPY.csv", "SPY_max.csv"], args.allow_yfinance)
    source_rows = [price_audit_row("SPY", spy)]
    panel = spy.copy()
    for sym, names in {
        "VIX": ["^VIX_20y.csv", "VIX_20y.csv"],
        "QQQ": ["QQQ_20y.csv", "QQQ.csv"],
        "GLD": ["GLD_20y.csv", "GLD.csv"],
        "SOXX": ["SOXX_20y.csv", "SOXX.csv"],
        "XLU": ["XLU_20y.csv", "XLU.csv"],
    }.items():
        px = load_price(sym, names)
        if sym == "VIX":
            source_rows.append(price_audit_row("VIX local CSV", px))
        if len(px):
            panel = panel.merge(px[["date", f"{sym}_close"]], on="date", how="left")
    panel = panel.sort_values("date").ffill()
    close = panel["SPY_close"]
    panel["spy_close"] = close
    panel["spy_return_1d"] = close.pct_change()
    for h in [5, 10, 21, 42, 63]:
        panel[f"spy_return_{h}d"] = close.pct_change(h)
    for w in [20, 50, 100, 200]:
        panel[f"MA{w}"] = close.rolling(w, min_periods=max(5, w // 2)).mean()
        panel[f"distance_to_MA{w}"] = close / panel[f"MA{w}"] - 1
    panel["MA20_slope_5d"] = panel["MA20"].diff(5)
    panel["MA50_slope_10d"] = panel["MA50"].diff(10)
    panel["MA100_slope_21d"] = panel["MA100"].diff(21)
    panel["realized_vol_21d"] = panel["spy_return_1d"].rolling(21, min_periods=10).std() * math.sqrt(252)
    panel["realized_vol_63d"] = panel["spy_return_1d"].rolling(63, min_periods=30).std() * math.sqrt(252)
    for w in [21, 42, 63]:
        panel[f"trailing_{w}d_low"] = close.rolling(w, min_periods=5).min()
    panel["drawdown_from_63d_high"] = close / close.rolling(63, min_periods=10).max() - 1
    panel["drawdown_from_126d_high"] = close / close.rolling(126, min_periods=20).max() - 1
    if "VIX_close" in panel:
        panel["vix_level"] = panel["VIX_close"]
    for h in [5, 10, 21]:
        panel[f"vix_change_{h}d"] = panel["vix_level"].diff(h) if "vix_level" in panel else np.nan
    panel["vix_10d_ma"] = panel["vix_level"].rolling(10, min_periods=5).mean() if "vix_level" in panel else np.nan
    panel["vix_above_10d_ma"] = panel["vix_level"] > panel["vix_10d_ma"] if "vix_level" in panel else False
    panel["vix_falling_5d"] = panel["vix_change_5d"] < 0
    panel["vix_falling_10d"] = panel["vix_change_10d"] < 0
    out = panel.set_index("date")
    out.attrs["source_audit_rows"] = source_rows
    return out


def add_regime_layer(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    downside = out["spy_return_1d"].clip(upper=0)
    out["downside_dev"] = np.sqrt((downside**2).ewm(halflife=20, adjust=False, min_periods=20).mean())
    out["downside_threshold"] = out["downside_dev"].rolling(252, min_periods=120).quantile(0.80)
    out["downside_risk_active"] = (out["downside_dev"] > out["downside_threshold"]) & (out["spy_return_21d"] < 0)
    pos1 = out["spy_return_1d"].gt(0.01).rolling(10, min_periods=5).max().astype(bool)
    neg1 = out["spy_return_1d"].lt(-0.01).rolling(10, min_periods=5).max().astype(bool)
    abs5 = out["spy_return_1d"].abs().gt(0.01).rolling(5, min_periods=3).sum()
    out["recent_whipsaw"] = (pos1 & neg1) | (abs5 >= 2)
    rv_med = out["realized_vol_21d"].rolling(252, min_periods=120).median()
    rv_high = out["realized_vol_21d"] > rv_med
    riskoff = out["spy_close"].lt(out["MA200"]) & (rv_high | out["downside_risk_active"])
    prebreak = out["distance_to_MA200"].between(-0.02, 0.03) | ((out["spy_close"] < out["MA100"]) & (out["distance_to_MA200"] > -0.03)) | (out["downside_risk_active"] & out["distance_to_MA200"].between(-0.05, 0.05))
    yellow = (out["spy_close"] > out["MA200"]) & (rv_high | out["recent_whipsaw"] | (out["spy_return_21d"] < 0) | (out["vix_change_10d"] > 0) | (out["distance_to_MA100"] < 0.01))
    green = (out["spy_close"] > out["MA100"]) & (out["spy_close"] > out["MA200"]) & (~out["downside_risk_active"]) & (~rv_high.fillna(False))
    regime = pd.Series("SPY_GREEN", index=out.index)
    regime.loc[yellow] = "SPY_YELLOW"
    regime.loc[prebreak] = "SPY_PREBREAK"
    regime.loc[riskoff] = "SPY_RISKOFF"
    regime.loc[~(riskoff | prebreak | yellow | green)] = "SPY_YELLOW"
    out["macro_spy_regime_normalized"] = regime
    out["spy_final_state_label"] = regime
    out["spy_final_daily_label"] = regime
    return out


def add_macro_causes(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    for col in ["DGS10", "T10YIE", "DFII10", "DCOILWTICO", "DGS2"]:
        if col not in out:
            out[col] = np.nan
    out["DGS10_chg_21d"] = out["DGS10"].diff(21)
    out["T10YIE_chg_21d"] = out["T10YIE"].diff(21)
    out["DFII10_chg_21d"] = out["DFII10"].diff(21)
    out["WTI_ret_21d"] = out["DCOILWTICO"].pct_change(21)
    out["WTI_ret_63d"] = out["DCOILWTICO"].pct_change(63)
    wti_abs = out["DCOILWTICO"].pct_change().abs()
    qqq_rel = out["QQQ_close"].pct_change(63) - out["spy_return_63d"] if "QQQ_close" in out else pd.Series(np.nan, index=out.index)
    gld_rel = out["GLD_close"].pct_change(63) - out["spy_return_63d"] if "GLD_close" in out else pd.Series(np.nan, index=out.index)
    causes = []
    for idx, r in out.iterrows():
        cause = "CAUSE_MACRO_CALM"
        if wti_abs.loc[idx] >= 0.05 or (wti_abs.loc[:idx].tail(21) >= 0.03).sum() >= 3:
            cause = "CAUSE_WTI_TAIL"
        elif r["WTI_ret_21d"] > 0.15 or r["WTI_ret_63d"] > 0.20:
            cause = "CAUSE_OIL_INFLATION"
        elif r["DFII10_chg_21d"] > 0.15:
            cause = "CAUSE_REAL_YIELD_TIGHTENING"
        elif r["DGS10_chg_21d"] > 0.15 and r["spy_return_21d"] < 0:
            cause = "CAUSE_HOSTILE_RATE_UP"
        elif r["DGS10_chg_21d"] < -0.15 and r["T10YIE_chg_21d"] < 0 and r["spy_return_21d"] < 0:
            cause = "CAUSE_RECESSION_FEAR"
        elif r["DGS10"] >= 4.75 or (r["DGS10"] >= 4.5 and (out["DGS10"].loc[:idx].tail(20) >= 4.5).all()):
            cause = "CAUSE_RATE_BURDEN"
        elif pd.notna(gld_rel.loc[idx]) and gld_rel.loc[idx] > 0.03 and r.get("vix_change_21d", 0) > 0:
            cause = "CAUSE_DEFENSIVE_ROTATION"
        elif r["DGS10_chg_21d"] > 0 and r["T10YIE_chg_21d"] > 0 and r["spy_return_21d"] > 0 and r["DFII10_chg_21d"] < 0.15:
            cause = "CAUSE_GROWTH_REFLATION"
        elif (pd.notna(qqq_rel.loc[idx]) and qqq_rel.loc[idx] > 0.03) or (r["spy_close"] > r["MA100"] and r["spy_close"] > r["MA200"]):
            cause = "CAUSE_TECH_LEADERSHIP"
        causes.append(cause)
    out["macro_cause_primary"] = causes
    out["current_macro_cause"] = out["macro_cause_primary"]
    return out


def build_weekly_features(daily: pd.DataFrame, outdir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w = daily.resample("W-FRI").last()
    feat = pd.DataFrame(index=w.index)
    inv = []

    def add(name: str, s: pd.Series, group: str) -> None:
        if s is not None and s.notna().sum() >= 80:
            feat[name] = s.replace([np.inf, -np.inf], np.nan)
            inv.append({"feature": name, "group": group, "non_null": int(s.notna().sum())})

    for col, group in [
        ("DGS2", "rates"), ("DGS10", "rates"), ("DFII10", "real_yield"), ("T5YIE", "inflation"), ("T10YIE", "inflation"),
        ("DCOILWTICO", "oil"), ("VIXCLS", "vix"), ("vix_level", "vix"), ("BAA10Y", "credit"), ("BAMLH0A0HYM2", "credit"),
        ("NFCI", "credit"), ("ANFCI", "credit"), ("FEDFUNDS", "rates"), ("WALCL", "liquidity"), ("M2SL", "liquidity"),
        ("UMCSENT", "sentiment"), ("UNRATE", "labor"), ("ICSA", "labor"), ("INDPRO", "growth"), ("CFNAI", "growth"),
        ("HOUST", "housing"), ("PERMIT", "housing"),
    ]:
        if col in w:
            s = w[col]
            add(f"{col}_level", s, group)
            add(f"{col}_diff_4w", s.diff(4), group)
            if (s > 0).mean() > 0.8:
                add(f"{col}_log", np.log(s.where(s > 0)), group)
                add(f"{col}_pct_4w", s.pct_change(4), group)
                add(f"{col}_pct_13w", s.pct_change(13), group)
            add(f"{col}_ewma_hl4", s.ewm(halflife=4, adjust=False, min_periods=4).mean(), group)
            add(f"{col}_ewma_hl8", s.ewm(halflife=8, adjust=False, min_periods=8).mean(), group)
            add(f"{col}_roll52_z", (s - s.rolling(52, min_periods=30).mean()) / s.rolling(52, min_periods=30).std(), group)
            exp_mu = s.expanding(52).mean().shift(1)
            exp_sd = s.expanding(52).std().shift(1)
            add(f"{col}_expanding_z", (s - exp_mu) / exp_sd, group)
    if {"DGS10", "DGS2"}.issubset(w.columns):
        slope = w["DGS10"] - w["DGS2"]
        add("DGS10_DGS2_slope", slope, "yield_curve")
        add("DGS10_DGS2_slope_4w_change", slope.diff(4), "yield_curve")
    if "DCOILWTICO" in w:
        add("oil_ret_4w", w["DCOILWTICO"].pct_change(4), "oil")
        add("oil_ret_13w", w["DCOILWTICO"].pct_change(13), "oil")
    add("SPY_ret_4w", w["spy_close"].pct_change(4), "market")
    add("SPY_ret_13w", w["spy_close"].pct_change(13), "market")
    add("SPY_distance_to_MA200", w["distance_to_MA200"], "market")
    add("SPY_realized_vol_21d", w["realized_vol_21d"], "market")
    feat = feat.dropna(axis=1, how="all")
    feat.index.name = "date"
    inv_df = pd.DataFrame(inv)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        feat.to_csv(outdir / "macro_feature_panel_weekly.csv")
        inv_df.to_csv(outdir / "macro_feature_inventory.csv", index=False)
    return feat, inv_df


def feature_group(name: str) -> str:
    n = name.lower()
    if "vix" in n:
        return "vix"
    if any(x in n for x in ["dgs", "fedfunds"]):
        return "rates"
    if "dfii" in n:
        return "real_yield"
    if any(x in n for x in ["baa", "baml", "nfci", "anfci"]):
        return "credit"
    if "oil" in n or "dcoil" in n:
        return "oil"
    if any(x in n for x in ["walcl", "m2sl"]):
        return "liquidity"
    if any(x in n for x in ["unrate", "icsa"]):
        return "labor"
    if any(x in n for x in ["indpro", "cfnai"]):
        return "growth"
    if any(x in n for x in ["houst", "permit"]):
        return "housing"
    if "umcsent" in n:
        return "sentiment"
    if any(x in n for x in ["t5yie", "t10yie", "cpi"]):
        return "inflation"
    if any(x in n for x in ["spy", "realized_vol"]):
        return "market"
    return "other"


def universe_features(feat: pd.DataFrame, universe: str) -> List[str]:
    cols = []
    for c in feat.columns:
        g = feature_group(c)
        if universe == "VIX_ONLY" and g == "vix":
            cols.append(c)
        elif universe == "FULL_PUBLIC_MACRO":
            cols.append(c)
        elif universe == "NO_VIX" and g not in {"vix"} and "realized_vol" not in c.lower():
            cols.append(c)
        elif universe == "RATES_CREDIT_ONLY" and g in {"rates", "real_yield", "credit", "yield_curve"}:
            cols.append(c)
        elif universe == "GROWTH_ONLY" and g in {"growth", "labor", "sentiment", "housing", "liquidity"}:
            cols.append(c)
    # prefer stable columns with broad coverage
    coverage = feat[cols].notna().sum().sort_values(ascending=False) if cols else pd.Series(dtype=int)
    return coverage.head(12).index.tolist()


def transform_for_cjm(raw: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    x = raw[features].replace([np.inf, -np.inf], np.nan).copy()
    cols = [c for c in x.columns if x[c].notna().sum() >= 156]
    x = x[cols].ffill()
    x = x.fillna(x.median())
    try:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        arr = pt.fit_transform(x)
        x = pd.DataFrame(arr, index=x.index, columns=x.columns)
    except Exception:
        pass
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns).dropna()


def fit_cjm(raw: pd.DataFrame, universe: str, k: int, lam: float, random_state: int) -> CJMFit:
    features = universe_features(raw, universe)
    if len(features) < 2:
        raise RuntimeError(f"Not enough features for {universe}")
    x = transform_for_cjm(raw, features)
    if len(x) < 156:
        raise RuntimeError(f"Not enough weekly rows for {universe}")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=30)
    km.fit(x)
    dist = ((x.values[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    temp = max(float(np.nanmedian(dist)), 1e-12)
    raw_prob = softmax(-dist / temp, axis=1)
    alpha = lam / (lam + 100.0)
    probs = np.zeros_like(raw_prob)
    probs[0] = raw_prob[0]
    for i in range(1, len(raw_prob)):
        probs[i] = (1 - alpha) * raw_prob[i] + alpha * probs[i - 1]
        probs[i] = probs[i] / probs[i].sum()
    prob_df = pd.DataFrame(probs, index=x.index, columns=[f"state_{i}_probability" for i in range(k)])
    labels = pd.Series(probs.argmax(axis=1), index=x.index, name="cjm_state")
    return CJMFit(labels=labels, probabilities=prob_df, features=features)


def label_frame(universe: str, fit: CJMFit) -> pd.DataFrame:
    out = fit.labels.to_frame().join(fit.probabilities)
    out.insert(0, "universe", universe)
    out["max_probability"] = fit.probabilities.max(axis=1)
    out["transition_flag"] = out["cjm_state"].ne(out["cjm_state"].shift()).fillna(False)
    ages, age = [], 0
    for flag in out["transition_flag"]:
        age = 0 if flag else age + 1
        ages.append(age)
    out["weeks_since_transition"] = ages
    out.index.name = "date"
    cols = ["universe", "cjm_state", "max_probability"] + [c for c in out.columns if c.startswith("state_")] + ["transition_flag", "weeks_since_transition"]
    return out[cols].reset_index()


def characterize_states(daily: pd.DataFrame, weekly: pd.DataFrame, fits: Dict[str, CJMFit]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, meta = [], []
    wdiag = daily.resample("W-FRI").last()
    for universe, fit in fits.items():
        joined = wdiag.reindex(fit.labels.index).join(fit.labels.rename("state"))
        state_means = joined.groupby("state").mean(numeric_only=True)
        for state, g in joined.groupby("state"):
            row = {"universe": universe, "cjm_state": int(state), "count": len(g)}
            for col in ["vix_level", "vix_change_21d", "spy_return_13w", "realized_vol_21d", "DGS10", "DFII10", "DGS10_DGS2_slope", "BAMLH0A0HYM2", "BAA10Y", "oil_ret_13w", "UNRATE", "ICSA", "WALCL", "M2SL", "distance_to_MA200"]:
                row[f"mean_{col}"] = g[col].mean() if col in g else np.nan
            rows.append(row)
        temp = pd.DataFrame([r for r in rows if r["universe"] == universe])
        if temp.empty:
            continue
        high_vix_state = temp["mean_vix_level"].idxmax() if "mean_vix_level" in temp and temp["mean_vix_level"].notna().any() else None
        low_vix_state = temp["mean_vix_level"].idxmin() if "mean_vix_level" in temp and temp["mean_vix_level"].notna().any() else None
        for idx, r in temp.iterrows():
            name, broad, sev = "mixed_transition", "MIXED", "pressure"
            spy_ok = np.nanmean([r.get("mean_spy_return_13w", np.nan), r.get("mean_distance_to_MA200", np.nan)]) >= 0
            if universe == "VIX_ONLY":
                if idx == high_vix_state and r.get("mean_vix_change_21d", 0) > 0:
                    name, broad, sev = "vix_stress", "VIX_STRESS", "stress"
                elif idx == high_vix_state:
                    name, broad, sev = "vix_repair", "VIX_REPAIR", "repair"
                elif idx == low_vix_state:
                    name, broad, sev = "low_vol_expansion", "CALM", "calm"
            elif universe == "RATES_CREDIT_ONLY":
                credit_high = r.get("mean_BAMLH0A0HYM2", -np.inf) >= temp.get("mean_BAMLH0A0HYM2", pd.Series(dtype=float)).median()
                if credit_high and not spy_ok:
                    name, broad, sev = "rates_credit_stress", "RATES_CREDIT_STRESS", "stress"
                elif spy_ok:
                    name, broad, sev = "high_rate_growth", "HIGH_RATE_GROWTH", "growth"
                else:
                    name, broad, sev = "macro_pressure", "MACRO_PRESSURE", "pressure"
            elif universe == "GROWTH_ONLY":
                labor_high = r.get("mean_ICSA", -np.inf) >= temp.get("mean_ICSA", pd.Series(dtype=float)).median()
                name, broad, sev = ("labor_stress", "MACRO_STRESS", "stress") if labor_high and not spy_ok else ("macro_pressure", "MACRO_PRESSURE", "pressure")
            else:
                labor_credit = np.nanmean([r.get("mean_ICSA", np.nan), r.get("mean_UNRATE", np.nan), r.get("mean_BAMLH0A0HYM2", np.nan)])
                oil_high = r.get("mean_oil_ret_13w", -np.inf) >= temp.get("mean_oil_ret_13w", pd.Series(dtype=float)).quantile(0.67)
                if labor_credit >= np.nanmedian([np.nanmean([x.get("mean_ICSA", np.nan), x.get("mean_UNRATE", np.nan), x.get("mean_BAMLH0A0HYM2", np.nan)]) for _, x in temp.iterrows()]) and not spy_ok:
                    name, broad, sev = "macro_stress", "MACRO_STRESS", "stress"
                elif oil_high:
                    name, broad, sev = "oil_reflation", "OIL_REFLATION", "growth" if spy_ok else "pressure"
                elif spy_ok and r.get("mean_DGS10", 0) >= temp.get("mean_DGS10", pd.Series(dtype=float)).median():
                    name, broad, sev = "high_rate_growth", "HIGH_RATE_GROWTH", "growth"
                else:
                    name, broad, sev = "macro_pressure", "MACRO_PRESSURE", "pressure"
            meta.append({"universe": universe, "cjm_state": int(r["cjm_state"]), "cjm_state_name": name, "broad_meta_label": broad, "severity_label": sev})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def map_cjm_daily(daily: pd.DataFrame, fits: Dict[str, CJMFit], meta: pd.DataFrame, outdir: Optional[Path] = None) -> pd.DataFrame:
    out = pd.DataFrame(index=daily.index)
    audits = []
    dates = pd.DataFrame({"date": daily.index})
    for u, fit in fits.items():
        lab = label_frame(u, fit).merge(meta[meta["universe"].eq(u)], on=["universe", "cjm_state"], how="left")
        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            lab.to_csv(outdir / f"cjm_labels_{u}.csv", index=False)
            fit.probabilities.reset_index().assign(universe=u).to_csv(outdir / f"cjm_probabilities_{u}.csv", index=False)
        mapped = pd.merge_asof(dates, lab.sort_values("date"), on="date", direction="backward").set_index("date")
        for col in ["cjm_state", "broad_meta_label", "severity_label", "max_probability", "transition_flag", "weeks_since_transition"]:
            out[f"{u}_{'state' if col == 'cjm_state' else col}"] = mapped[col]
        audits.append({"universe": u, "weekly_rows": len(lab), "daily_rows": mapped[col].notna().sum(), "features": "|".join(fit.features)})
    if outdir is not None:
        pd.DataFrame(audits).to_csv(outdir / "cjm_universe_daily_mapping_audit.csv", index=False)
    return out


def bucket(row: pd.Series) -> str:
    vix, full = str(row.get("VIX_ONLY_severity_label", "")).lower(), str(row.get("FULL_PUBLIC_MACRO_severity_label", "")).lower()
    vb = str(row.get("VIX_ONLY_broad_meta_label", ""))
    if vix == "calm" and full in {"calm", "growth"}:
        return "BOTH_CALM"
    if vix in {"pressure", "repair"} and full in {"pressure", "repair"}:
        return "BOTH_PRESSURE"
    if vix == "stress" and full == "stress":
        return "BOTH_STRESS"
    if full == "pressure" and vix in {"calm", "growth"}:
        return "MACRO_PRESSURE__VIX_CALM"
    if full == "stress" and vix in {"calm", "growth"}:
        return "MACRO_STRESS__VIX_CALM"
    if vb == "VIX_STRESS" and full == "pressure":
        return "VIX_STRESS__MACRO_PRESSURE"
    if vb in {"VIX_REPAIR", "MIXED"} and full == "pressure":
        return "VIX_REPAIR__MACRO_PRESSURE"
    if vb in {"VIX_REPAIR", "MIXED"} and full == "stress":
        return "VIX_REPAIR__MACRO_STRESS"
    return "OTHER_DISAGREEMENT"


def phase_reference_table() -> pd.DataFrame:
    rows = [
        ("CLEAN_GREEN", "Clean/normal tape.", "dashboard_price_tape", "normal", "calm", "SPY tape is healthy and VIX is calm. Macro pressure may exist but is not disrupting price/volatility behavior.", "It does not mean macro risk is absent.", "SPY_GREEN", "SPY tape is clean and market volatility is calm."),
        ("MACRO_PRESSURE_ACCEPTED", "Macro pressure exists, but price/tape is still accepting it.", "macro_cjm", "slow", "pressure", "Full macro or no-VIX CJM shows pressure, but SPY regime and VIX layer are not confirming active stress.", "It is not an active crash or direct bearish label.", "SPY_GREEN/SPY_YELLOW", "Macro pressure is present, but the market is still absorbing it."),
        ("MACRO_STRESS_NO_MARKET_CONFIRMATION", "Macro stress is active, but price/VIX confirmation is weak.", "macro_cjm", "slow", "pressure", "Full macro CJM remains stressed while SPY tape and VIX are not confirming a clean market stress state.", "It does not mean immediate downside or active crash.", "SPY_GREEN/SPY_YELLOW", "Macro stress remains, but market-volatility confirmation is weak."),
        ("FAST_VIX_SHOCK", "Fast volatility shock.", "vix_cjm", "fast", "shock", "VIX_ONLY CJM moved into stress while SPY dashboard is yellow/prebreak/riskoff.", "It is not a forward return forecast; it is a movement-risk / fast-shock phase.", "SPY_YELLOW/SPY_PREBREAK/SPY_RISKOFF", "Fast volatility shock is active."),
        ("VIX_SHOCK_INSIDE_MACRO_PRESSURE", "Early VIX shock while slow macro pressure is present.", "vix_cjm", "fast", "pressure", "VIX layer shows stress, but macro layer is pressure rather than full stress. Often captures fast volatility impulse before broader confirmation.", "It does not automatically mean the tape is in broad stress.", "SPY_YELLOW/SPY_PREBREAK", "VIX shock is active inside a macro-pressure background."),
        ("BROAD_STRESS", "Broad stress agreement.", "all_confirm", "mixed", "broad_stress", "VIX and full macro CJM both show stress; often coincides with prebreak/riskoff tape or stress climax.", "It is not a directional forecast. It may appear near stress exhaustion as well as during active stress.", "SPY_PREBREAK/SPY_RISKOFF", "VIX and macro CJM both confirm stress."),
        ("LATE_STRESS_REPAIR", "Post-shock repair under remaining macro pressure.", "vix_cjm_repair", "repair", "repair", "VIX is falling or no longer in clean stress, but macro pressure/stress remains. Often a late-stress or repair phase.", "It does not mean macro pressure is gone.", "SPY_YELLOW/SPY_PREBREAK/SPY_RISKOFF", "Volatility is repairing, but macro pressure remains."),
        ("OTHER_REPAIR_CHOP", "Unresolved repair/chop state.", "mixed", "repair", "mixed", "Model layers disagree, VIX is not rising, and price behavior is stabilizing or choppy.", "It is not clean stress by itself.", "Mixed", "Mixed repair/chop state; VIX is not confirming renewed stress."),
        ("OTHER_STRESS_RELAPSE", "Unresolved disagreement with renewed VIX pressure.", "mixed", "fast", "pressure", "Model layers disagree, but VIX is rising again and SPY tape is not clean.", "It is not broad stress unless macro and VIX both confirm stress.", "SPY_YELLOW/SPY_PREBREAK/SPY_RISKOFF", "Mixed disagreement with renewed VIX pressure."),
        ("YELLOW_VOL_WARNING", "Yellow tape with volatility warning.", "dashboard_price_tape + vix_cjm", "fast", "pressure", "SPY is in yellow/whipsaw mode and VIX layer is pressure/stress.", "It is not full risk-off or broad stress.", "SPY_YELLOW", "Yellow tape with volatility warning."),
        ("HIGH_RATE_GROWTH", "High-rate environment still absorbed by market.", "rates_credit_cjm", "slow", "growth", "Rates/credit CJM sees high-rate growth, while price and VIX are not confirming stress.", "It is not a risk-off label by itself.", "SPY_GREEN/SPY_YELLOW", "High-rate growth regime; market is still absorbing rates."),
        ("MIXED_TRANSITION", "Unstable transition bucket.", "mixed", "mixed", "mixed", "Price/tape, VIX, and macro layers do not align cleanly. Historically this bucket has weaker daily behavior and needs context.", "It is not a precise regime label. It should not be overinterpreted.", "Mixed", "Mixed transition state; model layers do not align cleanly."),
    ]
    return pd.DataFrame(rows, columns=["phase", "plain_english_description", "primary_driver", "time_scale", "stress_level", "what_it_means", "what_it_does_not_mean", "typical_dashboard_regime", "daily_report_phrase"])


def composite_phase(row: pd.Series) -> Tuple[str, str]:
    b = row["vix_macro_disagreement_bucket_v2"]
    vix_sev = str(row.get("VIX_ONLY_severity_label", "")).lower()
    full_sev = str(row.get("FULL_PUBLIC_MACRO_severity_label", "")).lower()
    vix_b = row.get("VIX_ONLY_broad_meta_label", "")
    regime = row.get("macro_spy_regime_normalized", "")
    down_ok = not bool(row.get("downside_risk_active", False))
    if b == "BOTH_STRESS" and vix_sev == "stress" and full_sev == "stress":
        return "BROAD_STRESS", "BOTH_STRESS override: VIX and full macro CJM both show stress."
    if b == "VIX_STRESS__MACRO_PRESSURE":
        return "VIX_SHOCK_INSIDE_MACRO_PRESSURE", "VIX stress inside macro-pressure bucket."
    if regime in {"SPY_YELLOW", "SPY_PREBREAK", "SPY_RISKOFF"} and vix_b == "VIX_STRESS":
        return "FAST_VIX_SHOCK", "VIX stress with non-clean SPY tape."
    if regime in {"SPY_GREEN", "SPY_YELLOW"} and vix_sev in {"calm", "pressure"} and full_sev == "pressure" and down_ok:
        return "MACRO_PRESSURE_ACCEPTED", "Macro pressure accepted by price/VIX layer."
    if regime in {"SPY_GREEN", "SPY_YELLOW"} and full_sev == "stress" and vix_sev in {"calm", "pressure"} and down_ok:
        return "MACRO_STRESS_NO_MARKET_CONFIRMATION", "Macro stress without price/VIX confirmation."
    if regime in {"SPY_PREBREAK", "SPY_RISKOFF", "SPY_YELLOW"} and vix_b in {"VIX_REPAIR", "MIXED"} and full_sev in {"pressure", "stress"} and (row.get("vix_change_10d", 0) < 0 or row.get("vix_change_21d", 0) < 0):
        return "LATE_STRESS_REPAIR", "VIX repair while macro pressure/stress remains."
    if b == "OTHER_DISAGREEMENT" and row.get("vix_change_10d", np.nan) <= 0 and (row.get("spy_return_5d", np.nan) > 0 or row.get("spy_return_10d", np.nan) > 0) and down_ok:
        return "OTHER_REPAIR_CHOP", "OTHER_DISAGREEMENT with non-rising VIX and positive short-term tape."
    if b == "OTHER_DISAGREEMENT" and row.get("vix_change_10d", np.nan) > 0 and regime in {"SPY_YELLOW", "SPY_PREBREAK", "SPY_RISKOFF"}:
        return "OTHER_STRESS_RELAPSE", "OTHER_DISAGREEMENT with rising VIX and non-clean tape."
    if row.get("RATES_CREDIT_ONLY_broad_meta_label") == "HIGH_RATE_GROWTH" and regime in {"SPY_GREEN", "SPY_YELLOW"} and vix_sev != "stress":
        return "HIGH_RATE_GROWTH", "Rates-credit CJM shows high-rate growth."
    if regime == "SPY_GREEN" and vix_sev == "calm" and full_sev in {"calm", "growth", "pressure"} and down_ok:
        return "CLEAN_GREEN", "Clean SPY tape and calm VIX layer."
    if regime == "SPY_YELLOW" and vix_sev in {"pressure", "stress"}:
        return "YELLOW_VOL_WARNING", "Yellow tape with VIX pressure/stress."
    return "MIXED_TRANSITION", "Fallback mixed transition state."


def stress_level(phase: str) -> str:
    if phase == "BROAD_STRESS":
        return "broad_stress"
    if "SHOCK" in phase:
        return "shock" if phase == "FAST_VIX_SHOCK" else "pressure"
    if "REPAIR" in phase:
        return "repair"
    if "PRESSURE" in phase or "WARNING" in phase:
        return "pressure"
    if phase in {"CLEAN_GREEN", "HIGH_RATE_GROWTH"}:
        return "calm" if phase == "CLEAN_GREEN" else "growth"
    return "mixed"


def time_scale(phase: str) -> str:
    if "SHOCK" in phase or "VIX" in phase or "YELLOW" in phase:
        return "fast"
    if "MACRO" in phase or "RATE" in phase:
        return "slow"
    if "REPAIR" in phase:
        return "repair"
    return "mixed"


def add_composite(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["vix_macro_disagreement_bucket_v2"] = out.apply(bucket, axis=1)
    out["vix_macro_agree_flag"] = out["VIX_ONLY_severity_label"].eq(out["FULL_PUBLIC_MACRO_severity_label"])
    out["vix_macro_disagreement_flag"] = ~out["vix_macro_agree_flag"]
    phases = out.apply(composite_phase, axis=1)
    out["composite_phase_label"] = [p[0] for p in phases]
    out["composite_phase_label_patched"] = out["composite_phase_label"]
    out["phase_patch_reason"] = [p[1] for p in phases]
    out["broad_stress_override_flag"] = out["composite_phase_label"].eq("BROAD_STRESS") & out["vix_macro_disagreement_bucket_v2"].eq("BOTH_STRESS")
    out["composite_stress_level"] = out["composite_phase_label"].map(stress_level)
    out["composite_time_scale"] = out["composite_phase_label"].map(time_scale)
    v, f = out["VIX_ONLY_max_probability"], out["FULL_PUBLIC_MACRO_max_probability"]
    out["composite_confidence"] = np.where((v >= 0.60) & (f >= 0.60), "high", np.where((v >= 0.45) & (f >= 0.45), "medium", "low"))
    return out


def add_forward_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["spy_close"]
    for h in FORWARD_HORIZONS:
        out[f"forward_{h}d_return"] = close.shift(-h) / close - 1
        vals = []
        for i in range(len(out)):
            vals.append(max_drawdown(close.iloc[i : i + h + 1]) if i + h < len(out) else np.nan)
        out[f"forward_{h}d_max_drawdown"] = vals
    return out


def max_drawdown(s: pd.Series) -> float:
    p = s.dropna()
    return float((p / p.cummax() - 1).min()) if len(p) else np.nan


def weekday_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if start >= end:
        return 0
    return int(sum(d.weekday() < 5 for d in pd.date_range(start + pd.Timedelta(days=1), end, freq="D")))


def latest_snapshot(df: pd.DataFrame, ref: pd.DataFrame, max_stale_trading_days: int = 2) -> pd.DataFrame:
    latest = df.iloc[-1]
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    latest_date = pd.Timestamp(latest.name)
    stale = weekday_count(latest_date, today)
    fresh = "fresh" if stale <= max_stale_trading_days else ("mildly_stale" if stale <= max_stale_trading_days + 3 else "stale")
    refd = ref.set_index("phase").to_dict("index").get(latest["composite_phase_label"], {})
    low_conf = latest["composite_confidence"] == "low" or latest["VIX_ONLY_max_probability"] < 0.45 or latest["FULL_PUBLIC_MACRO_max_probability"] < 0.45
    missing_macro = pd.isna(latest.get("macro_cause_primary"))
    stale_vix = pd.isna(latest.get("vix_level"))
    stale_cjm = pd.isna(latest.get("VIX_ONLY_broad_meta_label")) or pd.isna(latest.get("FULL_PUBLIC_MACRO_broad_meta_label"))
    bits = [fresh, "missing macro cause" if missing_macro else "macro cause present", "low CJM confidence" if low_conf else f"CJM confidence {latest['composite_confidence']}"]
    row = {
        "report_generated_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latest_data_date": latest_date.date().isoformat(),
        "latest_spy_close": latest["spy_close"],
        "data_staleness_calendar_days": (today - latest_date).days,
        "data_staleness_trading_days_estimate": stale,
        "data_freshness_flag": fresh,
        "macro_spy_regime": latest["macro_spy_regime_normalized"],
        "macro_cause": latest["macro_cause_primary"],
        "composite_phase_label_patched": latest["composite_phase_label"],
        "composite_stress_level": latest["composite_stress_level"],
        "composite_time_scale": latest["composite_time_scale"],
        "composite_confidence": latest["composite_confidence"],
        "phase_plain_english_description": refd.get("plain_english_description"),
        "phase_daily_report_phrase": refd.get("daily_report_phrase"),
        "phase_what_it_means": refd.get("what_it_means"),
        "phase_what_it_does_not_mean": refd.get("what_it_does_not_mean"),
        "vix_level": latest.get("vix_level"),
        "vix_change_10d": latest.get("vix_change_10d"),
        "vix_change_21d": latest.get("vix_change_21d"),
        "vix_macro_disagreement_bucket_v2": latest["vix_macro_disagreement_bucket_v2"],
        "broad_stress_override_flag": latest["broad_stress_override_flag"],
        "low_confidence_flag": bool(low_conf),
        "missing_macro_cause_flag": bool(missing_macro),
        "stale_spy_flag": fresh == "stale",
        "stale_vix_flag": bool(stale_vix),
        "stale_cjm_flag": bool(stale_cjm),
        "data_quality_summary": "; ".join(bits),
        "no_trading_advice_note": NO_ADVICE,
    }
    for u in UNIVERSES:
        for suffix in ["broad_meta_label", "severity_label", "max_probability"]:
            row[f"{u}_{suffix}"] = latest.get(f"{u}_{suffix}")
    return pd.DataFrame([row])


def data_quality(spy_loaded: bool, fred_log: pd.DataFrame, df: pd.DataFrame, snapshot: pd.DataFrame, fits: Dict[str, CJMFit], source_audit: pd.DataFrame) -> pd.DataFrame:
    s = snapshot.iloc[0]
    spy_audit = source_audit[source_audit["source_name"].eq("SPY")]
    vix_audit = source_audit[source_audit["source_name"].eq("VIX")]
    spy_detail = spy_audit.iloc[0].to_dict() if len(spy_audit) else {}
    spy_warning = spy_audit["warning"].iloc[0] if len(spy_audit) else ""
    vix_detail = vix_audit.iloc[0].to_dict() if len(vix_audit) else {}
    rows = [
        ("SPY source loaded", "ok" if spy_loaded else "error", "SPY price data is required."),
        ("SPY source freshness", "warning" if spy_warning else "ok", spy_warning or f"source={spy_detail.get('path_or_provider', '')}; latest={spy_detail.get('latest_date', s['latest_data_date'])}."),
        ("FRED cache loaded/fetched", "ok" if (fred_log["status"].astype(str).str.contains("failed").sum() < len(fred_log)) else "warning", f"Loaded/fetched {len(fred_log) - fred_log['status'].astype(str).str.contains('failed').sum()} series."),
        ("FRED series failures", "warning" if fred_log["status"].astype(str).str.contains("failed").any() else "ok", "; ".join(fred_log.loc[fred_log["status"].astype(str).str.contains("failed"), "series_id"].astype(str).tolist())),
        ("VIX source loaded", "ok" if df["vix_level"].notna().any() else "warning", f"source={vix_detail.get('path_or_provider', '')}; latest={vix_detail.get('latest_date', '')}; {vix_detail.get('warning', '')}".strip()),
        ("latest data not stale", "ok" if s["data_freshness_flag"] == "fresh" else "warning", str(s["data_freshness_flag"])),
        ("CJM labels built for all universes", "ok" if set(fits) == set(UNIVERSES) else "warning", ",".join(sorted(fits))),
        ("macro regime built", "ok" if df["macro_spy_regime_normalized"].notna().any() else "error", "Dashboard-like regime computed."),
        ("macro cause built", "ok" if df["macro_cause_primary"].notna().any() else "warning", "Macro cause labels computed."),
        ("composite phase exists", "ok" if df["composite_phase_label"].notna().any() else "error", "Composite labels computed."),
        ("no duplicate dates", "ok" if not df.index.duplicated().any() else "warning", f"duplicates={df.index.duplicated().sum()}"),
        ("latest snapshot complete", "ok" if snapshot.notna().sum(axis=1).iloc[0] > 10 else "warning", "Snapshot generated."),
        ("low confidence warning", "warning" if s["low_confidence_flag"] else "ok", f"confidence={s['composite_confidence']}"),
    ]
    return pd.DataFrame(rows, columns=["check_name", "status", "detail"])


def summarize_behavior(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    sample = df[df.index >= start]
    rows = []
    for phase, g in sample.groupby("composite_phase_label"):
        rows.append({
            "phase": phase, "count_days": len(g), "share": len(g) / max(len(sample), 1),
            "mean_daily_return": g["spy_return_1d"].mean(), "daily_volatility": g["spy_return_1d"].std(),
            "average_vix": g["vix_level"].mean(), "mean_forward_21d_return": g["forward_21d_return"].mean(),
            "mean_forward_21d_max_drawdown": g["forward_21d_max_drawdown"].mean(),
            "mean_forward_63d_return": g["forward_63d_return"].mean(),
            "mean_forward_63d_max_drawdown": g["forward_63d_max_drawdown"].mean(),
        })
    return pd.DataFrame(rows).sort_values("count_days", ascending=False)


def episode_outputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = df["composite_phase_label"].astype(str)
    groups = list(s.groupby(s.ne(s.shift()).cumsum()))
    rows = []
    for i, (_, gs) in enumerate(groups):
        st, en = gs.index[0], gs.index[-1]
        g = df.loc[st:en]
        rows.append({"start_date": st, "end_date": en, "trading_days": len(g), "phase": gs.iloc[0], "prior_phase": groups[i - 1][1].iloc[0] if i else np.nan, "next_phase": groups[i + 1][1].iloc[0] if i < len(groups) - 1 else np.nan, "SPY_return": g["spy_close"].iloc[-1] / g["spy_close"].iloc[0] - 1 if len(g) > 1 else 0, "max_drawdown": max_drawdown(g["spy_close"])})
    ep = pd.DataFrame(rows)
    post = ep[ep["start_date"] >= pd.Timestamp("2018-01-01")]
    trans = pd.crosstab(post["prior_phase"], post["phase"]).reset_index()
    dur = post.groupby("phase").agg(episodes=("phase", "size"), median_duration=("trading_days", "median"), max_duration=("trading_days", "max")).reset_index()
    return ep, trans, dur


def trough_alignment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    close = df["spy_close"]
    for w in [21, 42, 63]:
        trough = close.eq(close.rolling(w * 2 + 1, center=True, min_periods=w).min())
        for dt in close.index[trough.fillna(False) & (close.index >= pd.Timestamp("2018-01-01"))]:
            pos = close.index.get_loc(dt)
            rows.append({"trough_date": dt, "trough_type": f"trough_{w}d", "drawdown_from_63d_high": close.loc[dt] / close.iloc[max(0, pos - 63):pos + 1].max() - 1, "phase": df.loc[dt, "composite_phase_label"], "VIX_layer": df.loc[dt, "VIX_ONLY_broad_meta_label"], "macro_layer": df.loc[dt, "FULL_PUBLIC_MACRO_broad_meta_label"], "forward_21d_return": df.loc[dt, "forward_21d_return"]})
    return pd.DataFrame(rows)


def layer_attribution(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "BROAD_STRESS": ("all_confirm", "mixed"),
        "FAST_VIX_SHOCK": ("vix_cjm", "fast"),
        "VIX_SHOCK_INSIDE_MACRO_PRESSURE": ("vix_cjm", "fast"),
        "MACRO_PRESSURE_ACCEPTED": ("macro_cjm", "slow"),
        "MACRO_STRESS_NO_MARKET_CONFIRMATION": ("macro_cjm", "slow"),
        "LATE_STRESS_REPAIR": ("vix_cjm_repair", "repair"),
        "HIGH_RATE_GROWTH": ("rates_credit_cjm", "slow"),
        "CLEAN_GREEN": ("dashboard_price_tape", "normal"),
    }
    return pd.DataFrame([{"phase": p, "primary_driver": mapping.get(p, ("mixed", "mixed"))[0], "time_scale": mapping.get(p, ("mixed", "mixed"))[1], "count_days": len(g)} for p, g in df.groupby("composite_phase_label")])


def github_summary(snap: pd.DataFrame) -> str:
    s = snap.iloc[0]
    return f"""Daily Composite Regime Summary
==============================

**Latest data date:** {s['latest_data_date']}
**Data freshness:** {s['data_freshness_flag']}
**SPY close:** {s['latest_spy_close']:.2f}
**Macro SPY regime:** {s['macro_spy_regime']}
**Macro cause:** {s['macro_cause']}

**Composite phase:** {s['composite_phase_label_patched']}
**Plain-English read:** {s['phase_daily_report_phrase']}
**Stress level:** {s['composite_stress_level']}
**Time scale:** {s['composite_time_scale']}
**Confidence:** {s['composite_confidence']}

Layer Details
-------------

**VIX CJM:** {s['VIX_ONLY_broad_meta_label']} / {s['VIX_ONLY_severity_label']} / {s['VIX_ONLY_max_probability']:.3f}
**Full macro CJM:** {s['FULL_PUBLIC_MACRO_broad_meta_label']} / {s['FULL_PUBLIC_MACRO_severity_label']} / {s['FULL_PUBLIC_MACRO_max_probability']:.3f}
**No-VIX CJM:** {s['NO_VIX_broad_meta_label']} / {s['NO_VIX_severity_label']} / {s['NO_VIX_max_probability']:.3f}
**Rates-credit CJM:** {s['RATES_CREDIT_ONLY_broad_meta_label']} / {s['RATES_CREDIT_ONLY_severity_label']} / {s['RATES_CREDIT_ONLY_max_probability']:.3f}
**Growth CJM:** {s['GROWTH_ONLY_broad_meta_label']} / {s['GROWTH_ONLY_severity_label']} / {s['GROWTH_ONLY_max_probability']:.3f}

Interpretation
--------------

**What it means:** {s['phase_what_it_means']}
**What it does not mean:** {s['phase_what_it_does_not_mean']}

Data Quality
------------

**Quality summary:** {s['data_quality_summary']}

**No trading advice generated. Regime classification only.**
"""


def md_table(df: pd.DataFrame, n: int = 10) -> str:
    if df is None or df.empty:
        return "No rows."
    x = df.head(n).fillna("")
    lines = ["| " + " | ".join(map(str, x.columns)) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    for _, r in x.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in x.columns) + " |")
    return "\n".join(lines)


def report(snap: pd.DataFrame, ref: pd.DataFrame, qual: pd.DataFrame, beh: pd.DataFrame, dur: pd.DataFrame, trough: pd.DataFrame) -> str:
    s = snap.iloc[0]
    dist = beh[["phase", "share"]].merge(ref[["phase", "plain_english_description"]], on="phase", how="left")
    return f"""# Daily Composite Regime Panel — From Raw Build

## 1. Latest Snapshot

**Latest data date:** {s['latest_data_date']}

**SPY close:** {s['latest_spy_close']:.2f}

**Macro SPY regime:** {s['macro_spy_regime']}

**Macro cause:** {s['macro_cause']}

**Composite phase:** {s['composite_phase_label_patched']}

**Stress level:** {s['composite_stress_level']}

**Time scale:** {s['composite_time_scale']}

**Confidence:** {s['composite_confidence']}

**Plain-English read:** {s['phase_daily_report_phrase']}

**What it means:** {s['phase_what_it_means']}

**What it does not mean:** {s['phase_what_it_does_not_mean']}

## 2. Data Quality

{md_table(qual, 12)}

## 3. Post-2018 Phase Distribution

{md_table(dist.rename(columns={'plain_english_description': 'description'}), 12)}

## 4. Phase Behavior

{md_table(beh[['phase','mean_daily_return','daily_volatility','average_vix','mean_forward_21d_max_drawdown']], 12)}

## 5. Transition / Duration

{md_table(dur, 12)}

## 6. Trough Alignment

{md_table(trough.tail(12), 12)}

## 7. Final Notes

- Built directly from raw local market data and cached/FRED macro data.
- CJM labels are rebuilt without cvxpy and without prior composite CSVs.
- Forward returns and drawdowns are descriptive only.

**No trading advice generated. Regime classification only.**
"""


def safe_write_csv(df: pd.DataFrame, path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False if df.index.name is None else True)


def maybe_write_intermediate(df: pd.DataFrame, name: str, subfolder: Path, mode: str, write_intermediate: bool = False) -> None:
    if mode == "research" or write_intermediate:
        safe_write_csv(df, subfolder / name, mode)


def clean_daily_output_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for path in outdir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def add_phase_context(df: pd.DataFrame, ref: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "composite_phase_label_patched" not in out:
        out["composite_phase_label_patched"] = out["composite_phase_label"]
    phase_desc = ref.set_index("phase")["plain_english_description"].to_dict()
    out["phase_plain_english_description"] = out["composite_phase_label_patched"].map(phase_desc)
    out["data_quality_summary"] = snap.iloc[0].get("data_quality_summary", "")
    return out


def daily_tail_panel(df: pd.DataFrame, ref: pd.DataFrame, snap: pd.DataFrame, tail_days: int) -> pd.DataFrame:
    cols = [
        "date", "spy_close", "spy_return_1d", "spy_return_5d", "spy_return_21d",
        "vix_level", "vix_change_10d", "vix_change_21d",
        "macro_spy_regime_normalized", "macro_cause_primary", "downside_risk_active",
        "VIX_ONLY_broad_meta_label", "VIX_ONLY_severity_label", "VIX_ONLY_max_probability",
        "FULL_PUBLIC_MACRO_broad_meta_label", "FULL_PUBLIC_MACRO_severity_label", "FULL_PUBLIC_MACRO_max_probability",
        "NO_VIX_broad_meta_label", "NO_VIX_severity_label", "NO_VIX_max_probability",
        "RATES_CREDIT_ONLY_broad_meta_label", "RATES_CREDIT_ONLY_severity_label", "RATES_CREDIT_ONLY_max_probability",
        "GROWTH_ONLY_broad_meta_label", "GROWTH_ONLY_severity_label", "GROWTH_ONLY_max_probability",
        "vix_macro_disagreement_bucket_v2", "composite_phase_label_patched",
        "composite_stress_level", "composite_time_scale", "composite_confidence",
        "phase_plain_english_description", "data_quality_summary",
    ]
    tail = add_phase_context(df, ref, snap).tail(max(int(tail_days), 1)).reset_index()
    for col in cols:
        if col not in tail:
            tail[col] = np.nan
    return tail[cols]


def daily_markdown_report(snap: pd.DataFrame, qual: pd.DataFrame, tail: pd.DataFrame) -> str:
    s = snap.iloc[0]
    layer = pd.DataFrame([
        {"Layer": "Macro SPY", "Label": s["macro_spy_regime"], "Severity": "", "Confidence": ""},
        {"Layer": "VIX CJM", "Label": s["VIX_ONLY_broad_meta_label"], "Severity": s["VIX_ONLY_severity_label"], "Confidence": f"{s['VIX_ONLY_max_probability']:.3f}"},
        {"Layer": "Full macro CJM", "Label": s["FULL_PUBLIC_MACRO_broad_meta_label"], "Severity": s["FULL_PUBLIC_MACRO_severity_label"], "Confidence": f"{s['FULL_PUBLIC_MACRO_max_probability']:.3f}"},
        {"Layer": "No-VIX CJM", "Label": s["NO_VIX_broad_meta_label"], "Severity": s["NO_VIX_severity_label"], "Confidence": f"{s['NO_VIX_max_probability']:.3f}"},
        {"Layer": "Rates-credit CJM", "Label": s["RATES_CREDIT_ONLY_broad_meta_label"], "Severity": s["RATES_CREDIT_ONLY_severity_label"], "Confidence": f"{s['RATES_CREDIT_ONLY_max_probability']:.3f}"},
        {"Layer": "Growth CJM", "Label": s["GROWTH_ONLY_broad_meta_label"], "Severity": s["GROWTH_ONLY_severity_label"], "Confidence": f"{s['GROWTH_ONLY_max_probability']:.3f}"},
    ])
    warn = qual[qual["status"].isin(["warning", "error"])][["check_name", "status", "detail"]]
    warnings_md = "**Warnings:** None." if warn.empty else md_table(warn, 12)
    hist = tail.tail(20)[["date", "composite_phase_label_patched", "macro_spy_regime_normalized", "VIX_ONLY_broad_meta_label", "FULL_PUBLIC_MACRO_broad_meta_label"]].copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date.astype(str)
    hist = hist.rename(columns={
        "date": "Date",
        "composite_phase_label_patched": "Phase",
        "macro_spy_regime_normalized": "SPY regime",
        "VIX_ONLY_broad_meta_label": "VIX layer",
        "FULL_PUBLIC_MACRO_broad_meta_label": "Macro layer",
    })
    return f"""# Daily Composite Regime Report

## Latest Snapshot

**Latest data date:** {s['latest_data_date']}

**Composite phase:** {s['composite_phase_label_patched']}

**Macro SPY regime:** {s['macro_spy_regime']}

**Macro cause:** {s['macro_cause']}

**Data freshness:** {s['data_freshness_flag']}

## Layer Stack

{md_table(layer, 10)}

## Plain-English Read

**What it means:** {s['phase_what_it_means']}

**What it does not mean:** {s['phase_what_it_does_not_mean']}

## Data Quality

{warnings_md}

## Recent Phase History

{md_table(hist, 20)}

## Final Note

**No trading advice generated. Regime classification only.**
"""


def series_latest_date(df: pd.DataFrame, col: str) -> str:
    if col not in df:
        return ""
    idx = df.index[df[col].notna()]
    return idx.max().date().isoformat() if len(idx) else ""


def build_data_source_audit(market: pd.DataFrame, fred_daily: pd.DataFrame, daily: pd.DataFrame, vix_source: str, vix_warning: str) -> pd.DataFrame:
    rows = list(market.attrs.get("source_audit_rows", []))
    rows = [row for row in rows if row.get("source_name") != "VIX local CSV"]
    rows.append(
        {
            "source_name": "VIX",
            "path_or_provider": vix_source,
            "latest_date": series_latest_date(daily, "vix_level"),
            "rows": int(daily["vix_level"].notna().sum()) if "vix_level" in daily else 0,
            "status": "ok" if "vix_level" in daily and daily["vix_level"].notna().any() else "warning",
            "warning": vix_warning,
        }
    )
    rows.append(
        {
            "source_name": "FRED macro panel",
            "path_or_provider": "FRED cache/fetch",
            "latest_date": fred_daily.index.max().date().isoformat() if len(fred_daily) else "",
            "rows": int(len(fred_daily)),
            "status": "ok" if len(fred_daily) else "warning",
            "warning": "" if len(fred_daily) else "FRED macro panel is empty.",
        }
    )
    rows.append(
        {
            "source_name": "final daily panel",
            "path_or_provider": "in-memory daily build",
            "latest_date": daily.index.max().date().isoformat() if len(daily) else "",
            "rows": int(len(daily)),
            "status": "ok" if len(daily) else "error",
            "warning": "",
        }
    )
    return pd.DataFrame(rows, columns=["source_name", "path_or_provider", "latest_date", "rows", "status", "warning"])


def write_daily_outputs(outdir: Path, daily: pd.DataFrame, ref: pd.DataFrame, snap: pd.DataFrame, qual: pd.DataFrame, source_audit: pd.DataFrame, tail_days: int) -> List[Path]:
    clean_daily_output_dir(outdir)
    tail = daily_tail_panel(daily, ref, snap, tail_days)
    files = [
        outdir / "latest_composite_regime_snapshot.csv",
        outdir / "github_action_summary.txt",
        outdir / "daily_composite_regime_reviewer_report.md",
        outdir / "data_quality_flags.csv",
        outdir / "composite_phase_reference_table.csv",
        outdir / "daily_composite_regime_panel_tail.csv",
        outdir / "data_source_audit.csv",
    ]
    snap.to_csv(files[0], index=False)
    files[1].write_text(github_summary(snap), encoding="utf-8")
    files[2].write_text(daily_markdown_report(snap, qual, tail), encoding="utf-8")
    qual.to_csv(files[3], index=False)
    ref.to_csv(files[4], index=False)
    tail.to_csv(files[5], index=False)
    source_audit.to_csv(files[6], index=False)
    return files


def snapshot_diff(base_dir: Path, snap: pd.DataFrame) -> pd.DataFrame:
    prior_path = base_dir / "latest_composite_regime_snapshot.csv"
    rows = []
    current = snap.iloc[0]
    if prior_path.exists():
        try:
            prior = pd.read_csv(prior_path).iloc[0]
            for col in ["latest_data_date", "composite_phase_label_patched", "macro_spy_regime", "macro_cause", "data_freshness_flag"]:
                rows.append({"field": col, "prior": prior.get(col), "current": current.get(col), "changed": prior.get(col) != current.get(col)})
        except Exception as exc:
            rows.append({"field": "prior_snapshot_read", "prior": "", "current": str(exc), "changed": True})
    else:
        rows.append({"field": "prior_snapshot", "prior": "", "current": "not_found", "changed": False})
    return pd.DataFrame(rows)


def write_research_outputs(
    outdir: Path,
    daily: pd.DataFrame,
    fred_daily: pd.DataFrame,
    fred_weekly: pd.DataFrame,
    weekly_feat: pd.DataFrame,
    inv: pd.DataFrame,
    fits: Dict[str, CJMFit],
    meta: pd.DataFrame,
    char: pd.DataFrame,
    ref: pd.DataFrame,
    snap: pd.DataFrame,
    qual: pd.DataFrame,
    beh18: pd.DataFrame,
    beh20: pd.DataFrame,
    ep: pd.DataFrame,
    trans: pd.DataFrame,
    dur: pd.DataFrame,
    attr: pd.DataFrame,
    trough: pd.DataFrame,
    fred_cache: Path,
) -> List[Path]:
    subdirs = {name: outdir / name for name in ["current", "panels", "cjm", "diagnostics", "reference"]}
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    current = subdirs["current"]
    panels = subdirs["panels"]
    cjm = subdirs["cjm"]
    diag = subdirs["diagnostics"]
    refdir = subdirs["reference"]

    for path, df, index in [
        (current / "latest_composite_regime_snapshot.csv", snap, False),
        (current / "data_quality_flags.csv", qual, False),
        (panels / "daily_composite_regime_panel.csv", daily, True),
        (panels / "daily_composite_regime_panel_patched.csv", daily, True),
        (panels / "macro_feature_panel_weekly.csv", weekly_feat, True),
        (panels / "fred_macro_daily_panel.csv", fred_daily, True),
        (panels / "fred_macro_weekly_panel.csv", fred_weekly, True),
        (cjm / "cjm_universe_meta_labels.csv", meta, False),
        (cjm / "cjm_universe_state_characterization.csv", char, False),
        (diag / "composite_phase_behavior_summary_patched.csv", beh18, False),
        (diag / "composite_phase_behavior_post2020_patched.csv", beh20, False),
        (diag / "composite_phase_episode_summary_patched.csv", ep, False),
        (diag / "composite_phase_transition_matrix_patched.csv", trans, False),
        (diag / "composite_phase_duration_summary_patched.csv", dur, False),
        (diag / "composite_phase_layer_attribution_patched.csv", attr, False),
        (diag / "composite_phase_trough_alignment_patched.csv", trough, False),
        (diag / "snapshot_diff_vs_prior_patch.csv", snapshot_diff(BASE_DIR, snap), False),
        (refdir / "composite_phase_reference_table.csv", ref, False),
        (refdir / "macro_feature_inventory.csv", inv, False),
    ]:
        df.to_csv(path, index=index)
        files.append(path)

    manifest = fred_cache / "metadata" / "fred_series_manifest.csv"
    if manifest.exists():
        target = refdir / "fred_series_manifest.csv"
        target.write_bytes(manifest.read_bytes())
        files.append(target)
    for u, fit in fits.items():
        label_path = cjm / f"cjm_labels_{u}.csv"
        prob_path = cjm / f"cjm_probabilities_{u}.csv"
        label_frame(u, fit).merge(meta[meta["universe"].eq(u)], on=["universe", "cjm_state"], how="left").to_csv(label_path, index=False)
        fit.probabilities.reset_index().assign(universe=u).to_csv(prob_path, index=False)
        files.extend([label_path, prob_path])
    summary_path = current / "github_action_summary.txt"
    report_path = current / "daily_composite_regime_reviewer_report.md"
    summary_path.write_text(github_summary(snap), encoding="utf-8")
    report_path.write_text(report(snap, ref, qual, beh18, dur, trough), encoding="utf-8")
    files.extend([summary_path, report_path])
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["daily", "research"], default="daily")
    ap.add_argument("--tail-days", type=int, default=252)
    ap.add_argument("--write-intermediate", type=parse_bool, default=False)
    ap.add_argument("--keep-cache", type=parse_bool, default=True)
    ap.add_argument("--start-date", default="2000-01-01")
    ap.add_argument("--primary-report-start", default="2018-01-01")
    ap.add_argument("--secondary-report-start", default="2020-01-01")
    ap.add_argument("--refresh-fred", action="store_true")
    ap.add_argument("--fred-cache-dir", default=str(DATA_DIR / "fred_cache"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--fail-if-stale", action="store_true")
    ap.add_argument("--max-stale-trading-days", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lambda-cjm", type=float, default=100.0)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--use-existing-feature-sets", action="store_true")
    ap.add_argument("--allow-yfinance", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    default_out = BASE_DIR / "outputs" / ("daily_composite_regime" if args.mode == "daily" else "daily_composite_regime_research")
    outdir, fred_cache = Path(args.output_dir) if args.output_dir else default_out, Path(args.fred_cache_dir)
    ensure_dirs(outdir, fred_cache)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    log(f"Run timestamp UTC: {run_timestamp}")
    log(f"Output directory: {outdir}")
    market = build_market_panel(args)
    fred_daily, fred_weekly, fred_log = build_fred_panel(market.index, args)
    daily = market.join(fred_daily, how="left")
    vix_source = "local VIX CSV"
    vix_warning = ""
    if "vix_level" not in daily:
        daily["vix_level"] = np.nan
    local_vix_latest = daily.index[daily["vix_level"].notna()].max() if daily["vix_level"].notna().any() else pd.NaT
    fred_vix_latest = daily.index[daily["VIXCLS"].notna()].max() if "VIXCLS" in daily and daily["VIXCLS"].notna().any() else pd.NaT
    if "VIXCLS" in daily and pd.notna(fred_vix_latest) and (pd.isna(local_vix_latest) or fred_vix_latest > local_vix_latest):
        daily["vix_level"] = daily["VIXCLS"].combine_first(daily["vix_level"])
        vix_source = "FRED VIXCLS"
        if pd.notna(local_vix_latest):
            vix_warning = f"Local VIX latest date {local_vix_latest.date().isoformat()} was older than FRED VIXCLS {fred_vix_latest.date().isoformat()}; used FRED VIXCLS."
            log(f"WARNING: {vix_warning}")
    elif daily["vix_level"].isna().all():
        vix_source = "missing"
        vix_warning = "VIX data missing from local CSV and FRED VIXCLS."
        log(f"WARNING: {vix_warning}")
    for h in [5, 10, 21]:
        daily[f"vix_change_{h}d"] = daily["vix_level"].diff(h)
    daily["vix_10d_ma"] = daily["vix_level"].rolling(10, min_periods=5).mean()
    daily["vix_above_10d_ma"] = daily["vix_level"] > daily["vix_10d_ma"]
    daily["vix_falling_5d"] = daily["vix_change_5d"] < 0
    daily["vix_falling_10d"] = daily["vix_change_10d"] < 0
    daily = add_macro_causes(add_regime_layer(daily))

    weekly_feat, inv = build_weekly_features(daily)
    fits: Dict[str, CJMFit] = {}
    for u in UNIVERSES:
        try:
            fits[u] = fit_cjm(weekly_feat, u, args.k, args.lambda_cjm, args.random_state)
        except Exception as exc:
            log(f"CJM fit failed for {u}: {exc}")
    char, meta = characterize_states(daily, weekly_feat, fits)
    daily = daily.join(map_cjm_daily(daily, fits, meta))
    daily = add_forward_metrics(add_composite(daily))

    ref = phase_reference_table()
    snap = latest_snapshot(daily, ref, args.max_stale_trading_days)
    source_audit = build_data_source_audit(market, fred_daily, daily, vix_source, vix_warning)
    qual = data_quality(True, fred_log, daily, snap, fits, source_audit)
    beh18 = summarize_behavior(daily, pd.Timestamp(args.primary_report_start))
    beh20 = summarize_behavior(daily, pd.Timestamp(args.secondary_report_start))
    ep, trans, dur = episode_outputs(daily)
    trough = trough_alignment(daily)
    attr = layer_attribution(daily)

    daily.index.name = "date"
    if args.mode == "daily":
        written = write_daily_outputs(outdir, daily, ref, snap, qual, source_audit, args.tail_days)
    else:
        written = write_research_outputs(
            outdir, daily, fred_daily, fred_weekly, weekly_feat, inv, fits, meta, char, ref, snap, qual,
            beh18, beh20, ep, trans, dur, attr, trough, fred_cache
        )

    s = snap.iloc[0]
    report_file = outdir / "github_action_summary.txt" if args.mode == "daily" else outdir / "current" / "github_action_summary.txt"
    print("Daily Composite Regime Build Complete\n")
    print(f"Run timestamp UTC: {run_timestamp}")
    print(f"Latest date: {s['latest_data_date']}")
    print(f"Latest SPY data date: {source_audit.loc[source_audit['source_name'].eq('SPY'), 'latest_date'].iloc[0]}")
    print(f"Latest final panel date: {s['latest_data_date']}")
    print(f"Composite phase: {s['composite_phase_label_patched']}")
    print(f"Macro SPY regime: {s['macro_spy_regime']}")
    print(f"Macro cause: {s['macro_cause']}")
    print(f"VIX layer: {s['VIX_ONLY_broad_meta_label']} / {s['VIX_ONLY_severity_label']} / {s['VIX_ONLY_max_probability']:.3f}")
    print(f"Full macro layer: {s['FULL_PUBLIC_MACRO_broad_meta_label']} / {s['FULL_PUBLIC_MACRO_severity_label']} / {s['FULL_PUBLIC_MACRO_max_probability']:.3f}")
    print(f"Data freshness: {s['data_freshness_flag']}")
    print(f"Quality summary: {s['data_quality_summary']}")
    print(f"Output directory: {outdir}")
    print(f"Notification report file: {report_file}")
    print(f"FRED series loaded: {(~fred_log['status'].astype(str).str.contains('failed')).sum()}")
    print(f"FRED series failed: {fred_log['status'].astype(str).str.contains('failed').sum()}")
    print("Files written:")
    if args.mode == "daily":
        for path in written:
            print(f"- {path.name}")
    else:
        for path in [
            outdir / "current" / "latest_composite_regime_snapshot.csv",
            outdir / "current" / "github_action_summary.txt",
            outdir / "current" / "daily_composite_regime_reviewer_report.md",
            outdir / "current" / "data_quality_flags.csv",
            outdir / "panels",
            outdir / "cjm",
            outdir / "diagnostics",
            outdir / "reference",
        ]:
            print(f"- {path.relative_to(outdir)}")
    print(f"\n{NO_ADVICE}")
    if args.fail_if_stale and s["data_freshness_flag"] == "stale":
        print("Latest report is stale; notification should not be sent.", file=sys.stderr)
        return 2
    if args.mode == "daily":
        print("Daily mode complete: 7 files written.")
    else:
        print("Research mode complete: full diagnostics written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
