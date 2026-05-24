#!/usr/bin/env python3
"""
Standalone macro-enhanced SPY dashboard.

Monitoring only. No buy/sell rules, no position sizing, no return optimization.
SPY baseline remains the action gate; macro overlays provide context.
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


MIN_N_OK = 50
MIN_N_LOW = 20

FAST_RISK_OVERLAY = [
    "RECESSION_FEAR_RATE_DOWN",
    "RECESSION_FEAR_EARLY",
    "HOSTILE_RATE_UP",
    "WTI_TAIL_STRESS_ACTIVATION",
    "WTI_TAIL_STRESS_FAST",
    "RATE_BURDEN",
    "HIGH_RATE_4_75",
]

PERSISTENT_BACKGROUND_OVERLAY = [
    "WTI_TAIL_STRESS_PERSISTENT_BACKGROUND",
]

SLOW_BUILDUP_OVERLAY = [
    "OIL_PERSISTENT_PRESSURE",
    "OIL_BREAKEVEN_CONFIRMATION",
    "OIL_RATE_CONFIRMATION",
    "OIL_REAL_YIELD_CONFIRMATION",
    "REAL_YIELD_TIGHTENING",
    "GLD_DEFENSIVE_PERSISTENCE",
    "GLD_RESILIENT_AGAINST_RATES",
    "GLD_COMMODITY_STRESS_CONFIRMATION",
]

BENIGN_OR_CONFLICT = [
    "GROWTH_REFLATION_ACCEPTED",
    "GLD_REFLATION_SUPPORT",
    "POLICY_RELIEF_RATE_DOWN",
    "RECESSION_FEAR_MIDDLE",
    "RECESSION_FEAR_LATE",
]

ALL_OVERLAYS = list(dict.fromkeys(FAST_RISK_OVERLAY + SLOW_BUILDUP_OVERLAY + BENIGN_OR_CONFLICT + [
    "OIL_FRESH_SHOCK",
    "WTI_TAIL_STRESS",
    "WTI_TAIL_STRESS_EXTENDED",
    "WTI_TAIL_STRESS_PERSISTENT_BACKGROUND",
    "WTI_TAIL_STRESS_PERSISTENT",
    "WTI_STRESS_ACTIVE",
    "WTI_STRESS_TOO_PERSISTENT",
    "FRESH_HIGH_RATE_4_5",
    "PERSISTENT_HIGH_RATE_4_5_SHORT",
    "PERSISTENT_HIGH_RATE_4_5_MEDIUM",
    "PERSISTENT_HIGH_RATE_4_5_LONG",
    "HIGH_RATE_5_0",
    "GLD_BULLISH_IMPULSE",
    "GLD_WHIPSAW_REBOUND",
    "GLD_SAFE_HAVEN_CONFIRMATION",
    "GLD_NO_SIGNAL",
]))

CAUSE_PRIORITY = [
    "CAUSE_OIL_TIGHTENING",
    "CAUSE_HOSTILE_RATE_UP",
    "CAUSE_RECESSION_FEAR",
    "CAUSE_REAL_YIELD_TIGHTENING",
    "CAUSE_RATE_BURDEN",
    "CAUSE_OIL_INFLATION",
    "CAUSE_WTI_TAIL_PERSISTENT",
    "CAUSE_WTI_TAIL_EXTENDED",
    "CAUSE_WTI_TAIL_FRESH",
    "CAUSE_WTI_TAIL",
    "CAUSE_GLD_SAFE_HAVEN",
    "CAUSE_GLD_RATE_RESILIENCE",
    "CAUSE_GROWTH_REFLATION",
    "CAUSE_MACRO_CALM",
]

STATIC_PATH = {
    "FAST": {"path": "REBOUND_DOMINANT", "bias": "REBOUND_DOMINANT", "cluster": "IID_LIKE", "note": "+1% gain more common than -1% drawdown"},
    "BASELINE_YELLOW": {"path": "RECOVERY_DOMINANT", "bias": "REBOUND_DOMINANT", "p_gain_100": 82.7, "p_dd_100": 55.3, "p_up_first_100": 64.0, "p_down_first_100": 34.7},
    "OIL_PERSISTENT_PRESSURE": {"path": "THREE_WEEK_DANGER"},
    "OIL_BREAKEVEN_CONFIRMATION": {"path": "FOUR_TO_EIGHT_WEEK_BUILDUP"},
    "OIL_RATE_CONFIRMATION": {"path": "FOUR_TO_EIGHT_WEEK_BUILDUP"},
    "OIL_REAL_YIELD_CONFIRMATION": {"path": "EARLY_DANGER"},
    "REAL_YIELD_TIGHTENING": {"path": "THREE_WEEK_DANGER"},
    "GLD_DEFENSIVE_PERSISTENCE": {"path": "FLAT_BACKGROUND"},
    "GLD_RESILIENT_AGAINST_RATES": {"path": "EARLY_DANGER"},
    "GLD_COMMODITY_STRESS_CONFIRMATION": {"path": "FOUR_TO_EIGHT_WEEK_BUILDUP"},
}


@dataclass
class Audit:
    source: str
    found: bool
    rows: int = 0
    start_date: str = ""
    end_date: str = ""
    key_columns: str = ""
    missing_columns: str = ""
    status: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output-dir", default="outputs/macro_enhanced_spy_dashboard_standalone")
    p.add_argument("--as-of-date", default=None)
    p.add_argument("--post-2020-reference", action="store_true")
    p.add_argument("--allow-download", action="store_true", default=False)
    return p.parse_args()


def find_date_col(df: pd.DataFrame) -> str | None:
    for c in ["Date", "date", "DATE", "timestamp"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    dc = find_date_col(df)
    if dc is None:
        raise ValueError("no date column")
    if dc != "date":
        df = df.rename(columns={dc: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return df.reset_index(drop=True)


def first_existing(base: Path, candidates: list[str]) -> Path | None:
    for c in candidates:
        p = base / c
        if p.exists():
            return p
    return None


def load_source(base: Path, name: str, candidates: list[str], required: list[str]) -> tuple[pd.DataFrame | None, Audit]:
    path = first_existing(base, candidates)
    if path is None:
        return None, Audit(name, False, missing_columns="|".join(required), status="missing")
    try:
        df = read_csv(path)
        key = [c for c in required if c in df.columns]
        missing = [c for c in required if c not in df.columns]
        return df, Audit(
            name,
            True,
            len(df),
            df["date"].min().date().isoformat(),
            df["date"].max().date().isoformat(),
            "|".join(key),
            "|".join(missing),
            "ok" if not missing else "degraded_missing_columns",
        )
    except Exception as exc:
        return None, Audit(name, True, status=f"failed: {exc}")


def close_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in ["adj close", "adj_close", "adjusted_close", "close"]:
        if c in lower:
            return lower[c]
    raise ValueError("close column missing")


def prep_price(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    c = close_col(df)
    out = df[["date", c]].rename(columns={c: f"{prefix}_close"}).copy()
    out[f"{prefix}_close"] = pd.to_numeric(out[f"{prefix}_close"], errors="coerce")
    return out.dropna(subset=[f"{prefix}_close"])


def rolling_age(active: pd.Series) -> pd.Series:
    vals = active.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(vals), dtype=int)
    age = 0
    for i, v in enumerate(vals):
        age = age + 1 if v else 0
        out[i] = age
    return pd.Series(out, index=active.index)


def wti_tail_age_bucket(age: int) -> str:
    if age <= 0:
        return "inactive"
    if age == 1:
        return "activation_1d"
    if age <= 5:
        return "fresh_2_5d"
    if age <= 10:
        return "fast_window_6_10d"
    if age <= 21:
        return "extended_11_21d"
    if age <= 42:
        return "persistent_22_42d"
    if age <= 63:
        return "long_persistent_43_63d"
    return "structural_64d_plus"


def add_spy_baseline(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["spy_close"]
    x["spy_ret_1d"] = c.pct_change()
    x["spy_ret_21d"] = c.pct_change(21)
    for w in [50, 100, 200]:
        x[f"MA{w}"] = c.rolling(w).mean()
    x["distance_to_MA200"] = c / x["MA200"] - 1.0
    x["below_ma200"] = c < x["MA200"]
    abs100 = x["spy_ret_1d"].abs() >= 0.01
    x["abs100_count_5d"] = abs100.rolling(5).sum()
    x["abs100_count_5d_bucket"] = np.select(
        [x["abs100_count_5d"].eq(0), x["abs100_count_5d"].eq(1), x["abs100_count_5d"] >= 2],
        ["n0", "n1", "n2plus"],
        default="unknown",
    )
    up10 = (x["spy_ret_1d"] >= 0.01).rolling(10).sum() > 0
    down10 = (x["spy_ret_1d"] <= -0.01).rolling(10).sum() > 0
    x["recent_whipsaw_10d_100"] = up10 & down10
    x["RV21"] = x["spy_ret_1d"].rolling(21).std() * np.sqrt(252)
    rv_median = x["RV21"].rolling(252, min_periods=63).median()
    x["RV21_state"] = np.where(x["RV21"] > rv_median, "high_vol", "low_vol")
    x["rv21_rising_5d"] = x["RV21"] > x["RV21"].shift(5)
    x["rv21_rising_10d"] = x["RV21"] > x["RV21"].shift(10)
    x["vol_transition_state"] = np.select(
        [
            x["RV21_state"].eq("high_vol") & x["rv21_rising_5d"],
            x["RV21_state"].eq("high_vol") & ~x["rv21_rising_5d"],
            x["RV21_state"].eq("low_vol") & x["rv21_rising_5d"],
        ],
        ["high_and_rising", "high_but_falling", "low_and_rising"],
        default="low_flat_or_falling",
    )
    dist = x["distance_to_MA200"]
    high = x["RV21_state"].eq("high_vol")
    n2 = x["abs100_count_5d_bucket"].eq("n2plus")
    n1n2 = x["abs100_count_5d_bucket"].isin(["n1", "n2plus"])
    label = np.full(len(x), "mixed", dtype=object)
    label[x["below_ma200"] & high & n2] = "confirmed_stress"
    mask = dist.between(0, 0.02) & n2 & high & x["recent_whipsaw_10d_100"]
    label[mask] = "prebreak_riskoff_transition"
    mask = (label == "mixed") & dist.between(0, 0.02) & n2 & high
    label[mask] = "prebreak_warning"
    mask = (label == "mixed") & dist.between(0, 0.02) & n1n2
    label[mask] = "prebreak_watch"
    mask = (label == "mixed") & (x["recent_whipsaw_10d_100"] | n2)
    label[mask] = "whipsaw_aware"
    mask = (label == "mixed") & (dist > 0.05) & x["RV21_state"].eq("low_vol") & x["abs100_count_5d_bucket"].isin(["n0", "n1"])
    label[mask] = "active_but_safe"
    mask = (dist > 0.05) & x["RV21_state"].eq("low_vol") & x["abs100_count_5d_bucket"].eq("n0")
    label[mask] = "calm_state"
    x["spy_final_state_label"] = label
    x["spy_final_daily_label"] = np.select(
        [
            x["spy_final_state_label"].eq("confirmed_stress"),
            x["spy_final_state_label"].isin(["prebreak_watch", "prebreak_warning", "prebreak_riskoff_transition"]),
            x["spy_final_state_label"].eq("whipsaw_aware"),
            x["spy_final_state_label"].isin(["calm_state", "active_but_safe"]),
        ],
        ["RED_CONFIRMED_RISK_OFF", "ORANGE_PREBREAK_WATCH", "YELLOW_WHIPSAW_NOT_RISK_OFF", "GREEN_NORMAL"],
        default="MIXED_NEUTRAL",
    )
    x["spy_baseline_group"] = np.select(
        [
            x["spy_final_state_label"].isin(["calm_state", "active_but_safe"]),
            x["spy_final_state_label"].eq("whipsaw_aware"),
            x["spy_final_state_label"].isin(["prebreak_watch", "prebreak_warning", "prebreak_riskoff_transition"]),
            x["spy_final_state_label"].eq("confirmed_stress"),
        ],
        ["SPY_GREEN", "SPY_YELLOW", "SPY_PREBREAK", "SPY_RISKOFF"],
        default="UNKNOWN",
    )
    return x


def add_wti_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "wti_close" not in x.columns:
        for col in ["WTI", "oil_close"]:
            if col in x.columns:
                x["wti_close"] = pd.to_numeric(x[col], errors="coerce")
                break
    if "wti_close" not in x.columns:
        for f in [
            "WTI_TAIL_STRESS",
            "WTI_TAIL_STRESS_ACTIVATION",
            "WTI_TAIL_STRESS_FAST",
            "WTI_TAIL_STRESS_EXTENDED",
            "WTI_TAIL_STRESS_PERSISTENT_BACKGROUND",
            "WTI_TAIL_STRESS_PERSISTENT",
            "WTI_STRESS_ACTIVE",
            "WTI_STRESS_TOO_PERSISTENT",
        ]:
            x[f] = False
        x["wti_tail_stress_age_days"] = 0
        x["wti_tail_stress_age_bucket"] = "inactive"
        return x
    c = x["wti_close"]
    x["wti_ret_1d"] = c.pct_change()
    for h in [5, 21, 63, 126]:
        x[f"wti_ret_{h}d"] = c.pct_change(h)
    x["wti_ma63"] = c.rolling(63).mean()
    x["wti_ma126"] = c.rolling(126).mean()
    x["abs200_count_10d"] = (x["wti_ret_1d"].abs() >= 0.02).rolling(10).sum()
    x["abs300_count_21d"] = (x["wti_ret_1d"].abs() >= 0.03).rolling(21).sum()
    x["abs500_count_21d"] = (x["wti_ret_1d"].abs() >= 0.05).rolling(21).sum()
    x["WTI_TAIL_STRESS"] = (x["abs500_count_21d"] >= 2) | (x["abs300_count_21d"] >= 3)
    x["wti_tail_stress_age_days"] = rolling_age(x["WTI_TAIL_STRESS"])
    x["wti_tail_stress_age_bucket"] = x["wti_tail_stress_age_days"].astype(int).map(wti_tail_age_bucket)
    x["WTI_TAIL_STRESS_ACTIVATION"] = x["WTI_TAIL_STRESS"] & ~x["WTI_TAIL_STRESS"].shift(1, fill_value=False)
    x["WTI_TAIL_STRESS_FAST"] = x["WTI_TAIL_STRESS"] & x["wti_tail_stress_age_days"].between(1, 10)
    x["WTI_TAIL_STRESS_EXTENDED"] = x["WTI_TAIL_STRESS"] & x["wti_tail_stress_age_days"].between(11, 21)
    x["WTI_TAIL_STRESS_PERSISTENT_BACKGROUND"] = x["WTI_TAIL_STRESS"] & (x["wti_tail_stress_age_days"] >= 22)
    x["WTI_TAIL_STRESS_PERSISTENT"] = x["WTI_TAIL_STRESS"] & (x["wti_tail_stress_age_days"] >= 5)
    x["WTI_STRESS_ACTIVE"] = (x["abs200_count_10d"] >= 3) | x["WTI_TAIL_STRESS"]
    x["WTI_STRESS_TOO_PERSISTENT"] = x["WTI_STRESS_ACTIVE"].mean(skipna=True) > 0.80
    return x


def add_rates_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for col in ["DGS2", "DGS10", "T5YIE", "T10YIE", "DFII10"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[f"{col}_chg_21d_bps_calc"] = (x[col] - x[col].shift(21)) * 100
        else:
            x[f"{col}_chg_21d_bps_calc"] = np.nan
    dgs10 = x["DGS10"] if "DGS10" in x.columns else pd.Series(np.nan, index=x.index)
    x["OIL_FRESH_SHOCK"] = (x.get("wti_ret_5d", pd.Series(np.nan, index=x.index)) >= 0.08) | (x.get("wti_ret_21d", pd.Series(np.nan, index=x.index)) >= 0.15)
    x["OIL_PERSISTENT_PRESSURE"] = ((x.get("wti_ret_63d", pd.Series(np.nan, index=x.index)) >= 0.20) & (x.get("wti_close", np.nan) > x.get("wti_ma63", np.nan))) | ((x.get("wti_ret_126d", pd.Series(np.nan, index=x.index)) >= 0.25) & (x.get("wti_close", np.nan) > x.get("wti_ma126", np.nan)))
    oil_pressure = x["OIL_FRESH_SHOCK"] | x["OIL_PERSISTENT_PRESSURE"]
    x["OIL_BREAKEVEN_CONFIRMATION"] = oil_pressure & ((x["T5YIE_chg_21d_bps_calc"] > 5) | (x["T10YIE_chg_21d_bps_calc"] > 5))
    x["OIL_RATE_CONFIRMATION"] = oil_pressure & ((x["DGS2_chg_21d_bps_calc"] > 10) | (x["DGS10_chg_21d_bps_calc"] > 10))
    x["OIL_REAL_YIELD_CONFIRMATION"] = oil_pressure & (x["DFII10_chg_21d_bps_calc"] > 10)
    x["GROWTH_REFLATION_ACCEPTED"] = (x["DGS10_chg_21d_bps_calc"] > 15) & (x["T10YIE_chg_21d_bps_calc"] > 5) & (x["spy_ret_21d"] > 0) & (x["DFII10_chg_21d_bps_calc"] < 15)
    x["REAL_YIELD_TIGHTENING"] = (x["DGS10_chg_21d_bps_calc"] > 15) & (x["DFII10_chg_21d_bps_calc"] > 15)
    x["HOSTILE_RATE_UP"] = (x["DGS10_chg_21d_bps_calc"] > 15) & (x["spy_ret_21d"] < 0)
    x["RECESSION_FEAR_RATE_DOWN"] = (x["DGS10_chg_21d_bps_calc"] < -15) & (x["T10YIE_chg_21d_bps_calc"] < 0) & (x["spy_ret_21d"] < 0)
    x["POLICY_RELIEF_RATE_DOWN"] = (x["DGS10_chg_21d_bps_calc"] < -15) & (x["spy_ret_21d"] >= 0)
    age = rolling_age(x["RECESSION_FEAR_RATE_DOWN"])
    x["RECESSION_FEAR_EARLY"] = x["RECESSION_FEAR_RATE_DOWN"] & age.between(1, 5)
    x["RECESSION_FEAR_MIDDLE"] = x["RECESSION_FEAR_RATE_DOWN"] & age.between(6, 15)
    x["RECESSION_FEAR_LATE"] = x["RECESSION_FEAR_RATE_DOWN"] & (age >= 16)
    above45 = dgs10 >= 4.5
    above475 = dgs10 >= 4.75
    x["days_above_4_5"] = rolling_age(above45)
    x["days_above_4_75"] = rolling_age(above475)
    x["FRESH_HIGH_RATE_4_5"] = above45 & x["days_above_4_5"].between(1, 4)
    x["PERSISTENT_HIGH_RATE_4_5_SHORT"] = x["days_above_4_5"].between(5, 20)
    x["PERSISTENT_HIGH_RATE_4_5_MEDIUM"] = x["days_above_4_5"].between(21, 62)
    x["PERSISTENT_HIGH_RATE_4_5_LONG"] = x["days_above_4_5"] >= 63
    x["HIGH_RATE_4_75"] = dgs10 >= 4.75
    x["HIGH_RATE_5_0"] = dgs10 >= 5.0
    x["RATE_BURDEN"] = x["PERSISTENT_HIGH_RATE_4_5_SHORT"] | x["PERSISTENT_HIGH_RATE_4_5_MEDIUM"] | x["PERSISTENT_HIGH_RATE_4_5_LONG"] | x["HIGH_RATE_4_75"]
    return x


def add_gld_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "gld_close" not in x.columns:
        for f in ["GLD_BULLISH_IMPULSE", "GLD_WHIPSAW_REBOUND", "GLD_DEFENSIVE_PERSISTENCE", "GLD_RESILIENT_AGAINST_RATES", "GLD_SAFE_HAVEN_CONFIRMATION", "GLD_REFLATION_SUPPORT", "GLD_COMMODITY_STRESS_CONFIRMATION", "GLD_NO_SIGNAL"]:
            x[f] = False
        x["GLD_NO_SIGNAL"] = True
        return x
    c = x["gld_close"]
    x["gld_ret_1d"] = c.pct_change()
    for h in [5, 21, 63]:
        x[f"gld_ret_{h}d"] = c.pct_change(h)
    for w in [50, 100, 200]:
        x[f"gld_ma{w}"] = c.rolling(w).mean()
    x["gld_252_high"] = c.rolling(252, min_periods=63).max()
    x["gld_drawdown_252"] = c / x["gld_252_high"] - 1
    up10 = (x["gld_ret_1d"] >= 0.01).rolling(10).sum() > 0
    down10 = (x["gld_ret_1d"] <= -0.01).rolling(10).sum() > 0
    supportive = (c > x["gld_ma100"]) | (c > x["gld_ma200"])
    x["GLD_BULLISH_IMPULSE"] = ((x["gld_ret_1d"] >= 0.01) & (x["gld_ret_5d"] > 0)) | (supportive & (x["gld_ret_21d"] > 0))
    x["GLD_WHIPSAW_REBOUND"] = up10 & down10 & (x["gld_ret_5d"] > 0)
    x["GLD_DEFENSIVE_PERSISTENCE"] = (x["gld_ret_21d"] > 0) & supportive & (x["gld_drawdown_252"] > -0.08)
    x["GLD_RESILIENT_AGAINST_RATES"] = x["GLD_DEFENSIVE_PERSISTENCE"] & x["REAL_YIELD_TIGHTENING"]
    x["GLD_SAFE_HAVEN_CONFIRMATION"] = x["GLD_DEFENSIVE_PERSISTENCE"] & x["RECESSION_FEAR_RATE_DOWN"]
    x["GLD_REFLATION_SUPPORT"] = x["GLD_DEFENSIVE_PERSISTENCE"] & x["GROWTH_REFLATION_ACCEPTED"]
    x["GLD_COMMODITY_STRESS_CONFIRMATION"] = x["GLD_DEFENSIVE_PERSISTENCE"] & (x["WTI_TAIL_STRESS"] | x["OIL_BREAKEVEN_CONFIRMATION"] | x["OIL_RATE_CONFIRMATION"])
    constructive = x["GLD_BULLISH_IMPULSE"] | x["GLD_WHIPSAW_REBOUND"] | x["GLD_DEFENSIVE_PERSISTENCE"] | x["GLD_RESILIENT_AGAINST_RATES"] | x["GLD_SAFE_HAVEN_CONFIRMATION"] | x["GLD_REFLATION_SUPPORT"] | x["GLD_COMMODITY_STRESS_CONFIRMATION"]
    x["GLD_NO_SIGNAL"] = ~constructive
    return x


def add_causes(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    get = lambda c: x[c] if c in x.columns else pd.Series(False, index=x.index)
    x["CAUSE_OIL_TIGHTENING"] = get("OIL_RATE_CONFIRMATION") | get("OIL_REAL_YIELD_CONFIRMATION")
    x["CAUSE_HOSTILE_RATE_UP"] = get("HOSTILE_RATE_UP")
    x["CAUSE_RECESSION_FEAR"] = get("RECESSION_FEAR_RATE_DOWN") | get("RECESSION_FEAR_EARLY")
    x["CAUSE_REAL_YIELD_TIGHTENING"] = get("REAL_YIELD_TIGHTENING")
    x["CAUSE_RATE_BURDEN"] = get("RATE_BURDEN") | get("HIGH_RATE_4_75")
    x["CAUSE_OIL_INFLATION"] = get("OIL_FRESH_SHOCK") | get("OIL_PERSISTENT_PRESSURE") | get("OIL_BREAKEVEN_CONFIRMATION")
    x["CAUSE_WTI_TAIL_FRESH"] = get("WTI_TAIL_STRESS_ACTIVATION") | get("WTI_TAIL_STRESS_FAST")
    x["CAUSE_WTI_TAIL_EXTENDED"] = get("WTI_TAIL_STRESS_EXTENDED")
    x["CAUSE_WTI_TAIL_PERSISTENT"] = get("WTI_TAIL_STRESS_PERSISTENT_BACKGROUND")
    x["CAUSE_WTI_TAIL"] = get("WTI_TAIL_STRESS")
    x["CAUSE_GLD_SAFE_HAVEN"] = get("GLD_DEFENSIVE_PERSISTENCE") | get("GLD_SAFE_HAVEN_CONFIRMATION")
    x["CAUSE_GLD_RATE_RESILIENCE"] = get("GLD_RESILIENT_AGAINST_RATES")
    x["CAUSE_GROWTH_REFLATION"] = get("GROWTH_REFLATION_ACCEPTED")
    any_cause = pd.Series(False, index=x.index)
    for c in CAUSE_PRIORITY[:-1]:
        any_cause |= x[c]
    x["CAUSE_MACRO_CALM"] = ~any_cause
    labels = []
    for _, row in x.iterrows():
        labels.append(next((c for c in CAUSE_PRIORITY[:-1] if bool(row.get(c, False))), "CAUSE_MACRO_CALM"))
    x["current_macro_cause"] = labels
    return x


def load_inputs(base: Path) -> tuple[pd.DataFrame, list[Audit]]:
    audits: list[Audit] = []
    spy, a = load_source(base, "SPY", ["data/SPY_20y.csv", "SPY_20y.csv"], ["date"])
    audits.append(a)
    if spy is None:
        raise FileNotFoundError("SPY price input is required")
    panel = prep_price(spy, "spy")
    gld, a = load_source(base, "GLD", ["data/GLD_20y.csv", "GLD_20y.csv"], ["date"])
    audits.append(a)
    if gld is not None:
        panel = panel.merge(prep_price(gld, "gld"), on="date", how="left")
    wti, a = load_source(base, "WTI", ["data/CL=F_20y.csv", "data/DCOILWTICO.csv", "data/WTI_20y.csv", "outputs/oil_price_escalation_memory_bayes/oil_native_daily_state_dataset.csv"], ["date"])
    audits.append(a)
    if wti is not None:
        if "oil_close" in wti.columns:
            wti = wti[["date", "oil_close"]].rename(columns={"oil_close": "wti_close"})
        else:
            wti = prep_price(wti, "wti")
        panel = panel.merge(wti, on="date", how="left")
    aligned, a = load_source(base, "aligned_oil_rates_panel", ["outputs/macro_daily_aligned_panel.csv", "outputs/oil_inflation_rates_lead_lag_study/aligned_oil_rates_daily_panel.csv"], ["date", "DGS10"])
    audits.append(a)
    if aligned is not None:
        keep = [c for c in ["date", "WTI", "DGS2", "DGS10", "T5YIE", "T10YIE", "DFII10"] if c in aligned.columns]
        panel = panel.merge(aligned[keep], on="date", how="left")
    for series in ["DGS2", "DGS10", "T5YIE", "T10YIE", "DFII10"]:
        df, a = load_source(base, series, [f"data/{series}.csv", f"{series}.csv"], ["date"])
        audits.append(a)
        if df is not None:
            val_cols = [c for c in df.columns if c != "date"]
            if val_cols:
                tmp = df[["date", val_cols[0]]].rename(columns={val_cols[0]: series})
                panel = panel.drop(columns=[series], errors="ignore").merge(tmp, on="date", how="left")
    vix, a = load_source(base, "VIX", ["data/^VIX_20y.csv", "^VIX_20y.csv"], ["date"])
    audits.append(a)
    if vix is not None:
        panel = panel.merge(prep_price(vix, "vix"), on="date", how="left")
    return panel.sort_values("date").reset_index(drop=True), audits


def active_list(row: pd.Series, names: list[str]) -> list[str]:
    return [n for n in names if bool(row.get(n, False))]


def slow_stage(signal: str, age: int) -> str:
    if age <= 0:
        return "INACTIVE"
    if signal == "OIL_PERSISTENT_PRESSURE":
        return "WATCH" if age <= 10 else "THREE_WEEK_DANGER" if age <= 21 else "ELEVATED_CLUSTERING_RISK" if age <= 42 else "LATE_PERSISTENT_PRESSURE"
    if signal in ["OIL_BREAKEVEN_CONFIRMATION", "OIL_RATE_CONFIRMATION"]:
        return "WATCH" if age <= 10 else "BUILDING" if age <= 21 else "FOUR_TO_EIGHT_WEEK_BUILDUP" if age <= 42 else "LATE_BUILDUP"
    if signal == "OIL_REAL_YIELD_CONFIRMATION":
        return "EARLY_DANGER_ACTIVE" if age <= 10 else "HIGH_ATTENTION" if age <= 21 else "PERSISTENT_PRESSURE"
    if signal == "REAL_YIELD_TIGHTENING":
        return "WATCH" if age <= 10 else "THREE_WEEK_DANGER" if age <= 21 else "PERSISTENT_TIGHTENING"
    if signal == "GLD_DEFENSIVE_PERSISTENCE":
        return "BACKGROUND_CONFIRMATION_ONLY"
    if signal == "GLD_RESILIENT_AGAINST_RATES":
        return "EARLY_DANGER_ACTIVE" if age <= 10 else "HIGH_ATTENTION" if age <= 21 else "PERSISTENT_DEFENSIVE_CONFIRMATION"
    if signal == "GLD_COMMODITY_STRESS_CONFIRMATION":
        return "WATCH" if age <= 10 else "BUILDING" if age <= 21 else "FOUR_TO_EIGHT_WEEK_BUILDUP"
    return "WATCH"


def overlay_layer(name: str) -> str:
    if name.startswith("WTI"):
        return "WTI_PRICE_ONLY"
    if name.startswith("OIL"):
        return "OIL_TRANSMISSION"
    if name.startswith("GLD"):
        return "GLD_PRICE_ONLY"
    if name in ["RATE_BURDEN", "HIGH_RATE_4_75", "HIGH_RATE_5_0", "FRESH_HIGH_RATE_4_5"] or name.startswith("PERSISTENT_HIGH_RATE"):
        return "RATE_PERSISTENCE"
    return "RATES_DRIVER"


def current_age(panel: pd.DataFrame, signal: str, idx: int) -> int:
    if signal not in panel.columns:
        return 0
    active = panel[signal].fillna(False).astype(bool).iloc[: idx + 1].to_numpy()
    age = 0
    for v in active[::-1]:
        if v:
            age += 1
        else:
            break
    return age


def wti_timing_note(age: int, activation: bool, fast: bool, extended: bool, persistent: bool) -> str:
    if activation:
        return "WTI tail-stress just activated. Historical fast-monitor window is 5-10 trading days."
    if fast:
        return "WTI tail-stress is still inside the fast 1-2 week monitoring window."
    if extended:
        return "WTI tail-stress is beyond the first 2 weeks. Treat as extended stress, not a fresh trigger."
    if persistent:
        return "WTI tail-stress has persisted beyond 21 trading days. Treat as commodity-stress background, not a fresh fast trigger."
    return "WTI tail-stress is inactive."


def dashboard_verdict(group: str, fast: list[str], slow: list[str], benign: list[str], background: list[str]) -> str:
    has_wti_background = "WTI_TAIL_STRESS_PERSISTENT_BACKGROUND" in background
    if group == "SPY_YELLOW" and fast and has_wti_background:
        return "FAST_OVERLAY_WITH_PERSISTENT_WTI_BACKGROUND"
    if group == "SPY_YELLOW" and not fast and not slow and has_wti_background:
        return "BASELINE_YELLOW_WITH_PERSISTENT_WTI_BACKGROUND"
    if group == "SPY_YELLOW" and not (fast or slow or benign):
        return "BASELINE_ONLY_YELLOW"
    if group == "SPY_GREEN" and not (fast or slow or benign):
        return "BASELINE_ONLY_GREEN"
    if group == "SPY_YELLOW" and fast and slow:
        return "FAST_PLUS_SLOW_MONITOR"
    if group == "SPY_YELLOW" and fast:
        return "FAST_RISK_MONITOR"
    if group == "SPY_YELLOW" and slow:
        return "SLOW_BUILDUP_MONITOR"
    if group == "SPY_PREBREAK" and (fast or slow or benign):
        return "MACRO_CONFIRMED_PREBREAK"
    if group == "SPY_PREBREAK":
        return "TAPE_ONLY_PREBREAK"
    if group == "SPY_RISKOFF" and (fast or slow or benign):
        return "RISKOFF_WITH_MACRO_EXPLANATION"
    if group == "SPY_RISKOFF":
        return "BASELINE_RISKOFF"
    if group == "SPY_GREEN" and (fast or slow):
        return "MACRO_CONFLICT_WATCH"
    if group in ["SPY_YELLOW", "SPY_PREBREAK", "SPY_RISKOFF"] and any(x in benign for x in ["GROWTH_REFLATION_ACCEPTED", "GLD_REFLATION_SUPPORT"]):
        return "BENIGN_REFLATION_CONFLICT"
    return "MIXED_MONITOR"


def build_dashboard(panel: pd.DataFrame, audits: list[Audit], as_of: str | None, output_dir: Path) -> None:
    if as_of:
        panel = panel[panel["date"] <= pd.Timestamp(as_of)].copy()
    panel = add_spy_baseline(panel)
    panel = add_wti_features(panel)
    panel = add_rates_oil_features(panel)
    panel = add_gld_features(panel)
    panel = add_causes(panel)
    latest = panel.iloc[-1]
    idx = panel.index[-1]
    fast = active_list(latest, FAST_RISK_OVERLAY)
    slow = active_list(latest, SLOW_BUILDUP_OVERLAY)
    benign = active_list(latest, BENIGN_OR_CONFLICT)
    background = active_list(latest, PERSISTENT_BACKGROUND_OVERLAY)
    background_display = ["WTI_TAIL_STRESS" if x == "WTI_TAIL_STRESS_PERSISTENT_BACKGROUND" else x for x in background]
    wti_age = int(latest.get("wti_tail_stress_age_days", 0) or 0)
    wti_bucket = str(latest.get("wti_tail_stress_age_bucket", "inactive"))
    wti_note = wti_timing_note(
        wti_age,
        bool(latest.get("WTI_TAIL_STRESS_ACTIVATION", False)),
        bool(latest.get("WTI_TAIL_STRESS_FAST", False)),
        bool(latest.get("WTI_TAIL_STRESS_EXTENDED", False)),
        bool(latest.get("WTI_TAIL_STRESS_PERSISTENT_BACKGROUND", False)),
    )
    if fast and "WTI_TAIL_STRESS_PERSISTENT_BACKGROUND" in background:
        risk_timing = "FAST_OVERLAY_WITH_PERSISTENT_WTI_BACKGROUND"
    elif fast and slow:
        risk_timing = "FAST_OVERLAY_WITH_SLOW_PRESSURE_BACKGROUND"
    elif fast:
        risk_timing = "FAST_1_TO_2_WEEK_MONITOR"
    elif slow:
        risk_timing = "SLOW_BUILDUP_MONITOR"
    elif bool(latest.get("WTI_TAIL_STRESS_EXTENDED", False)):
        risk_timing = "EXTENDED_WTI_TAIL_MONITOR"
    elif "WTI_TAIL_STRESS_PERSISTENT_BACKGROUND" in background:
        risk_timing = "PERSISTENT_WTI_TAIL_BACKGROUND"
    else:
        risk_timing = "BASELINE_ONLY_NO_MACRO_CONFIRMATION"
    ages = []
    for s in slow:
        age = current_age(panel, s, idx)
        ages.append(f"{s}:{age}d:{slow_stage(s, age)}")
    if fast:
        path = STATIC_PATH["FAST"]["path"]
        bias = STATIC_PATH["FAST"]["bias"]
        cluster = STATIC_PATH["FAST"]["cluster"]
    elif latest["spy_baseline_group"] == "SPY_YELLOW" and not (fast or slow):
        path = STATIC_PATH["BASELINE_YELLOW"]["path"]
        bias = STATIC_PATH["BASELINE_YELLOW"]["bias"]
        cluster = "IID_LIKE" if background else "NONE"
    elif slow:
        path = "|".join([STATIC_PATH.get(s, {}).get("path", "MONITOR_ONLY") for s in slow])
        bias = "MIXED_WHIPSAW"
        cluster = "IID_LIKE"
    else:
        path = "NO_SIGNAL"
        bias = "NO_SIGNAL"
        cluster = "NONE"
    verdict = dashboard_verdict(str(latest["spy_baseline_group"]), fast, slow, benign, background)
    data_quality = "ok" if all(a.status.startswith("ok") or a.source not in ["SPY"] for a in audits) else "degraded"
    validation_failed = wti_age > 21 and risk_timing == "FAST_1_TO_2_WEEK_MONITOR" and not fast
    if validation_failed:
        print("warning: WTI timing patch failed; persistent tail stress was labeled as fresh fast risk.")
        data_quality = "TIMING_PATCH_FAILED"
    out = pd.DataFrame([{
        "date": latest["date"],
        "spy_close": latest["spy_close"],
        "spy_final_state_label": latest["spy_final_state_label"],
        "spy_final_daily_label": latest["spy_final_daily_label"],
        "spy_baseline_group": latest["spy_baseline_group"],
        "current_macro_cause": latest["current_macro_cause"],
        "active_fast_overlays": "|".join(fast) if fast else "NONE",
        "active_slow_overlays": "|".join(slow) if slow else "NONE",
        "active_benign_or_conflict_overlays": "|".join(benign) if benign else "NONE",
        "risk_timing": risk_timing,
        "wti_tail_stress_age_days": wti_age,
        "wti_tail_stress_age_bucket": wti_bucket,
        "wti_tail_stress_timing_note": wti_note,
        "persistent_background_overlays": "|".join(background_display) if background_display else "NONE",
        "fresh_fast_overlays": "|".join(fast) if fast else "NONE",
        "slow_signal_age_summary": "|".join(ages) if ages else "NONE",
        "path_anatomy_label": path,
        "drawdown_rebound_bias": bias,
        "clustering_status": cluster,
        "current_dashboard_verdict": verdict,
        "monitoring_note": "Monitoring only. No trading rule. SPY baseline remains action gate.",
        "data_quality_flag": data_quality,
    }])
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "macro_enhanced_current_dashboard.csv", index=False)
    signal_status(panel, latest, idx).to_csv(output_dir / "macro_overlay_signal_status.csv", index=False)
    cause_reference_table().to_csv(output_dir / "macro_cause_reference_table.csv", index=False)
    pd.DataFrame([a.__dict__ for a in audits]).to_csv(output_dir / "dashboard_file_audit.csv", index=False)
    text = render_text(out.iloc[0], fast, slow, benign, background)
    (output_dir / "macro_enhanced_current_dashboard.txt").write_text(text)
    summary = render_summary(out.iloc[0])
    (output_dir / "github_action_summary.txt").write_text(summary)
    (output_dir / "github_action_summary.html").write_text(render_summary_html(out.iloc[0]))
    print(summary)
    print("WTI age patch complete. Monitoring only. No trading rules generated.")


def signal_status(panel: pd.DataFrame, latest: pd.Series, idx: int) -> pd.DataFrame:
    rows = []
    for o in ALL_OVERLAYS:
        active = bool(latest.get(o, False))
        timing = "FAST_RISK_OVERLAY" if o in FAST_RISK_OVERLAY else "SLOW_BUILDUP_OVERLAY" if o in SLOW_BUILDUP_OVERLAY else "PERSISTENT_BACKGROUND" if o in PERSISTENT_BACKGROUND_OVERLAY else "BENIGN_OR_CONFLICT"
        age = current_age(panel, o, idx) if active else 0
        overlay_age = int(latest.get("wti_tail_stress_age_days", 0) or 0) if o == "WTI_TAIL_STRESS" else age
        age_bucket = str(latest.get("wti_tail_stress_age_bucket", "inactive")) if o == "WTI_TAIL_STRESS" else ("inactive" if overlay_age <= 0 else "active")
        is_fresh = bool(latest.get("WTI_TAIL_STRESS_ACTIVATION", False)) if o == "WTI_TAIL_STRESS" else (active and age == 1)
        is_background = bool(latest.get("WTI_TAIL_STRESS_PERSISTENT_BACKGROUND", False)) if o == "WTI_TAIL_STRESS" else o in PERSISTENT_BACKGROUND_OVERLAY and active
        if o == "WTI_TAIL_STRESS" and is_background:
            timing = "PERSISTENT_BACKGROUND"
            risk_stage = "STRUCTURAL_WTI_TAIL_BACKGROUND" if overlay_age >= 64 else "PERSISTENT_WTI_TAIL_BACKGROUND"
        elif o == "WTI_TAIL_STRESS" and bool(latest.get("WTI_TAIL_STRESS_EXTENDED", False)):
            timing = "EXTENDED_WTI_TAIL_MONITOR"
            risk_stage = "EXTENDED_WTI_TAIL_MONITOR"
        elif o == "WTI_TAIL_STRESS" and bool(latest.get("WTI_TAIL_STRESS_FAST", False)):
            timing = "FAST_RISK_OVERLAY"
            risk_stage = "FAST_1_TO_2_WEEK_MONITOR"
        elif o == "WTI_TAIL_STRESS_PERSISTENT_BACKGROUND" and active:
            risk_stage = "STRUCTURAL_WTI_TAIL_BACKGROUND" if int(latest.get("wti_tail_stress_age_days", 0) or 0) >= 64 else "PERSISTENT_WTI_TAIL_BACKGROUND"
        else:
            risk_stage = slow_stage(o, age) if o in SLOW_BUILDUP_OVERLAY else ("FAST_1_TO_2_WEEK_MONITOR" if active and o in FAST_RISK_OVERLAY else "INACTIVE")
        rows.append({
            "overlay": o,
            "active_today": active,
            "layer": overlay_layer(o),
            "timing_class": timing,
            "current_age_days": age,
            "overlay_age_days": overlay_age,
            "overlay_age_bucket": age_bucket,
            "is_fresh_activation": is_fresh,
            "is_persistent_background": is_background,
            "timing_note": wti_timing_note(
                int(latest.get("wti_tail_stress_age_days", 0) or 0),
                bool(latest.get("WTI_TAIL_STRESS_ACTIVATION", False)),
                bool(latest.get("WTI_TAIL_STRESS_FAST", False)),
                bool(latest.get("WTI_TAIL_STRESS_EXTENDED", False)),
                bool(latest.get("WTI_TAIL_STRESS_PERSISTENT_BACKGROUND", False)),
            ) if o.startswith("WTI_TAIL_STRESS") else "",
            "risk_stage": risk_stage,
            "best_yellow_lead_window": "1w" if o in ["RECESSION_FEAR_RATE_DOWN", "RECESSION_FEAR_EARLY"] else "",
            "best_prebreak_lead_window": "1w-2w" if o in FAST_RISK_OVERLAY else "3w-8w" if o in SLOW_BUILDUP_OVERLAY else "",
            "path_anatomy": STATIC_PATH.get(o, STATIC_PATH["FAST"] if o in FAST_RISK_OVERLAY else {}).get("path", ""),
            "historical_clustering_verdict": "IID_LIKE" if o in FAST_RISK_OVERLAY + SLOW_BUILDUP_OVERLAY + PERSISTENT_BACKGROUND_OVERLAY or o == "WTI_TAIL_STRESS" else "",
            "current_clustering_status": "IID_LIKE" if active and (o in FAST_RISK_OVERLAY + SLOW_BUILDUP_OVERLAY + PERSISTENT_BACKGROUND_OVERLAY or o == "WTI_TAIL_STRESS") else "",
            "sample_flag": "static_reference",
            "notes": "monitoring context only",
        })
    return pd.DataFrame(rows)


def cause_reference_table() -> pd.DataFrame:
    rows = [
        ("CAUSE_RECESSION_FEAR", "rates falling because growth/risk fear is active", "fast", "1 week / 5-10 trading days", "rebound-dominant but elevated movement risk", "leads yellow, can confirm prebreak"),
        ("CAUSE_OIL_TIGHTENING", "oil pressure has transmitted into rates or real yields", "slow-to-medium", "3-8 weeks depending on age", "elevated prebreak risk, not one-way bearish", "prebreak lead/confirmation"),
        ("CAUSE_RATE_BURDEN", "high-rate persistence or 4.75% pressure", "fast/burden", "1-2 weeks for confirmation; longer as burden", "rebound-dominant but elevated drawdown risk", "yellow/prebreak confirmation"),
        ("CAUSE_WTI_TAIL_FRESH", "WTI tail-stress newly activated or active for <=10 trading days", "fast", "1-2 weeks", "rebound-dominant but elevated movement risk", "fast monitor"),
        ("CAUSE_WTI_TAIL_EXTENDED", "WTI tail-stress active for 11-21 trading days", "extended", "already beyond initial fast window", "extended commodity stress, not a fresh trigger", "watch for persistence, do not treat as fresh trigger"),
        ("CAUSE_WTI_TAIL_PERSISTENT", "WTI tail-stress active for 22+ trading days", "persistent background", "background stress state", "persistent commodity-stress background", "context only unless paired with other active macro overlays"),
        ("CAUSE_WTI_TAIL", "WTI tail-stress pressure is active", "age-dependent", "depends on WTI tail-stress age", "use fresh/extended/persistent split for timing", "cause classifier"),
        ("CAUSE_GLD_SAFE_HAVEN", "GLD defensive persistence or safe-haven confirmation", "slow/background", "background / 4-8 weeks", "confirmation, not standalone", "confirms yellow/prebreak"),
        ("CAUSE_GROWTH_REFLATION", "rates up but equity tape accepts it", "benign/conflict", "none", "suppresses rate-panic interpretation", "monitor only"),
        ("CAUSE_MACRO_CALM", "no major macro cause active", "baseline-only", "none", "tape/baseline driven", "SPY baseline only"),
    ]
    return pd.DataFrame(rows, columns=["cause", "meaning", "fast_or_slow", "typical_risk_window", "path_anatomy_summary", "dashboard_use"])


def render_text(row: pd.Series, fast: list[str], slow: list[str], benign: list[str], background: list[str]) -> str:
    fast_note = "Fast signals historically matter over 5-10 trading days, but path is rebound/whipsaw dominant, not one-way bearish."
    wti_background_note = "This is not a fresh 1-2 week WTI tail-stress trigger. It is persistent commodity-stress background. SPY baseline remains yellow; no fresh fast macro trigger is active unless another fast overlay fires."
    cluster_note = "Large moves may be frequent, but completed audit found mostly IID-like count structure after signals."
    return f"""# Macro-Enhanced SPY Dashboard Standalone

## Current SPY Baseline
- date: {row['date']}
- close: {row['spy_close']:.2f}
- state: {row['spy_final_state_label']}
- daily label: {row['spy_final_daily_label']}
- group: {row['spy_baseline_group']}

## Current Macro Cause
- cause: {row['current_macro_cause']}
- fast overlays: {row['active_fast_overlays']}
- slow overlays: {row['active_slow_overlays']}
- benign/conflict overlays: {row['active_benign_or_conflict_overlays']}
- persistent background overlays: {row['persistent_background_overlays']}

## Fast / Slow Risk Timing
- timing: {row['risk_timing']}
- WTI tail stress age: {row['wti_tail_stress_age_days']} trading days
- WTI tail stress age bucket: {row['wti_tail_stress_age_bucket']}
- WTI timing note: {row['wti_tail_stress_timing_note']}
- slow signal ages: {row['slow_signal_age_summary']}
- note: {wti_background_note if background and not fast else fast_note if fast else 'No active fast macro overlay.'}

## Expected Path Anatomy
- path anatomy: {row['path_anatomy_label']}
- drawdown/rebound bias: {row['drawdown_rebound_bias']}
- clustering: {row['clustering_status']}
- clustering note: {cluster_note if row['clustering_status'] == 'IID_LIKE' else 'No active overlay clustering signal.'}

## Dashboard Verdict
- {row['current_dashboard_verdict']}

## Monitoring Note
- Monitoring only, no trading rule.
- SPY baseline remains action gate.
- Macro overlay is context, confirmation, timing, and cause classification only.
"""


def render_summary(row: pd.Series) -> str:
    return "\n".join([
        f"latest date: {row['date']}",
        f"SPY baseline: {row['spy_final_state_label']} / {row['spy_final_daily_label']} / {row['spy_baseline_group']}",
        f"current cause: {row['current_macro_cause']}",
        f"fresh fast overlays: {row['fresh_fast_overlays']}",
        f"persistent background overlays: {row['persistent_background_overlays']}",
        f"active overlays: slow={row['active_slow_overlays']} benign={row['active_benign_or_conflict_overlays']}",
        f"WTI tail stress age: {row['wti_tail_stress_age_days']} trading days ({row['wti_tail_stress_age_bucket']})",
        f"risk timing: {row['risk_timing']}",
        f"dashboard verdict: {row['current_dashboard_verdict']}",
        "output files: macro_enhanced_current_dashboard.csv, macro_enhanced_current_dashboard.txt, macro_overlay_signal_status.csv, macro_cause_reference_table.csv, dashboard_file_audit.csv, github_action_summary.txt",
    ])


def html_value(value: object) -> str:
    return html.escape(str(value), quote=False)


def html_line(label: str, value: object) -> str:
    return f"<b>{html.escape(label, quote=False)}:</b> <code>{html_value(value)}</code>"


def render_summary_html(row: pd.Series) -> str:
    output_files = "macro_enhanced_current_dashboard.csv, macro_enhanced_current_dashboard.txt, macro_overlay_signal_status.csv, macro_cause_reference_table.csv, dashboard_file_audit.csv, github_action_summary.txt"
    return "\n".join([
        "<b>Macro-Enhanced SPY Dashboard</b>",
        html_line("latest date", row["date"]),
        html_line("SPY baseline", f"{row['spy_final_state_label']} / {row['spy_final_daily_label']} / {row['spy_baseline_group']}"),
        html_line("current cause", row["current_macro_cause"]),
        html_line("fresh fast overlays", row["fresh_fast_overlays"]),
        html_line("persistent background overlays", row["persistent_background_overlays"]),
        html_line("active overlays", f"slow={row['active_slow_overlays']} benign={row['active_benign_or_conflict_overlays']}"),
        html_line("WTI tail stress age", f"{row['wti_tail_stress_age_days']} trading days ({row['wti_tail_stress_age_bucket']})"),
        html_line("risk timing", row["risk_timing"]),
        html_line("dashboard verdict", row["current_dashboard_verdict"]),
        html_line("output files", output_files),
    ])


def main() -> None:
    args = parse_args()
    if args.allow_download:
        print("--allow-download was set, but downloads are not implemented in this standalone version.")
    base = Path(args.base_dir)
    panel, audits = load_inputs(base)
    build_dashboard(panel, audits, args.as_of_date, Path(args.output_dir))


if __name__ == "__main__":
    main()
