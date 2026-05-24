from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf


LOOKBACK_DAYS = 252
TICKERS = ("SPY", "GLD", "VDE")
RATE_TICKER = "^TNX"


@dataclass(frozen=True)
class Signal:
    name: str
    value: int
    label: str


def _download_close(ticker: str, period: str = "18mo") -> pd.Series:
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close", ticker] if ("Close", ticker) in data.columns else data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    close = close.dropna()
    if close.empty:
        raise RuntimeError(f"No close prices returned for {ticker}")

    close.name = ticker
    return close


def load_market_data() -> pd.DataFrame:
    closes = [_download_close(ticker) for ticker in TICKERS]
    prices = pd.concat(closes, axis=1).dropna()

    if prices.empty:
        raise RuntimeError("No overlapping yfinance data returned for SPY, GLD, and VDE")

    return prices.tail(LOOKBACK_DAYS)


def load_rates_yahoo() -> pd.Series:
    return _download_close(RATE_TICKER).tail(LOOKBACK_DAYS)


def trend_signal(series: pd.Series, short_window: int = 50, long_window: int = 200) -> Signal:
    if len(series) < long_window:
        raise RuntimeError(f"Need at least {long_window} observations for {series.name}")

    short_ma = series.rolling(short_window).mean().iloc[-1]
    long_ma = series.rolling(long_window).mean().iloc[-1]
    value = 1 if short_ma > long_ma else -1
    label = "Positive" if value > 0 else "Negative"
    return Signal(str(series.name), value, label)


def rates_signal(rates: pd.Series | None) -> Signal:
    if rates is None or rates.empty or len(rates.dropna()) < 21:
        return Signal("Rates", 0, "Neutral (^TNX unavailable)")

    latest = rates.dropna().iloc[-1]
    one_month_ago = rates.dropna().iloc[-21]
    value = -1 if latest > one_month_ago else 1
    label = "Negative (rates rising)" if value < 0 else "Positive (rates falling)"
    return Signal("Rates", value, label)


def realized_vol(series: pd.Series, window: int = 20) -> float:
    returns = series.pct_change().dropna()
    if len(returns) < window:
        raise RuntimeError(f"Need at least {window + 1} observations for realized volatility")

    return float(returns.tail(window).std() * math.sqrt(252))


def classify_regime(macro_score: int, vol_20d: float) -> tuple[str, str]:
    if macro_score >= 2 and vol_20d < 0.20:
        return "Risk-On", "100%"
    if macro_score >= 1:
        return "Constructive", "75%"
    if macro_score == 0:
        return "Neutral", "50%"
    if macro_score == -1:
        return "Defensive", "25%"
    return "Risk-Off", "0%"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    prices = load_market_data()
    as_of = prices.index[-1].date()

    spy_signal = trend_signal(prices["SPY"])
    gld_signal = trend_signal(prices["GLD"])
    vde_signal = trend_signal(prices["VDE"])

    try:
        rates = load_rates_yahoo()
        rate_signal = rates_signal(rates)
    except Exception as exc:
        rate_signal = Signal("Rates", 0, f"Neutral (^TNX fetch failed: {exc})")

    vol_20d = realized_vol(prices["SPY"])
    macro_score = int(np.sum([spy_signal.value, gld_signal.value, vde_signal.value, rate_signal.value]))
    regime, exposure = classify_regime(macro_score, vol_20d)

    print("Daily Macro Regime Classification")
    print(f"As of: {as_of}")
    print(f"SPY Signal: {spy_signal.label}")
    print(f"GLD Signal: {gld_signal.label}")
    print(f"VDE Signal: {vde_signal.label}")
    print(f"Rates Signal: {rate_signal.label}")
    print(f"20d Realized Vol: {format_percent(vol_20d)}")
    print(f"Macro Score: {macro_score}")
    print(f"Regime: {regime}")
    print(f"Final Exposure: {exposure}")
    print()
    print("=== TELEGRAM SUMMARY ===")
    print(f"Date: {as_of}")
    print(f"Macro Score: {macro_score}")
    print(f"Regime: {regime}")
    print(f"Final Exposure: {exposure}")
    print(f"20d Realized Vol: {format_percent(vol_20d)}")
    print(f"GLD Signal: {gld_signal.label}")
    print(f"VDE Signal: {vde_signal.label}")
    print(f"Rates Signal: {rate_signal.label}")


if __name__ == "__main__":
    main()
