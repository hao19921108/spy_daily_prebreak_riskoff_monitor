#!/usr/bin/env python3
"""Download market and macro inputs for the standalone SPY dashboard."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MARKET_SERIES = {
    "SPY": "SPY_20y.csv",
    "GLD": "GLD_20y.csv",
    "CL=F": "CL=F_20y.csv",
    "^VIX": "^VIX_20y.csv",
}

FRED_SERIES = {
    "DGS2": "DGS2.csv",
    "DGS10": "DGS10.csv",
    "T5YIE": "T5YIE.csv",
    "T10YIE": "T10YIE.csv",
    "DFII10": "DFII10.csv",
}

YFINANCE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass
class AuditRow:
    source: str
    output_file: str
    rows: int
    start_date: str
    end_date: str
    status: str


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(str(part) for part in col if part not in ("", None)).strip()
            for col in df.columns.to_flat_index()
        ]
    return df


def normalize_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rename_map = {}
    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in df.columns:
        col_text = str(col).strip()
        for base in expected:
            if col_text == base or col_text in {f"{ticker} {base}", f"{base} {ticker}"}:
                rename_map[col] = base

    df = df.rename(columns=rename_map)
    present_cols = ["Date"] + [col for col in expected if col in df.columns]
    return df[present_cols]


def write_market_csv(ticker: str, output_path: Path, start: date, end: date) -> AuditRow:
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
        actions=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"{ticker}: yfinance returned no rows")

    df = flatten_columns(df)
    df = df.reset_index()
    if "Date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.dropna(subset=["Date"])
    df = normalize_yfinance_columns(df, ticker)
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")

    for col in YFINANCE_COLUMNS:
        if col != "Date" and col not in df.columns:
            df[col] = np.nan
    df = df[YFINANCE_COLUMNS]

    if df.empty:
        raise ValueError(f"{ticker}: normalized market data has zero rows")

    df.to_csv(output_path, index=False)
    return audit_success(f"yfinance:{ticker}", output_path, df)


def write_fred_csv(series: str, output_path: Path, start: date, end: date) -> AuditRow:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    if df.empty:
        raise ValueError(f"{series}: FRED returned no rows")

    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    value_col = series if series in df.columns else df.columns[-1]
    df = df.rename(columns={date_col: "Date", value_col: series})
    df = df[["Date", series]]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df[series] = pd.to_numeric(df[series].replace(".", np.nan), errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")

    if df.empty:
        raise ValueError(f"{series}: normalized FRED data has zero rows")

    df.to_csv(output_path, index=False)
    return audit_success(f"FRED:{series}", output_path, df)


def audit_success(source: str, output_path: Path, df: pd.DataFrame) -> AuditRow:
    return AuditRow(
        source=source,
        output_file=str(output_path.relative_to(BASE_DIR)),
        rows=len(df),
        start_date=str(df["Date"].iloc[0]),
        end_date=str(df["Date"].iloc[-1]),
        status="ok",
    )


def audit_failure(source: str, output_path: Path, error: Exception) -> AuditRow:
    return AuditRow(
        source=source,
        output_file=str(output_path.relative_to(BASE_DIR)),
        rows=0,
        start_date="-",
        end_date="-",
        status=f"failed: {error}",
    )


def print_audit(rows: list[AuditRow]) -> None:
    headers = ["source", "output file", "rows", "start date", "end date", "status"]
    table_rows = [
        [row.source, row.output_file, str(row.rows), row.start_date, row.end_date, row.status]
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    print("\nData fetch audit summary")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start = date.today() - timedelta(days=365 * 20 + 10)
    end = date.today()
    audit_rows: list[AuditRow] = []

    for ticker, filename in MARKET_SERIES.items():
        output_path = DATA_DIR / filename
        try:
            audit_rows.append(write_market_csv(ticker, output_path, start, end))
        except Exception as exc:
            audit_rows.append(audit_failure(f"yfinance:{ticker}", output_path, exc))

    for series, filename in FRED_SERIES.items():
        output_path = DATA_DIR / filename
        try:
            audit_rows.append(write_fred_csv(series, output_path, start, end))
        except Exception as exc:
            audit_rows.append(audit_failure(f"FRED:{series}", output_path, exc))

    print_audit(audit_rows)

    failed = [row for row in audit_rows if row.rows == 0 or not row.status.startswith("ok")]
    if failed:
        print(f"\nFailed to create {len(failed)} required file(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
