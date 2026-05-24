# Macro Regime Classifier

This project contains the daily macro regime classifier:

```text
projects/macro_regime/regime_classifier.py
```

It is the project moved from the original `macro-regime-bot` repo. The linked repo
for historical context is:

```text
https://github.com/hao19921108/macro-regime-bot
```

## What It Does

The classifier downloads recent market data with `yfinance`, keeps a 252-trading-day
lookback, and classifies the current macro regime using:

- `SPY` for equity trend
- `GLD` for defensive/gold trend
- `VDE` for energy/equity-sector trend
- `^TNX` for the 10-year Treasury yield proxy

It builds simple trend signals for `SPY`, `GLD`, and `VDE`, adds a rates signal from
the recent direction of `^TNX`, calculates 20-day realized volatility for `SPY`, and
combines those inputs into a macro score. The score maps to a regime label such as
`Risk-On`, `Constructive`, `Neutral`, `Defensive`, or `Risk-Off`, plus a final
exposure percentage.

## Run Locally

From this folder:

```bash
python regime_classifier.py
```

The script prints a full console report and a compact Telegram block beginning with:

```text
=== TELEGRAM SUMMARY ===
```

## GitHub Actions

The repo-level workflow is:

```text
.github/workflows/daily_regime_telegram.yml
```

It runs Monday-Friday at `22:00 UTC` and can also be started manually with
`workflow_dispatch`.

Workflow order:

```text
python regime_classifier.py | tee daily_regime_output.txt
extract the === TELEGRAM SUMMARY === block to telegram_summary.txt
python shared/send_telegram_report.py --title "Daily Macro Regime" --message-file projects/macro_regime/telegram_summary.txt
```

Telegram delivery uses the shared helper at:

```text
shared/send_telegram_report.py
```

The helper sends the extracted text summary. The classifier itself does not send
Telegram messages directly.

## Outputs

Generated local workflow files are:

```text
projects/macro_regime/daily_regime_output.txt
projects/macro_regime/telegram_summary.txt
```

These are runtime artifacts, not source files.
