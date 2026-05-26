# Composite / CJM-Enhanced SPY Regime Monitor

This project contains the Composite / CJM-enhanced SPY regime monitor. It is a
downstream daily build and review layer for the baseline SPY risk monitor and
the macro-enhanced SPY dashboard.

The project is regime-classification research only. It does not generate trading
advice, position sizing, portfolio actions, buy/sell rules, or trading
inference. Forward return and drawdown fields are descriptive only.

## Main Script

```text
build_daily_composite_regime_from_raw.py
```

The script builds the daily composite regime panel directly from local/raw
market files plus cached or fetched FRED macro data. It prepares a latest
snapshot, data-quality checks, a phase reference table, a daily tail panel, a
Telegram-ready summary, and a reviewer report.

## Local Run

```bash
python build_daily_composite_regime_from_raw.py --mode daily --tail-days 252
```

## GitHub Actions

The GitHub Actions workflow uses `github_action_summary.txt` as the Telegram
message body and uploads generated CSV/MD files as artifacts. It shares the same
Telegram bot and GitHub Secrets as the other SPY regime classification
workflows:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

The project may share FRED and macro data sources later, but for now it should
remain self-contained. `FRED_API_KEY` may be available to future data-fetching
steps, but this daily builder does not require it.

## Expected Outputs

```text
outputs/daily_composite_regime/latest_composite_regime_snapshot.csv
outputs/daily_composite_regime/github_action_summary.txt
outputs/daily_composite_regime/daily_composite_regime_reviewer_report.md
outputs/daily_composite_regime/data_quality_flags.csv
outputs/daily_composite_regime/composite_phase_reference_table.csv
outputs/daily_composite_regime/daily_composite_regime_panel_tail.csv
```
