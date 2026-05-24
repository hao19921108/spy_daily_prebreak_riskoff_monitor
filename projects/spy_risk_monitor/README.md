# Baseline SPY Risk Monitor

This project contains the baseline SPY daily prebreak/risk-off monitor:

```text
projects/spy_risk_monitor/spy_daily_prebreak_riskoff_monitor.py
```

It is the simpler baseline project that the macro-enhanced dashboard builds on.
The project-local Telegram sender remains here because it formats baseline-specific
CSV fields and sends baseline-specific report attachments.

## Run Locally

From this folder:

```bash
python spy_daily_prebreak_riskoff_monitor.py \
  --download \
  --output-dir outputs/spy_daily_prebreak_monitor \
  --overwrite-log \
  --signal-log-mode latest_only \
  --quiet
```

To send the generated report to Telegram:

```bash
python send_telegram_report.py
```

The sender expects `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the environment.

## GitHub Actions

The repo-level workflow is:

```text
.github/workflows/daily_spy_monitor.yml
```

It runs Monday-Friday at `23:30 UTC` and can also be started manually with
`workflow_dispatch`.

Generated outputs live under:

```text
projects/spy_risk_monitor/outputs/spy_daily_prebreak_monitor/
```
