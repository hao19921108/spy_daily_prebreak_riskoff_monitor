# Macro-Enhanced SPY Dashboard

This folder contains the standalone macro-enhanced SPY dashboard:

```text
projects/spy_risk_monitor_macro_enhanced/macro_enhanced_spy_dashboard_standalone.py
```

The dashboard is the analysis engine. It reads prepared CSV inputs from this
project's `data/` folder and writes dashboard outputs to this project's `outputs/`
folder. Data downloading is handled separately by this project's `fetch_data.py`.

Project dependencies are listed in:

```text
projects/spy_risk_monitor_macro_enhanced/requirements.txt
```

## Relationship To Baseline Project

The macro-enhanced dashboard is based on the baseline SPY risk monitor in:

```text
projects/spy_risk_monitor/
```

The baseline project provides the SPY regime/risk-monitoring foundation. This macro
version keeps SPY baseline state as the action gate, then layers macro context on top:
rates, inflation breakevens, real yields, oil, gold, and VIX. The macro overlays are
used for cause classification, timing context, and monitoring detail.

Detailed research context is in the pinned chats named:

- `SPY regime study - Escalation`
- `SPY regime study - Macro Enhanced`

## Inputs

Run the project ingestion script before running the dashboard:

```bash
python fetch_data.py
```

Expected local input files:

```text
data/SPY_20y.csv
data/GLD_20y.csv
data/CL=F_20y.csv
data/DGS2.csv
data/DGS10.csv
data/T5YIE.csv
data/T10YIE.csv
data/DFII10.csv
data/^VIX_20y.csv
```

Market inputs come from `yfinance`:

- `SPY`
- `GLD`
- `CL=F`
- `^VIX`

Macro/rate inputs come from FRED CSV downloads:

- `DGS2`
- `DGS10`
- `T5YIE`
- `T10YIE`
- `DFII10`

## Run Locally

From this folder:

```bash
python fetch_data.py
python macro_enhanced_spy_dashboard_standalone.py --base-dir . --output-dir outputs/macro_enhanced_spy_dashboard_standalone
```

The `--base-dir .` argument tells the dashboard to read this folder's `data/`.

## Outputs

The dashboard writes:

```text
outputs/macro_enhanced_spy_dashboard_standalone/github_action_summary.txt
outputs/macro_enhanced_spy_dashboard_standalone/macro_enhanced_current_dashboard.csv
outputs/macro_enhanced_spy_dashboard_standalone/macro_enhanced_current_dashboard.txt
outputs/macro_enhanced_spy_dashboard_standalone/macro_overlay_signal_status.csv
outputs/macro_enhanced_spy_dashboard_standalone/macro_cause_reference_table.csv
outputs/macro_enhanced_spy_dashboard_standalone/dashboard_file_audit.csv
```

`github_action_summary.txt` is the file sent to Telegram by the macro Telegram
workflow.

## GitHub Actions

The repo-level macro Telegram workflow is:

```text
.github/workflows/macro_dashboard_telegram.yml
```

It runs Monday-Friday at `22:00 UTC` and can also be started manually with
`workflow_dispatch`.

Workflow order:

```text
python fetch_data.py
python macro_enhanced_spy_dashboard_standalone.py --base-dir . --output-dir outputs/macro_enhanced_spy_dashboard_standalone
send outputs/macro_enhanced_spy_dashboard_standalone/github_action_summary.txt to Telegram
commit updated data/ and outputs/ back to the repo
```

Telegram delivery uses the shared helper at:

```text
shared/send_telegram_report.py
```

The helper is generic. It sends `github_action_summary.txt` as the message body,
so it works for this two-step `fetch_data.py` plus dashboard pipeline without
needing dashboard-specific Telegram formatting.

There is also an artifact workflow:

```text
.github/workflows/macro_dashboard.yml
```

It runs Monday-Friday at `22:30 UTC`, can be started manually, and uploads the
dashboard output directory as a GitHub Actions artifact.

## Scope

This dashboard is monitoring context only. It does not generate buy/sell rules,
optimize returns, or create position sizing.
