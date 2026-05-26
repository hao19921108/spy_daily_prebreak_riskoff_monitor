# Frozen Macro Risk Monitoring Repo

This repo is a modular workspace for related SPY and macro risk-monitoring
projects. Each project lives under `projects/`, keeps its own source files and
project-level dependencies, and writes runtime artifacts inside its own project
folder. Repo-level GitHub Actions orchestrate the projects.

## File Architecture

```text
.
├── requirements.txt
├── shared/
│   └── send_telegram_report.py
├── projects/
│   ├── macro_regime/
│   │   ├── regime_classifier.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── spy_risk_monitor/
│   │   ├── spy_daily_prebreak_riskoff_monitor.py
│   │   ├── send_telegram_report.py
│   │   └── requirements.txt
│   ├── spy_risk_monitor_macro_enhanced/
│       ├── fetch_data.py
│       ├── macro_enhanced_spy_dashboard_standalone.py
│       ├── requirements.txt
│       ├── data/
│       ├── outputs/
│       └── README.md
│   └── spy_risk_monitor_macro_cjm_enhanced/
│       ├── build_daily_composite_regime_from_raw.py
│       ├── requirements.txt
│       ├── outputs/
│       └── README.md
└── .github/workflows/
```

The root `requirements.txt` is an aggregator only:

```text
-r projects/macro_regime/requirements.txt
-r projects/spy_risk_monitor/requirements.txt
-r projects/spy_risk_monitor_macro_enhanced/requirements.txt
-r projects/spy_risk_monitor_macro_cjm_enhanced/requirements.txt
```

This keeps GitHub Actions simple while allowing each project to remain independently
maintainable.

## Projects

### Macro Regime Classifier

Path: `projects/macro_regime/`

Main script: `regime_classifier.py`

This is the moved `macro-regime-bot` project. It uses a 252-trading-day lookback
and `yfinance` data for `SPY`, `GLD`, `VDE`, and the 10-year Treasury yield proxy
`^TNX`. It builds trend/rates signals, computes 20-day realized volatility for
`SPY`, combines the inputs into a macro score, and classifies the environment as
`Risk-On`, `Constructive`, `Neutral`, `Defensive`, or `Risk-Off`.

Workflow: `.github/workflows/daily_regime_telegram.yml`

Schedule: Monday-Friday at `22:00 UTC`, plus manual `workflow_dispatch`.

Telegram: the workflow extracts the `=== TELEGRAM SUMMARY ===` block from script
stdout and sends it through `shared/send_telegram_report.py`.

Historical repo: `https://github.com/hao19921108/macro-regime-bot`

### Baseline SPY Risk Monitor

Path: `projects/spy_risk_monitor/`

Main script: `spy_daily_prebreak_riskoff_monitor.py`

This is the baseline SPY risk monitor. It is the foundation for the macro-enhanced
SPY dashboard and remains useful as the simpler baseline project.

Workflow: `.github/workflows/daily_spy_monitor.yml`

Schedule: Monday-Friday at `23:30 UTC`, plus manual `workflow_dispatch`.

Telegram: this project still uses its project-local `send_telegram_report.py`
because that helper formats baseline-specific CSV fields and sends baseline-specific
attachments.

### Macro-Enhanced SPY Dashboard

Path: `projects/spy_risk_monitor_macro_enhanced/`

Main scripts:

- `fetch_data.py`
- `macro_enhanced_spy_dashboard_standalone.py`

The macro-enhanced dashboard is based on the baseline SPY risk monitor in
`projects/spy_risk_monitor/`. It keeps SPY baseline state as the action gate, then
adds macro context from rates, inflation breakevens, real yields, oil, gold, and
VIX. Detailed research context is in the pinned chats named `SPY regime study -
Escalation` and `SPY regime study - Macro Enhanced`.

Workflow: `.github/workflows/macro_dashboard_telegram.yml`

Schedule: Monday-Friday at `22:00 UTC`, plus manual `workflow_dispatch`.

Workflow order:

```bash
cd projects/spy_risk_monitor_macro_enhanced
python fetch_data.py
python macro_enhanced_spy_dashboard_standalone.py --base-dir . --output-dir outputs/macro_enhanced_spy_dashboard_standalone
```

Telegram: the workflow sends
`projects/spy_risk_monitor_macro_enhanced/outputs/macro_enhanced_spy_dashboard_standalone/github_action_summary.txt`
through `shared/send_telegram_report.py`.

### Composite / CJM-Enhanced SPY Regime Monitor

Path: `projects/spy_risk_monitor_macro_cjm_enhanced/`

Main script: `build_daily_composite_regime_from_raw.py`

Workflow: `.github/workflows/composite_cjm_telegram.yml`

Purpose: Downstream composite regime daily build/review layer. Sends
`github_action_summary.txt` to Telegram and uploads CSV/MD daily outputs as
artifacts.

Scope: Regime classification only. No trading advice.

## Macro-Enhanced Data Sources

Market series come from `yfinance`:

- `SPY` -> `projects/spy_risk_monitor_macro_enhanced/data/SPY_20y.csv`
- `GLD` -> `projects/spy_risk_monitor_macro_enhanced/data/GLD_20y.csv`
- `CL=F` -> `projects/spy_risk_monitor_macro_enhanced/data/CL=F_20y.csv`
- `^VIX` -> `projects/spy_risk_monitor_macro_enhanced/data/^VIX_20y.csv`

Macro and rate series come from FRED CSV download URLs:

- `DGS2` -> `projects/spy_risk_monitor_macro_enhanced/data/DGS2.csv`
- `DGS10` -> `projects/spy_risk_monitor_macro_enhanced/data/DGS10.csv`
- `T5YIE` -> `projects/spy_risk_monitor_macro_enhanced/data/T5YIE.csv`
- `T10YIE` -> `projects/spy_risk_monitor_macro_enhanced/data/T10YIE.csv`
- `DFII10` -> `projects/spy_risk_monitor_macro_enhanced/data/DFII10.csv`

The FRED URL pattern is:

```text
https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
```

## Shared Telegram Helper

`shared/send_telegram_report.py` is intentionally generic. It sends a text file as
the Telegram message body, supports an optional title, and can attach optional files.
It does not know project-specific dataframe columns or trading logic.

Example:

```bash
python shared/send_telegram_report.py \
  --title "Daily Macro Regime" \
  --message-file projects/macro_regime/telegram_summary.txt
```

Project-specific formatting should happen inside the project or workflow before the
shared helper is called.

## Maintenance Rules

- Keep source code inside the relevant `projects/<project_name>/` folder.
- Keep project dependencies in that project's `requirements.txt`.
- Keep the root `requirements.txt` as an aggregator of project requirement files.
- Keep generated data and outputs inside the project that owns them.
- Put reusable, project-agnostic helpers in `shared/`.
- Do not put analysis, signal, or portfolio logic in shared Telegram or workflow
  helpers.
- Use repo-level workflows only to install, run project commands, send summaries,
  upload artifacts, or commit generated project artifacts.

## Telegram Secrets

For workflows that send Telegram messages, define these GitHub Actions secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Paste only the raw values into GitHub secrets. Avoid quotes, spaces, or extra line
breaks.
