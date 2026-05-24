import os
import textwrap
from pathlib import Path

import pandas as pd
import requests


OUT_DIR = Path("outputs/spy_daily_prebreak_monitor")
REPORT_CSV = OUT_DIR / "current_state_report.csv"
REPORT_TXT = OUT_DIR / "current_state_report.txt"
AUDIT_CSV = OUT_DIR / "output_file_audit.csv"


def send_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    r.raise_for_status()


def send_document(token: str, chat_id: str, path: Path, caption: str = "") -> None:
    if not path.exists():
        return

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with path.open("rb") as f:
        r = requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption},
            files={"document": (path.name, f)},
            timeout=60,
        )
    r.raise_for_status()


def pct(x):
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "n/a"


def main() -> int:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    report = pd.read_csv(REPORT_CSV).iloc[0]

    audit_status = "unknown"
    if AUDIT_CSV.exists():
        audit = pd.read_csv(AUDIT_CSV)
        if {"exists", "required_columns_present"}.issubset(audit.columns):
            audit_status = "PASS" if bool(audit["exists"].all() and audit["required_columns_present"].all()) else "FAIL"

    text = f"""
<b>SPY Daily Risk Monitor</b>

Date: {report.get("date", "n/a")}
Close: {report.get("close", "n/a")}
Daily return: {pct(report.get("ret"))}

Final label: <b>{report.get("final_daily_label", "n/a")}</b>
State: {report.get("final_state_label", "n/a")}
Risk score: {report.get("risk_score", "n/a")} / {report.get("risk_score_bucket", "n/a")}

Distance to MA200: {pct(report.get("distance_to_MA200"))}
MA200 state: {report.get("MA200_state", "n/a")}
RV21 state: {report.get("RV21_state", "n/a")}
Vol transition: {report.get("vol_transition_state", "n/a")}
Whipsaw 10d 1%: {report.get("recent_whipsaw_10d_100", "n/a")}

Old MA/ATR state: {report.get("old_ma_atr_state", "n/a")}
Old MA/ATR signal: {report.get("old_ma_atr_signal", "n/a")}

Production ready: {report.get("production_ready", "n/a")}
Audit: {audit_status}

No trading recommendation is made.
""".strip()

    # Telegram message limit is 4096 chars; keep this compact.
    send_message(token, chat_id, textwrap.shorten(text, width=3900, placeholder="\n..."))

    send_document(token, chat_id, REPORT_TXT, "Full text report")
    send_document(token, chat_id, REPORT_CSV, "Current state CSV")
    send_document(token, chat_id, OUT_DIR / "current_state_probabilities.csv", "Current probabilities CSV")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())