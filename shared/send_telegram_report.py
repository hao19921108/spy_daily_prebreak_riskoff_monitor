#!/usr/bin/env python3
"""Send a text report and optional files to Telegram."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a report file to Telegram.")
    parser.add_argument("--message-file", required=True, help="Text file to send as the Telegram message.")
    parser.add_argument("--title", default="", help="Optional title prepended to the message.")
    parser.add_argument("--attach", action="append", default=[], help="Optional file attachment. Can be repeated.")
    parser.add_argument("--max-chars", type=int, default=3900, help="Maximum message length before truncation.")
    parser.add_argument("--parse-mode", choices=["HTML", "MarkdownV2"], default=None, help="Optional Telegram parse mode.")
    return parser.parse_args()


def clean_secret(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        sys.exit(f"Missing {name}")
    if any(ord(ch) < 32 for ch in value):
        sys.exit(f"{name} contains a control character. Re-save the GitHub secret without extra line breaks.")
    return value


def send_message(token: str, chat_id: str, text: str, parse_mode: str | None = None) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    response = requests.post(
        url,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()


def send_document(token: str, chat_id: str, path: Path, caption: str = "") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Attachment does not exist: {path}")

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with path.open("rb") as file_obj:
        response = requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption},
            files={"document": (path.name, file_obj)},
            timeout=60,
        )
    response.raise_for_status()


def main() -> int:
    args = parse_args()
    token = clean_secret("TELEGRAM_BOT_TOKEN")
    chat_id = clean_secret("TELEGRAM_CHAT_ID")

    message_path = Path(args.message_file)
    if not message_path.exists():
        sys.exit(f"Message file does not exist: {message_path}")

    body = message_path.read_text(encoding="utf-8").strip()
    if not body:
        sys.exit(f"Message file is empty: {message_path}")

    text = f"{args.title.strip()}\n\n{body}".strip() if args.title else body
    if len(text) > args.max_chars:
        text = text[: args.max_chars].rstrip() + "\n...[truncated]"
    send_message(token, chat_id, text, args.parse_mode)

    for attachment in args.attach:
        path = Path(attachment)
        send_document(token, chat_id, path, path.name)

    print("Telegram report sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
