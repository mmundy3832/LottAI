"""
telegram_bot.py - Send a message via Telegram bot.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
"""

import os, sys, requests


def send_message(text, parse_mode=None):
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the environment"
        )

    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    if parse_mode:
        data["parse_mode"] = parse_mode

    resp = requests.post(url, data=data, timeout=15)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "LottAI test message"
    result = send_message(msg)
    print("Sent." if result.get("ok") else f"Error: {result}")
