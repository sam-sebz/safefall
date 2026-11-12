"""Finds your Telegram chat id, saves it to .env, and sends a test message.

Steps: put your bot token in .env, send your bot a message, then run this script.
"""
import os
import sys

import requests
from dotenv import dotenv_values, set_key

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def call(token, method, **params):
    try:
        response = requests.post(f"https://api.telegram.org/bot{token}/{method}",
                                 data=params, timeout=15)
        return response.status_code, response.json()
    except (requests.RequestException, ValueError) as error:
        # The exception text contains the URL, and so the token: show only its type.
        sys.exit(f"Could not reach Telegram ({type(error).__name__}). Check your internet.")


def main():
    if not os.path.exists(ENV_PATH):
        sys.exit("No .env file found. Copy .env.example to .env first.")
    token = (dotenv_values(ENV_PATH).get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN is empty in .env. Paste your token there and save.")

    status, body = call(token, "getUpdates")
    if status in (401, 404):
        sys.exit("Telegram rejected the token. Paste the newest token from BotFather into .env.")
    if not body.get("ok"):
        sys.exit(f"Telegram returned an error: HTTP {status}.")

    chats = [u["message"]["chat"]["id"] for u in body["result"] if "message" in u]
    if not chats:
        sys.exit("No messages found. Send your bot a message in Telegram, then run this again.")

    chat_id = str(chats[-1])
    set_key(ENV_PATH, "TELEGRAM_CHAT_IDS", chat_id, quote_mode="never")
    print(f"Saved chat id {chat_id} to .env")

    status, body = call(token, "sendMessage", chat_id=chat_id,
                        text="SafeFall test: alerts will arrive in this chat.")
    if not body.get("ok"):
        sys.exit(f"Could not send the test message: HTTP {status}.")
    print("Test message sent. Check Telegram.")


if __name__ == "__main__":
    main()
