"""Sends fall alerts to phones through the app and Telegram, and optionally to MQTT."""
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

import paho.mqtt.client as mqtt
import requests

log = logging.getLogger("safefall.alerts")


def _chat_ids(name: str) -> List[str]:
    return [c.strip() for c in os.environ.get(name, "").split(",") if c.strip()]


def build_caption(location: str, camera: str, when: str, people_down: int,
                  escalation: bool = False) -> str:
    title = "STILL DOWN - NO RESPONSE YET" if escalation else "FALL DETECTED"
    return (f"{title}\n"
            f"Location: {location}\n"
            f"Camera: {camera}\n"
            f"Time: {when}\n"
            f"People down: {people_down}")


@dataclass
class Alert:
    caption: str
    jpeg: bytes
    escalation: bool = False


class TelegramAlerter:
    """Queues alerts and sends them on a background thread so detection never waits."""

    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_ids = _chat_ids("TELEGRAM_CHAT_IDS")
        self.escalation_chat_ids = _chat_ids("TELEGRAM_ESCALATION_CHAT_IDS") or self.chat_ids
        self.enabled = bool(self.token and self.chat_ids)
        self.queue: "queue.Queue[Alert]" = queue.Queue()
        if self.enabled:
            threading.Thread(target=self._run, daemon=True, name="telegram").start()
        else:
            log.warning("Telegram is not configured: set TELEGRAM_BOT_TOKEN and "
                        "TELEGRAM_CHAT_IDS. Falls will only be logged.")

    def send(self, alert: Alert) -> None:
        if self.enabled:
            self.queue.put(alert)
        else:
            log.warning("Alert not sent (Telegram not configured): %s",
                        alert.caption.replace("\n", " | "))

    def _run(self) -> None:
        while True:
            alert = self.queue.get()
            for chat_id in (self.escalation_chat_ids if alert.escalation else self.chat_ids):
                if not (self._post(chat_id, alert) or self._post(chat_id, alert)):
                    log.error("Giving up on Telegram alert to chat %s", chat_id)

    def _post(self, chat_id: str, alert: Alert) -> bool:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendPhoto",
                data={"chat_id": chat_id, "caption": alert.caption},
                files={"photo": ("fall.jpg", alert.jpeg, "image/jpeg")},
                timeout=15,
            )
        except requests.RequestException as error:
            # The exception text contains the URL, and so the token: log only its type.
            log.error("Telegram request failed: %s", type(error).__name__)
            return False
        if response.ok:
            log.info("Telegram alert sent to chat %s", chat_id)
            return True
        log.error("Telegram rejected the alert: HTTP %s", response.status_code)
        return False


@dataclass
class PushMessage:
    title: str
    body: str
    alert_id: int


class PushAlerter:
    """Queues alerts and sends them to the phone app through Expo's push service.

    Messages carry text and the alert id only. The photo stays on this server, and the app
    fetches it after signing in.
    """

    URL = "https://exp.host/--/api/v2/push/send"
    BATCH_SIZE = 100  # Expo's limit for one request

    def __init__(self, get_tokens: Callable[[], List[str]],
                 remove_token: Callable[[str], None]):
        self.get_tokens = get_tokens
        self.remove_token = remove_token
        self.queue: "queue.Queue[PushMessage]" = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="push").start()

    def send(self, message: PushMessage) -> None:
        self.queue.put(message)

    def _run(self) -> None:
        while True:
            message = self.queue.get()
            try:
                self.deliver(message)
            except Exception:  # one bad alert must not end this thread
                log.exception("Could not send push alert")

    def deliver(self, message: PushMessage) -> None:
        tokens = self.get_tokens()
        for start in range(0, len(tokens), self.BATCH_SIZE):
            batch = tokens[start:start + self.BATCH_SIZE]
            if not (self._post(batch, message) or self._post(batch, message)):
                log.error("Giving up on a push alert to %s phones", len(batch))

    def _post(self, tokens: List[str], message: PushMessage) -> bool:
        payload = [{"to": token, "title": message.title, "body": message.body,
                    "data": {"alertId": message.alert_id}, "sound": "default",
                    "priority": "high", "channelId": "falls"} for token in tokens]
        try:
            response = requests.post(self.URL, json=payload, timeout=15)
        except requests.RequestException as error:
            log.error("Push request failed: %s", type(error).__name__)
            return False
        if not response.ok:
            log.error("Push service rejected the alert: HTTP %s", response.status_code)
            return False
        for token, ticket in zip(tokens, response.json().get("data", [])):
            if ticket.get("status") == "error":
                log.error("Push not delivered: %s", ticket.get("message"))
                if ticket.get("details", {}).get("error") == "DeviceNotRegistered":
                    self.remove_token(token)
        return True


class MqttPublisher:
    """Optional. Only connects when MQTT_BROKER is set, and never stops the app from starting."""

    TOPIC = "Fall-Detected"

    def __init__(self):
        self.client: Optional[mqtt.Client] = None
        broker = os.environ.get("MQTT_BROKER", "").strip()
        if not broker:
            return
        try:
            client = mqtt.Client("Fall-Down")
            client.connect(broker)
            client.loop_start()
            self.client = client
        except (OSError, ValueError) as error:
            log.error("Could not connect to MQTT broker: %s", error)

    def publish(self, location: str) -> None:
        if self.client:
            self.client.publish(self.TOPIC, f"Fall Down Detected - {location}")
