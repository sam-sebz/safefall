import requests

import alerts
from alerts import PushAlerter, PushMessage


class FakeResponse:
    def __init__(self, tickets=(), ok=True, status_code=200):
        self.tickets = list(tickets)
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return {"data": self.tickets}


def make_alerter(tokens):
    removed = []
    return PushAlerter(lambda: tokens, removed.append), removed


MESSAGE = PushMessage("FALL DETECTED", "Aisle 4: 1 down", 12)


def test_message_is_high_priority_and_holds_no_photo(monkeypatch):
    sent = []
    monkeypatch.setattr(alerts.requests, "post", lambda url, json, timeout: (
        sent.extend(json), FakeResponse([{"status": "ok"}]))[1])
    make_alerter(["phone"])[0].deliver(MESSAGE)
    assert sent[0]["to"] == "phone"
    assert sent[0]["priority"] == "high"
    assert sent[0]["channelId"] == "falls"
    assert sent[0]["data"] == {"alertId": 12}
    assert not any("image" in key or "attachment" in key for key in sent[0])


def test_no_phones_sends_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(alerts.requests, "post", lambda *a, **k: calls.append(1))
    make_alerter([])[0].deliver(MESSAGE)
    assert calls == []


def test_large_groups_are_split(monkeypatch):
    sizes = []
    monkeypatch.setattr(alerts.requests, "post", lambda url, json, timeout: (
        sizes.append(len(json)), FakeResponse())[1])
    make_alerter([f"phone{i}" for i in range(250)])[0].deliver(MESSAGE)
    assert sizes == [100, 100, 50]


def test_uninstalled_phones_are_forgotten(monkeypatch):
    tickets = [{"status": "ok"},
               {"status": "error", "message": "gone", "details": {"error": "DeviceNotRegistered"}}]
    monkeypatch.setattr(alerts.requests, "post", lambda url, json, timeout: FakeResponse(tickets))
    alerter, removed = make_alerter(["keep", "gone"])
    alerter.deliver(MESSAGE)
    assert removed == ["gone"]


def test_other_ticket_errors_keep_the_phone(monkeypatch):
    tickets = [{"status": "error", "message": "busy", "details": {"error": "MessageRateExceeded"}}]
    monkeypatch.setattr(alerts.requests, "post", lambda url, json, timeout: FakeResponse(tickets))
    alerter, removed = make_alerter(["phone"])
    alerter.deliver(MESSAGE)
    assert removed == []


def test_network_failure_retries_once(monkeypatch):
    calls = []

    def fail(url, json, timeout):
        calls.append(1)
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(alerts.requests, "post", fail)
    make_alerter(["phone"])[0].deliver(MESSAGE)
    assert len(calls) == 2


def test_rejected_request_retries_once(monkeypatch):
    calls = []

    def reject(url, json, timeout):
        calls.append(1)
        return FakeResponse(ok=False, status_code=500)

    monkeypatch.setattr(alerts.requests, "post", reject)
    make_alerter(["phone"])[0].deliver(MESSAGE)
    assert len(calls) == 2
