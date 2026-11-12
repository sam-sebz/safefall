import logging
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerts import Alert, TelegramAlerter, build_caption
from detector import FallTracker, calculate_angle, is_down, torso_angle_from_keypoints


def run(tracker, seconds, observations, start=0.0, step=0.1):
    """Feed the same observations for a span of time and collect the events."""
    events = []
    for i in range(int(round(seconds / step)) + 1):
        events += tracker.update(observations, start + i * step)
    return events


def make_tracker():
    # With no boxes, no fall movement is seen, so unwitnessed_seconds is the timer in play.
    return FallTracker(down_seconds=1, escalate_seconds=60, grace_seconds=3, recovery_seconds=1,
                       unwitnessed_seconds=10, lost_hold_seconds=20)


def fall_then_lie(tracker, start=0.0):
    """Walk upright, drop to the floor quickly, then lie still. Returns the events."""
    events = []
    for i in range(10):
        events += tracker.update({1: False}, start + i * 0.1, {1: (100, 50, 180, 250)})
    for i in range(1, 6):
        events += tracker.update({1: False}, start + 0.9 + i * 0.1,
                                 {1: (100, 50 + i * 30, 180 + i * 20, 250)})
    return events


def test_witnessed_fall_alerts_after_the_short_timer():
    tracker = make_tracker()
    fall_then_lie(tracker)
    events = run(tracker, 1.2, {1: True}, start=1.5)
    assert [e.kind for e in events] == ["fall"]


def test_lying_down_with_no_fall_seen_waits_for_the_long_timer():
    tracker = make_tracker()
    box = {1: (100, 100, 300, 180)}
    events = []
    for i in range(30):
        events += tracker.update({1: True}, i * 0.1, box)
    assert events == []
    for i in range(30, 110):
        events += tracker.update({1: True}, i * 0.1, box)
    assert [e.kind for e in events] == ["fall"]


def test_slowly_sitting_down_is_not_a_fall_movement():
    tracker = make_tracker()
    for i in range(50):
        tracker.update({1: False}, i * 0.1, {1: (100, 50 + i, 180, 250)})
    assert tracker.tracks[1].drop_at is None


def test_person_lost_right_after_a_fall_is_still_alerted():
    tracker = make_tracker()
    fall_then_lie(tracker)
    run(tracker, 0.4, {1: True}, start=1.5)
    events = run(tracker, 5, {}, start=2.0)
    assert [e.kind for e in events] == ["fall"]
    assert tracker.confirmed_down() == [1]


def test_torso_angle_from_keypoints():
    upright = [[90, 100, .9], [110, 100, .9], [90, 200, .9], [110, 200, .9]]
    flat = [[300, 90, .9], [300, 110, .9], [200, 90, .9], [200, 110, .9]]
    hidden = [[0, 0, .1], [0, 0, .1], [90, 200, .9], [110, 200, .9]]
    assert torso_angle_from_keypoints(upright) == pytest.approx(0)
    assert torso_angle_from_keypoints(flat) == pytest.approx(90)
    assert torso_angle_from_keypoints(hidden) is None


def test_angle_upright_lying_and_inverted():
    assert calculate_angle((100, 100), (100, 200)) == pytest.approx(0)
    assert calculate_angle((200, 100), (100, 100)) == pytest.approx(90)
    assert calculate_angle((0, 100), (100, 100)) == pytest.approx(90)
    assert calculate_angle((100, 200), (100, 100)) == pytest.approx(180)


def test_is_down_uses_angle_then_box_shape():
    assert is_down(80, 10, 100)
    assert not is_down(20, 100, 10)
    assert is_down(None, 100, 10)
    assert not is_down(None, 10, 100)


def test_brief_stumble_gives_no_alert():
    tracker = make_tracker()
    assert run(tracker, 5, {1: True}) == []
    assert run(tracker, 5, {1: False}, start=5.1) == []


def test_staying_down_alerts_once():
    tracker = make_tracker()
    events = run(tracker, 20, {1: True})
    assert [(e.kind, e.track_id, e.people_down) for e in events] == [("fall", 1, 1)]


def test_getting_up_clears_state():
    tracker = make_tracker()
    run(tracker, 12, {1: True})
    assert tracker.confirmed_down() == [1]
    run(tracker, 2, {1: False}, start=12.1)
    assert tracker.confirmed_down() == []
    # A second fall by the same person alerts again.
    assert len(run(tracker, 11, {1: True}, start=14.2)) == 1


def test_one_frame_pose_flicker_does_not_reset_timer():
    tracker = make_tracker()
    run(tracker, 5, {1: True})
    tracker.update({1: False}, 5.1)
    assert len(run(tracker, 5, {1: True}, start=5.2)) == 1


def test_two_people_give_two_alerts_with_running_total():
    tracker = make_tracker()
    run(tracker, 5, {1: True, 2: False})
    events = run(tracker, 10, {1: True, 2: True}, start=5.1)
    assert [(e.track_id, e.people_down) for e in events] == [(1, 1), (2, 2)]


def test_person_standing_nearby_does_not_affect_person_down():
    tracker = make_tracker()
    events = run(tracker, 12, {1: True, 2: False})
    assert [e.track_id for e in events] == [1]


def test_dropout_inside_grace_period_keeps_state():
    tracker = make_tracker()
    run(tracker, 5, {1: True})
    run(tracker, 2, {}, start=5.1)
    assert len(run(tracker, 4, {1: True}, start=7.2)) == 1


def test_dropout_beyond_grace_period_forgets_person():
    tracker = make_tracker()
    run(tracker, 5, {1: True})
    run(tracker, 4, {}, start=5.1)
    assert 1 not in tracker.tracks


def test_new_id_for_same_person_gives_no_duplicate_alert():
    tracker = make_tracker()
    box = (100, 100, 300, 180)
    for i in range(121):
        tracker.update({1: True}, i * 0.1, {1: box})
    assert tracker.confirmed_down() == [1]
    events = []
    for i in range(121):
        events += tracker.update({7: True}, 12.5 + i * 0.1, {7: box})
    assert tracker.confirmed_down() == [7]
    assert [e.kind for e in events] == []


def test_new_id_elsewhere_counts_as_a_second_person():
    tracker = make_tracker()
    for i in range(121):
        tracker.update({1: True}, i * 0.1, {1: (100, 100, 300, 180)})
    events = []
    for i in range(111):
        events += tracker.update({1: True, 7: True}, 12.1 + i * 0.1,
                                 {1: (100, 100, 300, 180), 7: (500, 100, 700, 180)})
    assert [(e.track_id, e.people_down) for e in events] == [(7, 2)]


def test_person_standing_up_under_a_new_id_clears_the_fall():
    tracker = make_tracker()
    for i in range(121):
        tracker.update({1: True}, i * 0.1, {1: (100, 100, 300, 180)})
    for i in range(21):
        tracker.update({7: False}, 12.1 + i * 0.1, {7: (150, 60, 260, 180)})
    assert tracker.confirmed_down() == []


def test_lost_track_cannot_trigger_an_alert():
    tracker = FallTracker(down_seconds=1, escalate_seconds=2, grace_seconds=3)
    tracker.update({5: True}, 0.0)
    assert run(tracker, 2.5, {}, start=0.1) == []
    assert tracker.confirmed_down() == []


def test_escalation_fires_once():
    tracker = make_tracker()
    events = run(tracker, 90, {1: True})
    assert [e.kind for e in events] == ["fall", "escalation"]


def test_caption_has_location_time_and_count():
    caption = build_caption("Warehouse B, Aisle 4", "Cam 3", "2026-11-04 10:00:00", 2)
    for text in ("FALL DETECTED", "Warehouse B, Aisle 4", "Cam 3", "2026-11-04 10:00:00", "2"):
        assert text in caption


def test_token_is_never_logged(monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRET123")
    monkeypatch.setenv("TELEGRAM_CHAT_IDS", "1")

    def fail(url, **kwargs):
        raise requests.ConnectionError(f"failed to reach {url}")

    monkeypatch.setattr(requests, "post", fail)
    with caplog.at_level(logging.INFO):
        assert TelegramAlerter()._post("1", Alert("caption", b"jpeg")) is False
    assert "SECRET123" not in caplog.text


def test_missing_token_warns_and_does_not_crash(monkeypatch, caplog):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_IDS", raising=False)
    with caplog.at_level(logging.WARNING):
        alerter = TelegramAlerter()
        alerter.send(Alert("caption", b"jpeg"))
    assert not alerter.enabled
    assert "not configured" in caplog.text.lower()


def test_fall_seen_counts_as_a_witnessed_fall():
    tracker = FallTracker(down_seconds=1, escalate_seconds=100, unwitnessed_seconds=30)
    assert tracker.update({1: False}, 0.0, fall_seen={1}) == []
    assert tracker.tracks[1].drop_at == 0.0
    events = tracker.update({1: True}, 1.0)
    assert [e.kind for e in events] == ["fall"]


def test_person_without_fall_seen_keeps_the_longer_unwitnessed_wait():
    tracker = FallTracker(down_seconds=1, escalate_seconds=100, unwitnessed_seconds=30)
    tracker.update({1: True}, 0.0)
    assert tracker.update({1: True}, 1.0) == []


def test_fall_seen_briefly_then_standing_clears_the_person():
    tracker = FallTracker(down_seconds=5, escalate_seconds=100, recovery_seconds=1)
    tracker.update({1: False}, 0.0, fall_seen={1})
    tracker.update({1: False}, 0.5)
    tracker.update({1: False}, 1.6)
    assert tracker.tracks[1].down_since is None
