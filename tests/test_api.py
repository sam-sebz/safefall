import threading
from datetime import timedelta

import pytest
from flask import Flask
from werkzeug.security import generate_password_hash

import api as api_module
from api import LoginLimiter, create_api
from storage import ACKNOWLEDGED, RESOLVED, AlertRecord, Device, User, db, is_repeat, latest_alert, utc_now


class FakeWorker:
    def status(self):
        return {"name": "Warehouse", "location": "Aisle 4", "state": "running", "people": 2,
                "down_ids": [7]}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "login_limiter", LoginLimiter())
    flask_app = Flask(__name__)
    flask_app.secret_key = "test key"
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{tmp_path / 'test.db'}"
    db.init_app(flask_app)
    (tmp_path / "1.jpg").write_bytes(b"photo")
    blueprint = create_api({"cam1": FakeWorker()}, {"cam1": b"jpeg"}, threading.Lock(),
                           str(tmp_path))
    flask_app.register_blueprint(blueprint)
    with flask_app.app_context():
        db.create_all()
        db.session.add(User(username="sam", password=generate_password_hash(
            "correct horse", method="pbkdf2:sha256")))
        db.session.add(AlertRecord(camera_id="cam1", camera_name="Warehouse",
                                   location="Aisle 4", track_id=7, people_down=1))
        db.session.commit()
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth(client):
    response = client.post("/api/login", json={"username": "sam", "password": "correct horse"})
    return {"Authorization": f"Bearer {response.get_json()['token']}"}


@pytest.mark.parametrize("method,path", [
    ("get", "/api/cameras"), ("get", "/api/cameras/cam1/snapshot"), ("get", "/api/alerts"),
    ("get", "/api/alerts/1"), ("get", "/api/alerts/1/image"), ("post", "/api/alerts/1/ack"),
    ("post", "/api/alerts/1/resolve"), ("post", "/api/devices"), ("delete", "/api/devices"),
])
def test_every_route_needs_a_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_forged_token_is_rejected(client):
    response = client.get("/api/alerts", headers={"Authorization": "Bearer made.up.token"})
    assert response.status_code == 401


def test_token_for_a_deleted_user_is_rejected(app, client, auth):
    with app.app_context():
        User.query.delete()
        db.session.commit()
    assert client.get("/api/alerts", headers=auth).status_code == 401


def test_wrong_password_is_rejected(client):
    response = client.post("/api/login", json={"username": "sam", "password": "nope"})
    assert response.status_code == 401


def test_repeated_wrong_passwords_are_blocked(client):
    for _ in range(5):
        client.post("/api/login", json={"username": "sam", "password": "nope"})
    response = client.post("/api/login", json={"username": "sam", "password": "correct horse"})
    assert response.status_code == 429


def test_limiter_forgives_after_the_window():
    limiter = LoginLimiter(attempts=1, window=10)
    limiter.failed("sam", now=100)
    assert limiter.blocked("sam", now=105)
    assert not limiter.blocked("sam", now=111)


def test_cameras_are_listed(client, auth):
    camera = client.get("/api/cameras", headers=auth).get_json()[0]
    assert camera["id"] == "cam1"
    assert camera["down_ids"] == [7]


def test_snapshot(client, auth):
    response = client.get("/api/cameras/cam1/snapshot", headers=auth)
    assert response.data == b"jpeg"
    assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/api/cameras/none/snapshot", headers=auth).status_code == 404


def test_alert_list_and_filter(client, auth):
    assert len(client.get("/api/alerts", headers=auth).get_json()) == 1
    assert client.get("/api/alerts?status=resolved", headers=auth).get_json() == []
    assert client.get("/api/alerts?status=nonsense", headers=auth).status_code == 400


def test_alert_photo(client, auth):
    assert client.get("/api/alerts/1/image", headers=auth).data == b"photo"
    assert client.get("/api/alerts/99/image", headers=auth).status_code == 404


def test_acknowledge_records_who_and_keeps_the_first_name(app, client, auth):
    body = client.post("/api/alerts/1/ack", headers=auth).get_json()
    assert body["status"] == ACKNOWLEDGED
    assert body["acted_by"] == "sam"
    first_time = body["acted_at"]
    assert client.post("/api/alerts/1/ack", headers=auth).get_json()["acted_at"] == first_time


def test_resolved_alert_cannot_change(client, auth):
    assert client.post("/api/alerts/1/resolve", headers=auth).get_json()["status"] == RESOLVED
    assert client.post("/api/alerts/1/ack", headers=auth).status_code == 409


def test_unknown_alert(client, auth):
    assert client.post("/api/alerts/99/ack", headers=auth).status_code == 404


def test_device_registers_once_and_can_be_removed(app, client, auth):
    for _ in range(2):
        assert client.post("/api/devices", json={"token": "phone"}, headers=auth).status_code == 200
    with app.app_context():
        assert Device.query.count() == 1
    client.delete("/api/devices", json={"token": "phone"}, headers=auth)
    with app.app_context():
        assert Device.query.count() == 0


def test_device_needs_a_token_value(client, auth):
    assert client.post("/api/devices", json={}, headers=auth).status_code == 400


def test_repeat_rules(app):
    with app.app_context():
        alert = latest_alert("cam1")
        assert is_repeat(alert, 1, 120)
        assert not is_repeat(alert, 2, 120)  # another person went down
        assert not is_repeat(alert, 1, 120, now=alert.raised_at + timedelta(seconds=121))
        alert.status = RESOLVED
        assert not is_repeat(alert, 1, 120)
        assert not is_repeat(None, 1, 120)
        assert utc_now() >= alert.raised_at
