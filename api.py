"""The API used by the mobile app. Every route except login needs a sign-in token."""
import os
import threading
import time
from functools import wraps

from flask import Blueprint, Response, current_app, g, jsonify, request, send_file
from itsdangerous import BadSignature, URLSafeTimedSerializer
from werkzeug.security import check_password_hash

from storage import ACKNOWLEDGED, ACTIVE, RESOLVED, AlertRecord, Device, User, db, utc_now

TOKEN_MAX_AGE = 30 * 24 * 3600  # phones stay signed in for 30 days
MAX_ALERTS = 100
ALERT_STATUSES = (ACTIVE, ACKNOWLEDGED, RESOLVED)


class LoginLimiter:
    """Blocks a username for a while after repeated wrong passwords.

    Counted by username because requests through the tunnel all arrive from one local address.
    """

    def __init__(self, attempts: int = 5, window: float = 300.0):
        self.attempts = attempts
        self.window = window
        self.failures = {}
        self.lock = threading.Lock()

    def _recent(self, username, now):
        recent = [t for t in self.failures.get(username, []) if now - t < self.window]
        self.failures[username] = recent
        return recent

    def blocked(self, username, now=None) -> bool:
        with self.lock:
            return len(self._recent(username, now or time.monotonic())) >= self.attempts

    def failed(self, username, now=None) -> None:
        now = now or time.monotonic()
        with self.lock:
            self._recent(username, now).append(now)

    def succeeded(self, username) -> None:
        with self.lock:
            self.failures.pop(username, None)


login_limiter = LoginLimiter()


def _serializer():
    return URLSafeTimedSerializer(current_app.secret_key, salt="safefall-mobile")


def issue_token(user) -> str:
    return _serializer().dumps({"user": user.id})


def token_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Please sign in"}), 401
        try:
            data = _serializer().loads(header[len("Bearer "):], max_age=TOKEN_MAX_AGE)
        except BadSignature:
            return jsonify({"error": "Please sign in again"}), 401
        g.user = db.session.get(User, data["user"])
        if g.user is None:
            return jsonify({"error": "Please sign in again"}), 401
        return view(*args, **kwargs)
    return wrapped


def create_api(workers, frames, lock, alert_dir) -> Blueprint:
    api = Blueprint("api", __name__, url_prefix="/api")

    @api.post("/login")
    def login():
        body = request.get_json(silent=True) or {}
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        if login_limiter.blocked(username):
            return jsonify({"error": "Too many attempts. Try again in a few minutes."}), 429
        user = User.query.filter_by(username=username).first()
        if user is None or not check_password_hash(user.password, password):
            login_limiter.failed(username)
            return jsonify({"error": "Wrong username or password"}), 401
        login_limiter.succeeded(username)
        return jsonify({"token": issue_token(user), "username": user.username})

    @api.post("/devices")
    @token_required
    def add_device():
        token = str((request.get_json(silent=True) or {}).get("token", "")).strip()
        if not token or len(token) > 255:
            return jsonify({"error": "A push token is required"}), 400
        device = Device.query.filter_by(token=token).first()
        if device:
            device.user_id = g.user.id  # the phone may have changed hands
        else:
            db.session.add(Device(token=token, user_id=g.user.id))
        db.session.commit()
        return jsonify({"data": "registered"})

    @api.delete("/devices")
    @token_required
    def remove_device():
        token = str((request.get_json(silent=True) or {}).get("token", "")).strip()
        Device.query.filter_by(token=token, user_id=g.user.id).delete()
        db.session.commit()
        return jsonify({"data": "removed"})

    @api.get("/cameras")
    @token_required
    def cameras():
        with lock:
            current = dict(workers)
        return jsonify([dict(worker.status(), id=cam_id) for cam_id, worker in current.items()])

    @api.get("/cameras/<cam_id>/snapshot")
    @token_required
    def snapshot(cam_id):
        with lock:
            frame = frames.get(cam_id)
        if frame is None:
            return jsonify({"error": "No picture yet"}), 404
        return Response(frame, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @api.get("/alerts")
    @token_required
    def alerts():
        query = AlertRecord.query
        status = request.args.get("status")
        if status:
            if status not in ALERT_STATUSES:
                return jsonify({"error": "Unknown status"}), 400
            query = query.filter_by(status=status)
        limit = min(request.args.get("limit", MAX_ALERTS, type=int), MAX_ALERTS)
        return jsonify([a.to_dict() for a in
                        query.order_by(AlertRecord.id.desc()).limit(max(limit, 1))])

    @api.get("/alerts/<int:alert_id>")
    @token_required
    def alert(alert_id):
        record = db.session.get(AlertRecord, alert_id)
        if record is None:
            return jsonify({"error": "Alert not found"}), 404
        return jsonify(record.to_dict())

    @api.get("/alerts/<int:alert_id>/image")
    @token_required
    def alert_image(alert_id):
        path = os.path.join(alert_dir, f"{alert_id}.jpg")
        if not os.path.isfile(path):
            return jsonify({"error": "No photo for this alert"}), 404
        return send_file(path, mimetype="image/jpeg")

    def act_on(alert_id, status):
        record = db.session.get(AlertRecord, alert_id)
        if record is None:
            return jsonify({"error": "Alert not found"}), 404
        if record.status == RESOLVED:
            return jsonify({"error": "This alert is already resolved"}), 409
        if record.status == status:  # someone else got there first: keep their name
            return jsonify(record.to_dict())
        record.status = status
        record.acted_by = g.user.username
        record.acted_at = utc_now()
        db.session.commit()
        return jsonify(record.to_dict())

    @api.post("/alerts/<int:alert_id>/ack")
    @token_required
    def acknowledge(alert_id):
        return act_on(alert_id, ACKNOWLEDGED)

    @api.post("/alerts/<int:alert_id>/resolve")
    @token_required
    def resolve(alert_id):
        return act_on(alert_id, RESOLVED)

    return api
