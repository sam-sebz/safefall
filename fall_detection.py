# Essential Libraries
from datetime import datetime
from functools import wraps
import json
import logging
import os
import secrets
import threading
import time
import uuid

import cv2
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, jsonify
from flask import Response, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
# Pretrained Model Library
from ultralytics import YOLO

from alerts import (Alert, MqttPublisher, PushAlerter, PushMessage, TelegramAlerter,
                    build_caption)
from api import create_api, login_limiter
from detector import FallTracker, is_down, torso_angle_from_keypoints
from storage import ACTIVE, AlertRecord, Device, User, db, is_repeat, latest_alert

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("safefall")

# Detection settings, all overridable in .env
DOWN_SECONDS = float(os.environ.get("DOWN_SECONDS", 10))
ESCALATE_SECONDS = float(os.environ.get("ESCALATE_SECONDS", 60))
UNWITNESSED_SECONDS = float(os.environ.get("UNWITNESSED_SECONDS", 30))
GRACE_SECONDS = float(os.environ.get("GRACE_SECONDS", 3))
TORSO_ANGLE_THRESHOLD = float(os.environ.get("TORSO_ANGLE_THRESHOLD", 45))
DROP_RATIO = float(os.environ.get("DROP_RATIO", 0.5))
SHRINK_RATIO = float(os.environ.get("SHRINK_RATIO", 0))  # 0 is off; 0.6 suits full-speed analysis
USE_FALL_MODEL = os.environ.get("FALL_MODEL", "off").lower() == "on"
ALERT_COOLDOWN_SECONDS = float(os.environ.get("ALERT_COOLDOWN_SECONDS", 120))
ALLOW_REGISTRATION = os.environ.get("ALLOW_REGISTRATION", "0") == "1"
MODEL_FILE = "yolo11s-pose.pt"
DETECTION_CONFIDENCE = 0.1  # low, because people lying down score poorly
BODY_POINTS = (5, 6, 11, 12)  # COCO left/right shoulder, left/right hip
MAX_LAG_SECONDS = 0.05
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 320))
CAMERAS_FILE = os.path.join(basedir, os.environ.get("CAMERAS_FILE", "cameras.json"))
UPLOAD_DIR = os.path.join(basedir, "uploads")
ALERT_DIR = os.path.join(basedir, "instance", "alerts")
UPLOAD_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
STREAM_WIDTH = 640

os.makedirs(ALERT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize a Flask object
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
if not os.environ.get("SECRET_KEY"):
    log.warning("SECRET_KEY is not set: using a temporary key, so logins reset on restart.")
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(basedir, 'instance', 'users.db')}"
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB upload limit
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()

lock = threading.Lock()
frames = {}   # cam_id -> latest JPEG bytes
workers = {}  # cam_id -> CameraWorker

app.register_blueprint(create_api(workers, frames, lock, ALERT_DIR))


def push_tokens():
    with app.app_context():
        return [device.token for device in Device.query.all()]


def forget_push_token(token):
    with app.app_context():
        Device.query.filter_by(token=token).delete()
        db.session.commit()


telegram = TelegramAlerter()
push = PushAlerter(push_tokens, forget_push_token)
mqtt_publisher = MqttPublisher()

fall_network, fall_threshold = None, 1.0
if USE_FALL_MODEL:
    from fall_model import FallClassifier, load_network
    fall_network, fall_threshold = load_network()
    log.info("Fall model on, alerting at %.2f", fall_threshold)


def new_fall_tracker():
    # With the model on, it replaces the hand-written drop rule, as in evaluation.
    return FallTracker(DOWN_SECONDS, ESCALATE_SECONDS, GRACE_SECONDS,
                       unwitnessed_seconds=UNWITNESSED_SECONDS,
                       drop_ratio=float("inf") if USE_FALL_MODEL else DROP_RATIO,
                       shrink_ratio=0 if USE_FALL_MODEL else SHRINK_RATIO)


def new_fall_classifier():
    return FallClassifier(fall_network) if USE_FALL_MODEL else None


class CameraWorker(threading.Thread):
    """Watches one camera: tracks people, decides on falls and sends alerts."""

    def __init__(self, cam_id, name, location, source, loop=True, output_path=None):
        super().__init__(daemon=True, name=f"camera-{cam_id}")
        self.cam_id = cam_id
        self.camera_name = name
        self.location = location
        self.loop = loop
        self.output_path = output_path
        # A file path is relative to the project; anything else, such as rtsp://, is used as is.
        file_source = os.path.join(basedir, source)
        self.is_file = os.path.isfile(file_source)
        self.source = file_source if self.is_file else source
        self.tracker = new_fall_tracker()
        self.classifier = new_fall_classifier()
        self.open_alerts = {}  # track id -> alert id, for matching "still down" reminders
        self.state = "starting"
        self.people = 0

    def status(self):
        return {
            "name": self.camera_name,
            "location": self.location,
            "state": self.state,
            "people": self.people,
            "down_ids": self.tracker.confirmed_down(),
        }

    def run(self):
        try:
            self.watch()
        except Exception:
            self.state = "error"
            log.exception("Camera %s stopped unexpectedly", self.cam_id)

    def watch(self):
        # Each worker needs its own model: tracker state is per YOLO instance.
        model = YOLO(os.path.join(basedir, MODEL_FILE))
        writer = None
        frame_index = 0
        started = time.monotonic()

        while True:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                log.error("Camera %s could not open its video source", self.cam_id)
                self.state = "error"
                if self.is_file:
                    return
                time.sleep(5)
                continue

            self.state = "running"
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            while True:
                # Skip frames when behind, so slow hardware doesn't play video in slow motion.
                # Uploads keep every frame because they are saved to a video file.
                behind = time.monotonic() - (started + frame_index / fps)
                if self.is_file and not self.output_path and behind > MAX_LAG_SECONDS:
                    if not cap.grab():
                        break
                    frame_index += 1
                    continue

                ok, frame = cap.read()
                if not ok:
                    break
                now = frame_index / fps if self.is_file else time.monotonic()
                self.process(frame, now, model)

                if self.output_path:
                    if writer is None:
                        size = (frame.shape[1], frame.shape[0])
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(self.output_path, fourcc, fps, size)
                    writer.write(frame)

                frame_index += 1
                if self.is_file:  # play saved clips at real speed
                    delay = started + frame_index / fps - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            cap.release()
            if not self.loop:
                break
            # The picture jumps when a source restarts, so old people must not carry over.
            self.tracker = new_fall_tracker()
            self.classifier = new_fall_classifier()
            for byte_tracker in getattr(model.predictor, "trackers", None) or []:
                byte_tracker.reset()

        if writer is not None:
            writer.release()
        self.state = "finished"

    def process(self, frame, now, model):
        result = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml",
                             imgsz=IMAGE_SIZE, conf=DETECTION_CONFIDENCE, verbose=False)[0]
        people, keypoints = [], []
        if result.boxes.id is not None:
            for box, track_id, points in zip(result.boxes.xyxy.int().tolist(),
                                             result.boxes.id.int().tolist(),
                                             result.keypoints.data.tolist()):
                x1, y1, x2, y2 = box
                body = [points[i] for i in BODY_POINTS]
                angle = torso_angle_from_keypoints(body)
                down = is_down(angle, x2 - x1, y2 - y1, TORSO_ANGLE_THRESHOLD)
                people.append((track_id, (x1, y1, x2, y2), down))
                keypoints.append((track_id, (x1, y1, x2, y2), points))
                for x, y, confidence in body:
                    if confidence >= 0.3:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        fall_seen = set()
        if self.classifier:
            fall_seen = {track_id for track_id, chance in self.classifier.update(keypoints, now).items()
                         if chance >= fall_threshold}
        events = self.tracker.update({track_id: down for track_id, _, down in people}, now,
                                     {track_id: box for track_id, box, _ in people}, fall_seen)
        self.people = len(people)

        for track_id, (x1, y1, x2, y2), down in people:
            color = (0, 0, 255) if down else (255, 0, 0)
            label = f"ID {track_id} DOWN" if down else f"ID {track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.tracker.confirmed_down():
            cv2.putText(frame, "FALL DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 255), 3, cv2.LINE_AA)

        for event in events:
            self.send_alert(event, frame)

        scale = STREAM_WIDTH / frame.shape[1]
        small = cv2.resize(frame, (STREAM_WIDTH, int(frame.shape[0] * scale)))
        ok, encoded = cv2.imencode(".jpg", small)
        if ok:
            with lock:
                frames[self.cam_id] = encoded.tobytes()

    def record_alert(self, event, jpeg):
        """Save the alert. Returns its id, or None if nobody needs telling."""
        if event.kind == "escalation":
            known = self.open_alerts.get(event.track_id)
            record = db.session.get(AlertRecord, known) if known else latest_alert(self.cam_id)
            if record is None:  # unexpected, but a reminder must never be lost
                return 0
            if record.status != ACTIVE:
                log.info("No reminder for alert %s: already %s", record.id, record.status)
                return None
            record.escalated = True
            db.session.commit()
            return record.id

        previous = latest_alert(self.cam_id)
        if is_repeat(previous, event.people_down, ALERT_COOLDOWN_SECONDS):
            self.open_alerts[event.track_id] = previous.id
            log.info("Fall on %s repeats alert %s", self.cam_id, previous.id)
            return None
        record = AlertRecord(camera_id=self.cam_id, camera_name=self.camera_name,
                             location=self.location, track_id=event.track_id,
                             people_down=event.people_down)
        db.session.add(record)
        db.session.commit()
        self.open_alerts[event.track_id] = record.id
        with open(os.path.join(ALERT_DIR, f"{record.id}.jpg"), "wb") as photo:
            photo.write(jpeg)
        return record.id

    def send_alert(self, event, frame):
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            log.error("Could not encode the alert snapshot for %s", self.cam_id)
            return
        with app.app_context():
            alert_id = self.record_alert(event, encoded.tobytes())
        if alert_id is None:
            return
        escalation = event.kind == "escalation"
        when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caption = build_caption(self.location, self.camera_name, when, event.people_down, escalation)
        log.warning("%s: person %s at %s (%s down)", event.kind, event.track_id,
                    self.location, event.people_down)
        title = caption.split("\n")[0]
        push.send(PushMessage(title, f"{self.location}: {event.people_down} down", alert_id))
        telegram.send(Alert(caption, encoded.tobytes(), escalation))
        if not escalation:
            mqtt_publisher.publish(self.location)


def start_cameras():
    with open(CAMERAS_FILE, encoding="utf-8") as config:
        cameras = json.load(config)
    for camera in cameras:
        worker = CameraWorker(camera["id"], camera["name"], camera["location"], camera["source"])
        workers[camera["id"]] = worker
        worker.start()


@app.context_processor
def template_settings():
    return {"allow_registration": ALLOW_REGISTRATION}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" in session:
            return view(*args, **kwargs)
        if request.method == "POST" or request.path == "/status":
            return jsonify({"error": "Please log in"}), 401
        return redirect(url_for("login"))
    return wrapped


@app.route("/", methods=["GET", "POST"])
def project():
    return render_template("index.html")


@app.route("/hero")
def home():
    return render_template("index.html")


@app.route("/model")
@login_required
def model():
    return render_template("model.html")


@app.route("/reg", methods=["GET", "POST"])
def register():
    if not ALLOW_REGISTRATION:
        flash("Sign-ups are closed. Ask your administrator for an account.", "danger")
        return redirect(url_for("login"))

    if request.method == "POST":
        username = request.form["name"]
        password = request.form["password"]

        # Check if username already exists
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash("Username already exists. Please choose a different username.", "danger")
            return redirect(url_for("register"))

        # Hash the password before saving to the database
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

        # Create new user and add to database
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["name"]
        password = request.form["password"]
        if login_limiter.blocked(username):
            flash("Too many attempts. Try again in a few minutes.", "danger")
            return render_template("login.html"), 429

        # Check if the user exists in the database
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_limiter.succeeded(username)
            session["user_id"] = user.id
            flash("Login successful!", "success")
            return redirect(url_for("model"))

        login_limiter.failed(username)
        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))


# Runs an uploaded video as a temporary camera
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    video_file = request.files.get('video_file')
    if video_file is None or not video_file.filename:
        return jsonify({"error": "Video file is required"}), 400

    filename = secure_filename(video_file.filename)
    if os.path.splitext(filename)[1].lower() not in UPLOAD_EXTENSIONS:
        return jsonify({"error": "Please upload an mp4, avi, mov or mkv video"}), 400

    cam_id = f"upload-{uuid.uuid4().hex[:8]}"
    video_path = os.path.join(UPLOAD_DIR, f"{cam_id}-{filename}")
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    opened = cap.isOpened()
    cap.release()
    if not opened:
        os.remove(video_path)
        return jsonify({"error": "Couldn't open video file"}), 400

    # Finished uploads are cleared so the camera list doesn't keep growing.
    with lock:
        for old_id in [c for c, w in workers.items()
                       if c.startswith("upload-") and w.state in ("finished", "error")]:
            del workers[old_id]
            frames.pop(old_id, None)
        worker = CameraWorker(cam_id, filename, "Uploaded video", video_path, loop=False,
                              output_path=os.path.join(basedir, "4_out_video_alert.mp4"))
        workers[cam_id] = worker
    worker.start()
    return jsonify({"data": "video processing started", "camera": cam_id})


@app.route("/status")
@login_required
def status():
    with lock:
        current = dict(workers)
    return jsonify({cam_id: worker.status() for cam_id, worker in current.items()})


# MJPEG Streaming Function
def generate(cam_id):
    last = None
    while True:
        with lock:
            frame = frames.get(cam_id)
            worker = workers.get(cam_id)
        if worker is None:
            return
        if frame is not None and frame is not last:
            last = frame
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        elif worker.state in ("finished", "error"):
            return
        else:
            time.sleep(0.03)


# Video Feed Route
@app.route("/video_feed/<cam_id>")
@login_required
def video_feed(cam_id):
    return Response(generate(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    start_cameras()
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(debug=debug, use_reloader=False, threaded=True, host="0.0.0.0", port=5000)
