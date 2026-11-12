"""Database tables for users, phones and the alert history."""
from datetime import datetime, timedelta, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

ACTIVE = "active"
ACKNOWLEDGED = "acknowledged"
RESOLVED = "resolved"


def utc_now() -> datetime:
    """Current UTC time without a timezone, as SQLite stores it."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)


class Device(db.Model):
    """A phone that receives push alerts."""
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(255), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


class AlertRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.String(64), nullable=False, index=True)
    camera_name = db.Column(db.String(150), nullable=False)
    location = db.Column(db.String(255), nullable=False)
    track_id = db.Column(db.Integer, nullable=False)
    people_down = db.Column(db.Integer, nullable=False)
    raised_at = db.Column(db.DateTime, nullable=False, default=utc_now, index=True)
    status = db.Column(db.String(20), nullable=False, default=ACTIVE)
    escalated = db.Column(db.Boolean, nullable=False, default=False)
    acted_by = db.Column(db.String(150))
    acted_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "location": self.location,
            "people_down": self.people_down,
            "raised_at": self.raised_at.isoformat() + "Z",
            "status": self.status,
            "escalated": self.escalated,
            "acted_by": self.acted_by,
            "acted_at": self.acted_at.isoformat() + "Z" if self.acted_at else None,
        }


def latest_alert(camera_id):
    return (AlertRecord.query.filter_by(camera_id=camera_id)
            .order_by(AlertRecord.id.desc()).first())


def is_repeat(previous, people_down, cooldown_seconds, now=None) -> bool:
    """True if a new fall just repeats an open alert, as a looping or flickering camera can.

    More people down than before is always new.
    """
    if previous is None or previous.status == RESOLVED or people_down > previous.people_down:
        return False
    return (now or utc_now()) - previous.raised_at < timedelta(seconds=cooldown_seconds)
