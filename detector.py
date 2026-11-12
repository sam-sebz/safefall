"""Fall decision logic. Has no OpenCV or model dependency, so it can be unit tested."""
import math
from dataclasses import dataclass, field
from typing import Collection, Dict, List, Optional


def calculate_angle(shoulder_center, hip_center):
    """Return the torso's lean from vertical: 0 is upright, 90 is lying flat, 180 is inverted.

    Points are (x, y) in image coordinates, where y grows downwards.
    """
    dx = shoulder_center[0] - hip_center[0]
    dy = hip_center[1] - shoulder_center[1]
    return math.degrees(math.atan2(abs(dx), dy))


def torso_angle_from_keypoints(points, min_confidence: float = 0.3) -> Optional[float]:
    """Return the torso angle from [left shoulder, right shoulder, left hip, right hip].

    Each point is (x, y, confidence). Returns None if the shoulders or hips aren't visible.
    """
    def center(pair):
        visible = [p for p in pair if p[2] >= min_confidence]
        if not visible:
            return None
        return (sum(p[0] for p in visible) / len(visible),
                sum(p[1] for p in visible) / len(visible))

    shoulders, hips = center(points[:2]), center(points[2:])
    if shoulders is None or hips is None:
        return None
    return calculate_angle(shoulders, hips)


def is_down(torso_angle: Optional[float], box_width: float, box_height: float,
            angle_threshold: float = 60.0) -> bool:
    """Use the torso angle when pose was found, otherwise fall back to the box shape."""
    if torso_angle is not None:
        return torso_angle > angle_threshold
    return box_width > box_height


def box_overlap(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared)


@dataclass
class TrackState:
    last_seen: float
    down_since: Optional[float] = None
    last_seen_down: Optional[float] = None
    up_since: Optional[float] = None
    drop_at: Optional[float] = None  # when a sudden fall movement was last seen
    alerted: bool = False
    escalated: bool = False
    box: Optional[tuple] = None
    history: list = field(default_factory=list)  # recent (time, box top, box height)
    tallest: float = 0.0


@dataclass(frozen=True)
class FallEvent:
    kind: str  # "fall" or "escalation"
    track_id: int
    people_down: int


class FallTracker:
    """Tracks each person separately and decides when someone has stayed down too long.

    A fall seen happening alerts after down_seconds. Someone found lying still with no fall
    seen, such as a person who lay down slowly or fell out of view, alerts after the longer
    unwitnessed_seconds.
    """

    def __init__(self, down_seconds: float = 10.0, escalate_seconds: float = 60.0,
                 grace_seconds: float = 3.0, recovery_seconds: float = 1.0,
                 same_person_overlap: float = 0.3, unwitnessed_seconds: float = 30.0,
                 drop_ratio: float = 0.35, drop_window: float = 1.0,
                 drop_memory: float = 3.0, min_seen_down: float = 0.2,
                 lost_hold_seconds: float = 30.0, shrink_ratio: float = 0.0):
        self.down_seconds = down_seconds
        self.escalate_seconds = escalate_seconds
        self.grace_seconds = grace_seconds
        self.recovery_seconds = recovery_seconds  # ignores one-frame pose flicker
        self.same_person_overlap = same_person_overlap
        self.unwitnessed_seconds = unwitnessed_seconds
        self.drop_ratio = drop_ratio    # drop size, as a share of the person's height
        self.drop_window = drop_window  # the drop must happen within this many seconds
        self.drop_memory = drop_memory  # a drop only counts if they go down soon after
        self.min_seen_down = min_seen_down  # filters out one-frame misreadings
        self.lost_hold_seconds = lost_hold_seconds
        self.shrink_ratio = shrink_ratio  # after a drop, this much shorter also counts as down
        self.tracks: Dict[int, TrackState] = {}

    def _inherit_lost_track(self, track_id: int, box: tuple, seen_ids) -> None:
        """Give a new id the state of a lost person at the same spot, so nobody is counted twice."""
        lost = [t for t, state in self.tracks.items()
                if t not in seen_ids and state.box is not None]
        best = max(lost, key=lambda t: box_overlap(box, self.tracks[t].box), default=None)
        if best is not None and box_overlap(box, self.tracks[best].box) >= self.same_person_overlap:
            self.tracks[track_id] = self.tracks.pop(best)

    def _note_movement(self, state: TrackState, box: tuple, now: float) -> None:
        top, height = box[1], box[3] - box[1]
        state.tallest = max(state.tallest, height)
        state.history = [h for h in state.history if now - h[0] <= self.drop_window]
        state.history.append((now, top, height))
        highest_top = min(h[1] for h in state.history)
        peak_height = max(h[2] for h in state.history)
        if peak_height > 0 and (top - highest_top) / peak_height >= self.drop_ratio:
            state.drop_at = now

    def _is_lost_after_fall(self, state: TrackState) -> bool:
        return (state.drop_at is not None and state.down_since is not None
                and state.last_seen_down is not None
                and state.last_seen_down - state.down_since >= self.min_seen_down)

    def update(self, observations: Dict[int, bool], now: float,
               boxes: Optional[Dict[int, tuple]] = None,
               fall_seen: Collection[int] = ()) -> List[FallEvent]:
        """Take {track_id: is_down} and optional {track_id: box} for one frame.

        fall_seen holds the ids a fall model believes are falling right now. Each counts as
        down, with a witnessed fall. Returns any new events.
        """
        boxes = boxes or {}
        for track_id, down in observations.items():
            if track_id not in self.tracks and track_id in boxes:
                self._inherit_lost_track(track_id, boxes[track_id], observations)
            state = self.tracks.setdefault(track_id, TrackState(last_seen=now))
            state.last_seen = now
            if track_id in boxes:
                state.box = boxes[track_id]
                self._note_movement(state, state.box, now)
                height = state.box[3] - state.box[1]
                if (self.shrink_ratio and state.drop_at is not None
                        and height <= self.shrink_ratio * state.tallest):
                    down = True
            if track_id in fall_seen:
                state.drop_at = now
                down = True
            if down:
                state.up_since = None
                state.last_seen_down = now
                if state.down_since is None:
                    state.down_since = now
            elif state.down_since is not None:
                if state.up_since is None:
                    state.up_since = now
                if now - state.up_since >= self.recovery_seconds:
                    self.tracks[track_id] = TrackState(last_seen=now, box=state.box,
                                                       tallest=state.tallest)
            elif state.drop_at is not None and now - state.drop_at > self.drop_memory:
                state.drop_at = None

        # Forget people missing too long. Someone lost right after a fall is probably still on
        # the floor and just hard to detect, so they are held for longer.
        for track_id, state in list(self.tracks.items()):
            limit = self.lost_hold_seconds if self._is_lost_after_fall(state) else self.grace_seconds
            if now - state.last_seen > limit:
                del self.tracks[track_id]

        newly_down, to_escalate = [], []
        for track_id, state in self.tracks.items():
            # Decide only on someone visibly down now, or lost right after a fall.
            visibly_down = state.last_seen == now and state.up_since is None
            if state.down_since is None or not (visibly_down or self._is_lost_after_fall(state)):
                continue
            elapsed = now - state.down_since
            required = self.down_seconds if state.drop_at is not None else self.unwitnessed_seconds
            if not state.alerted and elapsed >= required:
                state.alerted = True
                newly_down.append(track_id)
            if state.alerted and not state.escalated and elapsed >= self.escalate_seconds:
                state.escalated = True
                to_escalate.append(track_id)

        people_down = len(self.confirmed_down())
        return ([FallEvent("fall", t, people_down) for t in newly_down]
                + [FallEvent("escalation", t, people_down) for t in to_escalate])

    def confirmed_down(self) -> List[int]:
        return sorted(t for t, state in self.tracks.items() if state.alerted)
