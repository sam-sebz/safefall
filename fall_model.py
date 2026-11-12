"""Learned fall recognition: a small temporal network over about 2 seconds of pose keypoints.

The same window-building code runs in training and in the live app, so the network always
sees the same input format. Windows are rebuilt from timestamps, which lets it cope with the
uneven frame rate of a busy CPU.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from detector import box_overlap

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fall_tcn.pt")
SAMPLE_RATE = 7.5     # frames per second the network expects
WINDOW_FRAMES = 15    # 2 seconds
MIN_SEEN_FRAMES = 4   # too few sightings to judge a movement
MIN_CONFIDENCE = 0.2  # keypoints below this are treated as not visible
KEYPOINTS = 17
FEATURES = KEYPOINTS * 3 + 5  # x, y, confidence for each point, box x, y, w, h, seen flag
SAME_PERSON_OVERLAP = 0.3
LEFT_RIGHT_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))

Sample = Tuple[float, tuple, list]  # time, box (x1, y1, x2, y2), 17 keypoints (x, y, confidence)


def build_window(samples: List[Sample], now: float) -> np.ndarray:
    """Turn a person's recent sightings into a WINDOW_FRAMES x FEATURES array.

    Everything is measured from where the person was at the start of the window and scaled by
    their tallest box in it. Normalising each frame alone would hide the drop in height, which
    is the movement that matters most.
    """
    step = 1.0 / SAMPLE_RATE
    window = np.zeros((WINDOW_FRAMES, FEATURES), dtype=np.float32)
    start = now - (WINDOW_FRAMES - 1) * step
    chosen = [None] * WINDOW_FRAMES
    best_gap = [step] * WINDOW_FRAMES
    for sample in samples:
        position = int(round((sample[0] - start) / step))
        if 0 <= position < WINDOW_FRAMES:
            gap = abs(sample[0] - (start + position * step))
            if gap < best_gap[position]:
                best_gap[position], chosen[position] = gap, sample
    seen = [s for s in chosen if s is not None]
    if not seen:
        return window

    first_box = seen[0][1]
    origin_x = (first_box[0] + first_box[2]) / 2
    origin_y = (first_box[1] + first_box[3]) / 2
    scale = max(max(s[1][3] - s[1][1] for s in seen), 1.0)

    for index, sample in enumerate(chosen):
        if sample is None:
            continue
        _, box, points = sample
        row = window[index]
        for k, (x, y, confidence) in enumerate(points[:KEYPOINTS]):
            if confidence >= MIN_CONFIDENCE:
                row[k * 3] = (x - origin_x) / scale
                row[k * 3 + 1] = (y - origin_y) / scale
                row[k * 3 + 2] = confidence
        row[-5] = ((box[0] + box[2]) / 2 - origin_x) / scale
        row[-4] = ((box[1] + box[3]) / 2 - origin_y) / scale
        row[-3] = (box[2] - box[0]) / scale
        row[-2] = (box[3] - box[1]) / scale
        row[-1] = 1.0
    return window


def seen_frames(window: np.ndarray) -> int:
    return int(window[:, -1].sum())


def flip_windows(windows: torch.Tensor) -> torch.Tensor:
    """Mirror a batch of windows left to right, for training."""
    flipped = windows.clone()
    flipped[:, :, 0::3][:, :, :KEYPOINTS] *= -1
    flipped[:, :, -5] *= -1
    for left, right in LEFT_RIGHT_PAIRS:
        for offset in range(3):
            a, b = left * 3 + offset, right * 3 + offset
            flipped[:, :, [a, b]] = flipped[:, :, [b, a]]
    return flipped


class FallNetwork(nn.Module):
    """Dilated 1D convolutions. Kept small because the training set is small."""

    def __init__(self, channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(FEATURES, channels, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=4), nn.ReLU(), nn.Dropout(dropout),
        )
        self.output = nn.Linear(channels * 2, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Take a batch shaped (people, frames, features); return one score for each person."""
        hidden = self.layers(windows.transpose(1, 2))
        pooled = torch.cat([hidden.mean(dim=2), hidden.amax(dim=2)], dim=1)
        return self.output(pooled).squeeze(1)


def load_network(path: str = MODEL_FILE) -> Tuple[FallNetwork, float]:
    saved = torch.load(path, map_location="cpu")
    network = FallNetwork()
    network.load_state_dict(saved["state"])
    network.eval()
    return network, float(saved["threshold"])


class FallClassifier:
    """Keeps each person's recent keypoints and scores how likely they are falling right now."""

    def __init__(self, network: FallNetwork):
        self.network = network
        self.buffers: Dict[int, List[Sample]] = {}

    def _inherit_lost_buffer(self, track_id: int, box: tuple, seen_ids) -> None:
        """Tracking often gives a falling person a new id, just when their history matters most."""
        lost = [t for t in self.buffers if t not in seen_ids]
        best = max(lost, key=lambda t: box_overlap(box, self.buffers[t][-1][1]), default=None)
        if best is not None and box_overlap(box, self.buffers[best][-1][1]) >= SAME_PERSON_OVERLAP:
            self.buffers[track_id] = self.buffers.pop(best)

    def update(self, people, now: float) -> Dict[int, float]:
        """Take [(track_id, box, 17 keypoints)] for one frame; return {track_id: fall chance}."""
        seen_ids = {person[0] for person in people}
        for track_id, box, points in people:
            if track_id not in self.buffers:
                self._inherit_lost_buffer(track_id, box, seen_ids)
            self.buffers.setdefault(track_id, []).append((now, tuple(box), points))

        keep_after = now - (WINDOW_FRAMES + 2) / SAMPLE_RATE
        for track_id in list(self.buffers):
            self.buffers[track_id] = [s for s in self.buffers[track_id] if s[0] >= keep_after]
            if not self.buffers[track_id]:
                del self.buffers[track_id]

        ids, windows = [], []
        for track_id in seen_ids:
            window = build_window(self.buffers[track_id], now)
            if seen_frames(window) >= MIN_SEEN_FRAMES:
                ids.append(track_id)
                windows.append(window)
        if not ids:
            return {}
        with torch.no_grad():
            scores = torch.sigmoid(self.network(torch.from_numpy(np.stack(windows))))
        return dict(zip(ids, scores.tolist()))
