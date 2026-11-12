"""Trains the fall model on odd-numbered UR Fall clips and picks its alert settings.

Settings are chosen by 5-fold cross-validation, so every odd clip is judged by a model that
never trained on it. Even-numbered clips are not read here: they stay held out for
`python evaluate_dataset.py --learned`.

Usage: python extract_keypoints.py, then python train_fall_model.py
"""
import csv
import json
import os
import random

import numpy as np
import torch
from torch import nn

from evaluate_dataset import (ANGLES, DATA_DIR, DOWN_TIMES, KEYPOINTS_CACHE, balance,
                              clip_fall_scores, clip_number, line, replay_learned, tally)
from fall_model import (MIN_SEEN_FRAMES, MODEL_FILE, FallNetwork, build_window, flip_windows,
                        seen_frames)

SEED = 7
FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
KEYPOINT_NOISE = 0.01
DROP_RATES = (0.0, 0.2, 0.4)  # lost frames, so the model copes with a slow CPU
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
LABEL_FILES = ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")
FALL_SETTLED_MARGIN = 0.25  # seconds into the fall before a window counts as showing it
FALL_TAIL = 1.5             # seconds after the fall still counted as showing it
HARD_NEGATIVE_WEIGHT = 2.0  # lying or crouching on purpose is what must not alert
VIDEO_FPS = 30.0


def load_labels():
    """Return {sequence: {frame: label}}. -1 is upright, 0 is going down, 1 is lying down."""
    labels = {}
    for filename in LABEL_FILES:
        with open(os.path.join(DATA_DIR, "labels", filename), encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    labels.setdefault(row[0], {})[int(row[1])] = int(row[2])
    return labels


def main_person(clip):
    """UR Fall clips show one person: follow the largest detection in each frame."""
    samples = []
    for index, frame in enumerate(clip["frames"]):
        if frame["people"]:
            _, box, points = max(frame["people"],
                                 key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))
            samples.append((index / clip["fps"], tuple(box), points))
    return samples


def fall_period(frame_labels):
    """Return the (start, end) seconds of the 'going down' frames."""
    falling = [f for f, label in frame_labels.items() if label == 0]
    return (min(falling) / VIDEO_FPS, max(falling) / VIDEO_FPS) if falling else None


def window_target(name, now, frame_labels):
    """Return (target, weight) for a window ending at `now`, or None to leave it out."""
    if name.startswith("adl"):
        label = frame_labels.get(int(round(now * VIDEO_FPS)), -1)
        return 0.0, HARD_NEGATIVE_WEIGHT if label >= 0 else 1.0
    period = fall_period(frame_labels)
    if period is None or now < period[0]:
        return 0.0, 1.0
    if now < period[0] + FALL_SETTLED_MARGIN:
        return None  # only the first instant of the fall is in view
    if now <= period[1] + FALL_TAIL:
        return 1.0, 1.0
    return None  # afterwards the person just lies still, which the tracker handles


def clip_windows(name, clip, frame_labels, rng):
    windows, targets, weights = [], [], []
    samples = main_person(clip)
    for rate in DROP_RATES:
        kept = [s for s in samples if rng.random() >= rate]
        for index, sample in enumerate(kept):
            result = window_target(name, sample[0], frame_labels)
            if result is None:
                continue
            window = build_window(kept[max(0, index - 20):index + 1], sample[0])
            if seen_frames(window) >= MIN_SEEN_FRAMES:
                windows.append(window)
                targets.append(result[0])
                weights.append(result[1])
    return windows, targets, weights


def train(names, clips, labels, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    windows, targets, weights = [], [], []
    for name in names:
        w, t, g = clip_windows(name, clips[name], labels[name.rsplit("-", 1)[0]], rng)
        windows += w
        targets += t
        weights += g
    x = torch.from_numpy(np.stack(windows))
    y = torch.tensor(targets, dtype=torch.float32)
    g = torch.tensor(weights, dtype=torch.float32)

    network = FallNetwork()
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none",
                                   pos_weight=(y == 0).sum() / (y == 1).sum().clamp(min=1))
    for _ in range(EPOCHS):
        network.train()
        order = torch.randperm(len(x))
        for start in range(0, len(x), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            inputs = x[batch]
            if rng.random() < 0.5:
                inputs = flip_windows(inputs)
            seen = inputs[:, :, -1:]  # noise only on frames where the person was seen
            inputs = inputs + torch.randn_like(inputs) * KEYPOINT_NOISE * seen
            loss = (loss_fn(network(inputs), y[batch]) * g[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    network.eval()
    print(f"  trained on {len(names)} clips: {len(x)} windows, {int(y.sum())} showing a fall",
          flush=True)
    return network


def split_folds(names):
    """Share falls and everyday clips evenly between folds."""
    folds = [[] for _ in range(FOLDS)]
    for group in ("adl", "fall"):
        members = sorted(n for n in names if n.startswith(group))
        for index, name in enumerate(members):
            folds[index % FOLDS].append(name)
    return folds


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    with open(KEYPOINTS_CACHE, encoding="utf-8") as f:
        clips = {n: c for n, c in json.load(f).items() if clip_number(n) % 2 == 1}
    labels = load_labels()
    names = sorted(clips)
    print(f"{len(names)} training clips", flush=True)

    # Every clip is scored by the model from the fold that left it out.
    fall_scores = {}
    for fold, left_out in enumerate(split_folds(names)):
        network = train([n for n in names if n not in left_out], clips, labels, SEED + fold)
        for name in left_out:
            fall_scores[name] = clip_fall_scores(clips[name], network)

    grid = [{"threshold": t, "down": d, "angle": a}
            for t in THRESHOLDS for d in DOWN_TIMES for a in ANGLES]
    tuned = []
    for settings in grid:
        alerted = {n for n in names if replay_learned(clips[n], fall_scores[n], settings)}
        tuned.append((settings, tally(alerted, names)))
    settings, result = max(tuned, key=lambda item: balance(item[1]))
    print("best settings:", settings)
    print(line("cross-validated odd clips", result))
    print("missed:", result["missed_clips"])
    print("false alarms:", result["false_alarm_clips"])

    network = train(names, clips, labels, SEED)
    torch.save({"state": network.state_dict(), "threshold": settings["threshold"],
                "settings": settings, "tuning": result}, MODEL_FILE)
    print("saved", MODEL_FILE)


if __name__ == "__main__":
    main()
