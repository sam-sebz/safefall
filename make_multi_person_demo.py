"""Builds a 2x2 grid of UR Fall clips to demo several people falling on one camera feed.

This is a composite: each person was filmed separately. It shows tracking and counting, and
is not evidence of accuracy in a real crowded scene.

Usage: python make_multi_person_demo.py
"""
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CLIPS = ["fall-01-cam0.mp4", "fall-05-cam0.mp4", "fall-12-cam0.mp4", "fall-20-cam0.mp4"]
OUTPUT = os.path.join(BASE, "demo_outputs", "multi_person_input.mp4")
HOLD_SECONDS = 8  # keeps everyone on the floor long enough to test the "still down" reminder


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    caps = [cv2.VideoCapture(os.path.join(BASE, "datasets", "urfall", c)) for c in CLIPS]
    fps = 30
    writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    latest = [None] * len(caps)
    while True:
        alive = False
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            if ok:
                latest[i] = frame[:, frame.shape[1] // 2:].copy()  # colour half only
                alive = True
        if not alive:
            break
        grid = np.vstack([np.hstack(latest[:2]), np.hstack(latest[2:])])
        writer.write(grid)
    for _ in range(HOLD_SECONDS * fps):
        writer.write(grid)
    writer.release()
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
