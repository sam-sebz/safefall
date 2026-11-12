"""Makes colour-only copies of UR Fall clips to use as simulated cameras in cameras.json.

Usage: python make_camera_clips.py
"""
import os

import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
CLIPS = ["fall-03-cam0.mp4", "fall-06-cam0.mp4", "fall-11-cam0.mp4", "adl-24-cam0.mp4"]
OUTPUT_DIR = os.path.join(BASE, "demo_outputs", "cameras")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for clip in CLIPS:
        cap = cv2.VideoCapture(os.path.join(BASE, "datasets", "urfall", clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            colour = frame[:, frame.shape[1] // 2:]  # right half is colour, left is depth
            if writer is None:
                size = (colour.shape[1], colour.shape[0])
                writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, clip),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            writer.write(colour)
        cap.release()
        writer.release()
        print("wrote", clip)


if __name__ == "__main__":
    main()
