"""Saves all 17 pose keypoints for every UR Fall clip, ready for training the fall model.

Uses the same detector settings as the live app on six cameras: 320 px, every 4th frame.
Finished clips are saved as it goes, so an interrupted run can resume.

Usage: python extract_keypoints.py
"""
import glob
import json
import multiprocessing
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "datasets", "urfall")
MODEL = "yolo11s-pose.pt"
CONFIDENCE = 0.1
IMAGE_SIZE = 320
FRAME_STEP = 4
CACHE = os.path.join(DATA_DIR, "keypoints_realtime.json")
WORKERS = 3


def extract_clip(path):
    """Track people in the colour half of a clip; return boxes and keypoints for each frame."""
    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    model = YOLO(os.path.join(BASE, MODEL))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    observed = []
    index = -1
    while True:
        index += 1
        if index % FRAME_STEP:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        frame = frame[:, width // 2:].copy()  # right half is colour, left is depth
        result = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml",
                             imgsz=IMAGE_SIZE, conf=CONFIDENCE, verbose=False)[0]
        people = []
        if result.boxes.id is not None:
            for box, track_id, points in zip(result.boxes.xyxy.int().tolist(),
                                             result.boxes.id.int().tolist(),
                                             result.keypoints.data.tolist()):
                people.append([track_id, box, [[round(v, 2) for v in p] for p in points]])
        observed.append({"source_frame": index, "people": people})
    cap.release()
    return os.path.basename(path), {"fps": fps / FRAME_STEP, "frames": observed}


def main():
    clips = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            clips = json.load(f)
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.mp4")))
    remaining = [p for p in paths if os.path.basename(p) not in clips]
    with multiprocessing.Pool(WORKERS) as pool:
        for name, clip in pool.imap_unordered(extract_clip, remaining):
            clips[name] = clip
            print(f"{len(clips)}/{len(paths)} {name}", flush=True)
            with open(CACHE + ".tmp", "w", encoding="utf-8") as f:
                json.dump(clips, f)
            os.replace(CACHE + ".tmp", CACHE)


if __name__ == "__main__":
    main()
