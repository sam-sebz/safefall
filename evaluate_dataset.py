"""Measures fall detection on the UR Fall dataset using the app's real detection code.

Each clip counts as a fall if the tracker raises at least one alert. Fall clips should alert;
everyday activity (ADL) clips should not. Detection results are cached, so settings can be
compared without running the models again.

Usage: python evaluate_dataset.py            scores the hand-written rules
       python evaluate_dataset.py --learned  scores the trained fall model on held-out clips
"""
import glob
import json
import multiprocessing
import os
import sys

os.environ["TELEGRAM_BOT_TOKEN"] = ""  # evaluation must never send alerts

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "datasets", "urfall")
MODEL = "yolo11s-pose.pt"
CONFIDENCE = 0.1
BODY_POINTS = (5, 6, 11, 12)  # COCO left/right shoulder, left/right hip
IMAGE_SIZE = 320
FRAME_STEP = 4  # 30 fps video analysed at about 7.5 fps
CACHE = os.path.join(DATA_DIR, "observations_realtime.json")
BASELINE = os.path.join(DATA_DIR, "results.json")  # results from the original detector
RESULTS = os.path.join(DATA_DIR, "results_realtime.json")
KEYPOINTS_CACHE = os.path.join(DATA_DIR, "keypoints_realtime.json")
LEARNED_RESULTS = os.path.join(DATA_DIR, "results_learned.json")
WORKERS = 3
ANGLES = (45, 60, 75)
DOWN_TIMES = (0.25, 0.5, 1.0)
DROP_RATIOS = (0.25, 0.35, 0.5)
SHRINK_RATIOS = (0.0, 0.6)


def observe_clip(path):
    """Run tracking and pose on the colour half of a clip; return per-frame detections."""
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
        if index % FRAME_STEP:  # matches the frame rate the app manages on six cameras
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[:, frame.shape[1] // 2:].copy()  # right half is colour, left is depth
        result = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml",
                             imgsz=IMAGE_SIZE, conf=CONFIDENCE, verbose=False)[0]
        people = []
        if result.boxes.id is not None:
            for box, track_id, points in zip(result.boxes.xyxy.int().tolist(),
                                             result.boxes.id.int().tolist(),
                                             result.keypoints.data.tolist()):
                body = [[round(v, 2) for v in points[i]] for i in BODY_POINTS]
                people.append([track_id, box, body])
        observed.append(people)
    cap.release()
    return os.path.basename(path), {"fps": fps / FRAME_STEP, "frames": observed}


def replay(clip, settings):
    """Return the alerts the app's tracker would raise for one clip at these settings."""
    from detector import FallTracker, is_down, torso_angle_from_keypoints

    tracker = FallTracker(down_seconds=settings["down"], escalate_seconds=1e9,
                          drop_ratio=settings["drop"], shrink_ratio=settings["shrink"])
    events = []
    for index, people in enumerate(clip["frames"]):
        states = {t: is_down(torso_angle_from_keypoints(body), b[2] - b[0], b[3] - b[1],
                             settings["angle"])
                  for t, b, body in people}
        boxes = {t: tuple(b) for t, b, _ in people}
        events += tracker.update(states, index / clip["fps"], boxes)
    return events


def clip_fall_scores(clip, network):
    """Run the fall model over a keypoint clip; return {track_id: fall chance} for each frame."""
    from fall_model import FallClassifier

    classifier = FallClassifier(network)
    return [classifier.update(frame["people"], index / clip["fps"])
            for index, frame in enumerate(clip["frames"])]


def replay_learned(clip, fall_scores, settings):
    """Return the alerts the app's tracker would raise with the fall model switched on."""
    from detector import FallTracker, is_down, torso_angle_from_keypoints

    tracker = FallTracker(down_seconds=settings["down"], escalate_seconds=1e9,
                          drop_ratio=float("inf"))
    events = []
    for index, (frame, chances) in enumerate(zip(clip["frames"], fall_scores)):
        states = {t: is_down(torso_angle_from_keypoints([points[i] for i in BODY_POINTS]),
                             b[2] - b[0], b[3] - b[1], settings["angle"])
                  for t, b, points in frame["people"]}
        boxes = {t: tuple(b) for t, b, _ in frame["people"]}
        seen = {t for t, chance in chances.items() if chance >= settings["threshold"]}
        events += tracker.update(states, index / clip["fps"], boxes, seen)
    return events


def tally(alerted, names):
    """Turn the set of clips that alerted into a result."""
    falls = [n for n in names if n.startswith("fall")]
    adls = [n for n in names if n.startswith("adl")]
    false_alarms = sorted(n for n in adls if n in alerted)
    return {
        "falls_caught": sum(n in alerted for n in falls), "falls_total": len(falls),
        "false_alarms": len(false_alarms), "adl_total": len(adls),
        "missed_clips": sorted(n for n in falls if n not in alerted),
        "false_alarm_clips": false_alarms,
    }


def score(clips, settings):
    return tally({name for name, clip in clips.items() if replay(clip, settings)}, clips)


def clip_number(name):
    return int(name.split("-")[1])


def balance(result):
    """Share of falls caught minus share of everyday clips that falsely alerted."""
    return (result["falls_caught"] / result["falls_total"]
            - result["false_alarms"] / result["adl_total"])


def line(label, result):
    return (f"{label}: falls {result['falls_caught']}/{result['falls_total']}, "
            f"false alarms {result['false_alarms']}/{result['adl_total']}")


def main():
    clips = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            clips = json.load(f)
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.mp4")))
    remaining = [p for p in paths if os.path.basename(p) not in clips]
    if remaining:
        with multiprocessing.Pool(WORKERS) as pool:
            for name, clip in pool.imap_unordered(observe_clip, remaining):
                clips[name] = clip
                print(f"{len(clips)}/{len(paths)} {name}", flush=True)
                with open(CACHE + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(clips, f)
                os.replace(CACHE + ".tmp", CACHE)  # an interrupted run can resume from here

    # Settings are chosen on odd-numbered clips and scored on even-numbered clips they never saw.
    tuning = {n: c for n, c in clips.items() if clip_number(n) % 2 == 1}
    held_out = {n: c for n, c in clips.items() if clip_number(n) % 2 == 0}
    grid = [{"angle": a, "down": d, "drop": r, "shrink": s}
            for a in ANGLES for d in DOWN_TIMES for r in DROP_RATIOS for s in SHRINK_RATIOS]
    tuned = [(settings, score(tuning, settings)) for settings in grid]
    best_settings, best_tuning = max(tuned, key=lambda item: balance(item[1]))
    held_result = score(held_out, best_settings)

    print("best settings:", best_settings)
    print(line("tuning clips", best_tuning))
    print(line("held-out clips", held_result))
    print("held-out missed:", held_result["missed_clips"])
    print("held-out false alarms:", held_result["false_alarm_clips"])

    with open(BASELINE, encoding="utf-8") as f:
        old = next(r for r in json.load(f) if r["angle"] == 60 and r["down_seconds"] == 1.0)
    old_missed = sum(clip_number(n) % 2 == 0 for n in old["missed_clips"])
    old_false = sum(clip_number(n) % 2 == 0 for n in old["false_alarm_clips"])
    print(f"old system on held-out clips: falls {15 - old_missed}/15, "
          f"false alarms {old_false}/20")

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump({"settings": best_settings, "tuning": best_tuning,
                   "held_out": held_result}, f, indent=2)


def main_learned():
    """Final score: run once, after training has fixed the model and its settings."""
    import torch

    from fall_model import MODEL_FILE, FallNetwork

    with open(KEYPOINTS_CACHE, encoding="utf-8") as f:
        clips = json.load(f)
    held_out = {n: c for n, c in clips.items() if clip_number(n) % 2 == 0}
    saved = torch.load(MODEL_FILE, map_location="cpu")
    network = FallNetwork()
    network.load_state_dict(saved["state"])
    network.eval()

    settings = saved["settings"]
    alerted = {name for name, clip in held_out.items()
               if replay_learned(clip, clip_fall_scores(clip, network), settings)}
    held_result = tally(alerted, held_out)
    with open(RESULTS, encoding="utf-8") as f:
        rules = json.load(f)["held_out"]

    print("settings:", settings)
    print(line("rules on held-out clips", rules))
    print(line("model on held-out clips", held_result))
    print("held-out missed:", held_result["missed_clips"])
    print("held-out false alarms:", held_result["false_alarm_clips"])
    with open(LEARNED_RESULTS, "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "tuning_cross_validated": saved["tuning"],
                   "held_out": held_result, "rules_held_out": rules}, f, indent=2)


if __name__ == "__main__":
    main_learned() if "--learned" in sys.argv else main()
