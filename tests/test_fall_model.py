import numpy as np
import torch

from fall_model import (FEATURES, KEYPOINTS, SAMPLE_RATE, WINDOW_FRAMES, FallClassifier,
                        FallNetwork, build_window, flip_windows, seen_frames)

STEP = 1 / SAMPLE_RATE
BOX = (100, 100, 150, 300)


def body(top, confidence=0.9):
    return [[100 + k, top + k * 10, confidence] for k in range(KEYPOINTS)]


def walk(frames, box=BOX):
    return [(i * STEP, box, body(box[1])) for i in range(frames)]


LAST = (WINDOW_FRAMES - 1) * STEP


def test_window_shape_and_seen_count():
    window = build_window(walk(WINDOW_FRAMES), LAST)
    assert window.shape == (WINDOW_FRAMES, FEATURES)
    assert seen_frames(window) == WINDOW_FRAMES


def test_window_ignores_where_the_person_is_in_the_picture():
    near = walk(WINDOW_FRAMES)
    moved = [(t, (b[0] + 40, b[1] + 25, b[2] + 40, b[3] + 25),
              [[x + 40, y + 25, c] for x, y, c in p]) for t, b, p in near]
    assert np.allclose(build_window(near, LAST), build_window(moved, LAST))


def test_window_keeps_the_drop_in_height():
    frames = [(i * STEP, (100, 100 + i * 10, 150, 300 + i * 10), body(100 + i * 10))
              for i in range(WINDOW_FRAMES)]
    window = build_window(frames, LAST)
    assert window[-1, -4] > window[0, -4]


def test_missing_frames_stay_empty_when_frame_rate_falls():
    frames = walk(WINDOW_FRAMES)[::3]
    assert seen_frames(build_window(frames, LAST)) == len(frames)


def test_low_confidence_points_are_hidden():
    frames = [(i * STEP, BOX, body(100, 0.05)) for i in range(WINDOW_FRAMES)]
    assert np.all(build_window(frames, LAST)[:, :KEYPOINTS * 3] == 0)


def test_flip_twice_returns_the_original():
    windows = torch.rand(2, WINDOW_FRAMES, FEATURES)
    assert torch.allclose(flip_windows(flip_windows(windows)), windows)


def test_flip_mirrors_x_and_swaps_left_and_right():
    windows = torch.rand(1, WINDOW_FRAMES, FEATURES)
    flipped = flip_windows(windows)
    # Keypoint 5 is the left shoulder and 6 the right one.
    assert torch.allclose(flipped[:, :, 5 * 3], -windows[:, :, 6 * 3])
    assert torch.allclose(flipped[:, :, 5 * 3 + 1], windows[:, :, 6 * 3 + 1])
    assert torch.allclose(flipped[:, :, -5], -windows[:, :, -5])
    assert torch.allclose(flipped[:, :, -4], windows[:, :, -4])


def test_network_gives_one_score_each():
    assert FallNetwork()(torch.zeros(3, WINDOW_FRAMES, FEATURES)).shape == (3,)


def feed(classifier, track_id, box, frames, start=0):
    result = {}
    for i in range(start, start + frames):
        result = classifier.update([(track_id, box, body(box[1]))], i * STEP)
    return result


def classifier():
    return FallClassifier(FallNetwork().eval())


def test_classifier_waits_for_enough_sightings():
    watcher = classifier()
    assert feed(watcher, 1, BOX, 3) == {}
    assert 0 <= feed(watcher, 1, BOX, 1, start=3)[1] <= 1


def test_new_id_at_the_same_spot_keeps_the_history():
    watcher = classifier()
    feed(watcher, 1, BOX, 5)
    feed(watcher, 2, BOX, 1, start=5)
    assert 1 not in watcher.buffers
    assert len(watcher.buffers[2]) == 6


def test_new_id_elsewhere_starts_fresh():
    watcher = classifier()
    feed(watcher, 1, BOX, 5)
    feed(watcher, 2, (400, 100, 450, 300), 1, start=5)
    assert len(watcher.buffers[1]) == 5
    assert len(watcher.buffers[2]) == 1


def test_people_gone_too_long_are_forgotten():
    watcher = classifier()
    feed(watcher, 1, BOX, 5)
    feed(watcher, 2, (400, 100, 450, 300), 1, start=50)
    assert 1 not in watcher.buffers


def test_empty_frame_is_fine():
    assert classifier().update([], 0.0) == {}
