import numpy as np

from functions_detection import (compute_heatmap_from_detections, merge_detections,
                                 slide_window)


def test_slide_window_geometry():
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    windows = slide_window(img, (0, 960), (270, 540), (64, 64), (0.5, 0.5))
    assert len(windows) > 0
    for (x_min, y_min), (x_max, y_max) in windows:
        assert x_max - x_min == 64
        assert y_max - y_min == 64
        assert 0 <= x_min < x_max <= 960
        assert 270 <= y_min < y_max <= 540


def test_slide_window_rejects_full_overlap():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        slide_window(img, (0, 100), (0, 100), (64, 64), (1.0, 1.0))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_heatmap_survives_more_than_255_hits():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = ((10, 10), (50, 50))
    heatmap, thresh = compute_heatmap_from_detections(frame, [box] * 300, threshold=5)
    assert heatmap.max() == 300          # int32: no wrap-around
    assert thresh.max() == 255


def test_merge_detections_groups_overlapping_windows():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    windows = [((100, 100), (164, 164)),
               ((108, 104), (172, 168)),
               ((40, 40), (60, 60))]    # small isolated blob -> dropped by min_area
    boxes, _, _ = merge_detections(frame, windows, threshold=1, min_area=1000)
    assert len(boxes) == 1
    (x_min, y_min), (x_max, y_max) = boxes[0]
    assert x_min <= 100 and x_max >= 164
    assert y_min <= 100 and y_max >= 164


def test_merge_detections_threshold_requires_multiple_hits():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    windows = [((100, 100), (164, 164)),
               ((108, 104), (172, 168))]
    # pixels need >= 2 hits: only the overlap of the two windows survives
    boxes, _, _ = merge_detections(frame, windows, threshold=2, min_area=400)
    assert len(boxes) == 1
    (x_min, y_min), (x_max, y_max) = boxes[0]
    assert 106 <= x_min <= 110 and 102 <= y_min <= 106
    assert 161 <= x_max <= 165 and 161 <= y_max <= 165
