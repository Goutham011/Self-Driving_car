import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from functions_feat_extraction import image_to_features
from project_5_utils import stitch_together


def draw_labeled_bounding_boxes(img, labeled_frame, num_objects, color=(255, 0, 0), thickness=6):
    """Draw one bounding box per connected component of `labeled_frame`."""
    for car_number in range(1, num_objects + 1):
        rows, cols = np.where(labeled_frame == car_number)
        if rows.size == 0:
            continue
        x_min, x_max = cols.min(), cols.max()
        y_min, y_max = rows.min(), rows.max()
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def compute_heatmap_from_detections(frame, hot_windows, threshold=5, verbose=False):
    """Accumulate hot windows into a heatmap and threshold it."""
    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.int32)

    for bbox in hot_windows:
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        heatmap[max(y_min, 0):max(y_max, 0), max(x_min, 0):max(x_max, 0)] += 1

    # keep pixels hit by at least `threshold` windows (int32 heatmap: no uint8
    # wrap-around when many windows fire on the same pixel)
    heatmap_thresh = np.where(heatmap >= threshold, 255, 0).astype(np.uint8)
    heatmap_thresh = cv2.morphologyEx(
        heatmap_thresh,
        cv2.MORPH_CLOSE,
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1
    )

    if verbose:
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(heatmap, cmap='hot')
        ax[2].imshow(heatmap_thresh, cmap='hot')
        plt.show()

    return heatmap, heatmap_thresh


def merge_detections(frame, hot_windows, threshold=5, min_area=400,
                     scores=None, min_blob_score=None, min_hot_windows=None):
    """
    Fuse overlapping hot windows into a small set of vehicle boxes.

    Returns (boxes, labeled_frame, num_objects) where boxes is a list of
    ((x_min, y_min), (x_max, y_max)) sorted by descending area.

    When `scores` (one classifier decision value per window) is given, a blob
    is only kept if the mean score of the windows supporting it reaches
    `min_blob_score` and at least `min_hot_windows` windows hit it - texture
    noise (curbs, grass) fires a handful of weak windows, real vehicles fire
    many confident ones.
    """
    _, heatmap_thresh = compute_heatmap_from_detections(frame, hot_windows, threshold=threshold)
    labeled_frame, num_objects = ndimage.label(heatmap_thresh)

    boxes = []
    for car_number in range(1, num_objects + 1):
        rows, cols = np.where(labeled_frame == car_number)
        if rows.size == 0:
            continue
        x_min, x_max = int(cols.min()), int(cols.max())
        y_min, y_max = int(rows.min()), int(rows.max())
        if (x_max - x_min) * (y_max - y_min) < min_area:
            continue

        if scores is not None and (min_blob_score is not None or
                                   min_hot_windows is not None):
            frame_h, frame_w = frame.shape[:2]
            supporting = []
            for window, score in zip(hot_windows, scores):
                cx = min(max((window[0][0] + window[1][0]) // 2, 0), frame_w - 1)
                cy = min(max((window[0][1] + window[1][1]) // 2, 0), frame_h - 1)
                if labeled_frame[cy, cx] == car_number:
                    supporting.append(float(score))
            if min_hot_windows is not None and len(supporting) < min_hot_windows:
                continue
            if min_blob_score is not None and supporting:
                if sum(supporting) / len(supporting) < min_blob_score:
                    continue
            elif min_blob_score is not None:
                continue

        boxes.append(((x_min, y_min), (x_max, y_max)))

    boxes.sort(key=lambda b: (b[1][0] - b[0][0]) * (b[1][1] - b[0][1]), reverse=True)
    return boxes, labeled_frame, num_objects


def compute_windows_multiscale(image, verbose=False):
    """Multi-scale sliding windows restricted to the lower part of the frame."""
    h, w = image.shape[:2]
    windows_multiscale = []

    windows_32 = slide_window(image, (0, w), (4 * h // 8, 5 * h // 8), (32, 32), (0.5, 0.5))
    windows_64 = slide_window(image, (0, w), (4 * h // 8, 6 * h // 8), (64, 64), (0.5, 0.5))
    windows_128 = slide_window(image, (0, w), (3 * h // 8, h), (128, 128), (0.5, 0.5))

    windows_multiscale.extend([windows_32, windows_64, windows_128])

    if verbose:
        img_32 = draw_boxes(image, windows_32, (0, 0, 255), 1)
        img_64 = draw_boxes(image, windows_64, (0, 255, 0), 1)
        img_128 = draw_boxes(image, windows_128, (255, 0, 0), 1)

        stitched = stitch_together([img_32, img_64, img_128], (1, 3), resize_dim=(1300, 500))
        cv2.imshow('', stitched)
        cv2.waitKey()

    return np.concatenate(windows_multiscale)


def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None),
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Slide a window of size xy_window over img with the given overlap fraction.

    Windows are clipped to the requested region: a window is only emitted when
    it fits entirely inside (x_start_stop, y_start_stop).
    """
    x_start_stop = list(x_start_stop)
    y_start_stop = list(y_start_stop)

    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    x_step = int(xy_window[0] * (1 - xy_overlap[0]))
    y_step = int(xy_window[1] * (1 - xy_overlap[1]))

    if x_step <= 0 or y_step <= 0:
        raise ValueError('xy_overlap must be < 1')

    def count(span, window, step):
        if span < window:
            return 0
        return (span - window) // step + 1

    n_x_windows = count(x_span, xy_window[0], x_step)
    n_y_windows = count(y_span, xy_window[1], y_step)

    window_list = []
    for i in range(n_y_windows):
        for j in range(n_x_windows):
            start_x = j * x_step + x_start_stop[0]
            start_y = i * y_step + y_start_stop[0]
            window_list.append(((start_x, start_y),
                                (start_x + xy_window[0], start_y + xy_window[1])))

    return window_list


def draw_boxes(img, bbox_list, color=(0, 0, 255), thick=6):
    img_copy = np.copy(img)
    for bbox in bbox_list:
        cv2.rectangle(img_copy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
    return img_copy


def build_feature_matrix(image, windows, feat_extraction_params):
    """
    Stack the per-window feature vectors into one (n_windows, n_features) matrix.

    Every window goes through functions_feat_extraction.image_to_features - the
    exact same code path used when the classifier was trained.
    """
    rows = []
    kept = []
    for window in windows:
        x_min, y_min = window[0]
        x_max, y_max = window[1]
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        rows.append(image_to_features(crop, feat_extraction_params))
        kept.append(window)
    if not rows:
        return np.empty((0, 0)), []
    return np.vstack(rows), kept


def predict_windows(image, windows, clf, scaler, feat_extraction_params,
                    return_scores=False):
    """Classify windows in one batched call; returns the windows predicted as vehicle.

    With return_scores=True returns (hits, scores) where scores holds one raw
    classifier decision value per hit window.
    """
    features, kept = build_feature_matrix(image, windows, feat_extraction_params)
    if not kept:
        return ([], []) if return_scores else []
    scaled = scaler.transform(features)

    if return_scores:
        if hasattr(clf, 'decision_function'):
            raw = np.asarray(clf.decision_function(scaled)).ravel()
            predictions = (raw > 0).astype(np.int64)
        else:
            raw = np.asarray(clf.predict_proba(scaled))[:, 1]
            predictions = (raw > 0.5).astype(np.int64)
        hits = [window for window, prediction in zip(kept, predictions)
                if prediction == 1]
        hit_scores = [float(value) for value, prediction in zip(raw, predictions)
                      if prediction == 1]
        return hits, hit_scores

    predictions = clf.predict(scaled)
    return [window for window, prediction in zip(kept, predictions) if prediction == 1]


def search_windows(img, windows, clf, scaler, feat_extraction_params):
    """Reference implementation: classify every window with clf, batched."""
    return predict_windows(img, windows, clf, scaler, feat_extraction_params)


def search_windows_fast(img, clf, scaler, feat_extraction_params,
                        scales=(1.0, 0.75, 0.55), stride=32, y_frac=(0.375, 1.0),
                        return_scores=False):
    """
    Multi-scale sliding-window search in one call.

    The frame is scanned at several scales with a fixed 64x64 window - the
    size all training crops are resized to - so a vehicle appears at the same
    "apparent size" the classifier learned, while the overlapping windows feed
    the heatmap post-processing in merge_detections().

    Windows are returned in the coordinates of the original (unscaled) image.
    With return_scores=True the (hot_windows, scores) tuple is returned, where
    scores holds one classifier decision value per hot window.
    """
    full_img_h, full_img_w = img.shape[:2]
    hot_windows = []
    hot_scores = []

    for scale in scales:
        scaled_w = int(round(full_img_w * scale))
        scaled_h = int(round(full_img_h * scale))
        if min(scaled_w, scaled_h) < 64:
            continue

        scaled = cv2.resize(img, (scaled_w, scaled_h),
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        band_start = int(scaled_h * y_frac[0])
        band_end = int(scaled_h * y_frac[1])
        if band_end - band_start < 64:
            continue

        windows = slide_window(scaled, (0, scaled_w), (band_start, band_end),
                               (64, 64), (1 - stride / 64.0, 1 - stride / 64.0))
        if return_scores:
            hits, scores = predict_windows(scaled, windows, clf, scaler,
                                           feat_extraction_params, return_scores=True)
        else:
            hits = predict_windows(scaled, windows, clf, scaler, feat_extraction_params)
            scores = None

        for (x_min, y_min), (x_max, y_max) in hits:
            hot_windows.append(((int(round(x_min / scale)), int(round(y_min / scale))),
                                (int(round(x_max / scale)), int(round(y_max / scale)))))
        if return_scores:
            hot_scores.extend(scores)

    if return_scores:
        return hot_windows, hot_scores
    return hot_windows
