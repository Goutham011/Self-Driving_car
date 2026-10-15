"""The fast pyramid search must be exactly the reference search at scale 1.0."""

import os
import pickle

import cv2
import numpy as np
import pytest

from config import model_dir
from functions_detection import (search_windows, search_windows_fast, slide_window)

MODEL_PICKLE = os.path.join(model_dir, 'svm_trained.pickle')


def load_artifacts():
    with open(MODEL_PICKLE, 'rb') as handle:
        svc = pickle.load(handle)
    with open(os.path.join(model_dir, 'feature_scaler.pickle'), 'rb') as handle:
        scaler = pickle.load(handle)
    with open(os.path.join(model_dir, 'feat_extraction_params.pickle'), 'rb') as handle:
        params = pickle.load(handle)
    return svc, scaler, params


def sample_frame():
    test_dir = os.path.join('data', 'lane', 'test_images')
    if os.path.isdir(test_dir):
        for name in sorted(os.listdir(test_dir)):
            image = cv2.imread(os.path.join(test_dir, name))
            if image is not None:
                return image
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(540, 960, 3), dtype=np.uint8)


@pytest.mark.skipif(not os.path.exists(MODEL_PICKLE), reason='train_svm.py has not been run')
def test_scale_one_matches_reference_search():
    svc, scaler, params = load_artifacts()
    frame = sample_frame()
    h, w = frame.shape[:2]

    band_start, band_end = int(h * 0.375), int(h * 1.0)
    windows = slide_window(frame, (0, w), (band_start, band_end), (64, 64), (0.5, 0.5))

    reference = search_windows(frame, windows, svc, scaler, params)
    fast = search_windows_fast(frame, svc, scaler, params,
                               scales=(1.0,), stride=32, y_frac=(0.375, 1.0))

    assert sorted(reference) == sorted(fast)


@pytest.mark.skipif(not os.path.exists(MODEL_PICKLE), reason='train_svm.py has not been run')
def test_fast_search_returns_in_bounds_windows():
    svc, scaler, params = load_artifacts()
    frame = sample_frame()
    hot = search_windows_fast(frame, svc, scaler, params, scales=(1.0, 0.75), stride=32)
    assert isinstance(hot, list)
    for (x_min, y_min), (x_max, y_max) in hot:
        assert 0 <= x_min < x_max <= frame.shape[1]
        assert 0 <= y_min < y_max <= frame.shape[0]


@pytest.mark.skipif(not os.path.exists(MODEL_PICKLE), reason='train_svm.py has not been run')
def test_scale_smaller_frame_still_works():
    svc, scaler, params = load_artifacts()
    frame = cv2.resize(sample_frame(), (320, 240))
    hot = search_windows_fast(frame, svc, scaler, params, scales=(1.0, 0.75, 0.55), stride=32)
    assert isinstance(hot, list)
    for (x_min, y_min), (x_max, y_max) in hot:
        assert 0 <= x_min < x_max <= frame.shape[1]
        assert 0 <= y_min < y_max <= frame.shape[0]
