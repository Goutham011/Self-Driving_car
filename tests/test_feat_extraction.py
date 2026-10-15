import numpy as np
import pytest

from functions_feat_extraction import (bin_spatial, color_hist, get_hog_features,
                                       image_to_features)

PARAMS = {
    'color_space': 'YCrCb',
    'spatial_size': (32, 32),
    'hist_bins': 32,
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': 'ALL',
    'spatial_feat': True,
    'hist_feat': True,
    'hog_feat': True,
    'resize_h': 64,
    'resize_w': 64
}


def sample_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(80, 90, 3), dtype=np.uint8)


def test_bin_spatial_shape():
    assert bin_spatial(sample_image(), (32, 32)).shape == (32 * 32 * 3,)


def test_color_hist_shape():
    assert color_hist(sample_image(), nbins=32).shape == (3 * 32,)


def test_hog_features_vector_length():
    gray = sample_image()[:, :, 0].astype(np.float64)
    feats = get_hog_features(gray, orient=9, pix_per_cell=8, cell_per_block=2)
    # 80x89 usable cells -> (10-1) x (11-1) blocks x 2x2 cells x 9 orientations
    assert feats.shape == ((9 * 10) * (2 * 2) * 9,)


def test_hog_feature_length_matches_block_grid():
    from functions_feat_extraction import get_hog_features as hog_features
    gray = sample_image()[:, :, 0].astype(np.float64)
    feats = hog_features(gray, orient=9, pix_per_cell=8, cell_per_block=2)
    # window shrinks to the largest (win - block) % stride == 0 size:
    # 90 px wide -> 88 px -> 10 block columns, 80 px high -> 9 block rows
    assert feats.shape == (9 * 10 * (2 * 2) * 9,)
    assert np.isfinite(feats).all()
    assert (feats >= 0).all()


def test_image_to_features_deterministic_length():
    img = sample_image()
    first = image_to_features(img, PARAMS)
    second = image_to_features(img, PARAMS)
    assert first.shape == second.shape
    assert np.allclose(first, second)
    assert first.shape == (32 * 32 * 3 + 3 * 32 + 9 * 2 * 2 * 3 * (64 // 8 - 1) ** 2,)


def test_image_to_features_hog_only():
    params = dict(PARAMS, spatial_feat=False, hist_feat=False)
    img = sample_image()
    feats = image_to_features(img, params)
    assert feats.shape[0] == 9 * 2 * 2 * 3 * (64 // 8 - 1) ** 2


def test_unknown_color_space_raises():
    with pytest.raises(AttributeError):
        image_to_features(sample_image(), dict(PARAMS, color_space='NOT_A_SPACE'))
