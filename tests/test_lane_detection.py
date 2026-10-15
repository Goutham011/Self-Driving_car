import os

import cv2
import numpy as np
import pytest

from lane_detection import color_frame_pipeline, get_lane_lines

TEST_IMAGES = os.path.join('data', 'lane', 'test_images')


def synthetic_road(width=960, height=540):
    """Dark road with two bright diagonal lane lines."""
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.line(img, (width // 2 - 220, height), (width // 2 - 30, height // 2), (230, 230, 230), 12)
    cv2.line(img, (width // 2 + 220, height), (width // 2 + 30, height // 2), (230, 230, 230), 12)
    return img


def test_get_lane_lines_finds_both_sides():
    lanes = get_lane_lines(synthetic_road(), solid_lines=True)
    assert len(lanes) == 2
    slopes = sorted(l.slope for l in lanes)
    assert slopes[0] < 0 < slopes[1]


def test_pipeline_returns_same_size_image():
    frame = synthetic_road()
    out = color_frame_pipeline([frame], solid_lines=True)
    assert out.shape == frame.shape


def test_pipeline_blanks_do_not_crash():
    blank = np.zeros((540, 960, 3), dtype=np.uint8)
    out = color_frame_pipeline([blank], solid_lines=True)
    assert out.shape == blank.shape


def test_temporal_smoothing_over_frames():
    frames = [synthetic_road() for _ in range(4)]
    out = color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True)
    assert out.shape == frames[0].shape


def test_resizes_arbitrary_input():
    frame = cv2.resize(synthetic_road(), (640, 360))
    out = color_frame_pipeline([frame], solid_lines=True)
    assert out.shape == frame.shape


@pytest.mark.skipif(not os.path.isdir(TEST_IMAGES), reason='sample images not downloaded')
def test_real_frames_detect_lanes():
    for name in sorted(os.listdir(TEST_IMAGES)):
        image = cv2.imread(os.path.join(TEST_IMAGES, name))
        if image is None:
            continue
        lanes = get_lane_lines(image, solid_lines=True)
        assert len(lanes) == 2, 'no lanes inferred for {}'.format(name)
