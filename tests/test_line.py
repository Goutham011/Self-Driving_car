import numpy as np

from Line import Line


def test_slope_and_bias():
    line = Line(0, 100, 100, 0)
    assert np.isclose(line.slope, -1.0)
    assert np.isclose(line.bias, 100.0)


def test_get_and_set_coords():
    line = Line(0, 0, 10, 10)
    line.set_coords(0, 20, 20, 0)
    assert np.allclose(line.get_coords(), [0, 20, 20, 0])
    assert np.isclose(line.slope, -1.0)
    assert np.isclose(line.bias, 20.0)


def test_vertical_line_does_not_blow_up():
    line = Line(5, 0, 5, 100)
    assert abs(line.slope) > 1e6


def test_draw_writes_pixels():
    import cv2
    img = np.zeros((100, 100), dtype=np.uint8)
    Line(10, 10, 90, 90).draw(img, color=255, thickness=3)
    assert img.max() == 255
