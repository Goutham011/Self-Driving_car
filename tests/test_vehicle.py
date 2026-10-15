import numpy as np

from vehicle import Vehicle


def test_coords_and_center():
    vehicle = Vehicle(10, 20, 50, 80)
    assert vehicle.coords == (10, 20, 50, 80)
    assert vehicle.center == (30, 50)


def test_contains():
    vehicle = Vehicle(10, 20, 50, 80)
    assert vehicle.contains(30, 50)
    assert not vehicle.contains(5, 5)


def test_iou():
    a = Vehicle(0, 0, 10, 10)
    assert a.iou(Vehicle(0, 0, 10, 10)) == 1.0
    assert a.iou(Vehicle(20, 20, 30, 30)) == 0.0
    assert abs(a.iou(Vehicle(5, 5, 15, 15)) - 25 / 175) < 1e-9
    assert a.iou((5, 5, 15, 15)) == a.iou(Vehicle(5, 5, 15, 15))


def test_merged_with_is_union():
    union = Vehicle(0, 0, 10, 10).merged_with(Vehicle(5, 5, 15, 20))
    assert union.coords == (0, 0, 15, 20)


def test_draw():
    import cv2
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = Vehicle(10, 10, 60, 60).draw(frame, color=(0, 255, 255), thickness=2)
    assert out is frame
    assert cv2.countNonZero(frame[:, :, 0]) == 0
    assert frame[:, :, 1].sum() > 0
