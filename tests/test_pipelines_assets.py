"""Asset / integration checks for the heavier pipelines (skipped when absent)."""

import os

import numpy as np
import pytest

MODEL_DIR = os.path.join('data', 'models')
SSD_MODEL_FILES = [os.path.join('data', 'ssd', n) for n in
                   ('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')]
YOLO_WEIGHTS = os.path.join('data', 'weights', 'yolov3-tiny.weights')


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, 'svm_trained.pickle')),
                    reason='train_svm.py has not been run')
def test_svm_model_predicts_on_window():
    import pickle

    import cv2

    from functions_feat_extraction import image_to_features

    with open(os.path.join(MODEL_DIR, 'svm_trained.pickle'), 'rb') as handle:
        svc = pickle.load(handle)
    with open(os.path.join(MODEL_DIR, 'feature_scaler.pickle'), 'rb') as handle:
        scaler = pickle.load(handle)
    with open(os.path.join(MODEL_DIR, 'feat_extraction_params.pickle'), 'rb') as handle:
        params = pickle.load(handle)

    rng = np.random.default_rng(0)
    window = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    features = image_to_features(window, params).reshape(1, -1)
    prediction = svc.predict(scaler.transform(features))
    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)


@pytest.mark.skipif(not all(os.path.exists(p) for p in SSD_MODEL_FILES),
                    reason='SSD model files not downloaded')
def test_ssd_detects_a_car_on_known_frame():
    import cv2

    from SSD import get_SSD_model, process_frame_bgr_with_SSD, VEHICLE_CLASS_IDS

    net, helper, _ = get_SSD_model()
    frame = cv2.imread(os.path.join('data', 'ssd', 'images', '000000097679.jpg'))
    if frame is None:
        pytest.skip('sample frame missing')
    detections = process_frame_bgr_with_SSD(frame, net, helper, allow_classes=VEHICLE_CLASS_IDS,
                                            min_confidence=0.3)
    assert len(detections) >= 1


@pytest.mark.skipif(not os.path.exists(YOLO_WEIGHTS), reason='yolov3-tiny weights not downloaded')
def test_yolo_assets_consistent():
    assert os.path.exists(os.path.join('cfg', 'yolov3-tiny.cfg'))
    with open(os.path.join('data', 'coco.names')) as handle:
        names = [line.strip() for line in handle if line.strip()]
    assert len(names) == 80
    assert names[9] == 'traffic light'
    assert names[2] == 'car'


def test_vendored_utils_present():
    for module in ('datasets', 'utils', 'torch_utils', 'google_utils', 'layers', 'parse_config'):
        assert os.path.exists(os.path.join('utils', module + '.py')), module
