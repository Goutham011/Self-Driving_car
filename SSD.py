"""SSD (Single Shot MultiBox Detector) vehicle detection helpers.

Model: MobileNet-SSD trained on PASCAL VOC (Caffe), from
https://github.com/chuanqi305/MobileNet-SSD
  - deploy.prototxt
  - mobilenet_iter_73000.caffemodel (~23 MB)

Label convention: background_label_id = 0, then the 20 VOC classes in order
(aeroplane=1 ... tvmonitor=20), so car=7, bus=6, motorbike=14, person=15.
"""

import os
import urllib.error
import urllib.request

import cv2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ssd')

PROTOTXT_URL = ('https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/'
                'master/deploy.prototxt')
CAFFEMODEL_URL = ('https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/'
                  'master/mobilenet_iter_73000.caffemodel')

PROTOTXT_PATH = os.path.join(DATA_DIR, 'deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(DATA_DIR, 'mobilenet_iter_73000.caffemodel')

CLASS_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

VEHICLE_CLASS_IDS = [CLASS_IDS['bus'], CLASS_IDS['car'], CLASS_IDS['motorbike']]

INPUT_SIZE = (300, 300)
SCALE = 0.007843
MEAN = (127.5, 127.5, 127.5)


class BBoxHelper:
    """Small helper carrying label information for decoded SSD boxes."""

    def __init__(self, class_names=CLASS_NAMES, vehicle_ids=VEHICLE_CLASS_IDS,
                 input_size=INPUT_SIZE):
        self.class_names = list(class_names)
        self.vehicle_ids = list(vehicle_ids)
        self.input_size = input_size

    def class_name(self, class_id):
        if 0 <= int(class_id) < len(self.class_names):
            return self.class_names[int(class_id)]
        return 'id:{}'.format(int(class_id))


def _make_color_palette(n_classes=len(CLASS_NAMES)):
    import numpy as np

    rng = np.random.RandomState(7)
    colors = rng.randint(60, 256, size=(n_classes, 3)).tolist()
    return [[int(c) for c in color] for color in colors]


def _open_url(url, timeout):
    import ssl

    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        return urllib.request.urlopen(url, timeout=timeout, context=context)
    except (ssl.SSLCertVerificationError, urllib.error.URLError) as exc:
        reason = getattr(exc, 'reason', exc)
        if not isinstance(exc, ssl.SSLCertVerificationError) and \
                not isinstance(reason, ssl.SSLCertVerificationError) and \
                'CERTIFICATE_VERIFY_FAILED' not in str(exc):
            raise
        print('warning: SSL certificate verification failed for {}, '
              'retrying without verification'.format(url))
        return urllib.request.urlopen(url, timeout=timeout,
                                      context=ssl._create_unverified_context())


def _download(url, dest, timeout=120):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    try:
        with _open_url(url, timeout) as response:
            total = response.headers.get('Content-Length')
            total = int(total) if total else None
            received = 0
            last_pct = -1
            with open(part, 'wb') as out:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
                    received += len(chunk)
                    if total:
                        pct = int(100.0 * received / total)
                        if pct != last_pct:
                            last_pct = pct
                            print('\r  downloading {} ({}%)'.format(
                                os.path.basename(dest), pct), end='')
        if total:
            print('')
        os.replace(part, dest)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        if os.path.exists(part):
            os.remove(part)
        raise RuntimeError(
            'Failed to download {}\n  -> {}\n'
            'Check your network connection or place the file manually at: {}'.format(
                url, exc, dest)) from exc


def _ensure_model_files():
    for url, dest in ((PROTOTXT_URL, PROTOTXT_PATH),
                      (CAFFEMODEL_URL, CAFFEMODEL_PATH)):
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        print('Downloading {} -> {}'.format(url, dest))
        _download(url, dest)
    return PROTOTXT_PATH, CAFFEMODEL_PATH


def get_SSD_model():
    """Load (and download on first use) the MobileNet-SSD Caffe network.

    Returns (net, bbox_helper, color_palette).
    """
    prototxt, caffemodel = _ensure_model_files()
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    except cv2.error as exc:
        raise RuntimeError(
            'Could not load MobileNet-SSD from\n  {}\n  {}\n'
            'Delete these files and re-run to re-download. OpenCV error: {}'.format(
                prototxt, caffemodel, exc)) from exc
    bbox_helper = BBoxHelper()
    color_palette = _make_color_palette()
    return net, bbox_helper, color_palette


def process_frame_bgr_with_SSD(frame_bgr, net, bbox_helper,
                               allow_classes=VEHICLE_CLASS_IDS,
                               min_confidence=0.3):
    """Run SSD on a BGR frame.

    ``allow_classes`` filters detections by integer class id (defaults to
    :data:`VEHICLE_CLASS_IDS`; ``None`` or an empty list allows all classes).

    Returns a list of rows ``(label, confidence, x_min, y_min, x_max, y_max)``
    where ``label`` is the integer class id and the coordinates are normalized
    floats in [0, 1].
    """
    import numpy as np

    if frame_bgr is None or frame_bgr.size == 0:
        return []

    input_size = getattr(bbox_helper, 'input_size', INPUT_SIZE)

    blob = cv2.dnn.blobFromImage(frame_bgr, SCALE, input_size, MEAN, swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    detections = detections.reshape(-1, detections.shape[-1])

    if allow_classes is None:
        allow_classes = []  # no class filtering
    allow_classes = set(int(c) for c in allow_classes)

    rows = []
    for detection in detections:
        _, label, confidence = detection[0], int(detection[1]), float(detection[2])
        if confidence < min_confidence:
            continue
        if allow_classes and label not in allow_classes:
            continue
        x_min = float(np.clip(detection[3], 0.0, 1.0))
        y_min = float(np.clip(detection[4], 0.0, 1.0))
        x_max = float(np.clip(detection[5], 0.0, 1.0))
        y_max = float(np.clip(detection[6], 0.0, 1.0))
        if x_max <= x_min or y_max <= y_min:
            continue
        rows.append((label, confidence, x_min, y_min, x_max, y_max))
    return rows


def show_SSD_results(bboxes, frame_bgr, color_palette=None):
    """Draw boxes and labels on ``frame_bgr`` in place and return it."""
    if color_palette is None:
        color_palette = _make_color_palette()

    h, w = frame_bgr.shape[:2]
    for row in bboxes:
        label, confidence, x_min, y_min, x_max, y_max = row
        p1 = (int(round(x_min * w)), int(round(y_min * h)))
        p2 = (int(round(x_max * w)), int(round(y_max * h)))
        color = tuple(int(c) for c in color_palette[int(label) % len(color_palette)])
        cv2.rectangle(frame_bgr, p1, p2, color, 2)
        text = '{} {:.0%}'.format(CLASS_NAMES[int(label)]
                                  if 0 <= int(label) < len(CLASS_NAMES)
                                  else 'id:{}'.format(int(label)), confidence)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = p1[1] - 6 if p1[1] - 6 - text_size[1] >= 0 else p1[1] + text_size[1] + 6
        cv2.rectangle(frame_bgr, (p1[0], ty - text_size[1] - 4),
                      (p1[0] + text_size[0], ty + 4), color, cv2.FILLED)
        cv2.putText(frame_bgr, text, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame_bgr
