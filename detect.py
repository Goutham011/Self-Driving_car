import argparse
import os
import queue
import subprocess
import threading
import time
import random
import shutil
import urllib.request

import cv2
import torch
import numpy as np
from pathlib import Path

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

# NOTE: imported *after* the star imports above: utils.utils re-exports sys.platform as `platform`,
# so `import platform` must come last for `platform.system()` to refer to the stdlib module.
import platform

# Known darknet weights + working mirrors (first successful download wins).
WEIGHT_MIRRORS = {
    'yolov3-tiny.weights': [
        'https://pjreddie.com/media/files/yolov3-tiny.weights',
        'https://github.com/ultralytics/yolov3/releases/download/v8/yolov3-tiny.weights',
        'https://huggingface.co/ggml-org/models-moved/resolve/main/yolo/yolov3-tiny.weights',
    ],
    'yolov3.weights': [
        'https://pjreddie.com/media/files/yolov3.weights',
        'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights',
        'https://github.com/ultralytics/yolov3/releases/download/v8/yolov3.weights',
    ],
}
EXPECTED_SIZE = {'yolov3-tiny.weights': 35434956, 'yolov3.weights': 248007048}
MIN_WEIGHT_BYTES = 1000000


def _ssl_context():
    # Some Python builds (e.g. python.org macOS installers) ship no CA store:
    # prefer certifi's bundle when it is importable, else the default context.
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _validate(path, expected):
    size = os.path.getsize(path)
    if size < MIN_WEIGHT_BYTES:
        raise IOError('file too small (%d bytes)' % size)
    if expected and size != expected:
        raise IOError('size %d != expected %d' % (size, expected))
    return size


def _fetch(url, dest, expected=None):
    """Download url -> dest. Tries urllib (certifi CA bundle), then curl (system trust store)."""
    tmp = dest + '.part'
    errors = []
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(tmp):
        os.remove(tmp)

    try:  # attempt 1: urllib
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        size = _validate(tmp, expected)
        os.replace(tmp, dest)
        return size
    except Exception as e:
        errors.append('urllib: %s' % e)
        if os.path.exists(tmp):
            os.remove(tmp)

    try:  # attempt 2: curl (uses the OS trust store; works even without certifi)
        r = subprocess.run(['curl', '-fL', '--retry', '3', '--max-time', '900',
                            '-A', 'Mozilla/5.0', '-o', tmp, url],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise IOError('curl exit %d: %s' % (r.returncode, r.stderr.decode('utf-8', 'ignore')[-200:]))
        size = _validate(tmp, expected)
        os.replace(tmp, dest)
        return size
    except Exception as e:
        errors.append('curl: %s' % e)
        if os.path.exists(tmp):
            os.remove(tmp)

    raise IOError(' + '.join(errors))


def ensure_weights(weights):
    """Auto-download known weights from mirrored sources; exit with a clear message if all fail."""
    name = os.path.basename(weights.strip())
    expected = EXPECTED_SIZE.get(name)
    if os.path.isfile(weights):
        size = os.path.getsize(weights)
        if size >= MIN_WEIGHT_BYTES and (expected is None or size == expected):
            return
        print('%s exists but looks invalid (%d bytes), re-downloading' % (weights, size))
    urls = WEIGHT_MIRRORS.get(name, [])
    if not urls:
        return  # unknown file: caller falls back to upstream attempt_download()
    print('%s missing, trying %d mirror(s)...' % (weights, len(urls)))
    tried = []
    for url in urls:
        tried.append(url)
        print('  %s ... ' % url, end='', flush=True)
        try:
            size = _fetch(url, weights, expected)
            print('OK (%d bytes)' % size)
            return
        except Exception as e:
            print('failed (%s)' % e)
    raise SystemExit(
        '\n[detect] ERROR: could not download %s.\nTried:\n  %s\n'
        'Download it manually and save it as:\n  %s\n'
        'Canonical source: https://pjreddie.com/media/files/%s' %
        (name, '\n  '.join(tried), os.path.abspath(weights), name))


class TextToSpeech:
    """Non-blocking TTS. Speaks only when the announced action changes; never raises into inference."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.last = None
        self._queue = queue.Queue(maxsize=4)
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as e:
            print('TTS disabled (init failed: %s)' % e)
            self.enabled = False
            return
        while True:
            action = self._queue.get()
            if action is None:
                return
            try:
                if getattr(engine, '_inLoop', False):
                    engine._inLoop = False  # pyttsx3 2.98 never clears this flag after runAndWait()
                engine.say(action)
                engine.runAndWait()
                print('[tts] spoken: %s' % action)
            except Exception:
                try:  # last resort: a fresh engine for this utterance
                    engine = pyttsx3.init()
                    engine.say(action)
                    engine.runAndWait()
                    print('[tts] spoken: %s' % action)
                except Exception as e:
                    print('TTS disabled (speak failed: %s)' % e)
                    self.enabled = False
                    return

    def say(self, action):
        if not self.enabled or action == self.last:
            return  # announce only on change of action (not every box / every frame)
        self.last = action
        try:
            self._queue.put_nowait(action)  # never block inference
            print('[tts] announce: %s' % action)
        except queue.Full:
            pass  # drop: a newer action supersedes queued speech

    def close(self):
        if self._thread is None:
            return
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)  # a hung engine must not hang the program


def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size
    out, source, weights, half, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # --view-img accepts on/off; default: show streams, stay headless for files
    if opt.view_img is None:
        view_img = webcam
    else:
        view_img = str(opt.view_img).lower() in ('1', 'true', 'on', 'yes')

    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    os.makedirs(out, exist_ok=True)

    tts = TextToSpeech(enabled=not opt.no_tts)

    model = Darknet(opt.cfg, imgsz)
    ensure_weights(weights)
    if not os.path.isfile(weights):  # unknown weights file: try the upstream (Google Drive) downloader
        try:
            attempt_download(weights)
        except Exception as e:
            raise SystemExit('[detect] ERROR: weights not found and download failed: %s' % e)
    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        load_darknet_weights(model, weights)

    model.to(device).eval()

    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])
        import onnx
        model = onnx.load(f)
        onnx.checker.check_model(model)
        print(onnx.helper.printable_graph(model.graph))
        return

    half = half and device.type != 'cpu'
    if half:
        model.half()

    vid_path, vid_writer = None, None
    if webcam:
        torch.backends.cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    t0 = time.time()
    frame, nframes, stop = 0, 0, False
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None
    for path, img, im0s, vid_cap, frame, nframes in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        if half:
            pred = pred.float()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])

                for *xyxy, conf, cls in det:
                    if int(cls) != 9:  # project feature: only act on traffic lights (COCO class 9)
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    roi = im0[y1:y2, x1:x2]

                    if roi.size == 0:
                        action, color = 'UNKNOWN', (255, 255, 255)
                    else:
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                        mask_red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                        mask_red2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                        mask_yellow = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
                        mask_green = cv2.inRange(hsv, (40, 100, 100), (90, 255, 255))

                        red_pixels = cv2.countNonZero(mask_red)
                        yellow_pixels = cv2.countNonZero(mask_yellow)
                        green_pixels = cv2.countNonZero(mask_green)

                        if red_pixels > yellow_pixels and red_pixels > green_pixels:
                            action = "STOP"
                            color = (0, 0, 255)
                        elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                            action = "WAIT"
                            color = (0, 255, 255)
                        elif green_pixels > red_pixels and green_pixels > yellow_pixels:
                            action = "GO"
                            color = (0, 255, 0)
                        else:
                            action = "UNKNOWN"
                            color = (255, 255, 255)

                    label = f'{action}'
                    plot_one_box(xyxy, im0, label=label, color=color)
                    print('  traffic light -> %s (conf %.2f)' % (action, float(conf)))
                    tts.say(action)

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if view_img:
                cv2.imshow(p, im0)
                if nframes == 1 and not webcam:
                    cv2.waitKey(0)  # single image: hold until a key is pressed
                elif nframes > 1 and (cv2.waitKey(1) & 0xFF == ord('q')):
                    print('Quit (q pressed).')
                    stop = True

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

            if stop:
                break
        if stop:
            break

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    tts.close()

    if save_txt or save_img:
        print('Results saved to %s' % os.path.abspath(out))
        if platform.system() == 'Darwin' and view_img:
            os.system('open "%s"' % out)

    elapsed = time.time() - t0
    print('Done. (%.3fs)' % elapsed)
    if nframes:
        print('Average FPS: %.2f' % (nframes / elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='data/weights/yolov3-tiny.weights',
                        help='weights path (auto-downloaded from mirrors if missing)')
    parser.add_argument('--source', type=str, default='data/traffic_light',
                        help='source: image file, directory of images, video file, stream url/txt, or 0 for webcam')
    parser.add_argument('--output', type=str, default='out/yolo', help='output folder')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', nargs='?', const='on', default=None,
                        help='display results: on/off (default: on for streams, off otherwise)')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-tts', action='store_true', help='disable text-to-speech announcements')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    opt.names = check_file(opt.names)
    print(opt)

    with torch.no_grad():
        detect()
