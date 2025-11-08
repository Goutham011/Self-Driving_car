import argparse
import os
import platform
import time
import random
import shutil

import cv2
import torch
import numpy as np
import pyttsx3
from pathlib import Path

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    os.makedirs(out, exist_ok=True)

    engine = pyttsx3.init()

    model = Darknet(opt.cfg, imgsz)
    attempt_download(weights)
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
        view_img = True
        torch.backends.cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    t0 = time.time()
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
                    if int(cls) != 9:
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    roi = im0[y1:y2, x1:x2]
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
                    engine.say(action)
                    engine.runAndWait()

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if view_img:
                cv2.imshow(p, im0)
                if nframes == 1:
                    cv2.waitKey(0)
                elif nframes > 1:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(f"Average FPS: {frame/(time.time() - t0)}")
                        raise StopIteration

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

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    print(f"Average FPS: {nframes/(time.time() - t0)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, required=True, help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')
    parser.add_argument('--output', type=str, default='outputs', help='output folder')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    opt.names = check_file(opt.names)
    print(opt)

    with torch.no_grad():
        detect()
