import argparse
import os
import os.path as path

import cv2
import numpy as np

from SSD import (get_SSD_model, process_frame_bgr_with_SSD, VEHICLE_CLASS_IDS)
from vehicle import Vehicle

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v')


def process_pipeline(frame, ssd_model, bbox_helper, color_palette=None,
                     allow_classes=VEHICLE_CLASS_IDS, min_confidence=0.3,
                     verbose=False, return_counts=False):
    if frame is None or frame.size == 0:
        return (frame, (0, 0)) if return_counts else frame

    detected_vehicles = []

    img_blend_out = frame.copy()

    ssd_bboxes = process_frame_bgr_with_SSD(frame, ssd_model, bbox_helper,
                                            allow_classes=allow_classes,
                                            min_confidence=min_confidence)
    for row in ssd_bboxes:
        label, confidence, x_min, y_min, x_max, y_max = row
        x_min = int(round(x_min * frame.shape[1]))
        y_min = int(round(y_min * frame.shape[0]))
        x_max = int(round(x_max * frame.shape[1]))
        y_max = int(round(y_max * frame.shape[0]))

        proposed_vehicle = Vehicle(x_min, y_min, x_max, y_max)

        # fuse near-duplicate detections of the same object (IoU-style NMS);
        # partially overlapping but distinct vehicles are kept separate
        best_iou, best_idx = 0.0, None
        for i, vehicle in enumerate(detected_vehicles):
            overlap = vehicle.iou(proposed_vehicle)
            if overlap > best_iou:
                best_iou, best_idx = overlap, i

        if best_idx is not None and best_iou >= 0.5:
            detected_vehicles[best_idx] = detected_vehicles[best_idx].merged_with(
                proposed_vehicle)
        else:
            detected_vehicles.append(proposed_vehicle)

    for vehicle in detected_vehicles:
        vehicle.draw(img_blend_out, color=(0, 255, 255), thickness=2)

    h, w = frame.shape[:2]
    off_x, off_y = 30, 30
    thumb_h, thumb_w = (96, 128)

    mask = cv2.rectangle(frame.copy(), (0, 0), (w, 2 * off_y + thumb_h),
                         (0, 0, 0), thickness=cv2.FILLED)
    img_blend_out = cv2.addWeighted(src1=mask, alpha=0.3,
                                    src2=img_blend_out, beta=0.8, gamma=0)

    vehicle_thumbs = []
    for vehicle in detected_vehicles:
        x_min, y_min, x_max, y_max = vehicle.coords
        x_min = max(0, min(x_min, w))
        x_max = max(0, min(x_max, w))
        y_min = max(0, min(y_min, h))
        y_max = max(0, min(y_max, h))
        thumb = frame[y_min:y_max, x_min:x_max, :]
        if thumb.size > 0 and thumb.shape[0] > 1 and thumb.shape[1] > 1:
            vehicle_thumbs.append(thumb)

    def thumb_score(thumb):
        return float(np.mean(thumb)) if thumb.size else 0.0

    for i, thumbnail in enumerate(sorted(vehicle_thumbs, key=thumb_score,
                                         reverse=True)):
        start_x = 300 + (i + 1) * off_x + i * thumb_w
        start_y = off_y
        if start_y + thumb_h > h or start_x + thumb_w > w:
            break
        vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
        img_blend_out[start_y:start_y + thumb_h,
                      start_x:start_x + thumb_w, :] = vehicle_thumb

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_blend_out, 'Vehicles in sight: {:02d}'.format(len(detected_vehicles)),
                (20, off_y + thumb_h // 2), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if verbose:
        print('  detections={} kept_vehicles={}'.format(len(ssd_bboxes),
                                                         len(detected_vehicles)))

    if return_counts:
        return img_blend_out, (len(ssd_bboxes), len(detected_vehicles))
    return img_blend_out


def iter_images(input_path):
    if path.isdir(input_path):
        for name in sorted(os.listdir(input_path)):
            if name.lower().endswith(IMAGE_EXTS):
                yield name, path.join(input_path, name)
    elif path.isfile(input_path) and input_path.lower().endswith(IMAGE_EXTS):
        yield path.basename(input_path), input_path


def main():
    parser = argparse.ArgumentParser(description='Run SSD vehicle detection.')
    parser.add_argument('--input', required=True,
                        help='path to an input image or video file')
    parser.add_argument('--output', default='out/ssd',
                        help='output directory (default: out/ssd)')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='minimum confidence threshold (default: 0.3)')
    parser.add_argument('--no-display', action='store_true',
                        help='do not open display windows')
    args = parser.parse_args()

    if not path.exists(args.input):
        parser.error('input not found: {}'.format(args.input))

    os.makedirs(args.output, exist_ok=True)

    ssd_model, bbox_helper, color_palette = get_SSD_model()

    display = not args.no_display

    if args.input.lower().endswith(VIDEO_EXTS):
        cap_in = cv2.VideoCapture(args.input)
        if not cap_in.isOpened():
            parser.error('could not open video: {}'.format(args.input))

        f_counter = 0
        while True:
            ret, frame = cap_in.read()
            if not ret:
                break
            f_counter += 1

            frame_out = process_pipeline(frame, ssd_model, bbox_helper,
                                         color_palette,
                                         allow_classes=VEHICLE_CLASS_IDS,
                                         min_confidence=args.conf)
            cv2.imwrite(path.join(args.output, '{:06d}.jpg'.format(f_counter)),
                        frame_out)

            if display:
                cv2.imshow('ssd', frame_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap_in.release()
        if display:
            cv2.destroyAllWindows()
        print('processed {} frames -> {}'.format(f_counter, args.output))
        return

    images = list(iter_images(args.input))
    if not images:
        parser.error('no images found at: {}'.format(args.input))

    for name, img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print('{}: FAILED TO READ'.format(name))
            continue

        frame_out, (n_raw, n_kept) = process_pipeline(
            frame, ssd_model, bbox_helper, color_palette,
            allow_classes=VEHICLE_CLASS_IDS, min_confidence=args.conf,
            return_counts=True)

        out_path = path.join(args.output, name)
        cv2.imwrite(out_path, frame_out)

        print('{}: {} vehicle(s) detected ({} merged) -> {}'.format(
            name, n_raw, n_kept, out_path))

        if display:
            cv2.imshow('ssd', frame_out)
            cv2.waitKey(1)

    if display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
