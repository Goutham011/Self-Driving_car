import argparse
import os
import os.path as path

import cv2

from SSD import (get_SSD_model, process_frame_bgr_with_SSD, show_SSD_results,
                 VEHICLE_CLASS_IDS)


def main():
    parser = argparse.ArgumentParser(
        description='Run SSD vehicle detection on a video and write frames.')
    parser.add_argument('--input', required=True, help='path to input video')
    parser.add_argument('--output', default='out/ssd/video',
                        help='output directory for frames '
                             '(default: out/ssd/video)')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='minimum confidence threshold (default: 0.3)')
    args = parser.parse_args()

    if not path.isfile(args.input):
        parser.error('input video not found: {}'.format(args.input))

    os.makedirs(args.output, exist_ok=True)

    SSD_net, bbox_helper, color_palette = get_SSD_model()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        parser.error('could not open video: {}'.format(args.input))

    counter = 0
    total_dets = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = process_frame_bgr_with_SSD(frame, SSD_net, bbox_helper,
                                            min_confidence=args.conf,
                                            allow_classes=VEHICLE_CLASS_IDS)
        total_dets += len(bboxes)
        show_SSD_results(bboxes, frame, color_palette=color_palette)

        out_file = path.join(args.output, '{:06d}.jpg'.format(counter))
        cv2.imwrite(out_file, frame)
        counter += 1

    cap.release()
    print('wrote {} frames to {} ({} vehicle detections total)'.format(
        counter, args.output, total_dets))


if __name__ == '__main__':
    main()
