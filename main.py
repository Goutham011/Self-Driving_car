"""
Lane detection entry point.

Examples:
    python main.py --input data/lane/test_images --output out/lane
    python main.py --input data/lane/test_video.mp4 --output out/lane --no-display
    python main.py --input data/lane/test_images/challenge_frame.jpg --no-display
"""

import argparse
import os
from collections import deque
from os.path import basename, join

import cv2

from lane_detection import color_frame_pipeline

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}


def process_image(in_path, out_path, display):
    image = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if image is None:
        print('Could not read image: {}'.format(in_path))
        return False

    result = color_frame_pipeline([image], solid_lines=True)
    cv2.imwrite(out_path, result)

    if display:
        cv2.imshow('lane detection', result)
        cv2.waitKey(0)

    print('{} -> {}'.format(in_path, out_path))
    return True


def process_video(in_path, out_path, display, buffer_size=10):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print('Could not open video: {}'.format(in_path))
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path,
                             fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                             fps=fps,
                             frameSize=(width, height))

    frame_buffer = deque(maxlen=buffer_size)
    frames_done = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        blended = color_frame_pipeline(frames=list(frame_buffer),
                                       solid_lines=True,
                                       temporal_smoothing=True)
        writer.write(blended)
        frames_done += 1

        if display:
            cv2.imshow('lane detection', blended)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print('{} -> {} ({} frames)'.format(in_path, out_path, frames_done))
    return frames_done > 0


def collect_images(path):
    if os.path.isfile(path):
        return [path]
    return sorted(join(path, name) for name in os.listdir(path)
                  if os.path.splitext(name)[1].lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description='Lane detection pipeline')
    parser.add_argument('--input', required=True,
                        help='image file, directory of images, or video file')
    parser.add_argument('--output', default='out/lane',
                        help='output directory (default: out/lane)')
    parser.add_argument('--no-display', action='store_true',
                        help='do not open preview windows')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print('Input does not exist: {}'.format(args.input))
        return 1

    os.makedirs(args.output, exist_ok=True)
    display = not args.no_display
    ext = os.path.splitext(args.input)[1].lower()

    if ext in VIDEO_EXTS:
        ok = process_video(args.input, join(args.output, 'lanes_' + basename(args.input)), display)
    elif os.path.isdir(args.input) or ext in IMAGE_EXTS:
        ok = False
        for image_path in collect_images(args.input):
            out_path = join(args.output, basename(image_path))
            ok = process_image(image_path, out_path, display) or ok
    else:
        print('Unsupported input type: {}'.format(args.input))
        return 1

    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
