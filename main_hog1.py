"""
Vehicle detection entry point (HOG features + linear SVM).

Examples:
    python main_hog1.py --input data/hog/project_video.mp4 --output out/hog_svm
    python main_hog1.py --input data/ssd/images/000000097679.jpg --output out/hog_svm
"""

import argparse
import os
import pickle
import time
from os.path import basename, join

import cv2

from config import detection_params, model_dir
from functions_detection import draw_boxes, merge_detections, search_windows_fast

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}


def load_model(directory):
    required = ['svm_trained.pickle', 'feature_scaler.pickle', 'feat_extraction_params.pickle']
    for name in required:
        path = join(directory, name)
        if not os.path.exists(path):
            raise SystemExit('Missing {} - run "python train_svm.py" first'.format(path))

    with open(join(directory, 'svm_trained.pickle'), 'rb') as handle:
        svc = pickle.load(handle)
    with open(join(directory, 'feature_scaler.pickle'), 'rb') as handle:
        scaler = pickle.load(handle)
    with open(join(directory, 'feat_extraction_params.pickle'), 'rb') as handle:
        params = pickle.load(handle)
    return svc, scaler, params


def detect_vehicles(frame, svc, scaler, params, threshold):
    hot_windows, scores = search_windows_fast(frame, svc, scaler, params,
                                              return_scores=True)
    boxes, _, _ = merge_detections(
        frame, hot_windows, threshold=threshold,
        min_area=detection_params['min_box_area'],
        scores=scores,
        min_blob_score=detection_params['min_blob_score'],
        min_hot_windows=detection_params['min_hot_windows'])
    return boxes, len(hot_windows)


def annotate(frame, boxes):
    output = draw_boxes(frame, boxes, color=(0, 0, 255), thick=3)
    cv2.putText(output, 'Vehicles: {:02d}'.format(len(boxes)), (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def process_image(path, out_path, svc, scaler, params, threshold, display):
    frame = cv2.imread(path)
    if frame is None:
        print('Could not read image: {}'.format(path))
        return False

    t0 = time.time()
    boxes, n_hot = detect_vehicles(frame, svc, scaler, params, threshold)
    annotated = annotate(frame, boxes)
    cv2.imwrite(out_path, annotated)
    print('{}: {} vehicle(s) from {} hot windows ({:.2f}s) -> {}'.format(
        basename(path), len(boxes), n_hot, time.time() - t0, out_path))

    if display:
        cv2.imshow('vehicle detection', annotated)
        cv2.waitKey(0)
    return True


def process_video(path, out_path, svc, scaler, params, threshold, display):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print('Could not open video: {}'.format(path))
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames_done = 0
    total_vehicles = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, _ = detect_vehicles(frame, svc, scaler, params, threshold)
        writer.write(annotate(frame, boxes))
        frames_done += 1
        total_vehicles += len(boxes)
        if frames_done % 10 == 0:
            elapsed = time.time() - t_start
            print('  frame {} | {:.1f} fps | {} vehicles so far'.format(
                frames_done, frames_done / max(elapsed, 1e-6), total_vehicles))
        if display:
            cv2.imshow('vehicle detection', annotate(frame, boxes))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print('{} -> {} ({} frames, {:.1f} fps, {} vehicles)'.format(
        path, out_path, frames_done, frames_done / max(elapsed, 1e-6), total_vehicles))
    return frames_done > 0


def main():
    parser = argparse.ArgumentParser(description='HOG + SVM vehicle detection')
    parser.add_argument('--input', required=True, help='image file, image directory, or video file')
    parser.add_argument('--output', default='out/hog_svm', help='output directory')
    parser.add_argument('--model-dir', default=model_dir, help='directory with the trained pickles')
    parser.add_argument('--threshold', type=int, default=detection_params['heatmap_threshold'],
                        help='minimum number of hot windows per vehicle')
    parser.add_argument('--no-display', action='store_true', help='do not open preview windows')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print('Input does not exist: {}'.format(args.input))
        return 1

    svc, scaler, params = load_model(args.model_dir)
    os.makedirs(args.output, exist_ok=True)
    display = not args.no_display
    ext = os.path.splitext(args.input)[1].lower()

    if ext in VIDEO_EXTS:
        ok = process_video(args.input, join(args.output, 'vehicles_' + basename(args.input)),
                           svc, scaler, params, args.threshold, display)
    elif os.path.isdir(args.input) or ext in IMAGE_EXTS:
        ok = False
        paths = ([join(args.input, n) for n in sorted(os.listdir(args.input))
                  if os.path.splitext(n)[1].lower() in IMAGE_EXTS]
                 if os.path.isdir(args.input) else [args.input])
        for path in paths:
            ok = process_image(path, join(args.output, basename(path)),
                               svc, scaler, params, args.threshold, display) or ok
    else:
        print('Unsupported input type: {}'.format(args.input))
        return 1

    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
