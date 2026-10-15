"""
Train the HOG + linear SVM vehicle classifier used by main_hog1.py.

Example:
    python download_dataset.py            # builds data/vehicles + data/non-vehicles
    python train_svm.py                   # writes data/models/*.pickle
"""

import argparse
import os
import pickle
import time

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import feat_extraction_params, training_params
from functions_feat_extraction import image_to_features

C_CANDIDATES = [0.03, 0.1, 0.3, 1.0, 3.0]
from functions_feat_extraction import image_to_features


def list_images(directory):
    if not os.path.isdir(directory):
        raise SystemExit('Directory not found: {} (run download_dataset.py first)'.format(directory))
    names = sorted(n for n in os.listdir(directory)
                   if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    return [os.path.join(directory, n) for n in names]


def extract_features(paths, params):
    features, used = [], 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        features.append(image_to_features(image, params))
        used += 1
    return np.array(features), used


def main():
    parser = argparse.ArgumentParser(description='Train the HOG + SVM vehicle classifier')
    parser.add_argument('--vehicles', default='data/vehicles', help='directory of vehicle crops')
    parser.add_argument('--non-vehicles', default='data/non-vehicles',
                        help='directory of non-vehicle crops')
    parser.add_argument('--output', default='data/models', help='where to store the model pickles')
    parser.add_argument('--max-per-class', type=int, default=3000,
                        help='cap images per class for a balanced training set')
    parser.add_argument('--min-accuracy', type=float, default=0.8,
                        help='exit non-zero when held-out accuracy falls below this')
    parser.add_argument('--seed', type=int, default=training_params['random_state'])
    args = parser.parse_args()

    vehicle_paths = list_images(args.vehicles)[:args.max_per_class]
    non_vehicle_paths = list_images(args.non_vehicles)[:args.max_per_class]
    print('{} vehicle images, {} non-vehicle images'.format(len(vehicle_paths), len(non_vehicle_paths)))

    t0 = time.time()
    vehicle_features, n_vehicles = extract_features(vehicle_paths, feat_extraction_params)
    non_vehicle_features, n_non_vehicles = extract_features(non_vehicle_paths, feat_extraction_params)
    print('Feature extraction: {:.1f}s ({:.1f} ms / image)'.format(
        time.time() - t0, 1000 * (time.time() - t0) / max(n_vehicles + n_non_vehicles, 1)))

    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
    y = np.concatenate((np.ones(n_vehicles), np.zeros(n_non_vehicles)))
    print('Feature matrix: {}  ({} features / image)'.format(X.shape, X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=training_params['test_size'], random_state=args.seed, stratify=y)

    # pick the regularization strength with 3-fold CV on the training split so
    # the held-out test set stays untouched until the very end
    print('Tuning C by 3-fold cross-validation ...')
    best_score, best_c = -1.0, C_CANDIDATES[0]
    for candidate in C_CANDIDATES:
        pipe = make_pipeline(StandardScaler(),
                             LinearSVC(C=candidate, dual='auto', max_iter=20000))
        scores = cross_val_score(pipe, X_train, y_train, cv=3)
        print('  C={:<5} cv accuracy: {:.4f} (+/- {:.4f})'.format(
            candidate, scores.mean(), scores.std()))
        if scores.mean() > best_score:
            best_score, best_c = scores.mean(), candidate
    print('Selected C = {}'.format(best_c))

    t0 = time.time()
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(C=best_c, dual='auto', max_iter=20000))
    clf.fit(X_train, y_train)
    print('Training: {:.1f}s'.format(time.time() - t0))

    y_pred = clf.predict(X_test)
    print('\nHeld-out evaluation ({} samples):'.format(len(y_test)))
    print(classification_report(y_test, y_pred, target_names=['non-vehicle', 'vehicle']))
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, y_pred)))

    os.makedirs(args.output, exist_ok=True)
    svc = clf.named_steps['linearsvc']
    scaler = clf.named_steps['standardscaler']

    artifacts = {
        'svm_trained.pickle': svc,
        'feature_scaler.pickle': scaler,
        'feat_extraction_params.pickle': feat_extraction_params
    }
    for name, payload in artifacts.items():
        path = os.path.join(args.output, name)
        with open(path, 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Wrote {}'.format(path))

    accuracy = float((y_pred == y_test).mean())
    return 0 if accuracy >= args.min_accuracy else 1


if __name__ == '__main__':
    raise SystemExit(main())
