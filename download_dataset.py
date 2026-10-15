"""
Build a vehicle / non-vehicle image dataset from the COCO validation set.

The classic HOG+SVM lane projects rely on the GTI vehicle dataset whose download
links have rotted away, so this script builds an equivalent dataset from COCO:

  * positive crops  -> tight crops around car / motorcycle / bus / train / truck
  * negative crops  -> random square patches from images that contain no vehicle

Images and annotations are cached under --cache (default: data/cache) so the
script only downloads once. Everything written into --output is meant for
training and is not committed to git.

Example:
    python download_dataset.py --num-vehicles 1500 --num-non-vehicles 1500
"""

import argparse
import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
ANNOTATIONS_ZIP = 'annotations_trainval2017.zip'
INSTANCE_JSONS = [('val2017', 'instances_val2017.json'), ('train2017', 'instances_train2017.json')]
IMAGE_URL = 'http://images.cocodataset.org/{}/{:012d}.jpg'

VEHICLE_CATEGORY_IDS = {3, 4, 5, 7, 8}   # car, motorcycle, bus, train, truck
MIN_SIDE = 48
MAX_ASPECT = 2.5
MIN_AREA = 70 * 70


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print('Downloading {} -> {}'.format(url, dest))

    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(dest, 'wb') as out:
                shutil.copyfileobj(response, out)
            return dest
        except Exception as error:      # noqa: BLE001 - report and retry
            last_error = error
            print('  attempt {} failed: {}'.format(attempt + 1, error))
    raise RuntimeError('Could not download {}: {}'.format(url, last_error))


def ensure_annotations(cache_dir):
    """Download the COCO annotation zip once and extract the instance JSONs we need."""
    zip_path = download(ANNOTATIONS_URL, os.path.join(cache_dir, ANNOTATIONS_ZIP))
    resolved = []

    with zipfile.ZipFile(zip_path) as archive:
        for split, name in INSTANCE_JSONS:
            member = [n for n in archive.namelist() if n.endswith(name)]
            if not member:
                print('warning: {} not found in annotation zip'.format(name))
                continue
            dest = os.path.join(cache_dir, member[0])
            if not os.path.exists(dest):
                print('Extracting {} ...'.format(name))
                archive.extract(member[0], cache_dir)
            resolved.append((split, dest))

    if not resolved:
        raise RuntimeError('No COCO instance annotations found in {}'.format(zip_path))
    return resolved


def load_coco_index(json_path, split):
    """Index one COCO instances file into (vehicle entries, clean image entries)."""
    with open(json_path) as handle:
        coco = json.load(handle)

    images = {img['id']: img for img in coco['images']}
    vehicle_boxes = {}
    seen_ids = set()

    for ann in coco['annotations']:
        if ann.get('iscrowd', 0):
            continue
        image_id = ann['image_id']
        seen_ids.add(image_id)
        if ann['category_id'] in VEHICLE_CATEGORY_IDS:
            vehicle_boxes.setdefault(image_id, []).append(ann['bbox'])

    vehicle_entries = [{'id': image_id, 'split': split, 'boxes': boxes}
                       for image_id, boxes in vehicle_boxes.items() if image_id in images]
    # a clean image has annotations but never a vehicle
    clean_entries = [{'id': image_id, 'split': split}
                     for image_id in seen_ids if image_id not in vehicle_boxes and image_id in images]
    return vehicle_entries, clean_entries


def valid_box(bbox):
    x, y, w, h = bbox
    if w < MIN_SIDE or h < MIN_SIDE or w * h < MIN_AREA:
        return False
    aspect = w / float(h)
    return 1.0 / MAX_ASPECT <= aspect <= MAX_ASPECT


def save_crop(image, box, dest):
    img_h, img_w = image.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x_min, y_min = max(x, 0), max(y, 0)
    x_max, y_max = min(x + w, img_w), min(y + h, img_h)
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return False
    return cv2.imwrite(dest, crop)


def random_square_crop(image, rng, forbidden=()):
    """
    Random square patch that does not touch any forbidden box.

    `forbidden` holds vehicle bboxes in (x, y, w, h) format; patches may not
    overlap them, which is what makes "hard negatives" (pedestrians, carts,
    signs next to cars) possible inside vehicle images.
    """
    img_h, img_w = image.shape[:2]
    side = min(rng.randint(64, 128), img_h, img_w)
    if side < 48:
        return None

    for _ in range(8):
        y = rng.randint(0, img_h - side)
        x = rng.randint(0, img_w - side)
        x_max, y_max = x + side, y + side
        clashes = False
        for fx, fy, fw, fh in forbidden:
            margin = 4
            if (x < fx + fw + margin and x_max > fx - margin and
                    y < fy + fh + margin and y_max > fy - margin):
                clashes = True
                break
        if not clashes:
            return image[y:y_max, x:x_max]
    return None


def fetch_image(image_info, image_dir, keep_source):
    image_id = image_info['id']
    dest = os.path.join(image_dir, '{}/{}.jpg'.format(image_info['split'], image_id))
    if not os.path.exists(dest):
        download(IMAGE_URL.format(image_info['split'], image_id), dest)
    image = cv2.imread(dest)
    if not keep_source and os.path.exists(dest):
        os.remove(dest)
    return image_id, image


def main():
    parser = argparse.ArgumentParser(description='Build vehicle/non-vehicle dataset from COCO')
    parser.add_argument('--output', default='data', help='dataset root (default: data)')
    parser.add_argument('--cache', default='data/cache', help='download cache directory')
    parser.add_argument('--num-vehicles', type=int, default=1500, help='number of vehicle crops')
    parser.add_argument('--num-non-vehicles', type=int, default=1500, help='number of background crops')
    parser.add_argument('--workers', type=int, default=12, help='parallel image downloads')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keep-source', action='store_true',
                        help='keep the downloaded source images in the cache')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    instances = []
    for split, json_path in ensure_annotations(args.cache):
        vehicle_part, clean_part = load_coco_index(json_path, split)
        instances.append((vehicle_part, clean_part))

    vehicle_entries = [e for part, _ in instances for e in part]
    clean_entries = [e for _, part in instances for e in part]
    rng.shuffle(vehicle_entries)
    rng.shuffle(clean_entries)
    print('{} COCO images with vehicles, {} without'.format(len(vehicle_entries), len(clean_entries)))

    vehicle_dir = os.path.join(args.output, 'vehicles')
    non_vehicle_dir = os.path.join(args.output, 'non-vehicles')
    image_cache = os.path.join(args.cache, 'images')
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(non_vehicle_dir, exist_ok=True)

    def process(entries, worker, quota=None):
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_image, e, image_cache, args.keep_source): e
                       for e in entries}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    image_id, image = future.result()
                except Exception as error:      # noqa: BLE001 - skip broken downloads
                    print('  skip image {}: {}'.format(entry.get('id'), error))
                    continue
                if image is None:
                    continue
                done += worker(image_id, image, entry)
                if done % 100 == 0 and done > 0:
                    print('  ... {} crops written'.format(done))
                if quota is not None and done >= quota:
                    for pending in futures:
                        pending.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
        return done

    hard_negatives = {'count': 0, 'cap': args.num_non_vehicles // 2}

    def vehicle_worker(image_id, image, entry):
        written = 0
        valid_boxes = [b for b in entry['boxes'] if valid_box(b)]
        for index, bbox in enumerate(entry['boxes']):
            if not valid_box(bbox):
                continue
            dest = os.path.join(vehicle_dir, '{}_{:012d}_{:02d}.jpg'.format(
                entry['split'], image_id, index))
            if save_crop(image, bbox, dest):
                written += 1

        # hard negative: patch from the same scene that contains no vehicle
        if hard_negatives['count'] < hard_negatives['cap']:
            crop = random_square_crop(image, rng, forbidden=valid_boxes)
            if crop is not None:
                dest = os.path.join(non_vehicle_dir, 'hard_{}_{:012d}.jpg'.format(
                    entry['split'], image_id))
                if cv2.imwrite(dest, crop):
                    hard_negatives['count'] += 1
        return written

    def clean_worker(image_id, image, entry):
        crop = random_square_crop(image, rng)
        if crop is None:
            return 0
        dest = os.path.join(non_vehicle_dir, '{}_{:012d}_00.jpg'.format(
            entry['split'], image_id))
        return 1 if cv2.imwrite(dest, crop) else 0

    print('Collecting vehicle crops ...')
    # keep pulling images until the quota is filled (entries with many boxes vary a lot)
    written = process(vehicle_entries, vehicle_worker, quota=args.num_vehicles)
    if written < args.num_vehicles:
        print('Warning: only {} vehicle crops collected (asked for {})'.format(
            written, args.num_vehicles))

    print('Collecting non-vehicle crops ...')
    remaining = max(args.num_non_vehicles - hard_negatives['count'], 0)
    process(clean_entries, clean_worker, quota=remaining)

    n_vehicles = len(os.listdir(vehicle_dir))
    n_non_vehicles = len(os.listdir(non_vehicle_dir))
    print('Done: {} vehicle crops in {}, {} non-vehicle crops in {}'.format(
        n_vehicles, vehicle_dir, n_non_vehicles, non_vehicle_dir))

    if n_vehicles == 0 or n_non_vehicles == 0:
        print('Dataset is empty - aborting', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
