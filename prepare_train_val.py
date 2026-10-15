"""
This python script prepares train.txt and val.txt for YOLOv3
training.

Usage:
    python prepare_train_val.py --images data/lisa/input/images --seed 42
"""

import argparse
import os
import random
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Split image list into train.txt / val.txt for YOLOv3 training.')
    parser.add_argument('--images', default='data/lisa/input/images',
                        help='folder with the training images (output of prepare_labels.py)')
    parser.add_argument('--out-train', default='data/train.txt', help='where to write the train list')
    parser.add_argument('--out-val', default='data/val.txt', help='where to write the validation list')
    parser.add_argument('--train-split', type=float, default=0.80, help='fraction of images used for training')
    parser.add_argument('--seed', type=int, default=42, help='shuffle seed (reproducible split)')
    return parser.parse_args()


opt = parse_args()

if not os.path.isdir(opt.images):
    sys.exit('Image folder not found: %s\n'
             'Run prepare_labels.py first (or pass --images <folder> pointing at the LISA images).\n'
             'No dataset is downloaded by this script.' % opt.images)

# get all the image file names from the images folder
image_files = sorted(f for f in os.listdir(opt.images)
                     if os.path.splitext(f)[-1].lower() in ('.jpg', '.jpeg', '.png', '.bmp'))

if not image_files:
    sys.exit('No images found in %s - nothing to split.' % opt.images)

# shuffle with a fixed seed, then take a clean disjoint 80/20 split
# (random.sample / slicing guarantees no duplicates and no overlap, unlike random.randint loops)
random.Random(opt.seed).shuffle(image_files)
n_train = int(len(image_files) * opt.train_split)
train_files = image_files[:n_train]
valid_files = image_files[n_train:]

print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(valid_files)}")

for out_path, files in ((opt.out_train, train_files), (opt.out_val, valid_files)):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:  # 'w' not 'a': re-running replaces the lists instead of duplicating them
        for name in files:
            f.write(f"{os.path.join(opt.images, name)}\n")

print(f"Wrote {opt.out_train} and {opt.out_val}")
