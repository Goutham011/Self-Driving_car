"""Assemble a folder of JPG frames into an mp4 clip (LISA daySequence helper).

Usage:
    python img_to_vid.py --images data/lisa/lisa-traffic-light-dataset/daySequence1/daySequence1/frames \
                         --output data/lisa/input/test_data/day_seq1.mp4 --fps 20
"""

import argparse
import glob
import os
import sys

import cv2

try:  # moviepy < 2.0
    from moviepy.editor import ImageSequenceClip
except ImportError:  # moviepy >= 2.0 removed moviepy.editor
    from moviepy import ImageSequenceClip


def parse_args():
    parser = argparse.ArgumentParser(description='Convert a folder of image frames into a video clip.')
    parser.add_argument('--images', default='data/lisa/lisa-traffic-light-dataset/daySequence1/daySequence1/frames',
                        help='folder with *.jpg frames')
    parser.add_argument('--output', default='data/lisa/input/test_data/day_seq1.mp4',
                        help='output video path (*.mp4)')
    parser.add_argument('--fps', type=float, default=20, help='frames per second of the output video')
    return parser.parse_args()


opt = parse_args()

image_paths = sorted(glob.glob(os.path.join(opt.images, '*.jpg')) + glob.glob(os.path.join(opt.images, '*.png')))
if not image_paths:
    sys.exit('No frames found in %s\n'
             'Point --images at a folder of LISA frames (no dataset is downloaded by this script).' % opt.images)

print(f'{len(image_paths)} frames from {opt.images}')
print(image_paths[:5])

# ImageSequenceClip requires uniform frame sizes: resize everything to the first frame
# (folders of mixed-size stills - e.g. COCO images - otherwise crash)
first = cv2.imread(image_paths[0])
if first is None:
    sys.exit('Could not read frame: %s' % image_paths[0])
target_h, target_w = first.shape[:2]
frames = []
mixed_sizes = False
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        sys.exit('Could not read frame: %s' % path)
    if img.shape[:2] != (target_h, target_w):
        mixed_sizes = True
        img = cv2.resize(img, (target_w, target_h))
    frames.append(img[:, :, ::-1])  # moviepy expects RGB, cv2 reads BGR
if mixed_sizes:
    print(f'Mixed frame sizes found: resized all frames to {target_w}x{target_h}')

out_dir = os.path.dirname(opt.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

clip = ImageSequenceClip(frames, fps=opt.fps)
clip.write_videofile(opt.output, fps=opt.fps)
print(f'DONE -> {opt.output}')
