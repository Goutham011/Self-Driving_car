import argparse
import os
import sys
import glob
import cv2

import pandas as pd

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LISA traffic-light CSV annotations to YOLO label files.')
    parser.add_argument('--annotation-root', default='data/lisa/lisa-traffic-light-dataset/Annotations/Annotations',
                        help='folder containing dayTrain/ and nightTrain/ annotation clip folders')
    parser.add_argument('--image-root', default='data/lisa/lisa-traffic-light-dataset',
                        help='folder containing dayTrain/ and nightTrain/ frames')
    parser.add_argument('--output', default='data/lisa/input',
                        help='output folder (images -> <output>/images, labels -> <output>/labels)')
    return parser.parse_args()


show_info = True
images_with_required_classes = 0
total_images = 0
labels = {
    'go': 0,
    'stop': 1,
    'stopLeft': 2,
    'goLeft': 3,
    'warning': 4,
    'warningLeft': 5
}

root_folder_names = ['dayTrain', 'nightTrain']
root_folder_name_mapper = {
    'dayTrain': 'dayClip',
    'nightTrain': 'nightClip'
}

opt = parse_args()
annotation_root = opt.annotation_root
image_root = opt.image_root
labels_dir = os.path.join(opt.output, 'labels')
images_dir = os.path.join(opt.output, 'images')

if not os.path.isdir(annotation_root) or not os.path.isdir(image_root):
    sys.exit(
        'LISA traffic-light dataset not found (no dataset is downloaded by this script).\n'
        '  looked for annotations: %s\n'
        '  looked for images:      %s\n'
        'Place the dataset under data/lisa/ (Kaggle: "lisa-traffic-light-dataset"), or pass\n'
        '--annotation-root / --image-root pointing at it.' % (annotation_root, image_root)
    )

os.makedirs(labels_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)


def get_coords(tag, x_min, y_min, x_max, y_max, images_with_required_classes):
        """
        We will return a single digit for each label.
        Also we will return normalized x_center, y_center, 
        width, and height. We will divice the x_center and width by 
        image width and y_center and height by image height to
        normalize. Each image is 1280 in width and 960 in height. 
        """
        if tag in labels:
            if tag == 'go':
                label = labels['go']
                color = (0, 255, 0)
            elif tag == 'stop':
                label = labels['stop']
                color = (0, 0, 255)
            elif tag == 'stopLeft':
                label = labels['stopLeft']
                color = (0, 0, 155)
            elif tag == 'goLeft':
                label = labels['goLeft']
                color = (0, 200, 200)
            elif tag == 'warning':
                label = labels['warning']
                color = (29, 118, 255)
            elif tag == 'warningLeft':
                label = labels['warningLeft']
                color = (0 , 118, 255)

            x_center = ((x_max + x_min) / 2) / 1280 
            y_center = ((y_max + y_min) / 2) / 960
            w = (x_max - x_min) / 1280
            h = (y_max - y_min) / 960
            return label, x_center, y_center, w, h
        else:
            label = ''
            x_center = ''
            y_center = ''
            w = ''
            h = ''
            return label, x_center, y_center, w, h

for root_folder_name in root_folder_names:
    root_path = f"{annotation_root}/{root_folder_name}"
    if not os.path.isdir(root_path):
        print(f"Skipping {root_folder_name}: not found at {root_path}")
        continue
    folder_names = os.listdir(root_path)
    num_folders = len(folder_names)
    mapped_clip = root_folder_name_mapper[root_folder_name]

    for i in range(1,  num_folders+1): 
        csv_path = f"{annotation_root}/{root_folder_name}/{mapped_clip}{i}/frameAnnotationsBOX.csv"
        if not os.path.isfile(csv_path):
            print(f"Skipping {mapped_clip}{i}: no annotations at {csv_path}")
            continue
        print('##### NEW CSV AND IMAGES ####')
        # read the annotation CSV file
        df = pd.read_csv(csv_path, delimiter=';')
        # get all image paths
        image_paths = glob.glob(f"{image_root}/{root_folder_name}/{root_folder_name}/{mapped_clip}{i}/frames/*.jpg")
        image_paths.sort()

        total_images += len(image_paths)

        if show_info:
            print('NUMBER OF IMAGE AND UNIQUE CSV FILE NAMES MAY NOT MATCH')
            print('NOT A PROBLEM')
            print(f"Total objects in current CSV file: {len(df)}")
            print(f"Unique Filenames: {len(df['Filename'].unique())}")
            print(df.head())
            print(f"Total images in current folder: {len(image_paths)}")

        tags = df['Annotation tag'].values
        x_min = df['Upper left corner X'].values
        y_min = df['Upper left corner Y'].values
        x_max = df['Lower right corner X'].values
        y_max = df['Lower right corner Y'].values

        file_counter = 0 # to counter through CSV file
        # iterate through all image paths
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            image_name = image_path.split(os.path.sep)[-1]
            label_path = os.path.join(labels_dir, image_name.split('.')[0] + '.txt')
            if os.path.exists(label_path):
                os.remove(label_path)  # idempotent: re-running must not duplicate labels
            # iterate through all CSV rows
            for j in range(len(df)):
                if file_counter < len(df):
                    file_name = df.loc[file_counter]['Filename'].split('/')[-1]
                    if file_name == image_name:
                        label, x, y, w, h = get_coords(tags[file_counter], 
                                                    x_min[file_counter],
                                                    y_min[file_counter],
                                                    x_max[file_counter],
                                                    y_max[file_counter], 
                                                    images_with_required_classes)
                        with open(label_path, 'a+') as f:
                            if type(label) == int:
                                f.writelines(f"{label} {x} {y} {w} {h}\n")
                                f.close()
                            else:
                                f.writelines(f"")
                                f.close()
                            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                            cv2.imwrite(os.path.join(images_dir, image_name), image)
                            file_counter += 1
                        # continue
                    if file_name != image_name:
                        break

print(f"Total images parsed through: {total_images}")
# print(f"Total images with desired classes: {images_with_required_classes}")