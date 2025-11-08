import numpy as np
import cv2
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Return HOG features and optional visualization.
    """
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    """Resize and flatten image"""
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Compute color histograms for each channel and concatenate them"""
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def image_to_features(image, feat_extraction_params):
    color_space = feat_extraction_params['color_space']
    spatial_size = feat_extraction_params['spatial_size']
    hist_bins = feat_extraction_params['hist_bins']
    orient = feat_extraction_params['orient']
    pix_per_cell = feat_extraction_params['pix_per_cell']
    cell_per_block = feat_extraction_params['cell_per_block']
    hog_channel = feat_extraction_params['hog_channel']
    spatial_feat = feat_extraction_params['spatial_feat']
    hist_feat = feat_extraction_params['hist_feat']
    hog_feat = feat_extraction_params['hog_feat']
    resize_h = feat_extraction_params.get('resize_h', 64)
    resize_w = feat_extraction_params.get('resize_w', 64)

    file_features = []
    feature_image = cv2.resize(image, (resize_w, resize_h))

    if color_space != 'RGB':
        conversion = getattr(cv2, f'COLOR_BGR2{color_space}')
        feature_image = cv2.cvtColor(feature_image, conversion)

    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)

    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)

    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient, pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)
        file_features.append(hog_features)

    return np.concatenate(file_features)


def extract_features_from_file_list(image_paths, feat_extraction_params):
    """Extract features from list of image file paths"""
    features = []
    for file in image_paths:
        image = cv2.imread(file)
        if image is not None:
            file_features = image_to_features(image, feat_extraction_params)
            features.append(file_features)
    return features
