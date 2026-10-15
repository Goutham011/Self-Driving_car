import numpy as np
import cv2
from skimage.feature import hog as skimage_hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True,
                     transform_sqrt=False):
    """
    Return HOG features (and optionally a visualization image) for a single
    channel image.

    Uses OpenCV's C++ HOG implementation (~10x faster than a pure-python/numpy
    fallback) with the classic Dalal-Triggs setup: unsigned orientations,
    L2-Hys block normalization, blocks sliding one cell at a time.

    Because OpenCV always evaluates a window as a standalone image, the
    features of a 64x64 crop are bit-identical no matter whether the crop was
    taken from a training image or from a live video frame - training and
    inference can never drift apart.

    `transform_sqrt` is accepted for backwards compatibility and ignored.
    `feature_vec=False` is accepted for backwards compatibility; the features
    are always returned flattened (this backend cannot produce block maps).
    """
    gray = np.asarray(img)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(np.uint8(np.clip(gray, 0, 255)))

    h, w = gray.shape[:2]
    block_px = pix_per_cell * cell_per_block
    if h < block_px or w < block_px:
        raise ValueError('image {}x{} smaller than HOG block {}'.format(w, h, block_px))

    # OpenCV requires (win - block) to be a multiple of the stride; shrink the
    # window by at most (stride - 1) pixels so any input size works
    win_w = block_px + (w - block_px) // pix_per_cell * pix_per_cell
    win_h = block_px + (h - block_px) // pix_per_cell * pix_per_cell

    descriptor = cv2.HOGDescriptor(_winSize=(win_w, win_h),
                                   _blockSize=(block_px, block_px),
                                   _blockStride=(pix_per_cell, pix_per_cell),
                                   _cellSize=(pix_per_cell, pix_per_cell),
                                   _nbins=orient)
    features = descriptor.compute(gray[:win_h, :win_w]).ravel().astype(np.float64)

    if vis:
        hog_image = skimage_hog(gray[:win_h, :win_w], orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                visualize=True, feature_vector=True)[1]
        return features, hog_image

    return features


def convert_color(image, color_space):
    """Convert a BGR image to the colour space used by the feature extractor."""
    if color_space == 'RGB':
        return image
    return cv2.cvtColor(image, getattr(cv2, 'COLOR_BGR2{}'.format(color_space)))


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
    feature_image = convert_color(feature_image, color_space)

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
