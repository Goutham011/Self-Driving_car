import os

# directories that contain the vehicle / non-vehicle training images
root_data_vehicle = os.environ.get('SDC_VEHICLES_DIR', 'data/vehicles')
root_data_non_vehicle = os.environ.get('SDC_NON_VEHICLES_DIR', 'data/non-vehicles')

# where the trained classifier artifacts are stored / loaded from
model_dir = os.environ.get('SDC_MODEL_DIR', 'data/models')

# parameters used in the phase of feature extraction
feat_extraction_params = {
    'color_space': 'YCrCb',     # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'spatial_size': (32, 32),   # Spatial binning dimensions
    'hist_bins': 32,            # Number of histogram bins
    'orient': 9,                # HOG orientations
    'pix_per_cell': 8,          # HOG pixels per cell
    'cell_per_block': 2,        # HOG cells per block
    'hog_channel': 'ALL',       # Can be 0, 1, 2, or "ALL"
    'spatial_feat': True,       # Spatial features on or off
    'hist_feat': True,          # Histogram features on or off
    'hog_feat': True,           # HOG features on or off
    'resize_h': 64,             # Window is resized to this size before features
    'resize_w': 64
}

# parameters used when training the vehicle classifier
training_params = {
    'test_size': 0.2,           # fraction of the data kept for evaluation
    'random_state': 42          # fixed seed for reproducible splits
}

# heatmap post-processing used at inference time
detection_params = {
    'heatmap_threshold': 3,     # a vehicle box needs at least this many hits
    'min_box_area': 2500,       # discard blobs smaller than 50x50 (ground/curb noise)
    'min_blob_score': 0.8,      # mean classifier score required per blob
    'min_hot_windows': 8        # minimum hot windows supporting a blob
}
