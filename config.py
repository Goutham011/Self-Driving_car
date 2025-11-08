# root directory that contain all vehicle images in nested subdirectories
root_data_vehicle = 'data/vehicles'

# root directory that contain all NON-vehicle images in nested subdirectories
root_data_non_vehicle = 'data/non-vehicles'

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
    'resize_h': 64,             # ✅ Add this
    'resize_w': 64              # ✅ Add this
}








