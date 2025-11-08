import cv2
import pickle
import numpy as np
from functions_detection import compute_windows_multiscale, search_windows, draw_boxes

# Load trained model, scaler, and parameters
with open('data/svm_trained.pickle', 'rb') as f:
    svc = pickle.load(f)
with open('data/feature_scaler.pickle', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('data/feat_extraction_params.pickle', 'rb') as f:
    feat_extraction_params = pickle.load(f)

# Load input video
video_path = 'test_images/project_video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[ERROR] Could not read {video_path}")
    exit()

# Setup video writer
output_path = 'output_project_video.avi'
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("🚗 Starting vehicle detection on video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles
    windows = compute_windows_multiscale(frame)
    hot_windows = search_windows(frame, windows, svc, feature_scaler, feat_extraction_params)
    output_frame = draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=3)

    # Write the frame
    out.write(output_frame)

    # Optional: Show live output
    cv2.imshow('Vehicle Detection', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Output saved as:", output_path)
