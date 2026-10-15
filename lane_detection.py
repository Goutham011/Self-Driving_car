import numpy as np
import cv2

from Line import Line


# processing resolution of the pipeline; all geometry below assumes it
PROC_W, PROC_H = 960, 540

# Canny edge thresholds
CANNY_LOW, CANNY_HIGH = 50, 80

# Hough line parameters
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LEN = 25
HOUGH_MAX_LINE_GAP = 120

# only lines whose slope angle lies in this range are considered lane candidates
MIN_LANE_SLOPE = 0.5
MAX_LANE_SLOPE = 2.0

# fraction of the frame that belongs to the lane region of interest
ROI_BOTTOM_MARGIN = 50.0 / PROC_W
ROI_TOP_LEFT_X = 450.0 / PROC_W
ROI_TOP_RIGHT_X = 490.0 / PROC_W
ROI_TOP_Y = 310.0 / PROC_H


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image, mask


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform. Returns [] when no line is found.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        return []
    return lines


def weighted_img(img, initial_img, alpha=0.8, beta=1.0, lam=0.0):
    """
    Returns resulting blend image computed as follows:

    initial_img * alpha + img * beta + lam
    """
    img = np.uint8(img)
    if len(img.shape) == 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

    return cv2.addWeighted(initial_img, alpha, img, beta, lam)


def compute_lane_from_candidates(line_candidates, img_shape):
    """
    Compute lines that approximate the position of both road lanes.

    :param line_candidates: lines from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: (left_lane, right_lane); either entry is None when the lane
             could not be inferred from the candidates
    """
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]

    img_h, img_w = img_shape[:2]

    left_lane = None
    if neg_lines:
        neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
        neg_slope = np.median([l.slope for l in neg_lines])
        if abs(neg_slope) > 1e-6:
            left_lane = Line(0, neg_bias, -np.int32(np.round(neg_bias / neg_slope)), 0)

    right_lane = None
    if pos_lines:
        lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
        lane_right_slope = np.median([l.slope for l in pos_lines])
        if abs(lane_right_slope) > 1e-6:
            right_lane = Line(0, lane_right_bias,
                              np.int32(np.round((img_h - lane_right_bias) / lane_right_slope)),
                              img_h)

    return left_lane, right_lane


def get_lane_lines(color_image, solid_lines=True):
    """
    Take a color road frame and try to infer the lane lines in it.

    :param color_image: input frame (any size, BGR or RGB - only gradients matter)
    :param solid_lines: if True, infer the two solid lane lines; if False,
                        return every candidate line from the Hough transform
    :return: list of Line objects (empty when nothing is found)
    """
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)
    img_edge = cv2.Canny(img_blur, CANNY_LOW, CANNY_HIGH)

    detected_lines = hough_lines_detection(img=img_edge,
                                           rho=HOUGH_RHO,
                                           theta=HOUGH_THETA,
                                           threshold=HOUGH_THRESHOLD,
                                           min_line_len=HOUGH_MIN_LINE_LEN,
                                           max_line_gap=HOUGH_MAX_LINE_GAP)

    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]

    if not solid_lines:
        return detected_lines

    candidate_lines = [line for line in detected_lines
                       if MIN_LANE_SLOPE <= np.abs(line.slope) <= MAX_LANE_SLOPE]

    left_lane, right_lane = compute_lane_from_candidates(candidate_lines, img_gray.shape)
    return [lane for lane in (left_lane, right_lane) if lane is not None]


def smoothen_over_time(lane_lines):
    """
    Smooth the lane line inference over a window of frames and return the average lines.

    Frames in which a lane could not be inferred are ignored instead of
    dragging the average towards zero.
    """
    if not lane_lines:
        return []

    n_slots = max(len(frame_lanes) for frame_lanes in lane_lines)
    averaged = []

    for slot in range(n_slots):
        coords = [frame_lanes[slot].get_coords()
                  for frame_lanes in lane_lines
                  if len(frame_lanes) > slot]
        if coords:
            averaged.append(Line(*np.mean(np.array(coords), axis=0)))

    return averaged


def get_roi_vertices(img_w, img_h):
    """Lane region of interest as a polygon, expressed in fractions of the frame."""
    return np.array([[(int(ROI_BOTTOM_MARGIN * img_w), img_h),
                      (int(ROI_TOP_LEFT_X * img_w), int(ROI_TOP_Y * img_h)),
                      (int(ROI_TOP_RIGHT_X * img_w), int(ROI_TOP_Y * img_h)),
                      (img_w - int(ROI_BOTTOM_MARGIN * img_w), img_h)]],
                    dtype=np.int32)


def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):
    """
    Entry point for the lane detection pipeline.

    Takes a list of frames and returns a single image (same colour space as the
    input) with the inferred road lanes overlaid. For a single image pass
    ``frames=[image]``.
    """
    if not frames:
        raise ValueError('color_frame_pipeline() needs at least one frame')

    orig_h, orig_w = frames[0].shape[:2]
    reference_frame = frames[-1]
    frames = [cv2.resize(frame, (PROC_W, PROC_H)) for frame in frames]
    img_h, img_w = frames[0].shape[:2]

    use_smoothing = temporal_smoothing and solid_lines and len(frames) > 1
    frames_to_process = frames if use_smoothing else [frames[-1]]

    lane_lines = [get_lane_lines(color_image=frame, solid_lines=solid_lines)
                  for frame in frames_to_process]

    lane_lines = smoothen_over_time(lane_lines) if use_smoothing else lane_lines[0]

    line_img = np.zeros(shape=(img_h, img_w))
    for lane in lane_lines:
        lane.draw(line_img)

    img_masked, _ = region_of_interest(line_img, get_roi_vertices(img_w, img_h))

    if (orig_w, orig_h) != (img_w, img_h):
        img_masked = cv2.resize(img_masked, (orig_w, orig_h))

    return weighted_img(img_masked, reference_frame, alpha=0.8, beta=1.0, lam=0.0)
