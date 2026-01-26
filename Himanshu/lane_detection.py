import cv2
import numpy as np

offset_history = []
HISTORY_SIZE = 5


def get_road_roi(frame):
    height, width = frame.shape[:2]
    vertices = np.array([[
        [width * 0.20, height * 0.92],  # FIXED: Narrower bottom
        [width * 0.42, height * 0.68],  # FIXED: Tighter left
        [width * 0.58, height * 0.68],  # FIXED: Tighter right
        [width * 0.80, height * 0.92]  # FIXED: Narrower bottom
    ]], dtype=np.int32)
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def preprocess(frame_roi):
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def detect_lane_lines(edges):
    lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi / 180, threshold=60, minLineLength=60, maxLineGap=40)
    if lines is None:
        return np.array([0, 0, 0, 0], dtype=np.int32), np.array([0, 0, 0, 0], dtype=np.int32)

    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
        if slope < 0 and x1 < 0.42 * edges.shape[1]:  # FIXED: Tighter bounds
            left_lines.append(line[0])
        elif slope > 0 and x2 > 0.58 * edges.shape[1]:  # FIXED: Tighter bounds
            right_lines.append(line[0])

    def average_lines(lines):
        if not lines: return np.array([0, 0, 0, 0], dtype=np.int32)
        return np.array([np.mean([l[0] for l in lines]),
                         np.mean([l[1] for l in lines]),
                         np.mean([l[2] for l in lines]),
                         np.mean([l[3] for l in lines])], dtype=np.int32)

    return average_lines(left_lines), average_lines(right_lines)


def draw_lanes(frame, left_line, right_line):
    height, width = frame.shape[:2]

    def extend_line(line):
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
        y_bottom = height * 0.92
        x_bottom = int(x1 + (y_bottom - y1) / slope * (x2 - x1)) if slope != 0 else x1
        return (int(x_bottom), int(y_bottom))

    # FIXED: Much narrower lane (0.30-0.70 instead of 0.25-0.75)
    left_bottom = (int(width * 0.30), height) if np.all(left_line == 0) else extend_line(left_line)
    right_bottom = (int(width * 0.70), height) if np.all(right_line == 0) else extend_line(right_line)

    lane_center_x = (left_bottom[0] + right_bottom[0]) // 2
    top_height = int(height * 0.62)  # FIXED: Slightly higher top

    # FIXED: Even narrower top (0.4 ratio)
    left_top = (int(left_bottom[0] + (lane_center_x - left_bottom[0]) * 0.4), top_height)
    right_top = (int(right_bottom[0] - (right_bottom[0] - lane_center_x) * 0.4), top_height)

    pts = np.array([left_bottom, left_top, right_top, right_bottom], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    result = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    cv2.line(result, (lane_center_x, height), (lane_center_x, int(height // 2)), (255, 255, 0), 4)

    return result, lane_center_x, pts


def detect_lanes(frame):
    global offset_history

    roi = get_road_roi(frame)
    edges = preprocess(roi)
    left_line_arr, right_line_arr = detect_lane_lines(edges)
    overlay, lane_center_x, lane_poly = draw_lanes(frame, left_line_arr, right_line_arr)

    frame_center = frame.shape[1] // 2

    # FIXED: CORRECT offset direction
    # Positive = car is RIGHT of lane center (STEER LEFT "<")
    # Negative = car is LEFT of lane center (STEER RIGHT ">")
    raw_offset = lane_center_x - frame_center  # FLIPPED: Now correct!

    offset_history.append(raw_offset)
    if len(offset_history) > HISTORY_SIZE:
        offset_history.pop(0)
    smoothed_offset = np.mean(offset_history)

    def format_line(line):
        if np.all(line == 0): return None
        return [(int(line[0]), int(line[1])), (int(line[2]), int(line[3]))]

    return overlay, format_line(left_line_arr), format_line(right_line_arr), smoothed_offset, lane_poly
