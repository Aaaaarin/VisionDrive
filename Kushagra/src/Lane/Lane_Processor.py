# 1. Lane Detection and Lane Area Estimation
import cv2
import numpy as np

class LaneProcessor:
    def __init__(self, roi_vertices):
        self.left_lane = None
        self.right_lane = None
        self.lane_center = None
        self.roi_vertices = roi_vertices

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        return frame, masked_edges

    def detect_lanes(self, frame, masked_edges):
        lines = cv2.HoughLinesP(
            masked_edges, 1, np.pi / 180,
            threshold=80, minLineLength=60, maxLineGap=150
        )

        if lines is None:
            return None

        left, right = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if slope < -0.5:
                left.append((slope, intercept))
            elif slope > 0.5:
                right.append((slope, intercept))

        if not left or not right:
            return None

        return np.vstack((np.mean(left, axis=0), np.mean(right, axis=0)))

    def draw_lanes(self, frame, lanes):
        left_slope, left_intercept = lanes[0]
        right_slope, right_intercept = lanes[1]

        h, w = frame.shape[0], frame.shape[1]
        y1, y2 = h, int(0.6 * h)

        lx1 = int((y1 - left_intercept) / left_slope)
        lx2 = int((y2 - left_intercept) / left_slope)
        rx1 = int((y1 - right_intercept) / right_slope)
        rx2 = int((y2 - right_intercept) / right_slope)

        lx1 = int(np.clip(lx1, 0, w - 1))
        lx2 = int(np.clip(lx2, 0, w - 1))
        rx1 = int(np.clip(rx1, 0, w - 1))
        rx2 = int(np.clip(rx2, 0, w - 1))


        overlay = frame.copy()
        pts = np.array([[lx1, y1], [lx2, y2], [rx2, y2], [rx1, y1]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.line(frame, (lx1, y1), (lx2, y2), (0, 255, 255), 4)
        cv2.line(frame, (rx1, y1), (rx2, y2), (0, 255, 255), 4)

        lane_center = (lx1 + rx1) // 2
        return frame, lane_center