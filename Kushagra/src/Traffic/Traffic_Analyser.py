# 3. Masked Region Of Interest (ROI) Processing and # 4. Vehicle Counting within ROI
import numpy as np
import cv2

class TrafficAnalyser:
    def __init__(self, roi_vertices, frame_height):
        self.roi_vertices = roi_vertices
        self.counted_ids = set()
        self.total_count = 0
        self.entry_line_y = int(0.75 * frame_height)

    def process_traffic(self, frame, detections):
       for cx, cy, w, h, id in detections:
        center = (int(cx), int(cy))
        inside_roi = cv2.pointPolygonTest(
            self.roi_vertices, center, False)>= 0

        if inside_roi and id not in self.counted_ids:
            self.counted_ids.add(id)

       self.total_count = len(self.counted_ids)

       cv2.putText(
        frame,
        f'Total Count: {self.total_count}',(10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

       return frame

