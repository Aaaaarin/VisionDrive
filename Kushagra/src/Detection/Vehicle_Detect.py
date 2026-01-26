# 2. Vehicle Detection Using YOLO
import cv2

class vehicle_detect:
    def __init__(self, model):
        self.model = model

    def detect_vehicles(self, frame, detections):
        for cx, cy, w, h, _ in detections:
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        return frame