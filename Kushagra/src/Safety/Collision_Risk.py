# 5. Forward Collision Risk Estimation
import numpy as np
import cv2

class CollisionRiskEstimator:
    def __init__(self, frame_height, distance_threshold, size_threshold):
        self.frame_height = frame_height

        # class-based normalization (YOLOv8)
        self.class_weight = {
            2: 1.2,   # car (boost)
            3: 1.1,   # motorcycle
            5: 0.8,   # bus
            7: 0.8    # truck
        }

    def estimate_collision_risk(self, frame, detections):
        collision_warning = False
        best_risk = 0
        threat = None
        frame_area = frame.shape[0] * frame.shape[1]

        for cx, cy, w, h, cls_id in detections:
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            dist_bottom = self.frame_height - y2
            area_ratio = (w * h) / frame_area

            # distance-dominant + class normalized risk
            weight = self.class_weight.get(cls_id, 1.0)
            risk = weight * (0.7 / (dist_bottom + 1) + 0.3 * area_ratio)

            if dist_bottom < 0.4 * self.frame_height and risk > 0.015:
                if risk > best_risk:
                   best_risk = risk
                   threat = (x1, y1, x2, y2)


        if threat:
            collision_warning = True
            x1, y1, x2, y2 = threat
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame,
                "WARNING: COLLISION RISK",
                (10, 90),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 0, 255),3)

        return frame, collision_warning