# 6. Lane Departure Warning
import cv2

class LaneDepartureWarning:
    def __init__(self, deviation_threshold):
        self.deviation_threshold = deviation_threshold

    def check_lane_departure(self, frame, lane_center, detections):
        if lane_center is None:
            return frame, False

        vehicle_center = frame.shape[1] // 2
        deviation = lane_center - vehicle_center

        if abs(deviation) > self.deviation_threshold:
            cv2.putText(frame, "WARNING: LANE DEPARTURE",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3)
            return frame, True

        return frame, False