import cv2
import numpy as np
from ultralytics import YOLO
import lane_detection
from my_tracker import Tracker

VIDEO_PATH = r"C:\VisionDrive Project\Test\Test\6.mp4"
MODEL_PATH = "yolov8n.pt"
OUTPUT_FILENAME = "VisionDrive_Demo.mp4"
RESIZE_W, RESIZE_H = 1280, 720
AREA_THRES_HIGH = 25000
AREA_THRES_MED = 14000
LANE_DEVIATION_LIMIT = 60


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    model = YOLO(MODEL_PATH)
    tracker = Tracker(max_disappeared=8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (RESIZE_W, RESIZE_H))
    print(f"Recording: {OUTPUT_FILENAME}");
    print("PRODUCTION MODE: Strict thresholds + narrow lane")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        annotated = frame.copy()

        _, left_line, right_line, offset, lane_poly = lane_detection.detect_lanes(frame)
        if left_line: cv2.line(annotated, tuple(left_line[0]), tuple(left_line[1]), (255, 0, 0), 2)
        if right_line: cv2.line(annotated, tuple(right_line[0]), tuple(right_line[1]), (255, 0, 0), 2)

        results = model(frame, stream=True, verbose=False, device='0')
        raw_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                if conf > 0.7 and cls in [2, 3, 5, 7]: raw_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        clean_detections = []
        for box in raw_boxes:
            overlap = False
            for existing in clean_detections:
                dx = abs((box[0] + box[2]) / 2 - (existing[0] + existing[2]) / 2)
                dy = abs((box[1] + box[3]) / 2 - (existing[1] + existing[3]) / 2)
                if dx < 40 and dy < 40: overlap = True; break
            if not overlap: clean_detections.append(box)

        tracked_objects = tracker.update(clean_detections)

        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(mask, [lane_poly], (0, 255, 0))
        annotated = cv2.addWeighted(annotated, 1, mask, 0.2, 0)
        cv2.polylines(annotated, [lane_poly], True, (0, 255, 0), 2)

        warnings = [];
        danger_count = 0
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cx, cy = (x1 + x2) // 2, y2
            in_central_corridor = (RESIZE_W * 0.30) < cx < (RESIZE_W * 0.70)
            inside_polygon = cv2.pointPolygonTest(lane_poly, (cx, cy), False) >= 0
            is_danger = inside_polygon and in_central_corridor
            color = (0, 255, 0);
            label = ""

            if is_danger:
                area = (x2 - x1) * (y2 - y1)
                danger_count += 1
                if area > AREA_THRES_HIGH:
                    color = (0, 0, 255); label = "COLLISION!"; warnings.append("BRAKE!")
                elif area > AREA_THRES_MED:
                    color = (0, 165, 255); label = "CAUTION"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if label: cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        lane_status_color = (0, 255, 0);
        lane_msg = "LANE: OK"
        if abs(offset) > LANE_DEVIATION_LIMIT:
            lane_status_color = (0, 0, 255)
            lane_msg = "DEPARTURE >" if offset > 0 else "< DEPARTURE"
            warnings.append("STEER CENTER")

        cv2.rectangle(annotated, (0, 0), (1280, 80), (0, 0, 0), -1)
        cv2.putText(annotated, lane_msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, lane_status_color, 2)
        if warnings: cv2.putText(annotated, " + ".join(warnings), (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                 3)
        cv2.putText(annotated, f"Vehicles: {len(tracked_objects)} | Threats: {danger_count}", (800, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(annotated)
        cv2.imshow("VisionDrive ADAS - Production", annotated)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break

    cap.release();
    out.release();
    cv2.destroyAllWindows()
    print("✅ Production video saved!")


if __name__ == "__main__": main()
