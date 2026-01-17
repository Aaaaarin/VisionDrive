import cv2
import numpy as np
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_VIDEO = os.path.join(BASE_DIR, "..", "input_video", "eg_1.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "..", "output_video", "output.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8n.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.25
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("Video not found")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

counted_ids = set()
vehicle_count = 0

cv2.namedWindow("FULL ADAS SYSTEM", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # LANE DETECTION
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lane_mask = np.zeros_like(edges)
    lane_triangle = np.array([
        [(260, height - 410),
         (910, 325),
         (width - 560, height - 410)]
    ])
    cv2.fillPoly(lane_mask, lane_triangle, 255)
    masked_edges = cv2.bitwise_and(edges, lane_mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=200
    )

    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 10)

    # SAFETY WARNING
    safety_roi_points = np.array([
        [(260, height - 400),
         (410, height - 470),
         (width - 660, height - 470),
         (width - 510, height - 400)]
    ])
    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        safety_roi_points,
        (0, 0, 255)
    )

    alpha = 0.25
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    safety_mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(safety_mask_img, safety_roi_points, 255)

    motion_mask = object_detector.apply(frame)
    _, motion_mask = cv2.threshold(motion_mask, 254, 255, cv2.THRESH_BINARY)
    detection_zone = cv2.bitwise_and(motion_mask, motion_mask, mask=safety_mask_img)

    contours, _ = cv2.findContours(detection_zone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    warning_active = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            warning_active = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # VEHICLE DETECTION AND COUNTING
    results = model.track(
        frame,
        conf=CONFIDENCE,
        classes=VEHICLE_CLASSES,
        persist=True,
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes

        for box, track_id, conf in zip(boxes.xyxy, boxes.id, boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            conf = float(conf)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            inside_lane = cv2.pointPolygonTest(lane_triangle, (cx, cy), False)

            if inside_lane >= 0 and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            color = (0, 255, 0) if inside_lane >= 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            label = f"ID {track_id} | {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    # MERGE + DISPLAY
    final_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.polylines(final_frame, lane_triangle, True, (255, 0, 143), 2)
    cv2.polylines(final_frame, safety_roi_points, True, (255, 0, 143), 2)

    if warning_active:
        cv2.putText(final_frame, "CRASH ALERT!!!",
                    (width // 2, height // 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
        cv2.rectangle(final_frame, (0, 0), (width, height), (0, 0, 255), 10)

    cv2.putText(final_frame, f"Vehicles in Lane: {vehicle_count}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 3,
                (255, 255, 0), 3)
    if warning_active:
        red_overlay = final_frame.copy()
        red_overlay[:] = (0, 0, 255)
        final_frame = cv2.addWeighted(
            red_overlay, 0.2,
            final_frame, 0.8,
            0
        )

    out.write(final_frame)
    cv2.imshow("FULL ADAS SYSTEM", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ DONE. Output saved to:", OUTPUT_VIDEO)
