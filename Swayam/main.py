import cv2
import numpy as np
import os
from ultralytics import YOLO

#PATHS

INPUT_VIDEO = "video/eg_1.mp4"
OUTPUT_VIDEO = "output_video/final_full_adas.mp4"

# YOLO CONFIG
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.25

model = YOLO("yolov8n.pt")

# VIDEO SETUP

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("Video not opening")
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

# SAFETY SYSTEM
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# VEHICLE COUNTING STATE
counted_ids = set()
vehicle_count = 0

cv2.namedWindow("FULL ADAS SYSTEM", cv2.WINDOW_NORMAL)

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. LANE LINE DETECTION
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Lane Masking
    lane_mask = np.zeros_like(edges)
    lane_triangle = np.array([
        [(260, height - 410),
         (910, 325),
         (width - 560, height - 410)]
    ])
    cv2.fillPoly(lane_mask, lane_triangle, 255)
    masked_edges = cv2.bitwise_and(edges, lane_mask)

    # Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=200
    )

    # Draw Lane Lines
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # 2. SAFETY MOTION DETECTION
    safety_roi_points = np.array([
        [(260, height - 400),
         (410, height - 470),
         (width - 660, height - 470),
         (width - 510, height - 400)]
    ])

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

    # 3. YOLO OBJECT TRACKING
    results = model.track(
        frame,
        conf=CONFIDENCE,
        classes=VEHICLE_CLASSES,
        persist=True,
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        track_ids = results[0].boxes.id
        confs = results[0].boxes.conf

        for box, track_id, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            score = float(conf)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if inside lane triangle
            inside_lane = cv2.pointPolygonTest(lane_triangle, (cx, cy), False)

            if inside_lane >= 0 and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            # Determine color: Green if inside lane, Red if outside
            color = (0, 255, 0) if inside_lane >= 0 else (0, 0, 255)

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw Confidence Score Label
            label = f"{int(score * 100)}%"
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 4. FINAL COMPOSITION
    final_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    if warning_active:
        text = "SAFETY WARNING"
        font_scale = 2.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 5
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(final_frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

        cv2.rectangle(final_frame, (0, 0), (width, height), (0, 0, 255), 10)

    # Display Counter
    cv2.putText(final_frame, f"Lane Vehicle Count: {vehicle_count}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 3)

    out.write(final_frame)
    cv2.imshow("FULL ADAS SYSTEM", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("DONE. Output saved to:", OUTPUT_VIDEO)