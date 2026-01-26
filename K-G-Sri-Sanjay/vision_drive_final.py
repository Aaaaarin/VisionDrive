import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ================= CUDA CHECK =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ================= LOAD MODEL =================
model = YOLO("yolov8l.pt").to(device)

# ================= LOAD VIDEO =================
cap = cv2.VideoCapture("Test/01244.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ================= VIDEO WRITER (SAVE OUTPUT) =================
out = cv2.VideoWriter(
    "output_adas1.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ================= ROI DEFINITION =================
roi = np.array([[
    (int(0.15 * w), h),
    (int(0.45 * w), int(0.6 * h)),
    (int(0.55 * w), int(0.6 * h)),
    (int(0.85 * w), h)
]], dtype=np.int32)

# ================= TRACKING STORAGE =================
counted_centers = []
vehicle_count = 0

# ================= WINDOW =================
cv2.namedWindow("ADAS Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ADAS Output", 1280, 720)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()

    # ================= LANE DETECTION =================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        50,
        minLineLength=60,
        maxLineGap=150
    )

    left_x, right_x = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < -0.5:
                left_x.extend([x1, x2])
            elif slope > 0.5:
                right_x.extend([x1, x2])

    lane_center = w // 2
    if left_x and right_x:
        left_lane = int(np.mean(left_x))
        right_lane = int(np.mean(right_x))
        lane_center = (left_lane + right_lane) // 2

        cv2.line(output, (left_lane, h), (left_lane, int(0.6 * h)), (0, 255, 0), 3)
        cv2.line(output, (right_lane, h), (right_lane, int(0.6 * h)), (0, 255, 0), 3)

    cv2.line(output, (lane_center, h), (lane_center, int(0.6 * h)), (255, 0, 0), 2)

    # ================= YOLO DETECTION (GPU) =================
    results = model(
        frame,
        device=device,
        conf=0.4,
        classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
        verbose=False
    )

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Ignore dashboard / hood region
            if y2 > int(0.9 * h):
                continue

            # ROI CHECK
            if cv2.pointPolygonTest(roi[0], (cx, cy), False) >= 0:

                # ---- UNIQUE VEHICLE COUNTING ----
                already_counted = False
                for (px, py) in counted_centers:
                    if abs(cx - px) < 80 and abs(cy - py) < 80:
                        already_counted = True
                        break

                if not already_counted:
                    vehicle_count += 1
                    counted_centers.append((cx, cy))

                # DRAW BOX + CONFIDENCE
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    output,
                    f"{conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                # COLLISION RISK
                if y2 > int(0.75 * h):
                    cv2.putText(
                        output,
                        "COLLISION RISK",
                        (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

    # ================= LANE DEPARTURE =================
    if abs(lane_center - w // 2) > 60:
        cv2.putText(
            output,
            "LANE DEPARTURE",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

    # ================= VISUALIZATION =================
    cv2.polylines(output, roi, True, (255, 255, 0), 2)
    cv2.putText(
        output,
        f"Vehicle Count: {vehicle_count}",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # ================= DISPLAY & SAVE =================
    display = cv2.resize(output, (1280, 720))
    cv2.imshow("ADAS Output", display)
    out.write(output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
cap.release()
out.release()
cv2.destroyAllWindows()

print("[INFO] Annotated video saved as output_adas.mp4")
