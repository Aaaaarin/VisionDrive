import cv2
import numpy as np
import torch
import time

VIDEO_PATH = "2.mp4"
OUTPUT_PATH = "adas_output.mp4"

YOLO_MODEL = "yolov5l"
TARGET_WIDTH = 1920

DISPLAY_SCALE = 1.4
LANE_DEPARTURE_THRESH = 80
COLLISION_AREA_THRESH = 20000
CONF_THRESH = 0.4

LANE_COLOR = (255, 255, 0)
ROI_COLOR = (0, 255, 255)
SAFE_COLOR = (0, 255, 0)
DANGER_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

model = torch.hub.load("ultralytics/yolov5", YOLO_MODEL, pretrained=True)
model.conf = CONF_THRESH
model.iou = 0.45
model.max_det = 50

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cap = cv2.VideoCapture(VIDEO_PATH)
fps_input = cap.get(cv2.CAP_PROP_FPS)
if fps_input <= 0:
    fps_input = 30

writer = None

vehicle_id_counter = 0
tracked_vehicles = {}
counted_vehicles = set()
vehicle_count = 0

prev_time = 0
frame_id = 0
last_detections = None

LANE_UPDATE_INTERVAL = 3
cached_left_lane = None
cached_right_lane = None

def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array([[
        (w*0.1, h),
        (w*0.45, h*0.6),
        (w*0.55, h*0.6),
        (w*0.9, h)
    ]], np.int32)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(edges, mask)

def average_lane_lines(lines, h):
    left, right = [], []
    if lines is None:
        return None, None
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2-y1)/(x2-x1)
        intercept = y1 - slope*x1
        if slope < -0.5:
            left.append((slope, intercept))
        elif slope > 0.5:
            right.append((slope, intercept))

    def make_line(params):
        slope, intercept = np.mean(params, axis=0)
        y1 = h
        y2 = int(h*0.6)
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        return (x1,y1,x2,y2)

    return (make_line(left) if left else None,
            make_line(right) if right else None)

def get_lane_center(left, right):
    if left is None or right is None:
        return None
    return (left[0] + right[0]) // 2

def is_vehicle_in_lane(cx, cy, left_lane, right_lane):
    if left_lane is None or right_lane is None:
        return False
    lx1,ly1,lx2,ly2 = left_lane
    rx1,ry1,rx2,ry2 = right_lane
    ml = (ly2-ly1)/(lx2-lx1)
    bl = ly1 - ml*lx1
    mr = (ry2-ry1)/(rx2-rx1)
    br = ry1 - mr*rx1
    return int((cy-bl)/ml) < cx < int((cy-br)/mr)

def get_trapezium_roi(w, h):
    return np.array([
        (int(w*0.3), int(h*0.5)),
        (int(w*0.7), int(h*0.5)),
        (int(w*0.95), h),
        (int(w*0.05), h)
    ], np.int32)

def point_inside_roi(cx, cy, roi):
    return cv2.pointPolygonTest(roi, (cx,cy), False) >= 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    scale = TARGET_WIDTH / frame.shape[1]
    frame = cv2.resize(frame, (TARGET_WIDTH, int(frame.shape[0]*scale)))
    h, w = frame.shape[:2]

    now = time.time()
    fps = int(1/(now-prev_time)) if prev_time else 0
    prev_time = now

    if frame_id % LANE_UPDATE_INTERVAL == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,150)
        roi_edges = region_of_interest(edges)
        lines = cv2.HoughLinesP(roi_edges,1,np.pi/180,50,100,50)
        cached_left_lane, cached_right_lane = average_lane_lines(lines, h)

    left_lane, right_lane = cached_left_lane, cached_right_lane

    overlay = frame.copy()
    if left_lane:
        cv2.line(overlay,left_lane[:2],left_lane[2:],LANE_COLOR,6)
    if right_lane:
        cv2.line(overlay,right_lane[:2],right_lane[2:],LANE_COLOR,6)

    if left_lane and right_lane:
        pts = np.array([
            [left_lane[0],left_lane[1]],
            [left_lane[2],left_lane[3]],
            [right_lane[2],right_lane[3]],
            [right_lane[0],right_lane[1]]
        ])
        cv2.fillPoly(overlay,[pts],(0,255,0))

    frame = cv2.addWeighted(frame,0.7,overlay,0.3,0)
    lane_center = get_lane_center(left_lane,right_lane)

    roi = get_trapezium_roi(w,h)
    cv2.polylines(frame,[roi],True,ROI_COLOR,2)

    if frame_id % 2 == 0:
        results = model(frame, size=640)
        last_detections = results.xyxy[0].cpu().numpy()

    detections = last_detections if last_detections is not None else []

    closest = None
    max_y = 0
    warning_active = False

    for x1,y1,x2,y2,conf,cls in detections:
        label = model.names[int(cls)]
        if label not in VEHICLE_CLASSES or conf < CONF_THRESH:
            continue

        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        area = (x2-x1)*(y2-y1)

        in_roi = point_inside_roi(cx,cy,roi)
        in_lane = is_vehicle_in_lane(cx,cy,left_lane,right_lane)

        vehicle_id = None
        for vid,(px,py) in tracked_vehicles.items():
            if abs(cx-px)<70 and abs(cy-py)<70:
                vehicle_id = vid
                break

        if vehicle_id is None:
            vehicle_id_counter += 1
            vehicle_id = vehicle_id_counter

        tracked_vehicles[vehicle_id] = (cx,cy)

        if in_roi and vehicle_id not in counted_vehicles:
            vehicle_count += 1
            counted_vehicles.add(vehicle_id)

        color = SAFE_COLOR
        if in_roi and in_lane and (area>COLLISION_AREA_THRESH or y2>h*0.85):
            color = DANGER_COLOR
            warning_active = True
            if cy > max_y:
                max_y = cy
                closest = (int(x1),int(y1),int(x2),int(y2))

        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
        cv2.putText(frame,f"{label} {conf:.2f}",
                    (int(x1),int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    if closest:
        x1,y1,x2,y2 = closest
        cv2.rectangle(frame,(x1,y1),(x2,y2),DANGER_COLOR,4)
        cv2.putText(frame,"PRIMARY TARGET",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,DANGER_COLOR,2)

    if warning_active:
        cv2.rectangle(frame,(w//2-300,20),(w//2+300,70),DANGER_COLOR,-1)
        cv2.putText(frame,"FORWARD COLLISION WARNING",
                    (w//2-280,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1,TEXT_COLOR,3)

    if lane_center:
        offset = (w//2)-lane_center
        direction = "LEFT" if offset>0 else "RIGHT"
        cv2.putText(frame,f"Lane Offset: {abs(offset)} px {direction}",
                    (20,140),cv2.FONT_HERSHEY_SIMPLEX,0.6,TEXT_COLOR,2)
        if abs(offset)>LANE_DEPARTURE_THRESH:
            cv2.putText(frame,"LANE DEPARTURE WARNING",
                        (20,170),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,DANGER_COLOR,2)

    cv2.putText(frame,f"Vehicle Count: {vehicle_count}",
                (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,TEXT_COLOR,2)
    cv2.putText(frame,f"FPS: {fps}",
                (20,70),cv2.FONT_HERSHEY_SIMPLEX,0.6,TEXT_COLOR,2)

    if writer is None:
        writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_input,
            (w,h)
        )

    writer.write(frame)

cap.release()
writer.release()
cv2.destroyAllWindows()
