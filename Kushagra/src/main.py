from ultralytics import YOLO
import math
import torch
import cv2
import numpy as np

from Detection.Vehicle_Detect import vehicle_detect
from Lane.Lane_Processor import LaneProcessor
from Safety.Collision_Risk import CollisionRiskEstimator
from Safety.Lane_Departure import LaneDepartureWarning
from Traffic.Traffic_Analyser import TrafficAnalyser

DISPLAY_THESE_FRAMES = [20, 35, 54, 80]

cap = cv2.VideoCapture('Data/Videos/01095.mp4')
model = YOLO('yolov8n.pt').to('cpu')

success, first_frame = cap.read()
if not success:
    exit()

h, w = first_frame.shape[0], first_frame.shape[1]
roi_vertices = np.array([[(int(0.1*w), h),
     (int(0.45*w), int(0.6*h)),
     (int(0.55*w), int(0.6*h)),
     (int(0.9*w), h)]],
     dtype=np.int32).reshape(-1,1,2)

out = cv2.VideoWriter(
    "ADAS_Output.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    cap.get(cv2.CAP_PROP_FPS), (w,h))

lane_detector = LaneProcessor(roi_vertices)
vehicle_detector = vehicle_detect(model)
traffic_analyser = TrafficAnalyser(roi_vertices, h)
collision_estimator = CollisionRiskEstimator(first_frame.shape[0], 0.8, 0.25)
lane_departure_warning = LaneDepartureWarning(int(0.05 * w))
frame_count = 0
lane_center = None
VEHICLE_CLASSES = {2, 3, 5, 7}  #car, motorcycle, bus, truck

while True:
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1
    detections = []

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    if results[0].boxes.id is not None:
       boxes = results[0].boxes.xywh.cpu().numpy()
       ids = results[0].boxes.id.cpu().numpy()

    cls = results[0].boxes.cls.cpu().numpy().astype(int)

    valid = [i for i,c in enumerate(cls) if c in VEHICLE_CLASSES]
    boxes = boxes[valid]
    ids = ids[valid]

    for (cx, cy, w, h), tid in zip(boxes, ids):
        detections.append((cx, cy, w, h, tid))

    frame, masked_edges = lane_detector.process_frame(frame)
    lanes = lane_detector.detect_lanes(frame, masked_edges)

    if lanes is not None:
        frame, lane_center = lane_detector.draw_lanes(frame, lanes)

    frame = vehicle_detector.detect_vehicles(frame, detections)
    frame = traffic_analyser.process_traffic(frame, detections)
    frame, collision_warning = collision_estimator.estimate_collision_risk(frame, detections)

    if lane_center is not None:
        frame, lane_departure_flag = lane_departure_warning.check_lane_departure(frame, lane_center, detections)

    out.write(frame)

    if frame_count in DISPLAY_THESE_FRAMES:
       print(f"Displaying frame {frame_count}")
       cv2.imshow(frame)


cap.release()
out.release()
cv2.destroyAllWindows()