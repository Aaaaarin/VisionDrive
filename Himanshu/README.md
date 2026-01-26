VisionDrive ADAS - Himanshu Khichar
Project Overview
This project implements a real-time Advanced Driver Assistance System (ADAS) focusing on Lane Detection and Forward Collision Warning (FCW). The system balances processing speed with detection accuracy to provide safety-critical alerts in real-time.

Technical Approach & Methodology
1. Lane Detection Pipeline
Preprocessing: Converted frames to grayscale to reduce dimensionality (matrices), followed by Gaussian Blur to remove noise.
Edge Detection: Implemented Canny Edge Detection to identify structural lane boundaries.
Region of Interest (ROI): Applied a polygon mask to focus only on the road ahead, ignoring sky and surroundings.
Line Extraction: Used Probabilistic Hough Transform to extract line coordinates from the edge map.
2. Object Detection & Tracking
Model: Utilized YOLOv8 for vehicle detection due to its superior inference speed (hitting ~22 FPS) compared to R-CNN variants.
Filtering:
Applied a high Confidence Threshold (0.7) to minimize false positives (shadows/reflections).
Used NMS (Non-Maximum Suppression) with IOU (Intersection Over Union) logic to eliminate duplicate bounding boxes for the same vehicle.
Tracking: Implemented object tracking to assign persistent IDs to cars, ensuring stable warnings rather than flickering alerts.
3. Threat Assessment Logic
Instead of complex depth estimation models (which are slow), I used Bounding Box Area as a proxy for distance:

Files Included
main.py: Core logic scripts.
output_video.mp4: Processed output with bounding boxes and overlays.
