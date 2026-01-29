# Vision-Based Advanced Driver Assistance System (ADAS)

## Overview
This project implements a **vision-based Advanced Driver Assistance System (ADAS)** using a single front-facing dashcam video. The system processes continuous video input to detect road lanes, identify vehicles, estimate drivable regions, and generate real-time safety warnings such as **Forward Collision Warning (FCW)** and **Lane Departure Warning (LDW)**.

The project follows a **hybrid approach**, combining **classical computer vision techniques** for lane detection with **deep learning–based object detection** using a pretrained YOLO model.

---

## Features
- Lane detection using Canny edge detection and Hough transform  
- Drivable lane area estimation and lane center computation  
- Vehicle detection using pretrained YOLOv5  
- Trapezium-shaped road-facing Region of Interest (ROI)  
- Vehicle counting within ROI (each vehicle counted once)  
- Lane-aware Forward Collision Warning  
- Lane Departure Warning based on lane center deviation  
- Unified visualization with overlays, warnings, FPS, and counters  
- Output saved as an annotated video file  

---

## System Pipeline
1. **Video Input**  
   Continuous frames are read from a dashcam video and resized for consistent processing.

2. **Lane Detection**  
   Classical image processing is applied to detect left and right lane boundaries, which are then averaged and visualized.

3. **Region of Interest (ROI)**  
   A trapezium-shaped ROI is defined to focus processing on the road ahead and reduce false detections.

4. **Vehicle Detection**  
   Vehicles are detected using YOLOv5, filtering only relevant classes (car, truck, bus, motorcycle).

5. **Vehicle Tracking and Counting**  
   Lightweight centroid-based tracking assigns IDs to vehicles and ensures each vehicle is counted only once when entering the ROI.

6. **Forward Collision Warning**  
   Collision risk is estimated using geometric cues such as bounding box size, vertical position, ROI entry, and lane alignment.

7. **Lane Departure Warning**  
   The vehicle’s lateral deviation from the lane center is monitored and warnings are triggered when thresholds are exceeded.

8. **Visualization and Output**  
   All information is rendered on a single video stream and saved as an MP4 file.

---

## Technologies Used
- Python  
- OpenCV  
- PyTorch  
- YOLOv5 (Ultralytics)  
- NumPy  

---

## Requirements
```bash
pip install opencv-python torch torchvision numpy
```

## Performance Optimizations
-Frame resizing for reduced computation

-YOLO inference on alternate frames

-Cached lane detection across frames

-ROI-based filtering

-Limited object classes

### These optimizations help maintain near real-time performance without sacrificing system stability.
---

## Limitations
- No depth estimation or speed measurement
- Simple centroid-based tracking (may fail in heavy traffic)
- Performance depends on lighting and road conditions
- Monocular vision only (no sensor fusion)

## Future Improvements
1. Time-to-Collision (TTC) estimation
2. Advanced tracking (SORT / DeepSORT)
3. Semantic lane segmentation using deep learning
4. Sensor fusion with radar or LiDAR
5. Multi-lane and curved road handling
---

## Learning Outcomes

- This project demonstrates how a practical ADAS system can be built by integrating deep learning with classical vision, emphasizing system design, geometric reasoning, and real-time optimization over purely model-centric solutions.
---
