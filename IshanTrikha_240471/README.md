#  Real-Time ADAS using Computer Vision

## Overview
This project implements a **real-time Advanced Driver Assistance System (ADAS)** using a monocular dashcam video. The system combines **deep learning–based vehicle detection** with **classical computer vision techniques** to detect vehicles, identify lane boundaries, estimate collision risks, and generate visual alerts.

The goal of this project is to simulate a **basic perception module** used in autonomous driving systems while maintaining real-time performance.

---

## Features
- Real-time vehicle detection using YOLOv5  
- Lane detection using classical computer vision  
- Forward collision risk estimation  
- Vehicle counting without double counting  
- Minimal and intuitive visual alerts  
- Optimized for real-time performance  
- Google Colab compatible (no GUI dependency)

---

## Tech Stack
- **Language:** Python  
- **Deep Learning:** PyTorch  
- **Object Detection:** YOLOv5  
- **Computer Vision:** OpenCV  
- **Numerical Computation:** NumPy  
- **Environment:** Google Colab  

---

## Input and Output

### Input
- Dashcam video (`.mp4`)
- Single front-facing monocular camera

### Output
- Annotated video containing:
  - Lane boundaries
  - Vehicle bounding boxes with labels
  - Total vehicle count
  - Collision warning alerts

---

## System Pipeline

1. Read dashcam video frame-by-frame  
2. Resize frames for faster processing  
3. Detect lane boundaries using classical CV  
4. Detect vehicles using YOLOv5  
5. Track vehicles using centroid-based tracking  
6. Count vehicles when entering forward ROI  
7. Evaluate collision risk using geometric rules  
8. Overlay warnings and statistics  
9. Save processed video to disk  

---

## Object Detection

- Uses **YOLOv5**, a single-stage object detector
- Pretrained on the COCO dataset
- Filters detections based on:
  - Confidence threshold
  - Relevant vehicle classes (car, truck, bus, motorcycle)

This ensures efficient and task-specific detection.

---

## Lane Detection

Lane detection is implemented using classical computer vision techniques:

- Grayscale conversion  
- Gaussian blur  
- Canny edge detection  
- Trapezoidal region masking  
- Hough Line Transform  
- Lane line averaging for stability  

Lane detection is performed periodically instead of every frame to improve efficiency.

---

## Region of Interest (ROI)

A trapezoidal **forward-facing Region of Interest** is defined to represent the road area directly ahead of the vehicle.

**Purpose:**
- Focus collision detection on relevant regions
- Reduce false positives from adjacent lanes
- Improve performance

---

## Vehicle Tracking and Counting

### Tracking
- Lightweight centroid-based tracking
- Vehicles are assigned IDs based on spatial proximity across frames

### Counting
- Vehicles are counted **only once**
- Counting occurs when a vehicle enters the forward ROI
- Prevents double counting across frames

---

## Collision Risk Assessment

Collision risk is estimated using geometric reasoning:

A vehicle is considered dangerous if:
- It lies inside the forward ROI  
- It is within the detected lane boundaries  
- Its bounding box area exceeds a threshold (proxy for distance)  
- It is close to the bottom of the frame  

The most threatening vehicle is highlighted, and a collision warning is displayed.

---

## Visualization

- Green bounding boxes → safe vehicles  
- Red bounding boxes → potential threats  
- Bold highlight for primary threat  
- Clear “Collision Imminent” warning banner  
- Text-only display of total vehicle count  

Visual clutter is intentionally minimized to improve readability.

---

## Performance Optimizations

- Frame resizing before inference  
- Lane detection every N frames  
- GPU acceleration when available  
- Half-precision (FP16) inference  
- No real-time GUI rendering  

These optimizations allow near real-time processing.

---

## Limitations

- Performance degrades in poor lighting or weather  
- Lane detection fails with faded or missing lane markings  
- Monocular vision limits accurate distance estimation  
- Simple tracking may fail in dense traffic  
- Sensitive to camera placement and calibration  

---

## Future Improvements

- Sensor fusion (camera + radar / LiDAR)  
- Advanced tracking (DeepSORT, Kalman Filters)  
- Distance and speed estimation  
- Learning-based lane detection  
- Deployment on embedded automotive hardware  
- Adaptive thresholds using temporal models  

---

## Conclusion

This project demonstrates the design and optimization of a **real-time ADAS perception pipeline** by integrating deep learning, classical computer vision, and geometric reasoning. It highlights system-level thinking, performance trade-offs, and practical challenges faced in real-world autonomous driving applications.

---

## Key Learnings

- End-to-end computer vision pipeline development  
- Real-time deep learning inference  
- Classical CV integration with neural networks  
- Geometric reasoning for decision-making  
- Optimization under real-world constraints  

---



