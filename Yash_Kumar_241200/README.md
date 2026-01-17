# VisionDrive: Full ADAS System using Computer Vision

## Project Overview

**VisionDrive** is a real-time **Advanced Driver Assistance System (ADAS)** built using **OpenCV** and **YOLOv8**.  
The goal of this project is to simulate core ADAS functionalities on dashcam footage, including:

- Lane detection  
- Vehicle detection and tracking  
- Vehicle counting inside lane boundaries  
- Forward collision / crash warning  
- Visual safety feedback using overlays and alerts  
---

## Main Pipeline
The pipeline processes each video frame through three main stages:

1. Lane Detection  
2. Safety Zone & Collision Warning  
3. Vehicle Detection, Tracking & Counting  


---

## Lane Detection

### Approach
- Converted frames to grayscale  
- Applied Gaussian blur to reduce noise  
- Used **Canny Edge Detection**  
- Restricted detection to a **triangular Region of Interest (ROI)** representing the road  
- Applied **Hough Line Transform** to detect lane lines   

### Output
- Yellow lane lines overlaid on the road  
- Lane polygon used later for vehicle counting  

---

## Safety Zone & Crash Warning System

### Approach
- Defined a **forward trapezoidal safety region** ahead of the vehicle  
- Applied **Background Subtraction (MOG2)** to detect motion inside this region  
- Detected contours with significant area to identify potential obstacles  
- Triggered a **crash warning** when motion exceeds a threshold  

### Output
- Semi-transparent **red safety zone**  
- Bounding boxes around detected obstacles  
- **CRASH ALERT!!!** text  
- Red border and full-frame red tint when warning is active  

---

## Vehicle Detection, Tracking & Counting

### Approach
- Used **YOLOv8** for detecting vehicles (cars, buses, trucks, motorcycles)  
- Enabled **tracking with unique IDs**  
- Calculated object centroids  
- Counted vehicles **only when their center lies inside the lane polygon**  
- Prevented double counting using a `counted_ids` set  


### Output
- Bounding boxes with:
  - Track ID  
  - Confidence score  
- Color-coded boxes:
  - Green → inside lane  
  - Red → outside lane  
- Live vehicle count display  



---


