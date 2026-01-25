# Vision-Based ADAS Projects Repository

## Student Name: AZAD BHARTI AHIRWAR
## ROLL NO. : 240250
## Institute: IIT Kanpur  
## Course: B.Tech (Mechanical Engineering)

## Project Description
This project implements a vision-based ADAS (Advanced Driver Assistance System)
using a front-facing camera. The system detects road lanes using classical image
processing techniques and vehicles using a pretrained YOLOv5 object detection model.

## Features
- Lane detection using OpenCV and Hough Transform
- Vehicle detection using pretrained YOLOv5
- Real-time video processing
- Annotated output video generation

## Technologies Used
- Python
- OpenCV
- PyTorch
- YOLOv5

## How to Run
1. Install Python 3.9+
2. Install dependencies:
   python -m pip install opencv-python numpy torch torchvision ultralytics pandas tqdm seaborn matplotlib pillow
3. Run the project:
   python main.py

## Output
The processed video is saved in:
output_video/output.mp4

