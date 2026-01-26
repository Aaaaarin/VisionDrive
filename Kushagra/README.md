This repository contains my final project submission developed as part of the Vision-Based ADAS (Advanced Driver Assistance Systems) learning track.

**Name:** Kushagra Chandra

**Roll Number:** 240585

*Abstract*

This project presents the design and implementation of a real-time, vision-based Advanced
Driver Assistance System (ADAS) using monocular front-facing driving video. The system
integrates classical computer vision techniques with deep learning-based object detection and
tracking to perform lane detection, vehicle detection, region-of-interest (ROI) filtering, vehicle
counting, collision risk estimation, and lane departure warning in a unified real-time pipeline.

*System Overview*

The ADAS pipeline operates on continuous video input and consists of the following stages:
• Image preprocessing and edge extraction
• Lane boundary detection and lane area estimation
• Vehicle detection using a pretrained YOLOv8 model
• Multi-object tracking using ByteTrack
• ROI-based filtering and vehicle counting
• Safety logic for collision risk and lane departure warnings
The system is designed with a safety-first philosophy, prioritizing reliable warnings over
aggressive detection

*Repository Architecture*

├── src/   # Source code

│   ├── Lane/                # Lane detection and lane geometry estimation

│   ├── Detection/           # YOLOv8 vehicle detection + tracking

│   ├── Safety/              # Collision risk & lane departure logic

│   ├── Traffic/             # ROI masking & vehicle counting

│   └── main.py              # End-to-end ADAS pipeline orchestrator

│

├── Weights/                 # Pretrained YOLO model weights

├── Videos/                  # Input / demo driving videos

│

├── VisionDrive_Final_Project_Kushagra_Chandra_240585.ipynb  # Experiment notebook

├── Project Documentation.pdf                                # Project report

├── README.md               # Project overview and usage instructions

├── requirements.txt        # Python dependencies

