🚗 Vision-Based ADAS Projects Repository

This repository contains the final project submissions developed by mentees as part of the Vision-Based ADAS (Advanced Driver Assistance Systems) learning track.

Each project implements a camera-based ADAS perception pipeline operating on real driving video data, focusing on lane detection, vehicle detection, spatial reasoning, and safety logic, similar to early-stage ADAS prototypes used in industry.

📌 Project Overview

The goal of the final project was to design and implement a vision-only ADAS system that processes continuous front-facing driving video and produces a unified, annotated output demonstrating:

Road lane understanding

Vehicle awareness using object detection

Region-based spatial reasoning using masking

Rule-based safety and warning logic

Integrated real-time visualization

The emphasis of the project was system design and integration, not model training.

🧠 Technical Scope & Constraints

Object Detection:
All projects use a pretrained YOLO model for vehicle detection.
Training or fine-tuning of models was explicitly not part of the task.

Lane Detection:
Implemented using classical image processing techniques such as edge detection, region masking, and line estimation.

Dataset Usage:
Publicly available front-facing driving videos (e.g., dashcam-style datasets) were used strictly as input video data, not for training.

Evaluation:
All systems were tested on the same fixed video(s) to ensure consistent comparison across submissions.

🧩 Expected System Capabilities

Each project in this repository implements the following components:

Detection and visualization of road lane boundaries and lane area

Vehicle detection using YOLO with bounding box overlays

A masked Region of Interest (ROI) ahead of the vehicle

Counting of unique vehicles entering the ROI over time

Forward collision risk estimation using rule-based logic

Lane departure warning based on lane center deviation

A single unified output video combining all visualizations

📁 Repository Structure

Each mentee has committed their work in a separate folder, following a consistent structure:

├── mentee_name/
│   ├── src/                # Source code
│   ├── models/             # YOLO weights / configs (if applicable)
│   ├── input_video/        # Test video(s)
│   ├── output_video/       # Annotated ADAS output
│   ├── README.md           # Project-specific explanation
│   └── requirements.txt    # Dependencies


Note: Folder names and exact structure may vary slightly based on individual implementation choices.

▶️ How to Run a Project

Each mentee has included a project-level README inside their folder with:

Setup instructions

Dependency installation

Command to run the pipeline

Please refer to the respective folder for execution details.

🎯 Learning Outcomes

Through this project, mentees demonstrated their understanding of:

Real-time video processing

Classical computer vision pipelines

Practical usage of pretrained YOLO models

ROI masking and spatial filtering

Temporal logic and decision-making

System-level thinking for ADAS applications

📎 Disclaimer

These projects are educational prototypes intended for learning purposes only.
They do not represent production-ready ADAS systems and should not be used in real driving scenarios.

🙌 Acknowledgements

This repository represents the culmination of a structured learning journey in computer vision for autonomous and driver assistance systems.
Great effort by all contributors in bringing together perception, logic, and visualization into complete working systems.
