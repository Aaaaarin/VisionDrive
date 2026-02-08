# Vision-Based ADAS Project - Utkarsh_241118

## Project Overview
This project implements a Vision-Based Advanced Driver Assistance System (ADAS) capable of processing front-facing driving videos to provide real-time safety features. The system is built using Python, OpenCV, and YOLOv8.

## Features
- **Lane Detection**: Detects and visualizes road lane boundaries using Canny Edge Detection and Hough Transform.
- **Vehicle Detection**: Uses a pretrained YOLOv8n model to detect vehicles (cars, trucks, buses, motorbikes).
- **Vehicle Tracking**: Implements the SORT (Simple Online and Realtime Tracking) algorithm to track vehicles across frames.
- **ROI Masking**: Focuses processing on a relevant Region of Interest (ROI) to improve accuracy and performance.
- **Collision Risk Estimation**: Estimates forward collision risk based on vehicle position and size within the ROI.
- **Lane Departure Warning**: Monitors the ego-vehicle's position relative to the lane center and issues warnings if drifting occurs.
- **Vehicle Counting**: Counts unique vehicles entering the monitored zone.

## Directory Structure
```
Utkarsh_241118/
├── src/
│   ├── main.py         # Core ADAS pipeline logic
│   └── tracker.py      # Tracker implementation (SORT + Kalman Filter)
├── input_video/
│   └── video_links.txt # Link to the input video used for testing
├── output_video/       # Directory for processed video output
├── requirements.txt    # Python dependencies
├── run_on_colab.ipynb  # Jupyter Notebook for running on Google Colab
└── README.md           # Project documentation
```

## Setup & Installation

1.  **Clone/Download the repository** and navigate to `Utkarsh_241118`.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare Input Video**:
    -   Download the test video from the link provided in `input_video/video_links.txt`.
    -   Place the video file in `input_video/` and name it `video.mp4`.

## How to Run

### Method 1: Local Execution
Run the main script from the `src` directory:
```bash
cd src
python main.py
```
The output video will be saved to `../output_video/output.mp4`.

### Method 2: Google Colab (Recommended)
If you do not have a GPU, use the provided notebook:
1.  Upload the entire `Utkarsh_241118` folder (or zip) to your Google Drive or Colab environment.
2.  Open `run_on_colab.ipynb`.
3.  Follow the instructions in the notebook to execute the pipeline.

## Dependencies
-   opencv-python
-   numpy
-   ultralytics
-   filterpy
-   lap (optional but recommended for tracker)
-   scipy

## Acknowledgements
-   YOLOv8 by Ultralytics for object detection.
-   SORT algorithm for object tracking.
