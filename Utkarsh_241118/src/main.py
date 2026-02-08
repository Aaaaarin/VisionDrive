import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Sort


class LaneDetector:
    """
    Handles lane detection using classical computer vision techniques.
    """
    def __init__(self):
        self.left_fit_average = []
        self.right_fit_average = []

    def canny_edge_detection(self, image):
        """Applies Canny edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def region_of_interest(self, image):
        """Masks the image to focus on the road area."""
        height = image.shape[0]
        polygons = np.array([
            [(200, height), (1100, height), (550, 250)]
        ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def make_coordinates(self, image, line_parameters):
        """Converts slope and intercept to coordinates."""
        if line_parameters is None:
            return None
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        try:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
        except ZeroDivisionError:
            return None
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, image, lines):
        """Averages lines to find a single left and right lane."""
        left_fit = []
        right_fit = []
        if lines is None:
            return None
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        
        if left_fit:
            self.left_fit_average = np.average(left_fit, axis=0)
        if right_fit:
            self.right_fit_average = np.average(right_fit, axis=0)
            
        left_line = self.make_coordinates(image, self.left_fit_average)
        right_line = self.make_coordinates(image, self.right_fit_average)
        return np.array([left_line, right_line])

    def display_lines(self, image, lines):
        """Draws lines on the image."""
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                if line is not None:
                    x1, y1, x2, y2 = line
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image
    
    def get_lane_center(self, lines, image_width):
        """Calculates deviation from lane center."""
        if lines is None or lines[0] is None or lines[1] is None:
            return None
        
        left_x2 = lines[0][2]
        right_x2 = lines[1][2]
        lane_center_x = (left_x2 + right_x2) // 2
        return lane_center_x


class ObjectDetector:
    """
    Handles vehicle detection and tracking using YOLO and SORT.
    """
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.class_names = self.model.names
        self.vehicle_classes = ["car", "truck", "bus", "motorbike"]

    def detect_and_track(self, frame, mask_limits=None):
        results = self.model(frame, stream=True, verbose=False)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                if current_class in self.vehicle_classes and conf > 0.3:
                     # Filter by ROI if provided
                    if mask_limits is not None:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if not (mask_limits[0] < cx < mask_limits[2] and mask_limits[1] < cy < mask_limits[3]):
                            continue
                            
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        tracker_results = self.tracker.update(detections)
        return tracker_results


class ADASSystem:
    """
    Main ADAS logic controller.
    """
    def __init__(self, input_path, output_path):
        self.cap = cv2.VideoCapture(input_path)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video file at {input_path}")
            # Try absolute path just in case
            abs_path = os.path.abspath(input_path)
            print(f"Trying absolute path: {abs_path}")
            self.cap = cv2.VideoCapture(abs_path)
            if not self.cap.isOpened():
                print("Error: Still could not open video file.")
                return

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video opened successfully: {self.width}x{self.height} @ {self.fps}fps")
        
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        
        self.lane_detector = LaneDetector()
        self.object_detector = ObjectDetector()
        
        self.total_count = []
        
        # ROI for counting
        self.roi_limits = [int(0.35 * self.width), int(0.4 * self.height), int(0.6 * self.width), self.height]


    def process_video(self):
        print("Starting video processing...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 1. Lane Detection
            # Using a simplified ROI for lanes to avoid noise
            canny_image = self.lane_detector.canny_edge_detection(frame)
            cropped_image = self.lane_detector.region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
            averaged_lines = self.lane_detector.average_slope_intercept(frame, lines)
            line_image = self.lane_detector.display_lines(frame, averaged_lines)
            
            # Blend lane lines
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

            # 2. Vehicle Detection & Tracking
            # Masking for vehicle detection ROI
            mask = np.zeros_like(frame)
            roi_poly = np.array([[(self.roi_limits[0], self.roi_limits[1]), (self.roi_limits[2], self.roi_limits[1]), 
                                  (self.roi_limits[2], self.roi_limits[3]), (self.roi_limits[0], self.roi_limits[3])]])
            cv2.fillPoly(mask, roi_poly, (255, 255, 255))
            masked_frame_for_yolo = cv2.bitwise_and(frame, mask)
            
            track_results = self.object_detector.detect_and_track(masked_frame_for_yolo, self.roi_limits)

            for result in track_results:
                x1, y1, x2, y2, id = map(int, result)
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.rectangle(combo_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(combo_image, f'ID: {id}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                # Counting logic
                if self.roi_limits[0] < cx < self.roi_limits[2] and self.roi_limits[1] < cy < self.roi_limits[3]:
                    if id not in self.total_count:
                        self.total_count.append(id)
                
                # Collision Warning Logic (Simple heuristic based on size/position)
                if cy > self.roi_limits[3] - 100 and self.roi_limits[0] + 100 < cx < self.roi_limits[2] - 100:
                     cv2.putText(combo_image, "WARNING: COLLISION RISK", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 3. Lane Departure Warning
            lane_center_x = self.lane_detector.get_lane_center(averaged_lines, self.width)
            if lane_center_x:
                image_center_x = self.width // 2
                deviation = image_center_x - lane_center_x
                if abs(deviation) > 50: # Threshold
                     cv2.putText(combo_image, "LANE DEPARTURE WARNING", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                
                # Visualizing Lane Center
                cv2.circle(combo_image, (lane_center_x, int(self.height*0.7)), 5, (255,0,0), -1)

            # Display Count
            cv2.putText(combo_image, f'Vehicles: {len(self.total_count)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            self.out.write(combo_image)
            
            # Optional: Display if running locally with a screen (commented out for headless)
            # cv2.imshow("ADAS", combo_image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        
        print("Processing complete.")
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import os
    
    input_video = "../input_video/video.mp4" # Local path for testing
    output_video = "../output_video/output.mp4"

    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        print("Please place 'video.mp4' in 'Utkarsh_241118/input_video/' for testing.")
    else:
        adas = ADASSystem(input_video, output_video)
        adas.process_video()
