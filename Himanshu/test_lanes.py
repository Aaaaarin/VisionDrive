import cv2
from lane_detection import detect_lanes

# Resource Path required
video_path = r"C:\VisionDrive Project\ADAS\videos\DataSet\drivingDataset\normalDay\nD_1.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    overlay, center_x = detect_lanes(frame)

    cv2.putText(overlay, f"Center X: {center_x}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Lane Detection", overlay)
    frame_count += 1
    print(f"Frame {frame_count}, Center: {center_x}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames")


