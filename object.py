import cv2

# Create a tracker using KCF (Kernelized Correlation Filters)
tracker = cv2.TrackerKCF_create()

# Function to select the ROI (Region of Interest) in the first frame
def select_roi(frame):
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    return bbox

# Open video file
video_path = 'dataset/video/plane.mp4'
video = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = video.read()

# Select ROI in the first frame
bbox = select_roi(frame)

# Initialize the tracker with the selected ROI in the first frame
tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Update the tracker for the current frame
    success, bbox = tracker.update(frame)

    if success:
        # Draw bounding box around the tracked object
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close all windows
video.release()
cv2.destroyAllWindows()
