import os
import cv2
import face_recognition
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Function to process a single video footage and extract recognized faces
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Define variables to store the recognized faces
    recognized_faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Iterate through the detected faces
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Add the recognized face to the list
            recognized_faces.append(frame)

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the frame with the detected faces
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()

    # Return the recognized faces
    return recognized_faces

# Define the list of video footages
video_files = ['final.mp4', 'gg.mp4', 'onew.mp4']

# Define a list to store the recognized faces from each video
all_recognized_faces = []

# Process each video footage and extract the recognized faces
for video_file in video_files:
    video_path = "output/" + video_file
    recognized_faces = process_video(video_path)
    all_recognized_faces.extend(recognized_faces)

# Define a list to store the video footages containing the same face
same_face_videos = []

# Load the known face to compare with
known_face_image = face_recognition.load_image_file("dataset/Dhoni/msd.jpg")
known_face_encoding = face_recognition.face_encodings(known_face_image)[0]

# Compare the recognized faces with the known face
for frame in all_recognized_faces:
    frame_face_encodings = face_recognition.face_encodings(frame)
    if len(frame_face_encodings) > 0:
        face_distances = face_recognition.face_distance(frame_face_encodings, known_face_encoding)
        min_distance = min(face_distances)
        if min_distance < 0.6:
            same_face_videos.append(frame)

# Create a directory to store the centralized video frames
output_directory = "centralized_footage/cf.mp4"
os.makedirs(output_directory, exist_ok=True)

# Save the frames with the same face as individual frames in the output directory
for i, frame in enumerate(same_face_videos):
    output_frame_path = os.path.join(output_directory, f"frame_{i}.jpg")
    cv2.imwrite(output_frame_path, frame)

# Create a list of all the saved frame paths
frame_paths = [os.path.join(output_directory, f"frame_{i}.jpg") for i in range(len(same_face_videos))]

# Create a video clip from the saved frames
video_clip = concatenate_videoclips
