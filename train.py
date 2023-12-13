import cv2
import numpy as np
import face_recognition
import os
import datetime 
from  PIL import Image


# Load the known faces and their names
known_faces = []
known_names = []
known_faces_dir = "dataset/images"
"""image_files = [
    "dataset/Jadeja",
    "dataset/Ambati Rayudu",
    "dataset/Dhoni",
]"""

folder_name = os.path.basename(known_faces_dir)
for file_name in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, file_name)
    face_image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_faces.append(face_encoding)
    known_names.append(folder_name)


# Load the video
video_path = "dataset/video/biriyani.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = "output/mul.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame in the video
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
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Draw a rectangle around the face and label it
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        #add timestamp to the frame
        timestamp = datetime.datetime.now().strftime("%Y-%M-%D %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Write the processed frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()

#print the path to the ouput video
print(f"Output video saved to: {output_video_path}")
