import cv2
import datetime
import os
import numpy as np
import tensorflow as tf

def load_object_detection_model():
    # Load the YOLOv4-tiny object detection model
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[layer[0] - 1] for layer in unconnected_out_layers]

    return net, classes, output_layers

def detect_object(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = classes[class_ids[i]]
        detected_objects.append((x, y, w, h, label))

    return detected_objects

def load_image_recognition_model():
    # Load the pre-trained MobileNetV2 image recognition model
    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    return model

def recognize_object(frame, x, y, w, h, model):
    roi = frame[int(y):int(y+h), int(x):int(x+w)]
    roi = cv2.resize(roi, (224, 224))
    roi = tf.keras.applications.mobilenet_v2.preprocess_input(roi)
    roi = np.expand_dims(roi, axis=0)
    predictions = model.predict(roi)
    label = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)[0][0][1]
    return label

def track_object_in_video(video_path, image_path, output_folder):
    net, classes, output_layers = load_object_detection_model()
    model = load_image_recognition_model()

    # Load the object image
    object_image = cv2.imread(image_path)
    object_height, object_width, _ = object_image.shape

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the text output file
    output_text_file = os.path.join(output_folder, 'object_tracking_output.txt')
    text_file = open(output_text_file, 'w')

    # Create the output video writer
    output_video_path = os.path.join(output_folder, 'output_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detect_object(frame, net, output_layers)

        for obj in detected_objects:
            x, y, w, h, _ = obj
            label = recognize_object(frame, x, y, w, h, model)

            # Draw a rectangle around the object and label it
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Add timestamp and frame number to the frame
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {frame_number}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the frame number and timestamp to the output text file
            text_file.write(f"Frame {frame_number}, Timestamp: {timestamp}, Object: {label}\n")

        # Increment the frame number
        frame_number += 1

        # Save the frame with the detected and recognized objects to the output video
        out_video.write(frame)

    # Release the resources
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # Close the text output file
    text_file.close()

# Example usage
video_path = 'dataset/video/msd-1.mp4'
image_path = 'dataset/images/trophy.jpg'
output_folder = 'output'
track_object_in_video(video_path, image_path, output_folder)
