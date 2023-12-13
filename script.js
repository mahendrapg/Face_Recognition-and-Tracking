document.addEventListener("DOMContentLoaded", function(event) {
    // Get the video element
    const video = document.getElementById("video");
    const outputVideo = document.getElementById("output-video");
    const videoFileInput = document.getElementById("video-file");
    const startButton = document.getElementById("start-btn");
    const stopButton = document.getElementById("stop-btn");

    let videoCapture;
    let outputVideoWriter;
    let isProcessing = false;

    // Start the video processing
    function startProcessing() {
        // Clear the previous output video
        outputVideo.src = "";
        
        // Disable the start button and file input
        startButton.disabled = true;
        videoFileInput.disabled = true;

        // Initialize the video capture
        const selectedFile = videoFileInput.files[0];
        if (selectedFile) {
            videoCapture = cv2.VideoCapture(selectedFile);
        } else {
            videoCapture = cv2.VideoCapture(0);  // Webcam
        }

        // Get the video properties
        const frameWidth = parseInt(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH));
        const frameHeight = parseInt(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT));
        const fps = videoCapture.get(cv2.CAP_PROP_FPS);

       // Define the codec for the output video
        const fourcc = 'mp4v';
        const outputVideoPath = "output/after.mp4";
        outputVideoWriter = new cv2.VideoWriter(outputVideoPath, fourcc, fps, {width: frameWidth, height: frameHeight});

        // Process each frame from the video
        isProcessing = true;
        processFrames();
    }

    // Stop the video processing
    function stopProcessing() {
        if (isProcessing) {
            isProcessing = false;

            // Release the resources
            videoCapture.release();
            outputVideoWriter.release();
        }

        // Enable the start button and file input
        startButton.disabled = false;
        videoFileInput.disabled = false;
    }

    // Process frames from the video
    function processFrames() {
        if (!isProcessing) {
            return;
        }

        // Capture a frame from the video
        const frame = videoCapture.read();
        const ret = videoCapture.read();

        if (!ret) {
            stopProcessing();
            return;
        }

        // Convert the frame from BGR to RGB
        const rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

        // Detect faces in the frame
        const faceLocations = face_recognition.face_locations(rgbFrame);
        const faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations);

        // Iterate through the detected faces
        for (const [index, faceEncoding] of faceEncodings.entries()) {
            // Compare the face with known faces
            const matches = face_recognition.compare_faces(known_faces, faceEncoding);
            let name = "Unknown";

            // Find the best match
            const faceDistances = face_recognition.face_distance(known_faces, faceEncoding);
            const bestMatchIndex = np.argmin(faceDistances);
            if (matches[bestMatchIndex]) {
                name = known_names[bestMatchIndex];
            }

            // Draw a rectangle around the face and label it
            const [top, right, bottom, left] = faceLocations[index];
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2);
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2);

            // Add timestamp to the frame
            const timestamp = datetime.datetime.now().strftime("%Y-%M-%D %H:%M:%S");
            cv2.putText(frame, timestamp, (10, frameHeight - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
        }

        // Write the processed frame to the output video
        outputVideoWriter.write(frame);

        // Display the resulting frame
        cv2.imshow('Video', frame);

        // Continue processing frames
        setTimeout(processFrames, 1000 / 30);  // Process frames at 30 FPS
    }

    // Add event listeners
    startButton.addEventListener("click", startProcessing);
    stopButton.addEventListener("click", stopProcessing);
});
