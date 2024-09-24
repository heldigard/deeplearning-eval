import urllib.request
import os

import urllib.request
import os
import cv2
import numpy as np

# URLs for the required files
urls = {
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
}

# Function to download a file from a URL
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} downloaded.")
    else:
        print(f"{filename} already exists.")

# Download each file
for filename, url in urls.items():
    download_file(url, filename)

# Check if coco.names file exists
coco_names_path = "coco.names"
if not os.path.exists(coco_names_path):
    raise FileNotFoundError(f"File {coco_names_path} not found. Please ensure the file is in the correct directory.")

# Load class names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame not captured, exit loop
    if not ret:
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Preprocess frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process each output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame
    cv2.imshow("Image", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Your existing code
import cv2
import numpy as np

# Check if coco.names file exists
coco_names_path = "coco.names"
if not os.path.exists(coco_names_path):
    raise FileNotFoundError(f"File {coco_names_path} not found. Please ensure the file is in the correct directory.")

# Load class names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame not captured, exit loop
    if not ret:
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Preprocess frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Show the frame
    cv2.imshow("Image", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
