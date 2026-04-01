import cv2
import numpy as np
from ultralytics import YOLO

# ====================== CONFIGURATION ======================
# Path to your trained model (change if needed)
MODEL_PATH = r"your model path"

# Confidence threshold - higher value = stricter detection
CONF_THRESHOLD = 0.5

# Webcam index (0 = default webcam, try 1 or 2 if it doesn't work)
WEBCAM_INDEX = 0
# ===========================================================

# Load the YOLOv11 model
print("Loading YOLOv11 model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")
print(f"Available classes: {model.names}\n")

# Open the webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)

# Set camera resolution (optional - improves performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Live Face Mask Detection Started")
print("Press 'x' on keyboard to exit the program\n")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera")
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    for result in results:
        if result.boxes is not None:
            # Convert YOLO tensor outputs to NumPy arrays for easier handling
            boxes = result.boxes.xyxy.cpu().numpy()        # Bounding box coordinates [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()     # Class IDs

            # Loop through each detected object
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                # Extract coordinates and convert to integers
                x1, y1, x2, y2 = map(int, box)
                
                # Get class name and confidence
                class_name = model.names[int(cls_id)]
                confidence = float(conf)

                # ====================== MASK DETECTION LOGIC ======================
                # Check actual class names from your model ("masked" and "no_masked")
                class_lower = class_name.lower()

                if "no_masked" in class_lower or "no_mask" in class_lower:
                    color = (0, 0, 255)      # Red color for No Mask
                    label = f"No Mask: {confidence:.2f}"
                    is_wearing_mask = False
                else:
                    color = (0, 255, 0)      # Green color for With Mask
                    label = f"With Mask: {confidence:.2f}"
                    is_wearing_mask = True
                # ===================================================================

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label above the box
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Print detection details in terminal (useful for debugging)
                print(f"Detected: {class_name} | Confidence: {confidence:.2f} | Wearing Mask: {is_wearing_mask}")

    # Display the frame with detections
    cv2.imshow("YOLOv11 Face Mask Detection - Live", frame)

    # Exit when 'x' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        print("\nExiting program...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera closed. Program terminated.")
