import cv2
import numpy as np
from ultralytics import YOLO

# ====================== CONFIGURATION ======================
MODEL_PATH = r"your model path"  

CONF_THRESHOLD = 0.5
WEBCAM_INDEX = 0                    # Change to 1 or 2 if default camera doesn't work
# ========================================================

# Load the model
model = YOLO(MODEL_PATH)
print("YOLOv11 Model loaded successfully!")
print("Model Classes:", model.names)

# Open webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Live Face Mask Detection Started... Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv11 detection
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    # Process detections using NumPy
    for result in results:
        if result.boxes is not None:
            # Convert YOLO tensors to NumPy arrays (This is where NumPy is used)
            boxes = result.boxes.xyxy.cpu().numpy()      # Bounding boxes [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()     # Class IDs

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls_id)]
                confidence = float(conf)

                # Determine if person is wearing mask
                is_wearing_mask = "mask" in class_name.lower() and "without" not in class_name.lower()

                # Choose color and label
                if is_wearing_mask:
                    color = (0, 255, 0)      # Green
                    label = f"With Mask: {confidence:.2f}"
                else:
                    color = (0, 0, 255)      # Red
                    label = f"No Mask: {confidence:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Optional: Print detection info in terminal
                print(f"Detected: {class_name} | Confidence: {confidence:.2f} | Mask: {is_wearing_mask}")

    # Show the live video
    cv2.imshow("YOLOv11 Face Mask Detection (with NumPy)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nProgram ended.")
