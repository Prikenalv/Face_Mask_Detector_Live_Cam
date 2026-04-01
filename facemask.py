from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO(r"model path")

print("Model Classes:", model.names)

cap = cv2.VideoCapture(0) #Change camera from 0 to 1 if it doesn't work

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls_id)]
                confidence = float(conf)

                class_lower = class_name.lower()

                if "no_masked" in class_lower or "no_mask" in class_lower:
                    color = (0, 0, 255)        # Red
                    label = f"No Mask: {confidence:.2f}"
                    is_mask = False
                else:
                    color = (0, 255, 0)        # Green
                    label = f"With Mask: {confidence:.2f}"
                    is_mask = True

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

                # Print to terminal for debugging
                print(f"Detected: {class_name} | Confidence: {confidence:.2f} | Mask: {is_mask}")

    cv2.imshow("YOLOv11 Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
