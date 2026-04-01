from ultralytics import YOLO
import cv2

# ====================== MODEL PATH ======================
model = YOLO(r"C:\Users\Prince\Downloads\face mask detector\best.pt")

print("✅ Model loaded!")
print("Class names in your model:", model.names)   # ← This will tell us the real class names

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)   # lowered conf a bit for testing

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]          # Get real class name
            confidence = float(box.conf[0])

            # Debug: Print every detection to console
            print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

            # === Decide color and label based on ACTUAL class name ===
            if "mask" in class_name.lower() and "without" not in class_name.lower() and "no" not in class_name.lower():
                color = (0, 255, 0)   # Green - With Mask
                label = f"With Mask: {confidence:.2f}"
            else:
                color = (0, 0, 255)   # Red - No Mask
                label = f"No Mask: {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection - Live Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()