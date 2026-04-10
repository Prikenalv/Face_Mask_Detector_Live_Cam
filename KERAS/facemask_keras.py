from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np

model = load_model(r"MODEL PATH")

CATEGORIES = ["Mask_Detected", "No_Mask_Detected"]  # match your folder names

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for model
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    label = CATEGORIES[class_id]

    # Display
    color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
    display = f"{label}: {confidence:.2f}"
    cv2.putText(frame, display, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()