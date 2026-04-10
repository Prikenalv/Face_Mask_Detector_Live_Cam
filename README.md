# Face Mask Detection System

A real-time face mask detection application with two implementations — one using **YOLOv11** and one using **Keras/MobileNetV2** — both running live through a webcam.

---

## Project Structure

```
Face-Mask-Detection-master/
│
├── YOLO/
│   ├── facemask_yolo.py       # Main script for YOLOv11-based detection
│   └── best.pt                # Trained YOLOv11 model weights
│
├── KERAS/
│   ├── facemask_keras.py      # Main script for Keras-based detection
│   └── mask_detector.model    # Trained MobileNetV2 Keras model
```

---

## Implementations

### 1. YOLOv11 (Ultralytics)
**Folder:** `YOLO/`

The YOLOv11 implementation uses object detection to locate faces in the frame and classify them with bounding boxes in real time.

- **Model:** `best.pt` — trained YOLOv11 weights on a custom face mask dataset
- **Script:** `facemask_yolo.py` — loads the model, opens the webcam, draws bounding boxes labeled **"With Mask"** (green) or **"No Mask"** (red) with confidence scores

### 2. Keras / MobileNetV2
**Folder:** `KERAS/`

The Keras implementation uses MobileNetV2 as a base model with a custom classification head, trained using TensorFlow/Keras.

- **Model:** `mask_detector.model` — trained MobileNetV2 model saved in Keras format
- **Script:** `facemask_keras.py` — loads the model, captures webcam frames, preprocesses each frame, and classifies it as **"with_mask"** or **"without_mask"**

---

## How It Works

Both scripts follow the same general flow:

1. Load the trained model
2. Open the webcam using OpenCV
3. Capture frames continuously
4. Run the model on each frame
5. Display the result with colored labels in real time
6. Press **`Q`** to quit

---

## Requirements

```bash
pip install tensorflow opencv-python ultralytics numpy matplotlib scikit-learn imutils
```

---

## Status

> **On Development** — planned extension includes serial communication to control hardware such as LEDs, buzzers, and servo motors via Proteus simulation.
