# Face-Mask-Detector-Live-Cam
Face Mask Detection System using YOLOv11
This project is a real-time face mask detection application developed using YOLOv11 (Ultralytics). The system uses a webcam to detect whether a person is wearing a face mask or not.

# Main Files

facemask.py - This is the main Python script that runs the live face mask detection. It opens the webcam, processes each frame using the trained YOLOv11 model, draws bounding boxes around detected faces, and labels them as "With Mask" (green) or "No Mask" (red) with confidence scores.

best.pt - This is the trained model file (weights) of the YOLOv11 model. It contains all the learned parameters from training on a custom face mask dataset in Google Colab. This file is the core of the detection system and is loaded by the Python script to perform object detection.

# How It Works
The Python script loads the best.pt model and continuously captures video from the webcam. For every frame, YOLOv11 detects faces and classifies them into two classes: with_mask or without_mask. The results are displayed in real-time with colored bounding boxes and text labels.

# Purpose
This project demonstrates the integration of deep learning (YOLOv11) with computer vision for a practical application — enforcing face mask compliance. It can be further extended by adding serial communication to control hardware such as LEDs, buzzers, and servo motors in Proteus simulation.

# Status
On development
