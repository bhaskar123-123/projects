import torch
import cv2
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv5s model via torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam (device 0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Starting detection — press 'q' in cv2 window to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame.")
            break

        results = model(frame)  # run inference

        annotated = results.render()[0]  # BGR with boxes drawn
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        plt.imshow(annotated_rgb)
        plt.axis('off')
        display(plt.gcf())
        clear_output(wait=True)

        cv2.imshow("YOLOv5 Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
