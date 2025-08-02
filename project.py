import torch
import cv2
import numpy as np

# Load YOLOv5s model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Start video capture (0 is the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

print("✅ Webcam access successful. Starting detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Convert frame from BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img_rgb)

    # Draw detections on the original BGR frame
    results.render()
    annotated_frame = np.squeeze(results.ims)

    # Display result
    cv2.imshow('YOLOv5s Webcam Detection', annotated_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



