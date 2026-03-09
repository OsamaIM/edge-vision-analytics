import cv2
from ultralytics import YOLO

# 1. Load a tiny model (it will download automatically)
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt") 

# 2. Open Webcam (0 is usually the default cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Starting video feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Detect Objects
    results = model(frame, verbose=False) # verbose=False keeps terminal clean

    # 4. Draw Boxes
    annotated_frame = results[0].plot()

    # 5. Show on Screen
    cv2.imshow("YOLOv8 Real-Time Inference", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()