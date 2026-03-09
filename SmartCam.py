import cv2
import time
import base64
import threading
import requests
import json
from ultralytics import YOLO

# --- CONFIG ---
# This points to your local Ollama "TV"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "moondream" 

print("--- LOADING AI SYSTEMS ---")
print("1. Loading YOLO (Eyes)...")
model = YOLO("yolov8n.pt")

print("2. Connecting to Camera...")
cap = cv2.VideoCapture(0)

# Global variables to share data between the AI thread and the UI
current_description = "Press SPACE to analyze scene..."
is_thinking = False

def query_moondream(frame):
    """Sends the image to Ollama for description"""
    global current_description, is_thinking
    is_thinking = True
    current_description = "Thinking... (AI is looking)"
    
    try:
        # 1. Convert image to format Ollama understands (Base64)
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # 2. Prepare the request
        payload = {
            "model": MODEL_NAME,
            "prompt": "Describe this image in one short sentence.",
            "images": [jpg_as_text],
            "stream": False
        }
        
        # 3. Send to Local Brain
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()
        
        if "response" in result:
            current_description = "AI: " + result["response"]
        else:
            current_description = "Error: No response from Ollama"
            
    except Exception as e:
        current_description = f"Error: {e}"
    
    is_thinking = False

print("--- SYSTEM READY ---")
print("Controls: [SPACE] = Analyze Scene | [Q] = Quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Run YOLO (Fast Loop)
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # 2. Draw the AI Description Box
    # (Black background rectangle at the top)
    cv2.rectangle(annotated_frame, (0, 0), (1280, 80), (0, 0, 0), -1) 
    
    # (Text)
    cv2.putText(annotated_frame, current_description, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 3. Show Display
    cv2.imshow("Project 2: Smart Vision", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and not is_thinking:
        # Run Moondream in a background thread so video doesn't freeze
        threading.Thread(target=query_moondream, args=(frame.copy(),)).start()

cap.release()
cv2.destroyAllWindows()