import cv2
import time
import os
import ctypes
from ultralytics import YOLO

# --- CONFIGURATION ---
TIME_LIMIT_SECONDS = 30  # Set to 30 seconds for quick testing! (Change to 7200 for 2 hours later)
HOSTS_PATH = r"C:\Windows\System32\drivers\etc\hosts"
REDIRECT_IP = "127.0.0.1"
SITES_TO_BLOCK = [
    "www.facebook.com", "facebook.com",
    "www.instagram.com", "instagram.com",
    "www.tiktok.com", "tiktok.com"
]

def is_admin():
    """Checks if Python has Windows Administrator privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def block_websites():
    """Injects the redirect IP into the Windows hosts file."""
    print("\n[!] TIME LIMIT REACHED. INITIATING LOCKDOWN...")
    try:
        with open(HOSTS_PATH, 'r+') as file:
            content = file.read()
            for site in SITES_TO_BLOCK:
                if site not in content:
                    file.write(f"{REDIRECT_IP} {site}\n")
        print("[+] Distractions blocked successfully.")
    except PermissionError:
        print("[-] PERMISSION DENIED: You must run this script as Administrator!")

def unblock_websites():
    """Removes the blocked sites from the hosts file when you quit."""
    print("\n[!] REMOVING LOCKDOWN. Restoring internet access...")
    try:
        with open(HOSTS_PATH, 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            for line in lines:
                if not any(site in line for site in SITES_TO_BLOCK):
                    file.write(line)
            file.truncate()
        print("[+] Sites unblocked.")
    except PermissionError:
        pass

# --- MAIN AI LOOP ---
print("--- LOADING EDGE VISION ENFORCER ---")
if not is_admin():
    print("\nWARNING: You are not running as Administrator. The blocker will fail!")
    print("Please open a terminal as Administrator to run this script.\n")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

active_seconds = 0
is_blocked = False
last_seen_time = time.time()

print("--- SYSTEM ARMED ---")
print("Press 'q' to quit and unblock sites.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Run YOLO (classes=[0] forces it to ONLY detect humans)
    results = model(frame, classes=[0], verbose=False)
    
    # 2. Check if a human is in the frame
    person_detected = len(results[0].boxes) > 0
    current_time = time.time()
    
    # 3. Update Timer
    if person_detected:
        # Add the exact time elapsed since the last frame
        time_delta = current_time - last_seen_time
        # Prevent massive jumps if the camera lags
        if time_delta < 1.0: 
            active_seconds += time_delta
            
    last_seen_time = current_time

    # 4. Trigger Lockdown
    if active_seconds >= TIME_LIMIT_SECONDS and not is_blocked:
        block_websites()
        is_blocked = True

    # 5. UI Overlay
    annotated_frame = results[0].plot()
    
    # Draw Status Box
    cv2.rectangle(annotated_frame, (0, 0), (600, 80), (0, 0, 0), -1) 
    
    color = (0, 0, 255) if is_blocked else (0, 255, 0)
    status_text = f"Screen Time: {int(active_seconds)}s / {TIME_LIMIT_SECONDS}s"
    
    cv2.putText(annotated_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    if is_blocked:
        cv2.putText(annotated_frame, "DISTRACTIONS BLOCKED!", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Edge Vision Analytics - Active Monitoring", annotated_frame)

    # 6. Safety Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
unblock_websites()
cap.release()
cv2.destroyAllWindows()