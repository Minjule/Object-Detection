import cv2
import os
from datetime import datetime

SAVE_DIR = "D:\\HAR\\additional"   
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("📷 Press 'c' to capture an image")
print("❌ Press 'q' to quit")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    cv2.imshow("Capture Window", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{count}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved: {filepath}")
        count += 1

    # ---- Quit ----
    if key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
