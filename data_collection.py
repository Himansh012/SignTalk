import cv2
import csv
import os
import HandTrackingModule as htm

# 📁 Folder to store data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ✏️ Set which letter you're collecting data for (A-Z)
TARGET_LETTER = "A"

# 📄 File to save to
file_path = os.path.join(DATA_DIR, f"data_{TARGET_LETTER}.csv")

# 🎥 Start webcam
cap = cv2.VideoCapture(0)
detector = htm.HandDectector()

print(f"[INFO] Collecting data for letter: {TARGET_LETTER}")
print("[INFO] Press 's' to save current hand landmarks")
print("[INFO] Press 'q' to quit")

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=True)

    if len(lmList) == 21:
        cv2.putText(img, f"Letter: {TARGET_LETTER}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    fimg = cv2.flip(img, 1)
    cv2.imshow("Data Collection - ASL", fimg)

    key = cv2.waitKey(1)

    if key == ord('s') and len(lmList) == 21:
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = []
            for lm in lmList:
                row.extend([lm[1], lm[2]])  # x and y only
            row.append(TARGET_LETTER)
            writer.writerow(row)
        print(f"[+] Saved sample for {TARGET_LETTER}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
