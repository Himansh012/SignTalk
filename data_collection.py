import cv2
import csv
import os
import time
import HandTrackingModule as htm
import string

# 📁 Folder to store collected data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 🧠 Settings
NUM_SAMPLES_PER_LETTER = 50
DELAY_BETWEEN_SAMPLES = 0.1  # in seconds

# 📦 Setup
cap = cv2.VideoCapture(0)
detector = htm.HandDectector()

for letter in string.ascii_uppercase:

    if letter in 'JZ':
        continue
    print(f"\n📸 Starting collection for letter: {letter}")
    print(f"Hold the sign for '{letter}' and press 's' to begin collecting {NUM_SAMPLES_PER_LETTER} samples...")

    # Wait for user to press 's'
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        fimg = cv2.flip(img, 1)
        cv2.putText(fimg, f"Letter: {letter} - Press 's' to start", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("ASL Data Collector", fimg)
        key = cv2.waitKey(1)
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Start collecting samples
    collected = 0
    file_path = os.path.join(DATA_DIR, f"data_{letter}.csv")
    while collected < NUM_SAMPLES_PER_LETTER:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) == 21:
            with open(file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                row = []
                for lm in lmList:
                    row.extend([lm[1], lm[2]])  # x and y
                row.append(letter)
                writer.writerow(row)
            collected += 1

            print(f"Collected sample {collected}/{NUM_SAMPLES_PER_LETTER} for {letter}")

            cv2.putText(img, f"{letter}: {collected}/{NUM_SAMPLES_PER_LETTER}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            time.sleep(DELAY_BETWEEN_SAMPLES)

        fimg = cv2.flip(img, 1)
        cv2.imshow("ASL Data Collector", fimg)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("✅ Data collection complete for all letters!")
cap.release()
cv2.destroyAllWindows()
