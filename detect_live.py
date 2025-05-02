import cv2
import joblib
import numpy as np
import HandTrackingModule as htm

# Load the trained model
model = joblib.load("asl_model.pkl")

# Start webcam
cap = cv2.VideoCapture(0)
detector = htm.HandDectector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fimg = cv2.flip(img, 1)

    # If hand is detected
    if len(lmList) == 21:
        features = []
        for lm in lmList:
            features.extend([lm[1], lm[2]])  # x and y

        # Predict the letter
        prediction = model.predict([features])[0]

        # Show the result on screen
        cv2.putText(fimg, f"Letter: {prediction}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("ASL Live Detection", fimg)
    cv2.waitKey(1)
    if cv2.getWindowProperty("ASL Live Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()
