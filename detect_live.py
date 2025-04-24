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

    # If hand is detected
    if len(lmList) == 21:
        features = []
        for lm in lmList:
            features.extend([lm[1], lm[2]])  # x and y

        # Predict the letter
        prediction = model.predict([features])[0]

        # Show the result on screen
        cv2.putText(img, f"Letter: {prediction}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Flip image and display
    fimg = cv2.flip(img, 1)
    cv2.imshow("ASL Live Detection", fimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
