import cv2
import joblib
import numpy as np
import HandTrackingModule as htm

# Load the trained model
model = joblib.load("asl_model1.pkl")

# Start webcam
cap = cv2.VideoCapture(0)
detector = htm.HandDectector()

s =" "
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fimg = cv2.flip(img, 1)

    key=cv2.waitKey(1) & 0xFF
    if(key==32):  ## Space Bar 
        s=s+" "
    if(key==27):  ## Escape Button
        s=" "
    if(key==8 and len(s)>0): ## Backspace
        s=s[:-1]
    cv2.putText(fimg, f" Sentence -{s}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # If hand is detected
    if len(lmList) == 21:
        features = []
        for lm in lmList:
            features.extend([lm[1], lm[2]])  # x and y

        # Predict the letter
        prediction = model.predict([features])[0]

        if(key==13):  ## Enter
            s=s+prediction
        # Show the result on screen
        cv2.putText(fimg, f"Letter: {prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Live Detection", fimg)
    if (cv2.getWindowProperty("ASL Live Detection", cv2.WND_PROP_VISIBLE) < 1) or key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
