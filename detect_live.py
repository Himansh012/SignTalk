import cv2
import joblib
import numpy as np
import HandTrackingModule as htm
import pyttsx3 as px
# Load the trained model
# model = joblib.load("asl_model1.pkl")

# Start webcam
cap = cv2.VideoCapture(0)
detector = htm.HandDectector()
s =" "
mode = 1
mode_name="Alphabets"
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
    if(key==ord('0')):
        mode=0
    if(key==ord('1')):
        mode=1
    if(key==ord('2')):
        mode=2
    if(key== ord('p')):
        eng=px.init()
        eng.say(s)
        eng.runAndWait()
    cv2.putText(fimg, f" Sentence -{s}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print("Sentence -"+s)
    if(mode==1):
        mode_name="Alphabets"
    elif(mode==0):
        mode_name="Numbers"
    elif(mode==2):
        mode_name="Sentence"
    cv2.putText(fimg,f"Mode - {mode_name}",(375,20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),1)
    # If hand is detected
    if len(lmList) == 21:
        features = []
        for lm in lmList:
            features.extend([lm[1], lm[2]])  # x and y
        if(mode==1):
            model = joblib.load("asl_model1.pkl")
        elif(mode==0):
            model = joblib.load("asl_model2.pkl")
        elif(mode==2):
            model = joblib.load("asl_model3.pkl")
        # Predict the letter
        prediction = model.predict([features])[0]

        if(key==13 and mode==1):  ## Enter
            s=s+prediction
            print("Sentence -"+s)
        if(key==13 and mode==0):
            s=s+str(prediction)
            print("Sentence -"+s)
        if(key==13 and mode==2):
            s=s+prediction
            print("Sentence -"+s)

        # Show the result on screen
        cv2.putText(fimg, f"Letter: {prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Live Detection", fimg)
    if (cv2.getWindowProperty("ASL Live Detection", cv2.WND_PROP_VISIBLE) < 1) or key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
