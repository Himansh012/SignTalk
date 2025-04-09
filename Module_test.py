import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDectector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList)!=0:
        print(lmList[4])
    ## FPS counter    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    fimg=cv2.flip(img,1)
    cv2.putText(fimg,"FPS->"+str(int(fps)),(20,50),cv2.FONT_HERSHEY_COMPLEX,.5,  
            (255,0,255),1)
    cv2.imshow("Image",fimg)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release