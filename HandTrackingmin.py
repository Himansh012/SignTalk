import cv2
import mediapipe as mp
import time
'''Use ChatGPT to understand what each line does'''
cap = cv2.VideoCapture(0)       ### Opens the camera controlled by variable "cap"

### Accesses the mediapipe hands module that helps in hand detection and tracking
mpHands = mp.solutions.hands     

### Initializes the hand tracking pipeline, creates hands object
hands = mpHands.Hands()

### Accesses the drawing tools in mediapipe tracking the 21 points in the hand         
mpDraw = mp.solutions.drawing_utils  

pTime = 0
cTime = 0

while True:
    success, img = cap.read()   
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:  
            for id, lm in enumerate(handlms.landmark):
                # print(id,lm)
                height, width, channel =img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                # print(id,cx,cy)
                if id in (4,8,12):
                    cv2.circle(img,(cx,cy),15,(203,192,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)        

    # flipimg = cv2.flip(img,1)                         
    
    cTime = time.time()
    fps = 1/(cTime-pTime) 
    pTime = cTime
    
    cv2.putText(img,"FPS->"+str(int(fps)),(20,50),cv2.FONT_HERSHEY_COMPLEX,.5,  
              (255,0,255),1)                                         
                                                                                  

    cv2.imshow("Image",img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release

## import inspect
## from mediapipe.python.solutions import hands
## print(inspect.getfile(hands))
