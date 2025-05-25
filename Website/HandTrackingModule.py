import cv2
import mediapipe as mp
import time
'''Use ChatGPT to understand what each line does'''

class HandDectector():
    def __init__(self, mode=False, maxHands = 2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity= complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands     
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:  
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
                
        return img
    def findPosition(self, img, handNo=0, draw=True):
        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                height, width, channel =img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                # print(id,cx,cy)
                lmList.append([id,cx,cy])
                # if id%4==0:
                #     cv2.circle(img,(cx,cy),10,(203,192,255),cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDectector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        key = cv2.waitKey(1)
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
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1 or key==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()