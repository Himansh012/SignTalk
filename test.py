import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    cv2.imshow("test",img)
    key = cv2.waitKey(1)
    if(key!=-1):
        print(key)
