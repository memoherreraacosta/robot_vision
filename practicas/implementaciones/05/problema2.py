import cv2
import numpy as np
import scipy.misc

vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print('Error opening video stream or file')


while vid.isOpened():
    ret, frame = vid.read()
    img = frame.copy()
    cv2.imshow('', img)
    if ret == True:
        blur = cv2.blur(img,(5,5))
        blur0=cv2.medianBlur(blur,5)
        blur1= cv2.GaussianBlur(blur0,(5,5),0)
        blur2= cv2.bilateralFilter(blur1,9,75,75)
        hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

        low_blue = np.array([55, 0, 0])
        high_blue = np.array([118, 255, 255])

        #low_green = np.array([0, 55, 0])
        #high_green = np.array([255, 118, 255])

        #low_red = np.array([0, 0, 55])
        #high_red = np.array([255, 255, 118])

        mask = cv2.inRange(hsv, low_blue, high_blue)
        res = cv2.bitwise_and(img, img, mask= mask)

        kernel = np.ones((5,5), np.uint8) 
        res = cv2.erode(res, kernel, iterations=5) 
        res = cv2.dilate(res, kernel, iterations=5)

        cv2.imshow('Blue filter', res)
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break

vid.release()
cv2.destroyAllWindows()