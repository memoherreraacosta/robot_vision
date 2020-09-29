import cv2 as cv
import cv2
import numpy as np
from skimage.filters import threshold_otsu
import scipy.misc

vid = cv2.VideoCapture(0)
value = 0

if not vid.isOpened():
    print('Error opening video stream or file')


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


alpha = 0.5

while vid.isOpened():
    ret, frame = vid.read()
    img = frame
    if ret == True:
        beta = 0
        frame = cv2.addWeighted(frame, alpha, np.zeros(
            img.shape, img.dtype), 0, beta)
        cv.imshow('', frame)
 
        if(cv.waitKey(25) & 0xFF == ord('w')):
            alpha -= 0.1
            print(alpha)
        if(cv.waitKey(25) & 0xFF == ord('e')):
            alpha += 0.1
            print

        if(cv.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break

vid.release()
cv.destroyAllWindows()