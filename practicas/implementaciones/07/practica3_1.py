import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    lower_reso = cv2.pyrDown(img)
    upper_reso = cv2.pyrUp(img)
    cv2.imshow("Classificador tamano original", img)
    cv2.imshow("Classificador tamano menor", lower_reso)
    cv2.imshow("Classificador tamano mayor", upper_reso)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc key
        break


with open("imagenes.txt", "w") as f:
    f.write("Imagen original:\n" + str(np.asarray(img)) + "\n\n")
    f.write("Imagen tamano menor:\n" + str(np.asarray(lower_reso)) + "\n\n")
    f.write("Imagen tamano mayor:\n" + str(np.asarray(upper_reso)))

cap.release()
cv2.destroyAllWindows()