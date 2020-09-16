import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
{ "description" : "The first Blender Open Movie from 2006",
  "sources" : [ "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4" ],
  "subtitle" : "By Blender Foundation",
  "title" : "Elephant Dream"
}
"""
video_name = "ElephantsDream.mp4"
video = cv.VideoCapture(video_name)

if(video.isOpened() == False):
    print("Error opening video stream or file")

while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        f = np.fft.fft2(grayFrame)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        normalized = cv.normalize(
            magnitude_spectrum,
            None,
            alpha=0,
            beta=1,
            norm_type=cv.NORM_MINMAX,
            dtype=cv.CV_32F
        )

        cv.imshow('Frame', grayFrame)
        cv.imshow('Magnitude Spectrum', normalized)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv.destroyAllWindows()
