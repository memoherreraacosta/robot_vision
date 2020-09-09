import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

fname = "pattern.png"
img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    # Python: cv.FindCornerSubPix(image, corners, win, zero_zone, criteria) → corners
    # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornersubpix#cv.FindCornerSubPix
    corners2 = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria
    )
    imgpoints.append(corners2)

    # Draw and display the corners
    # img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

# Python: cv.CalibrateCamera2(objectPoints, imagePoints, pointCounts, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags=0) → None
# https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv.CalibrateCamera2
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print(mtx)
cv2.destroyAllWindows()

"""
Output using "pattern.png":

$ python3 practice2.py
[[9.79805940e+07 0.00000000e+00 9.14500001e+02]
 [0.00000000e+00 9.79553435e+07 6.64500000e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

 like the "camera_calibration.png" image found at the current directory
"""