import cv2
import numpy as np

img = cv2.imread("intel.jpg", 0)
vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print('Error opening video stream or file')

def get_corrected_img(img1, img2):
    MIN_MATCHES = 50
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    print("IMG1 ", kp1, " ", des1)
    print("IMG1 ", kp2, " ", des2)

    index_params = dict(
        algorithm=6,
        table_number=6,
        key_size=12,
        multi_probe_level=2
    )
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = [
        m
        for m, n in matches
        if m.distance < 0.75 * n.distance
    ]

    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))
        return corrected_img
    return img2


while vid.isOpened():
    ret, frame = vid.read()
    img1 = frame.copy()
    #orb = cv2.ORB_create(nfeatures=500)
    #kp1, des1 = orb.detectAndCompute(img, None)
    #kp2, des2 = orb.detectAndCompute(img1, None)

    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #matches = bf.match(des1, des2)
    #matches = sorted(matches, key=lambda x: x.distance)
    # draw first 50 matches
    #match_img = cv2.drawMatches(img, kp1, img1, kp2, matches[:50], None)
    #cv2.imshow("Intel Logo Matching", match_img)
    
    cv2.imshow("Intel Logo Matching", get_corrected_img(img, img1))
    if(cv2.waitKey(25) & 0xFF == ord('q')):
        break

vid.release()
cv2.destroyAllWindows()