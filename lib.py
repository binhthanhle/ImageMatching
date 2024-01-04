from skimage import io
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from typing import Tuple

# Define the constants
MIN_MATCH_COUNT = 25
THRESHOLD = 0.9

def matching(path1:str = None, path2:str = None, draw:bool =False) -> int:
    """ Function to get the matching between two images
    """
    img1 = io.imread(path1)
    img2 = io.imread(path2)

    # Size:
    h,w,s = img1.shape
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
      src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
      M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 7.0)
      matchesMask = mask.ravel().tolist()
    else:
        matchesMask = None

    if draw:
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
        img_match = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        plt.figure(figsize=(5, 5))
        plt.imshow(img_match, 'gray'),plt.show()

    if matchesMask:
        if len(matchesMask)/len(good) > THRESHOLD:
            return 1
        else:
            return 0
    else:
        return -1