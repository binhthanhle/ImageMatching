import streamlit as st
from skimage import io
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from typing import Tuple

# Define the constants
MIN_MATCH_COUNT = 25
THRESHOLD = 0.9

def get_input(kind:str = 'path', value1 = None, value2 = None):
    if kind=="path":
        img1 = io.imread(value1)
        img2 = io.imread(value2)
    else:
        img1 = value1
        img2 = value2
    return img1, img2


def matching(path1 = None, path2 = None, kind:str = 'path', draw:bool =False) -> int:
    """ Function to get the matching between two images
    """
    img1, img2 = get_input(kind, path1, path2)
    fig, ax = plt.figure(figsize=(5, 5))
    plt.imshow(img_match, 'gray', ax=ax),
    st.pyplot(fig)
    # Size:
    # h,w,s = img1.shape
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
        fig, ax = plt.figure(figsize=(5, 5))
        ax.imshow(img_match, 'gray'),
        st.pyplot(fig)

    if matchesMask:
        if len(matchesMask)/len(good) > THRESHOLD:
            return 1
        else:
            return 0
    else:
        return -1
    
MATCH_LABEL = '<p style="font-family:Courier; color:Green; font-size: 20px;">MATCH</p>'
NOT_MATCH_LABEL = '<p style="font-family:Courier; color:Red; font-size: 20px;">NOT MATCH</p>'
UNCERTAIN_LABEL = '<p style="font-family:Courier; color:Blue; font-size: 20px;">UNCERTAIN_LABEL</p>'


## Header of the application
st.header("Application for Image Matching")


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


col1, col2 = st.columns(2)
on = st.toggle('Upload File')
st.write("---")
with col1:
    col1.header("Image 1")
    path_src = st.text_input(
        "Please input the path of image",
        key="inputpath_src",
        disabled=on,
    )
    uploaded_file_src = st.file_uploader("Choose a file", key="upload_src", disabled=not on)

with col2:
    col2.header("Image 2")
    path_des = st.text_input(
        "Please input the path of image",
        key="inputpath_des",
        disabled=on,
    )
    uploaded_file_des = st.file_uploader("Choose a file",key="upload_des", disabled=not on)


if st.button("Check", type="primary"):
    draw = st.toggle('Plot the Matching?', key="draw_bool")

    if on:
        if (uploaded_file_src is not None & uploaded_file_des is not None):
            result = matching(path1 = path_src, path2 = path_des, kind = 'path', draw = draw)
    else:
        if path_src is not None and path_des is not None:
            result = matching(path1 = uploaded_file_src, path2 = path_des, kind = 'file', draw = draw) 

    if result==1:
        st.markdown(MATCH_LABEL, unsafe_allow_html=True)
    elif result == -1:
        st.markdown(NOT_MATCH_LABEL, unsafe_allow_html=True)
    else:
        st.markdown(NOT_MATCH_LABEL, unsafe_allow_html=True)