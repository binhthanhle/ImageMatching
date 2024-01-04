import streamlit as st
from lib import matching

MATCH_LABEL = '<p style="font-family:Courier; color:Blue; font-size: 20px;">MATCH</p>'
NOT_MATCH_LABEL = '<p style="font-family:Courier; color:Red; font-size: 20px;">NOT MATCH</p>'


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

draw = st.toggle('Plot the Matching?', key="draw_bool")
if on:
    result = matching(path1 = path_src, path2 = path_des, kind = 'path', draw = draw)
else:
    result = matching(path1 = uploaded_file_src, path2 = path_des, kind = 'path', draw = draw) 

if result:
    st.markdown(MATCH_LABEL, unsafe_allow_html=True)
else:
    st.markdown(NOT_MATCH_LABEL, unsafe_allow_html=True)