import streamlit as st
#files upload
uploaded=st.file_uploader(
"upload a video",
type=["mp4","avi","mov","mkv","webm"]
)
