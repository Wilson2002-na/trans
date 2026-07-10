import streamlit as st
#files upload
uploaded=st.fileuploader(
"upload a video",
type=["mp4","avi","mov","mkv","webm"]
)
