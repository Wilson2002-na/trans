
# app.py
# Refactored structure showing DSA usage (Queue + HashMap cache)
from collections import deque
import os, tempfile, shutil, atexit
import streamlit as st
import whisper
from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs
from moviepy import VideoFileClip, AudioFileClip

st.title("AI English → Tamil Video Dubber (DSA Version)")

client = ElevenLabs(api_key="api_key = os.environ.get("sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7")
VOICE_ID="CwhRBWXzGAHq8TQ4Fs17"

if "tmpdir" not in st.session_state:
    st.session_state.tmpdir=None

def cleanup(path):
    if path and os.path.isdir(path):
        shutil.rmtree(path,ignore_errors=True)

atexit.register(lambda: cleanup(st.session_state.get("tmpdir")))

def transcribe(video_path):
    model=whisper.load_model("small")
    return model.transcribe(video_path,language="en",fp16=False)["segments"]

def translate_segments(segments):
    translator=GoogleTranslator(source="en",target="ta")
    q=deque(segments)          # Queue
    cache={}                   # HashMap
    out=[]
    while q:
        seg=q.popleft()
        txt=seg["text"].strip()
        if txt not in cache:
            cache[txt]=translator.translate(txt)
        out.append(cache[txt])
    return " ".join(out)

def tts(text,audio_path):
    audio=client.text_to_speech.convert(
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        text=text
    )
    with open(audio_path,"wb") as f:
        for chunk in audio:
            f.write(chunk)

def merge(video_path,audio_path,out_path):
    video=VideoFileClip(video_path)
    audio=AudioFileClip(audio_path)
    if audio.duration>video.duration:
        audio=audio.subclipped(0,video.duration)
    final=video.with_audio(audio)
    final.write_videofile(out_path,codec="libx264",audio_codec="aac",logger=None)
    video.close(); audio.close()

uploaded=st.file_uploader("Upload",type=["mp4","avi","mov","mkv","webm"])

if st.button("Start") and uploaded:
    cleanup(st.session_state.tmpdir)
    tmp=tempfile.mkdtemp()
    st.session_state.tmpdir=tmp

    video_path=os.path.join(tmp,"video"+os.path.splitext(uploaded.name)[1])
    audio_path=os.path.join(tmp,"ta.mp3")
    out_path=os.path.join(tmp,"Tamil_Dubbed.mp4")

    with open(video_path,"wb") as f:
        f.write(uploaded.read())

    segs=transcribe(video_path)
    tamil=translate_segments(segs)
    tts(tamil,audio_path)
    merge(video_path,audio_path,out_path)

    with open(out_path,"rb") as f:
        st.download_button("Download",f.read(),"Tamil_Dubbed.mp4","video/mp4")
