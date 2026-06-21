import streamlit as st
import whisper
import tempfile
import os
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs
from elevenlabs import save


st.title("🎬 AI English → Tamil Video Dubber")


# ElevenLabs
client = ElevenLabs(
    api_key="sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7"
)


video = st.file_uploader(
    "Upload video",
    type=["mp4","mov","mkv"]
)


if video:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as f:
        f.write(video.read())
        video_path=f.name


    st.video(video_path)


    if st.button("Convert Tamil Voice"):


        # 1. Extract audio
        clip = VideoFileClip(video_path)

        audio_path="original_audio.mp3"
        clip.audio.write_audiofile(audio_path)


        # 2. Whisper
        model = whisper.load_model("base")

        result=model.transcribe(audio_path)

        english_text=result["text"]



        # 3. Translate
        tamil_text = GoogleTranslator(
            source="en",
            target="ta"
        ).translate(
            english_text
        )



        # 4. ElevenLabs voice

        audio = client.text_to_speech.convert(
            voice_id="YOUR_VOICE_ID",
            model_id="eleven_multilingual_v2",
            text=tamil_text
        )


        tamil_audio="tamil.mp3"

        save(
            audio,
            tamil_audio
        )



        # 5. Replace audio using MoviePy

        video_clip = VideoFileClip(video_path)

        new_audio = AudioFileClip(
            tamil_audio
        )


        final = video_clip.with_audio(
            new_audio
        )


        output="tamil_video.mp4"

        final.write_videofile(
            output,
            codec="libx264",
            audio_codec="aac"
        )


        st.success("Done 🎉")

        st.video(output)
