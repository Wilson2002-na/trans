import streamlit as st
import whisper

from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator

import os


st.title("🎬 AI English → Tamil Video Dubber")


uploaded_file = st.file_uploader(
    "Upload English Video",
    type=["mp4", "mov", "avi"]
)



if st.button("Start Tamil Dubbing"):


    if uploaded_file:


        # Save uploaded video

        with open("video.mp4", "wb") as f:

            f.write(
                uploaded_file.read()
            )


        st.success("Video uploaded")



        # Whisper speech recognition

        st.info("Converting speech to text...")


        model = whisper.load_model(
            "small"
        )


        result = model.transcribe(

            "video.mp4",

            language="en",

            fp16=False

        )



        english = []


        for item in result["segments"]:

            english.append(
                item["text"]
            )





        # Translate English to Tamil

        st.info("Translating to Tamil...")


        translator = GoogleTranslator(

            source="en",

            target="ta"

        )


        tamil = []


        for sentence in english:

            tamil.append(

                translator.translate(sentence)

            )



        tamil_text = " ".join(tamil)






        # Tamil voice

        st.info("Generating Tamil voice...")


        voice = gTTS(

            text=tamil_text,

            lang="ta"

        )


        voice.save(
            "tamil_audio.mp3"
        )





        # Merge audio + video

        st.info("Creating dubbed video...")


        video = VideoFileClip(
            "video.mp4"
        )


        audio = AudioFileClip(
            "tamil_audio.mp3"
        )



        final = video.with_audio(
            audio
        )


        output = "Tamil_Dubbed.mp4"


        final.write_videofile(

            output,

            codec="libx264",

            audio_codec="aac"

        )



        st.success(
            "Tamil Dubbed Video Ready 🎉"
        )



        # Download


        with open(output, "rb") as file:


            st.download_button(

                label="⬇️ Download Tamil Video",

                data=file,

                file_name="Tamil_Dubbed.mp4",

                mime="video/mp4"

            )



    else:

        st.warning(
            "Upload a video first"
        )
