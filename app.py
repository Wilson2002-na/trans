import streamlit as st
import whisper
import os

from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator


st.title("🎬 AI English → Tamil Video Dubber")


uploaded_file = st.file_uploader(
    "Upload English Video",
    type=["mp4", "mov", "avi"]
)



if st.button("Start Tamil Dubbing"):


    if uploaded_file:


        # Save uploaded video

        with open(
            "video.mp4",
            "wb"
        ) as f:

            f.write(
                uploaded_file.read()
            )



        st.success("Video uploaded ✅")




        # 1. Speech To Text


        st.info(
            "Converting English speech to text..."
        )


        whisper_model = whisper.load_model(
            "small"
        )


        result = whisper_model.transcribe(

            "video.mp4",

            language="en",

            fp16=False

        )



        segments = result["segments"]



        english_sentences = []



        for item in segments:


            english_sentences.append(

                item["text"]

            )



        english_text = " ".join(

            english_sentences

        )



        st.subheader(
            "English Text"
        )


        st.write(
            english_text
        )





        # 2. English To Tamil


        st.info(
            "Translating to Tamil..."
        )


        translator = GoogleTranslator(

            source="en",

            target="ta"

        )



        tamil_sentences = []



        for sentence in english_sentences:


            tamil = translator.translate(

                sentence

            )


            tamil_sentences.append(

                tamil

            )



        tamil_text = " ".join(

            tamil_sentences

        )



        st.subheader(
            "Tamil Translation"
        )


        st.write(
            tamil_text
        )






        # 3. Tamil Voice


        st.info(
            "Creating Tamil voice..."
        )



        tts = gTTS(

            text=tamil_text,

            lang="ta",

            slow=False

        )



        tts.save(

            "tamil_audio.mp3"

        )







        # 4. Create Dubbed Video


        st.info(
            "Creating Tamil dubbed video..."
        )



        video = VideoFileClip(

            "video.mp4"

        )



        audio = AudioFileClip(

            "tamil_audio.mp3"

        )



        final_video = video.with_audio(

            audio

        )



        final_video.write_videofile(

            "Tamil_Dubbed.mp4",

            codec="libx264",

            audio_codec="aac"

        )





        st.success(
            "DONE ✅ Tamil Dubbed Video Ready"
        )





        # Download


        with open(

            "Tamil_Dubbed.mp4",

            "rb"

        ) as file:



            st.download_button(

                label="⬇️ Download Tamil Dubbed Video",

                data=file,

                file_name="Tamil_Dubbed.mp4",

                mime="video/mp4"

            )



    else:


        st.warning(
            "Upload a video first"
        )
