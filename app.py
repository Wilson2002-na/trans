import streamlit as st
import whisper
import yt_dlp

from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator


st.title("🎬 AI YouTube English → Tamil Video Dubber")


url = st.text_input(
    "Paste YouTube Video URL"
)



if st.button("Start Tamil Dubbing"):


    if url:


        # Download YouTube video

        st.info("Downloading YouTube video...")


        ydl_opts = {

            "format": "mp4",

            "outtmpl": "video.mp4"

        }


        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            ydl.download([url])



        st.success("Video downloaded")



        # Whisper

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



        # Translate

        st.info("Translating English to Tamil...")


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



        # Tamil audio

        st.info("Generating Tamil voice...")


        voice = gTTS(

            tamil_text,

            lang="ta"

        )


        voice.save(
            "tamil_audio.mp3"
        )



        # Merge

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



        # Download button

        with open(output,"rb") as file:


            st.download_button(

                "⬇️ Download Tamil Video",

                file,

                file_name="Tamil_Dubbed.mp4",

                mime="video/mp4"

            )


    else:


        st.warning(
            "Paste YouTube URL first"
        )
