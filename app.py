import streamlit as st
import whisper
import yt_dlp
import os

from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator


st.title("🎬 AI English → Tamil YouTube Video Dubber")


url = st.text_input(
    "Paste YouTube URL"
)


if st.button("Start Tamil Dubbing"):


    if url:


        # 1. Download YouTube Video


        st.info(
            "Downloading video..."
        )


        if os.path.exists("video.mp4"):
            os.remove("video.mp4")



        options = {


            "format": "best[ext=mp4]/best",


            "outtmpl": "video.mp4",


            "noplaylist": True,


            "quiet": False,


            "nocheckcertificate": True,


            "http_headers": {

                "User-Agent":
                "Mozilla/5.0"

            }

        }



        try:


            with yt_dlp.YoutubeDL(options) as ydl:

                ydl.download([url])


            st.success(
                "Video downloaded ✅"
            )


        except Exception as e:


            st.error(
                "Download failed"
            )

            st.write(e)

            st.stop()






        # 2. Whisper Speech To Text


        st.info(
            "Converting speech to text..."
        )


        model = whisper.load_model(
            "small"
        )


        result = model.transcribe(

            "video.mp4",

            language="en",

            fp16=False

        )



        english_sentences = []



        for item in result["segments"]:


            english_sentences.append(

                item["text"]

            )



        english_text = " ".join(

            english_sentences

        )



        st.subheader(
            "English"
        )


        st.write(
            english_text
        )







        # 3. Translate English → Tamil


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
            "Tamil"
        )


        st.write(
            tamil_text
        )







        # 4. Tamil Voice


        st.info(
            "Creating Tamil voice..."
        )



        tts = gTTS(

            text=tamil_text,

            lang="ta"

        )


        tts.save(

            "tamil_audio.mp3"

        )







        # 5. Create Dubbed Video


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
            "Paste YouTube URL first"
        )
