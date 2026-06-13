import streamlit as st

import yt_dlp
import whisper

from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip

from deep_translator import GoogleTranslator

import os



st.title("🎬 AI English → Tamil YouTube Video Dubber")



url = st.text_input(
    "Paste YouTube URL"
)



if st.button("Start Tamil Dubbing"):


    if url:


        # 1. Download YouTube video

        st.info("Downloading video...")


        options = {

            "format": "mp4",

            "outtmpl": "video.mp4",

            "http_headers": {

                "User-Agent":
                "Mozilla/5.0"

            },

            "extractor_args": {

                "youtube": {

                    "player_client":
                    ["android"]

                }

            }

        }



        with yt_dlp.YoutubeDL(options) as ydl:

            ydl.download([url])



        st.success("Downloaded")




        # 2. Whisper

        st.info("Converting speech to text...")


        model = whisper.load_model(
            "large-v3"
        )


        result = model.transcribe(

            "video.mp4",

            language="en",

            fp16=False

        )



        segments = result["segments"]



        english = []


        for item in segments:

            english.append(
                item["text"]
            )




        # 3. Translate


        st.info("Translating to Tamil...")


        translator = GoogleTranslator(

            source="en",

            target="ta"

        )


        tamil = []


        for sentence in english:


            translated = translator.translate(
                sentence
            )


            tamil.append(
                translated
            )



        tamil_text = " ".join(tamil)





        # 4. Tamil voice


        st.info("Generating Tamil voice...")


        voice = gTTS(

            text=tamil_text,

            lang="ta",

            slow=False

        )


        voice.save(
            "tamil_audio.mp3"
        )






        # 5. Merge


        st.info("Creating final video...")


        video = VideoFileClip(
            "video.mp4"
        )


        audio = AudioFileClip(
            "tamil_audio.mp3"
        )



        final = video.with_audio(
            audio
        )


        final.write_videofile(

            "Tamil_Dubbed.mp4",

            codec="libx264",

            audio_codec="aac"

        )



        st.success(
            "Tamil Dubbed Video Ready!"
        )



        # Download button


        with open(
            "Tamil_Dubbed.mp4",
            "rb"
        ) as file:


            st.download_button(

                label="Download Tamil Video",

                data=file,

                file_name="Tamil_Dubbed.mp4",

                mime="video/mp4"

            )


    else:

        st.warning(
            "Enter YouTube URL"
        )
