import streamlit as st
import yt_dlp
import whisper

from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator


st.title("🎬 AI English → Tamil YouTube Video Dubber")


url = st.text_input(
    "Paste YouTube URL"
)


if st.button("Start Tamil Dubbing"):


    if url:


        # 1. Download Video

        st.info("Downloading video...")


        options = {
            "format": "mp4",
            "outtmpl": "video.mp4"
        }


        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])


        st.success("Download completed ✅")



        # 2. Speech to Text


        st.info("Converting English speech to text...")


        whisper_model = whisper.load_model(
            "large-v3"
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


        st.subheader("English Text")

        st.write(
            english_text
        )




        # 3. Translate English to Tamil


        st.info("Translating to Tamil...")


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


        st.subheader("Tamil Translation")

        st.write(
            tamil_text
        )





        # 4. Tamil Voice


        st.info("Creating Tamil voice...")


        tts = gTTS(
            text=tamil_text,
            lang="ta",
            slow=False
        )


        tts.save(
            "tamil_audio.mp3"
        )





        # 5. Create Dubbed Video


        st.info("Creating Tamil dubbed video...")


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



        st.success("DONE ✅ Tamil Dubbed Video Created")



        # Download Button


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
            "Please paste YouTube URL"
        )
