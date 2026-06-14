import streamlit as st
import whisper
import yt_dlp
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator
import os
import tempfile

st.title("🎬 AI YouTube English → Tamil Video Dubber")

url = st.text_input("Paste YouTube Video URL")

if st.button("Start Tamil Dubbing"):
    if url:
        # Use a temp directory for all files
        tmpdir = tempfile.mkdtemp()
        video_path = os.path.join(tmpdir, "video.mp4")
        audio_path = os.path.join(tmpdir, "tamil_audio.mp3")
        output_path = os.path.join(tmpdir, "Tamil_Dubbed.mp4")

        try:
            # Download YouTube video
            st.info("Downloading YouTube video...")
            ydl_opts = {
                "format": "mp4",
                "outtmpl": video_path,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            st.success("Video downloaded ✅")

            # Whisper transcription
            st.info("Converting speech to text...")
            model = whisper.load_model("small")
            result = model.transcribe(
                video_path,   # ✅ real file path, not object
                language="en",
                fp16=False
            )
            english = [item["text"] for item in result["segments"]]
            st.success("Transcription done ✅")

            # Translate to Tamil
            st.info("Translating English to Tamil...")
            translator = GoogleTranslator(source="en", target="ta")
            tamil = [translator.translate(sentence) for sentence in english]
            tamil_text = " ".join(tamil)
            st.success("Translation done ✅")

            # Generate Tamil audio
            st.info("Generating Tamil voice...")
            voice = gTTS(tamil_text, lang="ta")
            voice.save(audio_path)
            st.success("Tamil audio generated ✅")

            # Merge video + audio
            st.info("Creating dubbed video...")
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)

            # Trim audio if longer than video
            if audio.duration > video.duration:
                audio = audio.subclipped(0, video.duration)

            final = video.with_audio(audio)
            final.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                logger=None   # suppresses noisy logs in Streamlit
            )
            video.close()
            audio.close()
            st.success("Tamil Dubbed Video Ready 🎉")

            # Download button
            with open(output_path, "rb") as file:
                st.download_button(
                    "⬇️ Download Tamil Video",
                    file,
                    file_name="Tamil_Dubbed.mp4",
                    mime="video/mp4"
                )

        except Exception as e:
            st.error(f"Something went wrong: {e}")

        finally:
            # Cleanup temp files
            for f in [video_path, audio_path, output_path]:
                if os.path.exists(f):
                    os.remove(f)

    else:
        st.warning("Paste YouTube URL first")
