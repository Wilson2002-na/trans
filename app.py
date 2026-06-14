import streamlit as st
import whisper
import os
import tempfile
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator

st.title("🎬 AI English → Tamil Video Dubber")
st.write("Upload a video file to get a Tamil dubbed version")

uploaded = st.file_uploader(
    "Upload Video File",
    type=["mp4", "mkv", "avi", "mov", "webm"]
)

if st.button("Start Tamil Dubbing"):
    if uploaded:
        tmpdir = tempfile.mkdtemp()
        ext = os.path.splitext(uploaded.name)[1]  # keep original extension
        video_path = os.path.join(tmpdir, f"video{ext}")
        audio_path = os.path.join(tmpdir, "tamil_audio.mp3")
        output_path = os.path.join(tmpdir, "Tamil_Dubbed.mp4")

        try:
            # Save uploaded file to disk
            st.info("Saving uploaded video...")
            with open(video_path, "wb") as f:
                f.write(uploaded.read())
            st.success("Video saved ✅")

            # Whisper transcription
            st.info("Converting speech to text... (this may take a few minutes)")
            model = whisper.load_model("small")
            result = model.transcribe(
                video_path,
                language="en",
                fp16=False
            )
            english = [item["text"] for item in result["segments"]]
            st.success("Transcription done ✅")

            # Show original English text
            with st.expander("📝 English Transcript"):
                st.write(" ".join(english))

            # Translate to Tamil
            st.info("Translating English to Tamil...")
            translator = GoogleTranslator(source="en", target="ta")
            tamil = [translator.translate(sentence) for sentence in english]
            tamil_text = " ".join(tamil)
            st.success("Translation done ✅")

            # Show Tamil text
            with st.expander("🔤 Tamil Translation"):
                st.write(tamil_text)

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
                logger=None
            )
            video.close()
            audio.close()
            st.success("Tamil Dubbed Video Ready 🎉")

            # Download button
            with open(output_path, "rb") as file:
                st.download_button(
                    "⬇️ Download Tamil Dubbed Video",
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
        st.warning("Please upload a video file first!")
