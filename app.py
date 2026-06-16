import streamlit as st
import whisper
import os
import tempfile
import atexit
import shutil
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator

st.title("🎬 AI English → Tamil Video Dubber")
st.write("Upload a video file to get a Tamil dubbed version")

# ── Session-state setup ──────────────────────────────────────────────────────
if "tmpdir" not in st.session_state:
    st.session_state.tmpdir = None
if "output_path" not in st.session_state:
    st.session_state.output_path = None
if "download_ready" not in st.session_state:
    st.session_state.download_ready = False
if "download_clicked" not in st.session_state:
    st.session_state.download_clicked = False

def cleanup_tmpdir(path: str):
    """Remove the entire temp directory and everything inside."""
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

# Register cleanup on interpreter exit as a safety net
def _atexit_cleanup():
    cleanup_tmpdir(st.session_state.get("tmpdir"))

atexit.register(_atexit_cleanup)

# ── Auto-cleanup after download was clicked ──────────────────────────────────
if st.session_state.download_clicked:
    cleanup_tmpdir(st.session_state.tmpdir)
    st.session_state.tmpdir = None
    st.session_state.output_path = None
    st.session_state.download_ready = False
    st.session_state.download_clicked = False
    st.success("✅ Temporary files deleted automatically.")

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Video File",
    type=["mp4", "mkv", "avi", "mov", "webm"]
)

# ── Process ──────────────────────────────────────────────────────────────────
if st.button("Start Tamil Dubbing"):
    if uploaded:
        # Clean up any previous run's files first
        cleanup_tmpdir(st.session_state.tmpdir)

        tmpdir = tempfile.mkdtemp()
        st.session_state.tmpdir = tmpdir
        st.session_state.download_ready = False
        st.session_state.download_clicked = False

        ext = os.path.splitext(uploaded.name)[1]
        video_path  = os.path.join(tmpdir, f"video{ext}")
        audio_path  = os.path.join(tmpdir, "tamil_audio.mp3")
        output_path = os.path.join(tmpdir, "Tamil_Dubbed.mp4")

        try:
            st.info("Saving uploaded video...")
            with open(video_path, "wb") as f:
                f.write(uploaded.read())
            st.success("Video saved ✅")

            st.info("Converting speech to text… (this may take a few minutes)")
            model  = whisper.load_model("small")
            result = model.transcribe(video_path, language="en", fp16=False)
            english = [item["text"] for item in result["segments"]]
            st.success("Transcription done ✅")

            with st.expander("📝 English Transcript"):
                st.write(" ".join(english))

            st.info("Translating English → Tamil…")
            translator = GoogleTranslator(source="en", target="ta")
            tamil      = [translator.translate(s) for s in english]
            tamil_text = " ".join(tamil)
            st.success("Translation done ✅")

            with st.expander("🔤 Tamil Translation"):
                st.write(tamil_text)

            st.info("Generating Tamil voice…")
            gTTS(tamil_text, lang="ta").save(audio_path)
            st.success("Tamil audio generated ✅")

            st.info("Creating dubbed video…")
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
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

            # Store path so the download widget below can read it
            st.session_state.output_path  = output_path
            st.session_state.download_ready = True

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            cleanup_tmpdir(tmpdir)
            st.session_state.tmpdir = None

    else:
        st.warning("Please upload a video file first!")

# ── Download (shown even after a re-run, until clicked) ─────────────────────
if st.session_state.download_ready and st.session_state.output_path:
    output_path = st.session_state.output_path
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            video_bytes = f.read()          # read into memory before deletion

        clicked = st.download_button(
            "⬇️ Download Tamil Dubbed Video",
            video_bytes,                    # serve from memory
            file_name="Tamil_Dubbed.mp4",
            mime="video/mp4",
        )

        if clicked:
            # Flag for cleanup on next Streamlit rerun
            st.session_state.download_clicked = True
            st.rerun()
    else:
        st.warning("Output file not found. Please re-run the dubbing process.")
        st.session_state.download_ready = False
