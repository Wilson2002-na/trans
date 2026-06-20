import streamlit as st
import whisper
import os
import sys
import subprocess
import tempfile
import atexit
import shutil

from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs


st.title("🎬 AI English → Tamil Lip-Synced Video Dubber")
st.write(
    "Upload a video file to get a Tamil dubbed version with the lips "
    "re-synced to the new audio (powered by Wav2Lip)."
)


# ── Settings ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    elevenlabs_key = st.text_input(
        "sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7",
        value=os.environ.get("sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7", ""),
        type="password",
        help=(
            "Set the ELEVENLABS_API_KEY environment variable instead of "
            "typing this each time. Never hardcode API keys in source code."
        ),
    )

    voice_id = st.text_input("ElevenLabs voice ID", value="CwhRBWXzGAHq8TQ4Fs17")

    st.markdown("---")
    st.subheader("Wav2Lip")

    wav2lip_repo = st.text_input(
        "Path to cloned Wav2Lip repo",
        value=os.environ.get("WAV2LIP_REPO", "Wav2Lip"),
    )

    checkpoint_path = st.text_input(
        "Path to checkpoint (.pth)",
        value=os.environ.get(
            "WAV2LIP_CHECKPOINT",
            os.path.join("Wav2Lip", "checkpoints", "wav2lip_gan.pth"),
        ),
    )

    resize_factor = st.slider(
        "Resize factor",
        min_value=1,
        max_value=4,
        value=2,
        help="Higher = faster but lower visual quality. On CPU, 2-3 is recommended.",
    )

    st.caption(
        "⚠️ Configured for CPU. Lip-sync is by far the slowest step here — "
        "budget roughly several minutes of processing per 10 seconds of footage, "
        "more for longer or higher-resolution clips."
    )


# ── Session-state setup ─────────────────────────────────────

if "tmpdir" not in st.session_state:
    st.session_state.tmpdir = None

if "output_path" not in st.session_state:
    st.session_state.output_path = None

if "download_ready" not in st.session_state:
    st.session_state.download_ready = False

if "download_clicked" not in st.session_state:
    st.session_state.download_clicked = False


def cleanup_tmpdir(path):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _atexit_cleanup():
    cleanup_tmpdir(st.session_state.get("tmpdir"))


atexit.register(_atexit_cleanup)


def run_wav2lip(video_path, audio_path, output_path, repo_dir, ckpt_path, resize_factor=2):
    """
    Runs Wav2Lip's inference.py as a subprocess to produce a lip-synced video.
    Wav2Lip handles audio via librosa/soundfile and the final mux via ffmpeg —
    no pydub is involved anywhere in this pipeline.
    """
    inference_script = os.path.join(repo_dir, "inference.py")

    if not os.path.isfile(inference_script):
        raise FileNotFoundError(
            f"Wav2Lip not found at '{repo_dir}'. Clone it first:\n"
            "git clone https://github.com/Rudrabha/Wav2Lip"
        )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'. Download wav2lip_gan.pth "
            "(or wav2lip.pth) and place it at that path."
        )

    cmd = [
        sys.executable,
        "inference.py",
        "--checkpoint_path", os.path.abspath(ckpt_path),
        "--face", os.path.abspath(video_path),
        "--audio", os.path.abspath(audio_path),
        "--outfile", os.path.abspath(output_path),
        "--resize_factor", str(resize_factor),
        "--nosmooth",
    ]

    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-3000:]
        raise RuntimeError(f"Wav2Lip failed:\n{tail}")

    if not os.path.isfile(output_path):
        raise RuntimeError("Wav2Lip ran but no output file was produced.")


# cleanup after download

if st.session_state.download_clicked:
    cleanup_tmpdir(st.session_state.tmpdir)
    st.session_state.tmpdir = None
    st.session_state.output_path = None
    st.session_state.download_ready = False
    st.session_state.download_clicked = False
    st.success("Temporary files deleted")


# upload

uploaded = st.file_uploader(
    "Upload Video File",
    type=["mp4", "mkv", "avi", "mov", "webm"],
)


if st.button("Start Tamil Dubbing + Lip-Sync"):

    if not elevenlabs_key:
        st.warning("Add your ElevenLabs API key in the sidebar first.")

    elif uploaded:

        cleanup_tmpdir(st.session_state.tmpdir)

        tmpdir = tempfile.mkdtemp()
        st.session_state.tmpdir = tmpdir

        ext = os.path.splitext(uploaded.name)[1]
        video_path = os.path.join(tmpdir, "video" + ext)
        audio_path = os.path.join(tmpdir, "tamil_audio.mp3")
        output_path = os.path.join(tmpdir, "Tamil_Dubbed_LipSync.mp4")

        try:
            # save video
            st.info("Saving video...")
            with open(video_path, "wb") as f:
                f.write(uploaded.read())
            st.success("Video saved")

            # whisper
            st.info("Converting speech to text...")
            model = whisper.load_model("small")
            result = model.transcribe(video_path, language="en", fp16=False)
            english_text = " ".join(x["text"] for x in result["segments"])
            st.success("Transcript done")

            with st.expander("English Transcript"):
                st.write(english_text)

            # translate
            st.info("Translating English → Tamil")
            translator = GoogleTranslator(source="en", target="ta")
            tamil_text = translator.translate(english_text)
            st.success("Tamil translation done")

            with st.expander("Tamil Text"):
                st.write(tamil_text)

            # ElevenLabs voice
            st.info("Generating Tamil AI voice...")
            client = ElevenLabs(api_key=elevenlabs_key)
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                text=tamil_text,
            )
            with open(audio_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)
            st.success("Tamil voice created")

            # lip-sync
            st.info("Running Wav2Lip lip-sync — this is the slow part on CPU, please wait...")
            run_wav2lip(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                repo_dir=wav2lip_repo,
                ckpt_path=checkpoint_path,
                resize_factor=resize_factor,
            )
            st.success("Tamil Lip-Synced Video Ready 🎉")

            st.session_state.output_path = output_path
            st.session_state.download_ready = True

        except Exception as e:
            st.error(f"Error: {e}")
            cleanup_tmpdir(tmpdir)
            st.session_state.tmpdir = None

    else:
        st.warning("Upload video first")


# download

if st.session_state.download_ready and st.session_state.output_path:

    if os.path.exists(st.session_state.output_path):

        with open(st.session_state.output_path, "rb") as f:
            video_bytes = f.read()

        clicked = st.download_button(
            "⬇️ Download Tamil Lip-Synced Video",
            video_bytes,
            file_name="Tamil_Dubbed_LipSync.mp4",
            mime="video/mp4",
        )

        if clicked:
            st.session_state.download_clicked = True
            st.rerun()
