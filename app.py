import streamlit as st
import whisper
import os
import tempfile
import atexit
import shutil
import gc

from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs


st.title("🎬 AI English → Tamil Video Dubber")
st.write("Upload video and convert English speech to Tamil voice")


# ---------------- Whisper cache ----------------

@st.cache_resource
def load_whisper():

    return whisper.load_model("small")


model = load_whisper()



# ---------------- ElevenLabs ----------------

client = ElevenLabs(
    api_key=st.secrets["sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7"]
)


VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"



# ---------------- Session ----------------

if "tmpdir" not in st.session_state:
    st.session_state.tmpdir = None

if "output_path" not in st.session_state:
    st.session_state.output_path = None

if "download_ready" not in st.session_state:
    st.session_state.download_ready = False



# ---------------- Cleanup ----------------

def cleanup_tmpdir(path):

    if path and os.path.exists(path):

        shutil.rmtree(
            path,
            ignore_errors=True
        )



@atexit.register
def cleanup():

    cleanup_tmpdir(
        st.session_state.get("tmpdir")
    )




# ---------------- Translation ----------------


translator = GoogleTranslator(
    source="en",
    target="ta"
)



def translate_long_text(text):

    result = []

    for i in range(0,len(text),4000):

        part = text[i:i+4000]

        result.append(
            translator.translate(part)
        )

    return " ".join(result)




# ---------------- Upload ----------------


uploaded = st.file_uploader(
    "Upload video",
    type=[
        "mp4",
        "mkv",
        "avi",
        "mov",
        "webm"
    ]
)




if st.button("Start Tamil Dubbing"):


    if uploaded:


        cleanup_tmpdir(
            st.session_state.tmpdir
        )


        tmpdir = tempfile.mkdtemp()

        st.session_state.tmpdir = tmpdir



        ext = os.path.splitext(
            uploaded.name
        )[1]



        video_path = os.path.join(
            tmpdir,
            "video"+ext
        )


        audio_path = os.path.join(
            tmpdir,
            "tamil_audio.mp3"
        )


        output_path = os.path.join(
            tmpdir,
            "Tamil_Dubbed.mp4"
        )



        video = None
        video_audio = None
        final = None



        try:


            # save video

            st.info("Saving video...")


            with open(video_path,"wb") as f:

                f.write(
                    uploaded.read()
                )



            # speech recognition

            st.info("Converting speech...")


            result = model.transcribe(
                video_path,
                language="en",
                fp16=False
            )



            english_text = " ".join(
                [
                    x["text"]
                    for x in result["segments"]
                ]
            )



            st.success(
                "Transcript completed"
            )



            with st.expander(
                "English Transcript"
            ):

                st.write(
                    english_text
                )



            # translation

            st.info(
                "Translating to Tamil..."
            )



            tamil_text = translate_long_text(
                english_text
            )



            st.success(
                "Tamil translation completed"
            )



            # elevenlabs

            st.info(
                "Generating Tamil voice..."
            )


            tts_audio = client.text_to_speech.convert(

                voice_id=VOICE_ID,

                model_id="eleven_multilingual_v2",

                text=tamil_text

            )



            with open(audio_path,"wb") as f:


                for chunk in tts_audio:

                    f.write(chunk)



            st.success(
                "Voice created"
            )




            # merge


            st.info(
                "Creating final video..."
            )



            video = VideoFileClip(
                video_path
            )


            video_audio = AudioFileClip(
                audio_path
            )



            if video_audio.duration > video.duration:


                video_audio = video_audio.subclipped(
                    0,
                    video.duration
                )



            final = video.with_audio(
                video_audio
            )



            final.write_videofile(

                output_path,

                codec="libx264",

                audio_codec="aac",

                logger=None

            )



            st.success(
                "Tamil video ready 🎉"
            )



            st.session_state.output_path = output_path

            st.session_state.download_ready = True



        except Exception as e:


            st.error(
                f"Error: {e}"
            )



        finally:


            try:

                if final:
                    final.close()

                if video:
                    video.close()

                if video_audio:
                    video_audio.close()


            except:

                pass



            gc.collect()



    else:

        st.warning(
            "Upload video first"
        )




# ---------------- Download ----------------


if st.session_state.download_ready:


    path = st.session_state.output_path


    if os.path.exists(path):


        with open(path,"rb") as f:

            data = f.read()



        st.download_button(

            "⬇️ Download Tamil Dubbed Video",

            data,

            file_name="Tamil_Dubbed.mp4",

            mime="video/mp4"

        )
